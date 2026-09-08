"""
Controle qualite RGPD par comparaison de DEUX fichiers : le pivot
ORIGINAL (jamais modifie) et le fichier de sortie ANONYMISE (ecrit par
`AnonymiserDatasetUseCase` dans un fichier separe). Croise par
`identifiant`, compare `texte_original` (pivot) et `texte_anonymise`
(sortie) champ par champ pour un echantillon stratifie.

Design (08/09/2026, decision du capitaine -- remplace un enganche en
direct dans la boucle d'anonymisation) : puisque le pivot original
n'est plus jamais mute, ce controle peut se relancer a tout moment sur
n'importe quelle tranche deja anonymisee -- y compris retroactivement
sur une vague anonymisee il y a longtemps, tant que le pivot original
existe encore. Aucune dependance au moment precis de l'anonymisation.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from chsa_triage.application.detection_pii_residuelle import (
    CATEGORIES_DETERMINISTES,
    CandidatRegex,
    detecter_candidats,
)
from chsa_triage.application.echantillonnage import echantillon_stratifie
from src.chsa_triage.application.use_cases.uc_03_anonymiser_dataset import StatistiquesSource
from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.ports import RepositoryLectureEcriture
from chsa_triage.domain.ports.verificateur_entites import (
    VerdictEntiteNommee,
    VerificateurEntitesNommees,
)

VERDICT_CONFIRME           = "confirme"
VERDICT_FAUX_POSITIF_REGEX = "faux_positif_regex_ecarte_par_spacy"
VERDICT_REVISION_HUMAINE   = "pendant_revision_humaine"

# Jeton de masquage par defaut de PresidioAnonymiseur (strategie
# "replace", la strategie retenue par la mission -- cf.
# `infrastructure/adapters/presidio_anonymiseur.py`). La detection de
# faux positifs de masquage ci-dessous ne fonctionne que pour cette
# strategie (jeton fixe identifiable dans un diff) ; documente comme
# limite connue si jamais --strategie mask/redact est utilisee.
JETON_MASQUE_DEFAUT = "<INFO_MASQUEE>"


@dataclass(frozen=True, slots=True)
class CandidatPiiResiduelle:
    """Un passage du texte ANONYMISE signale par regex comme PII potentiellement residuelle."""

    identifiant : str
    source       : str
    champ         : str
    langue         : str
    type_motif      : str
    passage          : str
    verdict           : str  # VERDICT_CONFIRME | VERDICT_FAUX_POSITIF_REGEX | VERDICT_REVISION_HUMAINE


@dataclass(frozen=True, slots=True)
class ExempleControle:
    """Un exemple reel original -> anonymise, conserve pour inspection manuelle."""

    identifiant       : str
    source             : str
    champ               : str
    texte_original        : str
    texte_anonymise        : str


@dataclass(frozen=True, slots=True)
class CandidatFauxPositifAnonymisation:
    """Un fragment du texte ORIGINAL masque dans l'anonymise, sans confirmation spaCy qu'il s'agissait d'une entite nommee."""

    identifiant       : str
    source             : str
    champ               : str
    fragment_masque       : str
    texte_original          : str
    texte_anonymise           : str
    verdict                    : str  # VERDICT_FAUX_POSITIF_REGEX | VERDICT_REVISION_HUMAINE (jamais CONFIRME -- cf. observer())


def _paires_champs(original: ExemplePivot, anonymise: ExemplePivot) -> list[tuple[str, str, str]]:
    """Aligne les champs texte libre de deux ExemplePivot (meme identifiant) : (nom_champ, original, anonymise)."""
    paires = [("symptomes", original.symptomes, anonymise.symptomes)]
    if original.antecedents:
        paires.append(("antecedents", original.antecedents, anonymise.antecedents or ""))
    for nom_groupe in ("prompt", "completion", "chosen", "rejected"):
        messages_original = getattr(original, nom_groupe)
        messages_anonymise = getattr(anonymise, nom_groupe)
        for index, (message_original, message_anonymise) in enumerate(zip(messages_original, messages_anonymise)):
            paires.append((f"{nom_groupe}[{index}]", message_original.contenu, message_anonymise.contenu))
    return paires


def _extraire_fragments_masques(
    texte_original: str, texte_anonymise: str, jeton_masque: str
) -> list[tuple[str, int, int]]:
    """
    Aligne texte_original/texte_anonymise (difflib) et retourne, pour
    chaque occurrence du jeton de masquage cote anonymise, le fragment
    original qu'il a remplace avec sa position dans texte_original.
    """
    if not texte_original or not texte_anonymise or jeton_masque not in texte_anonymise:
        return []
    matcher = difflib.SequenceMatcher(None, texte_original, texte_anonymise, autojunk=False)
    fragments: list[tuple[str, int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and texte_anonymise[j1:j2] == jeton_masque:
            fragments.append((texte_original[i1:i2], i1, i2))
    return fragments


@dataclass(slots=True)
class ControleQualiteAnonymisation:
    """Accumulateur des observations de controle qualite -- pas d'I/O ici."""

    verificateur_entites          : VerificateurEntitesNommees
    max_exemples_par_source         : int = 10
    max_faux_positifs_par_source     : int = 10
    jeton_masque                      : str = JETON_MASQUE_DEFAUT

    nombre_exemples_observes          : int = field(default=0, init=False)
    exemples_par_source                : dict[str, list[ExempleControle]] = field(default_factory=dict, init=False)
    candidats_pii                       : list[CandidatPiiResiduelle] = field(default_factory=list, init=False)
    candidats_faux_positifs              : list[CandidatFauxPositifAnonymisation] = field(default_factory=list, init=False)

    def observer(self, original: ExemplePivot, anonymise: ExemplePivot) -> None:
        """Compare un couple (original, anonymise) partageant le meme identifiant, champ par champ."""
        self.nombre_exemples_observes += 1
        langue = original.langue.value

        for nom_champ, texte_original, texte_anonymise in _paires_champs(original, anonymise):
            self._detecter_pii_residuelle(original, nom_champ, texte_anonymise, langue)
            self._detecter_candidats_faux_positifs(original, nom_champ, texte_original, texte_anonymise, langue)

        self._conserver_exemple_illustratif(original, anonymise)

    def _conserver_exemple_illustratif(self, original: ExemplePivot, anonymise: ExemplePivot) -> None:
        exemples_source = self.exemples_par_source.setdefault(original.source, [])
        if len(exemples_source) >= self.max_exemples_par_source:
            return
        for nom_champ, texte_original, texte_anonymise in _paires_champs(original, anonymise):
            if texte_original:
                exemples_source.append(
                    ExempleControle(
                        identifiant=original.identifiant,
                        source=original.source,
                        champ=nom_champ,
                        texte_original=texte_original,
                        texte_anonymise=texte_anonymise,
                    )
                )
                return

    def _detecter_pii_residuelle(self, original: ExemplePivot, nom_champ: str, texte_anonymise: str, langue: str) -> None:
        for candidat in detecter_candidats(texte_anonymise):
            verdict = self._trancher_candidat(candidat, texte_anonymise, langue)
            self.candidats_pii.append(
                CandidatPiiResiduelle(
                    identifiant=original.identifiant,
                    source=original.source,
                    champ=nom_champ,
                    langue=langue,
                    type_motif=candidat.type_motif,
                    passage=candidat.passage,
                    verdict=verdict,
                )
            )

    def _trancher_candidat(self, candidat: CandidatRegex, texte_anonymise: str, langue: str) -> str:
        if candidat.type_motif in CATEGORIES_DETERMINISTES:
            # email/telephone/url/date : match non ambigu, confirme directement.
            return VERDICT_CONFIRME
        verdict_spacy = self.verificateur_entites.verifier(texte_anonymise, langue, candidat.debut, candidat.fin)
        if verdict_spacy is VerdictEntiteNommee.ENTITE_PERTINENTE:
            return VERDICT_CONFIRME
        if verdict_spacy is VerdictEntiteNommee.AUCUNE_ENTITE:
            return VERDICT_FAUX_POSITIF_REGEX
        return VERDICT_REVISION_HUMAINE

    def _detecter_candidats_faux_positifs(
        self, original: ExemplePivot, nom_champ: str, texte_original: str, texte_anonymise: str, langue: str
    ) -> None:
        deja_pour_source = sum(1 for c in self.candidats_faux_positifs if c.source == original.source)
        if deja_pour_source >= self.max_faux_positifs_par_source:
            return
        for fragment, debut, fin in _extraire_fragments_masques(texte_original, texte_anonymise, self.jeton_masque):
            if not fragment.strip():
                continue
            verdict_spacy = self.verificateur_entites.verifier(texte_original, langue, debut, fin)
            if verdict_spacy is VerdictEntiteNommee.ENTITE_PERTINENTE:
                continue  # spaCy confirme une entite nommee -- masquage juge legitime, pas un faux positif
            verdict = (
                VERDICT_FAUX_POSITIF_REGEX
                if verdict_spacy is VerdictEntiteNommee.AUCUNE_ENTITE
                else VERDICT_REVISION_HUMAINE
            )
            self.candidats_faux_positifs.append(
                CandidatFauxPositifAnonymisation(
                    identifiant=original.identifiant,
                    source=original.source,
                    champ=nom_champ,
                    fragment_masque=fragment,
                    texte_original=texte_original,
                    texte_anonymise=texte_anonymise,
                    verdict=verdict,
                )
            )
            deja_pour_source += 1
            if deja_pour_source >= self.max_faux_positifs_par_source:
                break


@dataclass(slots=True)
class ControlerQualiteAnonymisationUseCase:
    """
    Orchestre le controle qualite : lit le pivot original et le
    fichier anonymise, preleve un echantillon stratifie parmi les
    exemples anonymises disponibles, et compare chaque couple via
    `ControleQualiteAnonymisation.observer`.
    """

    repository_original          : RepositoryLectureEcriture
    repository_anonymise           : RepositoryLectureEcriture
    verificateur_entites             : VerificateurEntitesNommees
    taille_echantillon                : int | None = 200
    graine_aleatoire                   : int = 42
    max_exemples_par_source              : int = 10
    max_faux_positifs_par_source          : int = 10
    jeton_masque                           : str = JETON_MASQUE_DEFAUT

    nombre_introuvables_dans_original        : int = field(default=0, init=False)

    def executer(self) -> ControleQualiteAnonymisation:
        originaux_par_id = {e.identifiant: e for e in self.repository_original.lister()}
        anonymises = list(self.repository_anonymise.lister())

        if self.taille_echantillon is not None and self.taille_echantillon < len(anonymises):
            echantillon = echantillon_stratifie(anonymises, self.taille_echantillon, self.graine_aleatoire)
        else:
            echantillon = anonymises

        controle = ControleQualiteAnonymisation(
            verificateur_entites=self.verificateur_entites,
            max_exemples_par_source=self.max_exemples_par_source,
            max_faux_positifs_par_source=self.max_faux_positifs_par_source,
            jeton_masque=self.jeton_masque,
        )

        self.nombre_introuvables_dans_original = 0
        for exemple_anonymise in echantillon:
            exemple_original = originaux_par_id.get(exemple_anonymise.identifiant)
            if exemple_original is None:
                # Garde-fou : ne devrait jamais arriver si --sortie a
                # bien ete alimente depuis --dataset par AnonymiserDatasetUseCase.
                self.nombre_introuvables_dans_original += 1
                continue
            controle.observer(exemple_original, exemple_anonymise)

        return controle


# ----------------------------------------------------------------------
# Serialisation JSON + presentation Markdown -- pas d'I/O fichier ici
# (a la charge de l'appelant, comme pour rapport_anonymisation.py).
# ----------------------------------------------------------------------


def controle_vers_dict(
    controle: ControleQualiteAnonymisation,
    horodatage: str,
    dataset_original: str,
    dataset_anonymise: str,
) -> dict:
    """Serialise l'accumulateur en dict JSON, pour inspection programmatique (pas seulement le Markdown)."""
    return {
        "horodatage": horodatage,
        "dataset_original": dataset_original,
        "dataset_anonymise": dataset_anonymise,
        "nombre_exemples_observes": controle.nombre_exemples_observes,
        "candidats_pii_residuelle": [
            {
                "identifiant": c.identifiant,
                "source": c.source,
                "champ": c.champ,
                "langue": c.langue,
                "type_motif": c.type_motif,
                "passage": c.passage,
                "verdict": c.verdict,
            }
            for c in controle.candidats_pii
        ],
        "candidats_faux_positifs": [
            {
                "identifiant": c.identifiant,
                "source": c.source,
                "champ": c.champ,
                "fragment_masque": c.fragment_masque,
                "texte_original": c.texte_original,
                "texte_anonymise": c.texte_anonymise,
                "verdict": c.verdict,
            }
            for c in controle.candidats_faux_positifs
        ],
        "exemples_par_source": {
            source: [
                {
                    "identifiant": e.identifiant,
                    "champ": e.champ,
                    "texte_original": e.texte_original,
                    "texte_anonymise": e.texte_anonymise,
                }
                for e in exemples
            ]
            for source, exemples in controle.exemples_par_source.items()
        },
    }


def formater_rapport_markdown(
    controle: ControleQualiteAnonymisation,
    horodatage: str,
    dataset_original: str,
    dataset_anonymise: str,
    taille_echantillon_demandee: int | None,
    total_anonymise_disponible: int,
    statistiques_cumulees: dict[str, StatistiquesSource],
) -> str:
    """
    Rapport Markdown du controle qualite pour L'ECHANTILLON compare
    lors de cette execution -- pas un cumul persistant (a la
    difference du rapport RGPD de la Partie 1) : chaque execution de
    `controler_qualite_anonymisation.py` compare un echantillon frais,
    stratifie parmi TOUT ce qui est anonymise au moment de l'appel
    (y compris retroactivement sur d'anciennes vagues, puisque le
    pivot original n'est jamais modifie).
    """
    total_confirmes = sum(1 for c in controle.candidats_pii if c.verdict == VERDICT_CONFIRME)
    total_faux_positifs_regex = sum(1 for c in controle.candidats_pii if c.verdict == VERDICT_FAUX_POSITIF_REGEX)
    total_revision_humaine = sum(1 for c in controle.candidats_pii if c.verdict == VERDICT_REVISION_HUMAINE)

    total_masquages_faux_positifs = sum(
        1 for c in controle.candidats_faux_positifs if c.verdict == VERDICT_FAUX_POSITIF_REGEX
    )
    total_masquages_a_revoir = sum(
        1 for c in controle.candidats_faux_positifs if c.verdict == VERDICT_REVISION_HUMAINE
    )

    lignes = [
        "# Rapport de controle qualite -- anonymisation (comparaison original/anonymise)",
        "",
        f"> Genere automatiquement le {horodatage} par `controler_qualite_anonymisation.py`, "
        f"a partir de `{dataset_original}` (original) compare a `{dataset_anonymise}` (anonymise).",
        ">",
        f"> **Portee explicite** : {controle.nombre_exemples_observes} exemples compares -- un "
        f"echantillon stratifie (type_exemple, source)"
        + (
            f" de taille demandee {taille_echantillon_demandee}"
            if taille_echantillon_demandee is not None
            else ""
        )
        + f" parmi les {total_anonymise_disponible} exemples disponibles dans `{dataset_anonymise}` au "
        "moment de cette execution -- PAS un cumul persistant entre executions (relancer ce script "
        "compare un nouvel echantillon frais a chaque fois). Comme le pivot original n'est jamais "
        "modifie, ce controle peut porter sur n'importe quelle vague deja anonymisee, y compris "
        "retroactivement.",
        "",
        "## 1. Compteurs par categorie d'entite anonymisee (rapport RGPD cumule, Partie 1)",
        "",
        "Reutilise directement `statistiques_cumulees` du rapport RGPD cumule -- aucun recalcul ici "
        "(ce controle ne relance pas Presidio, il compare des textes deja anonymises).",
        "",
        "| Source | Registres traites (cumule) | Avec >=1 entite | Entites par type |",
        "|---|---:|---:|---|",
    ]
    for source in sorted(statistiques_cumulees):
        stats = statistiques_cumulees[source]
        detail = ", ".join(
            f"{t} {n}" for t, n in sorted(stats.entites_par_type.items(), key=lambda kv: -kv[1])
        ) or "(aucune)"
        lignes.append(f"| {source} | {stats.registres_traites} | {stats.registres_avec_entite} | {detail} |")
    if not statistiques_cumulees:
        lignes.append("| (rapport RGPD cumule introuvable ou vide) | -- | -- | -- |")

    lignes += [
        "",
        "## 2. Candidats de PII residuelle (regex sur texte anonymise + seconde opinion spaCy)",
        "",
        f"- Candidats bruts detectes par regex : **{len(controle.candidats_pii)}**.",
        f"- Confirmes (match deterministe email/telephone/url/date, ou bigramme capitalise que "
        f"spaCy reconnait comme entite nommee) : **{total_confirmes}**.",
        f"- Ecartes par la seconde opinion spaCy (bigramme capitalise sans entite nommee detectee -- "
        f"probable terme medical, pas une PII) : **{total_faux_positifs_regex}**.",
        f"- En attente de revision humaine (spaCy detecte une entite mais d'un type non tranchant) : "
        f"**{total_revision_humaine}**.",
        "",
        "### Passages confirmes",
    ]
    lignes += _lister_candidats_pii(controle.candidats_pii, VERDICT_CONFIRME)
    lignes += ["", "### Passages en attente de revision humaine"]
    lignes += _lister_candidats_pii(controle.candidats_pii, VERDICT_REVISION_HUMAINE)

    lignes += [
        "",
        "## 3. Candidats de faux positifs de l'anonymisation (termes masques sans necessite)",
        "",
        f"Detectes par diff texte-original/texte-anonymise (fragments remplaces par le jeton "
        f"`{controle.jeton_masque}`), puis seconde opinion spaCy sur le fragment ORIGINAL masque : "
        "si spaCy ne reconnait aucune entite nommee a cet endroit, le masquage est probablement "
        "un faux positif (terme medical/scientifique pris pour un nom propre). **Ce ne sont pas des "
        "faux positifs confirmes** -- seulement des candidats a verifier humainement.",
        "",
        f"- Fragments masques trouves sans confirmation spaCy : **{len(controle.candidats_faux_positifs)}** "
        f"({total_masquages_faux_positifs} sans aucune entite detectee, {total_masquages_a_revoir} "
        "avec une entite d'un type non tranchant).",
        "",
    ]
    for candidat in controle.candidats_faux_positifs:
        lignes.append(
            f"- `{candidat.identifiant}` ({candidat.source}, champ `{candidat.champ}`, "
            f"verdict `{candidat.verdict}`) : fragment masque {candidat.fragment_masque!r}"
        )
    if not controle.candidats_faux_positifs:
        lignes.append("- (aucun)")

    lignes += ["", "## 4. Exemples reels (original -> anonymise) par source", ""]
    for source in sorted(controle.exemples_par_source):
        lignes.append(f"### {source}")
        for exemple in controle.exemples_par_source[source]:
            lignes.append(f"- `{exemple.identifiant}` (champ `{exemple.champ}`) :")
            lignes.append(f"  - original : {exemple.texte_original[:200]!r}")
            lignes.append(f"  - anonymise : {exemple.texte_anonymise[:200]!r}")
        lignes.append("")

    return "\n".join(lignes)


def _lister_candidats_pii(candidats: list[CandidatPiiResiduelle], verdict: str) -> list[str]:
    lignes = []
    for candidat in candidats:
        if candidat.verdict != verdict:
            continue
        lignes.append(
            f"- `{candidat.identifiant}` ({candidat.source}, champ `{candidat.champ}`, "
            f"motif `{candidat.type_motif}`) : {candidat.passage!r}"
        )
    if not lignes:
        lignes.append("- (aucun)")
    return lignes
