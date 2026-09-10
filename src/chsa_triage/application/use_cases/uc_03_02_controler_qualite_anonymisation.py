"""
Controle qualite RGPD par comparaison de DEUX fichiers : le pivot
ORIGINAL (jamais modifie) et le fichier de sortie ANONYMISE (ecrit par
`AnonymiserDatasetUseCase` dans un fichier separe). Croise par
`identifiant`, compare `texte_original` (pivot) et `texte_anonymise`
(sortie) champ par champ pour un echantillon stratifie.

Design (08/09/2026 ; remplace un enganche en
direct dans la boucle d'anonymisation) : puisque le pivot original
n'est plus jamais mute, ce controle peut se relancer a tout moment sur
n'importe quelle tranche deja anonymisee, y compris retroactivement
sur une vague anonymisee il y a longtemps, tant que le pivot original
existe encore. Aucune dependance au moment precis de l'anonymisation.
"""

from __future__ import annotations

import difflib  # diff par opcodes entre texte original et anonymise
import random   # tirage aleatoire du stratum "sans entite detectee"
from datetime import datetime, timezone  # horodatage du muestreo incremental persiste

# Structures de donnees immuables
from dataclasses import dataclass, field  # dataclasses figees (accumulateur, resultats)

# Detection de PII residuelle et seconde opinion sur les entites nommees
from chsa_triage.application.detection_pii_residuelle import (  # regex : motifs deterministes + candidats
    CATEGORIES_DETERMINISTES,
    CandidatRegex,
    detecter_candidats,
)
from chsa_triage.domain.ports.verificateur_entites import (  # port : seconde opinion spaCy
    VerdictEntiteNommee,
    VerificateurEntitesNommees,
)

# Echantillonnage stratifie de l'echantillon compare
from chsa_triage.application.echantillonnage import echantillon_stratifie  # tirage stratifie (type_exemple, source)

# Modele pivot, ports du domaine et statistiques RGPD cumulees
from chsa_triage.application.use_cases.uc_03_00_anonymiser_dataset import StatistiquesSource  # stats cumulees (Partie 1)
from chsa_triage.domain.model import (  # entite pivot comparee + cle stable des decisions humaines
    DECISION_ACCEPTE,
    DECISION_REJETE,
    SOURCE_CANDIDATS_FAUX_POSITIFS,
    SOURCE_CANDIDATS_PII,
    SOURCE_CANDIDATS_PII_SANS_ENTITE,
    CleCandidatRevision,
    ExemplePivot,
)
from chsa_triage.domain.ports import RepositoryLectureEcriture  # port de lecture/ecriture generique
from chsa_triage.domain.ports.registre_echantillons_controle_qualite import (  # port du muestreo incremental
    RegistreEchantillonsControleQualite,
)

VERDICT_CONFIRME           = "confirme"
VERDICT_FAUX_POSITIF_REGEX = "faux_positif_regex_ecarte_par_spacy"
VERDICT_REVISION_HUMAINE   = "pendant_revision_humaine"

# Strates du muestreo incremental persiste (cf. RegistreEchantillonsControleQualite) --
# INDEPENDANTES l'une de l'autre, comme les deux tirages qu'elles recouvrent
# (echantillon principal vs stratum dedie "sans entite detectee").
STRATUM_PRINCIPAL     = "principal"
STRATUM_SANS_ENTITE   = "sans_entite"

# Jeton de masquage par defaut de PresidioAnonymiseur (strategie
# "replace", la strategie retenue par la mission ; cf.
# `infrastructure/adapters/presidio_anonymiseur.py`). La detection de
# faux positifs de masquage ci-dessous ne fonctionne que pour cette
# strategie (jeton fixe identifiable dans un diff) ; documente comme
# limite connue si jamais --strategie mask/redact est utilisee.
JETON_MASQUE_DEFAUT = "<INFO_MASQUEE>"


@dataclass(frozen=True, slots=True)
class CandidatPiiResiduelle:
    """Un passage du texte ANONYMISE signale par regex comme PII potentiellement residuelle."""

    identifiant : str
    source      : str
    champ       : str
    langue      : str
    type_motif  : str
    passage     : str
    verdict     : str  # VERDICT_CONFIRME | VERDICT_FAUX_POSITIF_REGEX | VERDICT_REVISION_HUMAINE
    # Position [debut:fin] du match dans le texte ANONYMISE ; desambiguise
    # plusieurs matches du meme type_motif dans le meme champ (confirme sur
    # donnees reelles : jusqu'a 17 matches de bigramme_capitalise dans un
    # seul champ chosen[0]) pour la cle stable de revision humaine (cf.
    # `application.use_cases.uc_03_03_reviser_pii_residuelle.cle_candidat_pii` --
    # ce dataclass sert a la fois a `candidats_pii` et
    # `candidats_pii_sans_entite`, deux SOURCE_* differentes, donc la cle
    # complete ne peut pas etre calculee ici sans savoir dans quelle liste
    # l'appelant l'a range).
    debut       : int = 0
    fin         : int = 0


@dataclass(frozen=True, slots=True)
class ExempleControle:
    """Un exemple reel original -> anonymise, conserve pour inspection manuelle."""

    identifiant     : str
    source          : str
    champ           : str
    texte_original  : str
    texte_anonymise : str


@dataclass(frozen=True, slots=True)
class CandidatFauxPositifAnonymisation:
    """
    Un fragment du texte ORIGINAL masque dans l'anonymise, sans
    confirmation spaCy qu'il s'agissait d'une entite nommee.
    """

    identifiant     : str
    source          : str
    champ           : str
    fragment_masque : str
    texte_original  : str
    texte_anonymise : str
    verdict         : str  # VERDICT_FAUX_POSITIF_REGEX | VERDICT_REVISION_HUMAINE (jamais CONFIRME ; cf. observer())
    # Position [debut:fin] de fragment_masque dans texte_original (pas
    # regex-type ; type_motif="" dans la cle stable de revision humaine).
    debut           : int = 0
    fin             : int = 0


# ##############################################################################
# _paires_champs
# ##############################################################################
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


# ##############################################################################
# _est_exemple_sans_entite
# ##############################################################################
def _est_exemple_sans_entite(original: ExemplePivot, anonymise: ExemplePivot) -> bool:
    """
    True si AUCUN champ texte libre n'a change entre original et
    anonymise, c'est-a-dire que Presidio n'a RIEN detecte du tout
    sur cet exemple (item 3 : distingue explicitement "rien detecte"
    de "quelque chose detecte", ce que l'echantillonnage stratifie
    (type_exemple, source) seul ne fait pas ; il peut tres bien ne
    jamais tirer un exemple "propre" par pur hasard).
    """
    paires = _paires_champs(original, anonymise)
    if not paires:
        return True
    return all(texte_original == texte_anonymise for _, texte_original, texte_anonymise in paires)


# ##############################################################################
# _premier_champ_non_vide
# ##############################################################################
def _premier_champ_non_vide(original: ExemplePivot, anonymise: ExemplePivot) -> ExempleControle | None:
    for nom_champ, texte_original, texte_anonymise in _paires_champs(original, anonymise):
        if texte_original:
            return ExempleControle(
                identifiant=original.identifiant,
                source=original.source,
                champ=nom_champ,
                texte_original=texte_original,
                texte_anonymise=texte_anonymise,
            )
    return None


# ##############################################################################
# _extraire_fragments_masques
# ##############################################################################
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
    """Accumulateur des observations de controle qualite ; pas d'I/O ici."""

    verificateur_entites             : VerificateurEntitesNommees
    max_exemples_par_source          : int = 10
    # None = pas de plafond ; necessaire pour `ReviserPiiResiduelleUseCase`
    # (cf. uc_03_03) qui doit voir TOUS les candidats REVISION_HUMAINE, pas
    # seulement les `max_faux_positifs_par_source` premiers par source
    # (plafond pense pour la LISIBILITE du rapport Markdown, pas pour la
    # completude de la revue humaine).
    max_faux_positifs_par_source     : int | None = 10
    jeton_masque                     : str = JETON_MASQUE_DEFAUT

    nombre_exemples_observes : int = field(default=0, init=False)
    exemples_par_source      : dict[str, list[ExempleControle]] = field(default_factory=dict, init=False)
    candidats_pii            : list[CandidatPiiResiduelle] = field(default_factory=list, init=False)
    candidats_faux_positifs  : list[CandidatFauxPositifAnonymisation] = field(default_factory=list, init=False)

    # Stratum dedie "sans entite detectee" (item 3) ; compteurs et
    # listes SEPARES du reste : jamais melanges a candidats_pii /
    # nombre_exemples_observes ci-dessus. cf. `observer_sans_entite`.
    nombre_disponibles_sans_entite       : int = field(default=0, init=False)
    nombre_exemples_sans_entite_observes : int = field(default=0, init=False)
    candidats_pii_sans_entite            : list[CandidatPiiResiduelle] = field(default_factory=list, init=False)
    exemples_sans_entite                 : list[ExempleControle] = field(default_factory=list, init=False)

    # ##########################################################################
    # observer
    # ##########################################################################
    def observer(self, original: ExemplePivot, anonymise: ExemplePivot) -> None:
        """Compare un couple (original, anonymise) partageant le meme identifiant, champ par champ."""
        self.nombre_exemples_observes += 1
        langue = original.langue.value

        for nom_champ, texte_original, texte_anonymise in _paires_champs(original, anonymise):
            self._detecter_pii_residuelle(original, nom_champ, texte_anonymise, langue)
            self._detecter_candidats_faux_positifs(original, nom_champ, texte_original, texte_anonymise, langue)

        self._conserver_exemple_illustratif(original, anonymise)

    # ##########################################################################
    # observer_sans_entite
    # ##########################################################################
    def observer_sans_entite(self, original: ExemplePivot, anonymise: ExemplePivot) -> None:
        """
        Stratum dedie "sans entite detectee" (item 3, cf. rapport RGPD
        §7) : un couple (original, anonymise) ou AUCUN champ n'a
        change ; Presidio n'a rien detecte du tout. Comptabilise a
        part de `observer()` ci-dessus (jamais melange a
        nombre_exemples_observes / candidats_pii) : le but n'est pas
        de mesurer le sur-masquage mais de verifier, via la meme
        heuristique regex + seconde opinion spaCy appliquee au texte
        NON MODIFIE, si un motif de PII evident (email/telephone/
        url/date/bigramme capitalise) est neanmoins present, ce qui
        signalerait un FAUX NEGATIF COMPLET (rien detecte alors qu'il
        aurait fallu detecter quelque chose), distinct d'un sur/
        sous-masquage partiel.
        """
        self.nombre_exemples_sans_entite_observes += 1
        langue = original.langue.value

        for nom_champ, _texte_original, texte_anonymise in _paires_champs(original, anonymise):
            self.candidats_pii_sans_entite.extend(
                self._detecter_pii_residuelle_champ(original, nom_champ, texte_anonymise, langue)
            )

        exemple = _premier_champ_non_vide(original, anonymise)
        if exemple is not None:
            self.exemples_sans_entite.append(exemple)

    # ##########################################################################
    # _conserver_exemple_illustratif
    # ##########################################################################
    def _conserver_exemple_illustratif(self, original: ExemplePivot, anonymise: ExemplePivot) -> None:
        exemples_source = self.exemples_par_source.setdefault(original.source, [])
        if len(exemples_source) >= self.max_exemples_par_source:
            return
        exemple = _premier_champ_non_vide(original, anonymise)
        if exemple is not None:
            exemples_source.append(exemple)

    # ##########################################################################
    # _detecter_pii_residuelle
    # ##########################################################################
    def _detecter_pii_residuelle(self, original: ExemplePivot, nom_champ: str, texte_anonymise: str, langue: str) -> None:
        self.candidats_pii.extend(self._detecter_pii_residuelle_champ(original, nom_champ, texte_anonymise, langue))

    # ##########################################################################
    # _detecter_pii_residuelle_champ
    # ##########################################################################
    def _detecter_pii_residuelle_champ(
        self, original: ExemplePivot, nom_champ: str, texte_anonymise: str, langue: str
    ) -> list[CandidatPiiResiduelle]:
        resultats = []
        for candidat in detecter_candidats(texte_anonymise):
            verdict = self._trancher_candidat(candidat, texte_anonymise, langue)
            resultats.append(
                CandidatPiiResiduelle(
                    identifiant=original.identifiant,
                    source=original.source,
                    champ=nom_champ,
                    langue=langue,
                    type_motif=candidat.type_motif,
                    passage=candidat.passage,
                    verdict=verdict,
                    debut=candidat.debut,
                    fin=candidat.fin,
                )
            )
        return resultats

    # ##########################################################################
    # _trancher_candidat
    # ##########################################################################
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

    # ##########################################################################
    # _detecter_candidats_faux_positifs
    # ##########################################################################
    def _detecter_candidats_faux_positifs(
        self, original: ExemplePivot, nom_champ: str, texte_original: str, texte_anonymise: str, langue: str
    ) -> None:
        plafond = self.max_faux_positifs_par_source
        deja_pour_source = sum(1 for c in self.candidats_faux_positifs if c.source == original.source)
        if plafond is not None and deja_pour_source >= plafond:
            return
        for fragment, debut, fin in _extraire_fragments_masques(texte_original, texte_anonymise, self.jeton_masque):
            if not fragment.strip():
                continue
            verdict_spacy = self.verificateur_entites.verifier(texte_original, langue, debut, fin)
            if verdict_spacy is VerdictEntiteNommee.ENTITE_PERTINENTE:
                continue  # spaCy confirme une entite nommee, masquage juge legitime, pas un faux positif
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
                    debut=debut,
                    fin=fin,
                )
            )
            deja_pour_source += 1
            if plafond is not None and deja_pour_source >= plafond:
                break


@dataclass(slots=True)
class ControlerQualiteAnonymisationUseCase:
    """
    Orchestre le controle qualite : lit le pivot original et le
    fichier anonymise, preleve un echantillon stratifie INCREMENTAL
    parmi les exemples anonymises disponibles, et compare chaque
    couple via `ControleQualiteAnonymisation.observer`.

    Muestreo incremental (09/09/2026 ; NF2 du
    cahier des charges exige une revision humaine PERSISTEE, pas un
    echantillon aleatoire jete a chaque execution) : `registre_echantillons`
    (meme role, pour ce cas d'usage, que `RepositoryLectureEcriture.identifiants_existants()`
    pour `AnonymiserDatasetUseCase`) exclut du tirage les identifiants
    DEJA echantillonnes lors d'une execution precedente ; chaque
    execution tire `taille_echantillon` identifiants NOUVEAUX, jamais
    revus. Applique au stratum principal (STRATUM_PRINCIPAL) ET au
    stratum dedie "sans entite detectee" (STRATUM_SANS_ENTITE),
    independamment l'un de l'autre comme le reste de ces deux strates.
    """

    repository_original          : RepositoryLectureEcriture
    repository_anonymise         : RepositoryLectureEcriture
    verificateur_entites         : VerificateurEntitesNommees
    registre_echantillons        : RegistreEchantillonsControleQualite
    taille_echantillon           : int | None = 200
    graine_aleatoire             : int = 42
    max_exemples_par_source      : int = 10
    max_faux_positifs_par_source : int | None = 10
    jeton_masque                 : str = JETON_MASQUE_DEFAUT
    # Stratum dedie "sans entite detectee" (item 3) ; independant de
    # `taille_echantillon`/`graine_aleatoire` ci-dessus (graine
    # distincte pour ne pas correler les deux tirages). L'exigence est
    # de 30-50 exemples relus a la main/seconde
    # opinion spaCy pour ce stratum ; 40 par defaut (milieu de la
    # fourchette).
    taille_echantillon_sans_entite : int | None = 40
    graine_aleatoire_sans_entite   : int = 43

    nombre_introuvables_dans_original : int = field(default=0, init=False)

    # ##########################################################################
    # executer
    # ##########################################################################
    def executer(self) -> ControleQualiteAnonymisation:
        originaux_par_id = {e.identifiant: e for e in self.repository_original.lister()}
        anonymises = list(self.repository_anonymise.lister())
        horodatage = datetime.now(timezone.utc).isoformat(timespec="seconds")

        controle = ControleQualiteAnonymisation(
            verificateur_entites=self.verificateur_entites,
            max_exemples_par_source=self.max_exemples_par_source,
            max_faux_positifs_par_source=self.max_faux_positifs_par_source,
            jeton_masque=self.jeton_masque,
        )

        # ----------------------------------------------------------------------
        # Echantillon stratifie principal (type_exemple, source) --
        # EXCLUT les identifiants deja echantillonnes lors d'une
        # execution precedente (muestreo incremental, cf. docstring).
        # ----------------------------------------------------------------------
        deja_echantillonnes_principal = self.registre_echantillons.identifiants_vus(STRATUM_PRINCIPAL)
        candidats_principal = [a for a in anonymises if a.identifiant not in deja_echantillonnes_principal]

        if self.taille_echantillon is not None and self.taille_echantillon < len(candidats_principal):
            echantillon = echantillon_stratifie(candidats_principal, self.taille_echantillon, self.graine_aleatoire)
        else:
            echantillon = candidats_principal

        self.nombre_introuvables_dans_original = 0
        for exemple_anonymise in echantillon:
            exemple_original = originaux_par_id.get(exemple_anonymise.identifiant)
            if exemple_original is None:
                # Garde-fou : ne devrait jamais arriver si --sortie a
                # bien ete alimente depuis --dataset par
                # AnonymiserDatasetUseCase.
                self.nombre_introuvables_dans_original += 1
                continue
            controle.observer(exemple_original, exemple_anonymise)

        if echantillon:
            self.registre_echantillons.marquer_vus(
                STRATUM_PRINCIPAL, [e.identifiant for e in echantillon], horodatage
            )

        # ----------------------------------------------------------------------
        # Stratum dedie "sans entite detectee" (item 3) : independant
        # du tirage stratifie ci-dessus ; tire sur TOUS les couples
        # valides disponibles (pas seulement `echantillon`), pour ne
        # pas dependre du hasard du premier tirage. cf.
        # ControleQualiteAnonymisation.observer_sans_entite.
        # `nombre_disponibles_sans_entite` reste le total du stratum
        # (deja echantillonne ou non) ; seul le TIRAGE ci-dessous est
        # restreint aux identifiants pas encore vus.
        # ----------------------------------------------------------------------
        sans_entite = [
            (originaux_par_id[a.identifiant], a)
            for a in anonymises
            if a.identifiant in originaux_par_id and _est_exemple_sans_entite(originaux_par_id[a.identifiant], a)
        ]
        controle.nombre_disponibles_sans_entite = len(sans_entite)

        deja_echantillonnes_sans_entite = self.registre_echantillons.identifiants_vus(STRATUM_SANS_ENTITE)
        candidats_sans_entite = [
            (o, a) for (o, a) in sans_entite if a.identifiant not in deja_echantillonnes_sans_entite
        ]

        if self.taille_echantillon_sans_entite is not None and self.taille_echantillon_sans_entite < len(candidats_sans_entite):
            rng = random.Random(self.graine_aleatoire_sans_entite)
            echantillon_sans_entite = rng.sample(candidats_sans_entite, self.taille_echantillon_sans_entite)
        else:
            echantillon_sans_entite = candidats_sans_entite
        for exemple_original, exemple_anonymise in echantillon_sans_entite:
            controle.observer_sans_entite(exemple_original, exemple_anonymise)

        if echantillon_sans_entite:
            self.registre_echantillons.marquer_vus(
                STRATUM_SANS_ENTITE, [a.identifiant for _, a in echantillon_sans_entite], horodatage
            )

        return controle


# ----------------------------------------------------------------------
# Serialisation JSON + presentation Markdown ; pas d'I/O fichier ici
# (a la charge de l'appelant, comme pour rapport_anonymisation.py).
# ----------------------------------------------------------------------


# ##############################################################################
# _candidat_pii_vers_dict
# ##############################################################################
def _candidat_pii_vers_dict(c: CandidatPiiResiduelle) -> dict:
    return {
        "identifiant" : c.identifiant,
        "source"      : c.source,
        "champ"       : c.champ,
        "langue"      : c.langue,
        "type_motif"  : c.type_motif,
        "passage"     : c.passage,
        "verdict"     : c.verdict,
        "debut"       : c.debut,
        "fin"         : c.fin,
    }


# ##############################################################################
# _exemple_controle_vers_dict
# ##############################################################################
def _exemple_controle_vers_dict(e: ExempleControle) -> dict:
    return {
        "identifiant"     : e.identifiant,
        "champ"           : e.champ,
        "texte_original"  : e.texte_original,
        "texte_anonymise" : e.texte_anonymise,
    }


# ##############################################################################
# controle_vers_dict
# ##############################################################################
def controle_vers_dict(
    controle: ControleQualiteAnonymisation,
    horodatage: str,
    dataset_original: str,
    dataset_anonymise: str,
) -> dict:
    """Serialise l'accumulateur en dict JSON, pour inspection programmatique (pas seulement le Markdown)."""
    return {
        "horodatage"                    : horodatage,
        "dataset_original"              : dataset_original,
        "dataset_anonymise"             : dataset_anonymise,
        "nombre_exemples_observes"      : controle.nombre_exemples_observes,
        "candidats_pii_residuelle"      : [_candidat_pii_vers_dict(c) for c in controle.candidats_pii],
        "candidats_faux_positifs"       : [
            {
                "identifiant"     : c.identifiant,
                "source"          : c.source,
                "champ"           : c.champ,
                "fragment_masque" : c.fragment_masque,
                "texte_original"  : c.texte_original,
                "texte_anonymise" : c.texte_anonymise,
                "verdict"         : c.verdict,
                "debut"           : c.debut,
                "fin"             : c.fin,
            }
            for c in controle.candidats_faux_positifs
        ],
        "exemples_par_source"           : {
            source: [_exemple_controle_vers_dict(e) for e in exemples]
            for source, exemples in controle.exemples_par_source.items()
        },
        # Stratum dedie "sans entite detectee" (item 3) ; compteurs et
        # listes toujours SEPARES des cles ci-dessus, jamais fusionnes.
        "stratum_sans_entite_detectee"  : {
            "nombre_disponibles"       : controle.nombre_disponibles_sans_entite,
            "nombre_observes"          : controle.nombre_exemples_sans_entite_observes,
            "candidats_pii_residuelle" : [_candidat_pii_vers_dict(c) for c in controle.candidats_pii_sans_entite],
            "exemples"                 : [_exemple_controle_vers_dict(e) for e in controle.exemples_sans_entite],
        },
    }


# ##############################################################################
# formater_rapport_markdown
# ##############################################################################
def formater_rapport_markdown(
    controle: ControleQualiteAnonymisation,
    horodatage: str,
    dataset_original: str,
    dataset_anonymise: str,
    taille_echantillon_demandee: int | None,
    total_anonymise_disponible: int,
    statistiques_cumulees: dict[str, StatistiquesSource],
    decisions_par_cle: dict[CleCandidatRevision, str] | None = None,
) -> str:
    """
    Rapport Markdown du controle qualite pour L'ECHANTILLON compare
    lors de cette execution.

    Design incremental (09/09/2026 ; NF2 du
    cahier des charges) : chaque execution ne compare que des
    identifiants JAMAIS echantillonnes auparavant (cf.
    `RegistreEchantillonsControleQualite` / `ControlerQualiteAnonymisationUseCase`),
    donc CE rapport ne decrit que le LOT de cette execution ; il ne
    remplace pas un decompte cumule sur toutes les executions. Le
    statut cumule des decisions humaines (acceptees/rejetees/encore en
    attente, toutes executions confondues) vit dans
    `data/processed/decisions_revision_humaine.jsonl`, tenu a jour par
    `reviser_pii_residuelle.py` ; c'est la source de verite pour
    affirmer "0 PII residuelle confirmee", pas ce rapport a lui seul.

    `decisions_par_cle` (optionnel, cle stable `CleCandidatRevision` ->
    "accepte"/"rejete") permet d'annoter chaque candidat
    VERDICT_REVISION_HUMAINE de CETTE execution avec son statut de
    decision humaine, s'il en a deja une (typiquement rare pour un lot
    fraichement echantillonne ; une decision suppose une execution
    prealable de `reviser_pii_residuelle.py`).
    """
    decisions_par_cle = decisions_par_cle or {}

    total_confirmes = sum(1 for c in controle.candidats_pii if c.verdict == VERDICT_CONFIRME)
    total_faux_positifs_regex = sum(1 for c in controle.candidats_pii if c.verdict == VERDICT_FAUX_POSITIF_REGEX)
    total_revision_humaine = sum(1 for c in controle.candidats_pii if c.verdict == VERDICT_REVISION_HUMAINE)
    acceptes_pii, rejetes_pii, en_attente_pii = _repartir_revision_humaine(
        controle.candidats_pii, SOURCE_CANDIDATS_PII, decisions_par_cle
    )

    total_masquages_faux_positifs = sum(
        1 for c in controle.candidats_faux_positifs if c.verdict == VERDICT_FAUX_POSITIF_REGEX
    )
    total_masquages_a_revoir = sum(
        1 for c in controle.candidats_faux_positifs if c.verdict == VERDICT_REVISION_HUMAINE
    )
    acceptes_fp, rejetes_fp, en_attente_fp = _repartir_revision_humaine_faux_positifs(
        controle.candidats_faux_positifs, decisions_par_cle
    )

    # ----------------------------------------------------------------------
    # En-tete et portee de l'execution
    # ----------------------------------------------------------------------
    lignes = [
        "# Rapport de controle qualite : anonymisation (comparaison original/anonymise)",
        "",
        f"> Genere automatiquement le {horodatage} par `controler_qualite_anonymisation.py`, "
        f"a partir de `{dataset_original}` (original) compare a `{dataset_anonymise}` (anonymise).",
        ">",
        f"> **Portee explicite** : {controle.nombre_exemples_observes} exemples compares, un "
        f"echantillon stratifie (type_exemple, source)"
        + (
            f" de taille demandee {taille_echantillon_demandee}"
            if taille_echantillon_demandee is not None
            else ""
        )
        + f" parmi les {total_anonymise_disponible} exemples disponibles dans `{dataset_anonymise}` au "
        "moment de cette execution ; muestreo INCREMENTAL (09/09/2026) : les identifiants deja "
        "echantillonnes lors d'une execution precedente (cf. "
        "`data/processed/controle_qualite_identifiants_echantillonnes.jsonl`) sont exclus du tirage, "
        "chaque execution ne compare donc que des identifiants JAMAIS encore vus. Comme le pivot "
        "original n'est jamais modifie, ce controle peut porter sur n'importe quelle vague deja "
        "anonymisee, y compris retroactivement.",
        "",
        "## 1. Compteurs par categorie d'entite anonymisee (rapport RGPD cumule, Partie 1)",
        "",
        "Reutilise directement `statistiques_cumulees` du rapport RGPD cumule, aucun recalcul ici "
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
        lignes.append("| (rapport RGPD cumule introuvable ou vide) | — | — | — |")

    # ----------------------------------------------------------------------
    # PII residuelle detectee dans le texte anonymise
    # ----------------------------------------------------------------------
    lignes += [
        "",
        "## 2. Candidats de PII residuelle (regex sur texte anonymise + seconde opinion spaCy)",
        "",
        f"- Candidats bruts detectes par regex : **{len(controle.candidats_pii)}**.",
        f"- Confirmes (match deterministe email/telephone/url/date, ou bigramme capitalise que "
        f"spaCy reconnait comme entite nommee) : **{total_confirmes}**.",
        f"- Ecartes par la seconde opinion spaCy (bigramme capitalise sans entite nommee detectee, "
        f"probable terme medical, pas une PII) : **{total_faux_positifs_regex}**.",
        f"- Marques revision humaine par le regex+spaCy (spaCy detecte une entite mais d'un type non "
        f"tranchant) : **{total_revision_humaine}**, dont **{acceptes_pii}** deja acceptes (confirmes "
        f"non-PII par une personne), **{rejetes_pii}** deja rejetes (PII reelle confirmee), "
        f"**{en_attente_pii}** encore genuinement en attente d'une decision humaine "
        f"(`reviser_pii_residuelle.py --verify`).",
        "",
        "### Passages confirmes",
    ]
    lignes += _lister_candidats_pii(controle.candidats_pii, VERDICT_CONFIRME)
    lignes += ["", "### Passages marques revision humaine (statut de decision entre crochets)"]
    lignes += _lister_candidats_pii(controle.candidats_pii, VERDICT_REVISION_HUMAINE, SOURCE_CANDIDATS_PII, decisions_par_cle)

    # ----------------------------------------------------------------------
    # Faux positifs de masquage (termes masques sans necessite)
    # ----------------------------------------------------------------------
    lignes += [
        "",
        "## 3. Candidats de faux positifs de l'anonymisation (termes masques sans necessite)",
        "",
        f"Detectes par diff texte-original/texte-anonymise (fragments remplaces par le jeton "
        f"`{controle.jeton_masque}`), puis seconde opinion spaCy sur le fragment ORIGINAL masque : "
        "si spaCy ne reconnait aucune entite nommee a cet endroit, le masquage est probablement "
        "un faux positif (terme medical/scientifique pris pour un nom propre). **Ce ne sont pas des "
        "faux positifs confirmes** ; seulement des candidats a verifier humainement.",
        "",
        f"- Fragments masques trouves sans confirmation spaCy : **{len(controle.candidats_faux_positifs)}** "
        f"({total_masquages_faux_positifs} sans aucune entite detectee, {total_masquages_a_revoir} "
        "avec une entite d'un type non tranchant, dont "
        f"**{acceptes_fp}** deja acceptes, **{rejetes_fp}** deja rejetes, **{en_attente_fp}** encore "
        "en attente d'une decision humaine).",
        "",
    ]
    for candidat in controle.candidats_faux_positifs:
        suffixe = ""
        if candidat.verdict == VERDICT_REVISION_HUMAINE:
            decision = decisions_par_cle.get(cle_candidat_faux_positif(candidat))
            suffixe = f" [decision humaine : {decision or 'en attente'}]"
        lignes.append(
            f"- `{candidat.identifiant}` ({candidat.source}, champ `{candidat.champ}`, "
            f"verdict `{candidat.verdict}`) : fragment masque {candidat.fragment_masque!r}{suffixe}"
        )
    if not controle.candidats_faux_positifs:
        lignes.append("- (aucun)")

    # ----------------------------------------------------------------------
    # Exemples illustratifs par source
    # ----------------------------------------------------------------------
    lignes += ["", "## 4. Exemples reels (original -> anonymise) par source", ""]
    for source in sorted(controle.exemples_par_source):
        lignes.append(f"### {source}")
        for exemple in controle.exemples_par_source[source]:
            lignes.append(f"- `{exemple.identifiant}` (champ `{exemple.champ}`) :")
            lignes.append(f"  - original : {exemple.texte_original[:200]!r}")
            lignes.append(f"  - anonymise : {exemple.texte_anonymise[:200]!r}")
        lignes.append("")

    # ----------------------------------------------------------------------
    # Stratum dedie "sans entite detectee" (item 3)
    # ----------------------------------------------------------------------
    total_confirmes_sans_entite = sum(
        1 for c in controle.candidats_pii_sans_entite if c.verdict == VERDICT_CONFIRME
    )
    total_revision_sans_entite = sum(
        1 for c in controle.candidats_pii_sans_entite if c.verdict == VERDICT_REVISION_HUMAINE
    )
    acceptes_se, rejetes_se, en_attente_se = _repartir_revision_humaine(
        controle.candidats_pii_sans_entite, SOURCE_CANDIDATS_PII_SANS_ENTITE, decisions_par_cle
    )
    lignes += [
        "",
        "## 5. Stratum dedie : exemples SANS entite detectee (item 3)",
        "",
        "Stratum INDEPENDANT de l'echantillon stratifie (type_exemple, source) des sections "
        "1-4 ci-dessus : au lieu de tirer parmi TOUS les exemples anonymises (avec ou sans "
        "entite), ce stratum isole specifiquement les couples ou `texte_original == "
        "texte_anonymise` sur TOUS les champs texte libre, c'est-a-dire ceux ou Presidio n'a "
        "RIEN detecte du tout. L'exigence est que 30-50 de ces cas soient "
        "relus a la main/seconde opinion spaCy plutot que d'etre presumes corrects par defaut : "
        "\"rien detecte\" peut aussi bien signifier \"le texte ne contient reellement aucune PII\" "
        "que \"Presidio a rate une PII qui aurait du l'etre\".",
        "",
        f"- Exemples disponibles dans ce stratum (texte inchange par l'anonymisation) : "
        f"**{controle.nombre_disponibles_sans_entite}**.",
        f"- Exemples relus dans cette execution : **{controle.nombre_exemples_sans_entite_observes}**.",
        f"- Candidats de PII residuelle detectes sur ce sous-ensemble (texte NON modifie) : "
        f"**{len(controle.candidats_pii_sans_entite)}** ({total_confirmes_sans_entite} confirmes, "
        f"{total_revision_sans_entite} marques revision humaine, dont **{acceptes_se}** acceptes, "
        f"**{rejetes_se}** rejetes, **{en_attente_se}** encore en attente).",
        "",
        "### Candidats confirmes (faux negatif complet possible de Presidio)",
    ]
    lignes += _lister_candidats_pii(controle.candidats_pii_sans_entite, VERDICT_CONFIRME)
    lignes += ["", "### Candidats marques revision humaine (statut de decision entre crochets)"]
    lignes += _lister_candidats_pii(
        controle.candidats_pii_sans_entite, VERDICT_REVISION_HUMAINE, SOURCE_CANDIDATS_PII_SANS_ENTITE, decisions_par_cle
    )

    lignes += ["", "### Exemples relus (texte inchange)", ""]
    for exemple in controle.exemples_sans_entite:
        lignes.append(f"- `{exemple.identifiant}` ({exemple.source}, champ `{exemple.champ}`) : "
                       f"{exemple.texte_original[:200]!r}")
    if not controle.exemples_sans_entite:
        lignes.append("- (aucun)")

    return "\n".join(lignes)


# ##############################################################################
# cle_candidat_pii / cle_candidat_faux_positif
# ##############################################################################
def cle_candidat_pii(source_liste: str, c: CandidatPiiResiduelle) -> CleCandidatRevision:
    """
    Cle stable d'un `CandidatPiiResiduelle` ; `source_liste` doit etre
    SOURCE_CANDIDATS_PII ou SOURCE_CANDIDATS_PII_SANS_ENTITE selon la
    liste d'origine (le meme dataclass sert aux deux, cf. docstring de
    `CandidatPiiResiduelle.debut`). Reutilisee par
    `application.use_cases.uc_03_03_reviser_pii_residuelle`.
    """
    return CleCandidatRevision(source_liste, c.identifiant, c.champ, c.type_motif, c.debut, c.fin)


def cle_candidat_faux_positif(c: CandidatFauxPositifAnonymisation) -> CleCandidatRevision:
    """Cle stable d'un `CandidatFauxPositifAnonymisation` (type_motif="" ; pas issu d'une regex typee)."""
    return CleCandidatRevision(SOURCE_CANDIDATS_FAUX_POSITIFS, c.identifiant, c.champ, "", c.debut, c.fin)


# ##############################################################################
# _repartir_revision_humaine / _repartir_revision_humaine_faux_positifs
# ##############################################################################
def _repartir_revision_humaine(
    candidats: list[CandidatPiiResiduelle],
    source_liste: str,
    decisions_par_cle: dict[CleCandidatRevision, str],
) -> tuple[int, int, int]:
    """Parmi les candidats VERDICT_REVISION_HUMAINE, compte (acceptes, rejetes, encore en attente)."""
    acceptes = rejetes = en_attente = 0
    for c in candidats:
        if c.verdict != VERDICT_REVISION_HUMAINE:
            continue
        decision = decisions_par_cle.get(cle_candidat_pii(source_liste, c))
        if decision == DECISION_ACCEPTE:
            acceptes += 1
        elif decision == DECISION_REJETE:
            rejetes += 1
        else:
            en_attente += 1
    return acceptes, rejetes, en_attente


def _repartir_revision_humaine_faux_positifs(
    candidats: list[CandidatFauxPositifAnonymisation],
    decisions_par_cle: dict[CleCandidatRevision, str],
) -> tuple[int, int, int]:
    acceptes = rejetes = en_attente = 0
    for c in candidats:
        if c.verdict != VERDICT_REVISION_HUMAINE:
            continue
        decision = decisions_par_cle.get(cle_candidat_faux_positif(c))
        if decision == DECISION_ACCEPTE:
            acceptes += 1
        elif decision == DECISION_REJETE:
            rejetes += 1
        else:
            en_attente += 1
    return acceptes, rejetes, en_attente


# ##############################################################################
# _lister_candidats_pii
# ##############################################################################
def _lister_candidats_pii(
    candidats: list[CandidatPiiResiduelle],
    verdict: str,
    source_liste: str = "",
    decisions_par_cle: dict[CleCandidatRevision, str] | None = None,
) -> list[str]:
    lignes = []
    for candidat in candidats:
        if candidat.verdict != verdict:
            continue
        suffixe = ""
        if verdict == VERDICT_REVISION_HUMAINE and decisions_par_cle is not None and source_liste:
            decision = decisions_par_cle.get(cle_candidat_pii(source_liste, candidat))
            suffixe = f" [decision humaine : {decision or 'en attente'}]"
        lignes.append(
            f"- `{candidat.identifiant}` ({candidat.source}, champ `{candidat.champ}`, "
            f"motif `{candidat.type_motif}`) : {candidat.passage!r}{suffixe}"
        )
    if not lignes:
        lignes.append("- (aucun)")
    return lignes
