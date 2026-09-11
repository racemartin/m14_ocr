"""
Cas d'usage : fermer reellement l'exigence NF2 du cahier des charges
("anonymisation validee MANUELLEMENT, 0 PII residuelle sur echantillon
de controle") en donnant a une personne le moyen de trancher les
candidats que `E1_04_02_controler_qualite_anonymisation.py` marque
`VERDICT_REVISION_HUMAINE` ; ni le regex ni la seconde opinion spaCy
ne les tranchent seuls ; et de PERSISTER sa decision.

Design "replay cumule" (09/09/2026) : a cause du
muestreo incremental de `ControlerQualiteAnonymisationUseCase` (cf.
E1_04_02_controler_qualite_anonymisation), le rapport d'une execution donnee ne contient QUE les
candidats du lot fraichement echantillonne ; il ne suffit donc pas de
relire le dernier rapport JSON pour trouver TOUS les candidats en
attente de revision humaine. Ce cas d'usage recalcule plutot la
comparaison original/anonymise sur la TOTALITE des identifiants deja
echantillonnes, toutes executions confondues (`RegistreEchantillonsControleQualite.identifiants_vus`
pour les deux strates) : deterministe (memes textes, memes regex,
meme seconde opinion spaCy => memes candidats a chaque appel), donc
sans risque d'incoherence avec les decisions deja persistees.
"""

from __future__ import annotations

from dataclasses import dataclass

from chsa_triage.application.use_cases.E1_04_02_controler_qualite_anonymisation import (
    JETON_MASQUE_DEFAUT,
    STRATUM_PRINCIPAL,
    STRATUM_SANS_ENTITE,
    VERDICT_CONFIRME,
    VERDICT_REVISION_HUMAINE,
    ControleQualiteAnonymisation,
    _est_exemple_sans_entite,
    _paires_champs,
    cle_candidat_faux_positif,
    cle_candidat_pii,
)
from chsa_triage.domain.model import (
    DECISION_ACCEPTE,
    SOURCE_CANDIDATS_PII,
    SOURCE_CANDIDATS_PII_SANS_ENTITE,
    CleCandidatRevision,
    DecisionRevisionHumaine,
    ExemplePivot,
)
from chsa_triage.domain.ports import RepositoryLectureEcriture
from chsa_triage.domain.ports.decisions_revision_humaine import RegistreDecisionsRevisionHumaine
from chsa_triage.domain.ports.registre_echantillons_controle_qualite import (
    RegistreEchantillonsControleQualite,
)
from chsa_triage.domain.ports.verificateur_entites import VerificateurEntitesNommees


VERDICTS_REVISION_HUMAINE_SEUL: frozenset[str] = frozenset({VERDICT_REVISION_HUMAINE})

# Motifs d'exclusion de publication (cf. `ReviserPiiResiduelleUseCase.identifiants_a_exclure_publication`) ;
# "confirme" l'emporte sur "pendant_revision_humaine" quand un identifiant a les deux (motif le plus grave).
RAISON_CONFIRME                 = "confirme"
RAISON_PENDANT_REVISION_HUMAINE = "pendant_revision_humaine"


@dataclass(frozen=True, slots=True)
class CandidatARevoir:
    """Un candidat de PII residuelle, uniformise entre les 3 sources, pret a etre montre pour decision ou export."""

    cle          : CleCandidatRevision
    source_corpus: str  # ExemplePivot.source (nom du corpus, ex. "MediQAl") ; pas source_liste
    langue       : str  # "" pour SOURCE_CANDIDATS_FAUX_POSITIFS (pas de langue stockee sur ce candidat)
    passage      : str
    verdict      : str  # VERDICT_CONFIRME | VERDICT_FAUX_POSITIF_REGEX | VERDICT_REVISION_HUMAINE


# ##############################################################################
# candidats_a_revoir
# ##############################################################################
def candidats_a_revoir(
    controle: ControleQualiteAnonymisation, verdicts: frozenset[str] = VERDICTS_REVISION_HUMAINE_SEUL
) -> list[CandidatARevoir]:
    """
    Rassemble, uniformises, tous les candidats des 3 sources de
    `controle` dont le `verdict` figure dans `verdicts` (par defaut,
    uniquement VERDICT_REVISION_HUMAINE, pour la revue interactive).
    Generalise (11/09/2026) pour aussi couvrir VERDICT_CONFIRME, sans
    dupliquer cette collecte, au profit de
    `ReviserPiiResiduelleUseCase.identifiants_a_exclure_publication`.
    """
    resultats: list[CandidatARevoir] = []

    for c in controle.candidats_pii:
        if c.verdict in verdicts:
            resultats.append(
                CandidatARevoir(cle_candidat_pii(SOURCE_CANDIDATS_PII, c), c.source, c.langue, c.passage, c.verdict)
            )

    for c in controle.candidats_faux_positifs:
        if c.verdict in verdicts:
            resultats.append(
                CandidatARevoir(cle_candidat_faux_positif(c), c.source, "", c.fragment_masque, c.verdict)
            )

    for c in controle.candidats_pii_sans_entite:
        if c.verdict in verdicts:
            resultats.append(
                CandidatARevoir(
                    cle_candidat_pii(SOURCE_CANDIDATS_PII_SANS_ENTITE, c), c.source, c.langue, c.passage, c.verdict
                )
            )

    return resultats


# ##############################################################################
# texte_original_et_anonymise
# ##############################################################################
def texte_original_et_anonymise(
    original: ExemplePivot, anonymise: ExemplePivot, nom_champ: str
) -> tuple[str, str] | None:
    """
    Retrouve le texte COMPLET (original, anonymise) d'un champ nomme
    (ex. "symptomes", "chosen[0]") pour l'affichage etendu en mode
    verification ; le `passage` de 40 caracteres de contexte de
    chaque cote (cf. `detection_pii_residuelle.CONTEXTE_CARACTERES`)
    suffit dans la majorite des cas, mais une personne qui revise peut
    vouloir le champ entier pour juger un cas ambigu. Retourne None si
    `nom_champ` n'existe pas sur ce couple (ne devrait pas arriver pour
    un candidat produit par `ControleQualiteAnonymisation`).
    """
    for champ, texte_original, texte_anonymise in _paires_champs(original, anonymise):
        if champ == nom_champ:
            return texte_original, texte_anonymise
    return None


@dataclass(slots=True)
class ReviserPiiResiduelleUseCase:
    """
    Orchestre le mode verification de `E1_04_01_reviser_pii_residuelle.py` :
    recalcule les candidats REVISION_HUMAINE sur tout ce qui a deja ete
    echantillonne pour le controle qualite, exclut ceux ayant deja une
    decision humaine, et persiste chaque nouvelle decision IMMEDIATEMENT.
    """

    repository_original    : RepositoryLectureEcriture
    repository_anonymise    : RepositoryLectureEcriture
    verificateur_entites     : VerificateurEntitesNommees
    registre_echantillons    : RegistreEchantillonsControleQualite
    decisions                : RegistreDecisionsRevisionHumaine
    jeton_masque              : str = JETON_MASQUE_DEFAUT

    # ##########################################################################
    # _rejouer_controle
    # ##########################################################################
    def _rejouer_controle(self) -> ControleQualiteAnonymisation:
        """
        Recalcule (replay deterministe, cf. docstring du module)
        `ControleQualiteAnonymisation` sur TOUS les identifiants deja
        echantillonnes (les deux strates) ; moteur commun a
        `candidats_en_attente` et `identifiants_a_exclure_publication`,
        pour ne jamais dupliquer ce replay.
        """
        originaux_par_id = {e.identifiant: e for e in self.repository_original.lister()}
        anonymises_par_id = {e.identifiant: e for e in self.repository_anonymise.lister()}

        # Aucun plafond par source ici (`max_faux_positifs_par_source=None`) :
        # contrairement au rapport Markdown (pense pour la lisibilite),
        # la revue humaine doit voir TOUS les candidats, pas seulement
        # les premiers par source.
        controle = ControleQualiteAnonymisation(
            verificateur_entites=self.verificateur_entites,
            max_exemples_par_source=0,
            max_faux_positifs_par_source=None,
            jeton_masque=self.jeton_masque,
        )

        for identifiant in self.registre_echantillons.identifiants_vus(STRATUM_PRINCIPAL):
            original = originaux_par_id.get(identifiant)
            anonymise = anonymises_par_id.get(identifiant)
            if original is not None and anonymise is not None:
                controle.observer(original, anonymise)

        for identifiant in self.registre_echantillons.identifiants_vus(STRATUM_SANS_ENTITE):
            original = originaux_par_id.get(identifiant)
            anonymise = anonymises_par_id.get(identifiant)
            if original is not None and anonymise is not None and _est_exemple_sans_entite(original, anonymise):
                controle.observer_sans_entite(original, anonymise)

        return controle

    # ##########################################################################
    # candidats_en_attente
    # ##########################################################################
    def candidats_en_attente(self) -> list[CandidatARevoir]:
        """
        Candidats VERDICT_REVISION_HUMAINE sur tout ce qui a deja ete
        echantillonne, exclus ceux ayant deja une decision humaine
        persistee (peu importe laquelle : `verify` ne doit plus les
        montrer une fois tranches).
        """
        controle = self._rejouer_controle()
        deja_decides = self.decisions.cles_decidees()
        return [c for c in candidats_a_revoir(controle) if c.cle not in deja_decides]

    # ##########################################################################
    # identifiants_a_exclure_publication
    # ##########################################################################
    def identifiants_a_exclure_publication(self) -> dict[str, str]:
        """
        Identifiants a exclure d'une publication (ex. sous-ensemble SFT,
        cf. `interfaces/cli/E1_04_01_reviser_pii_residuelle.py exporter`) : au
        moins un candidat VERDICT_CONFIRME (fuite non ambigue, jamais
        soumise a decision humaine), ou au moins un candidat
        VERDICT_REVISION_HUMAINE sans decision DECISION_ACCEPTE
        persistee (candidat encore ouvert, ou explicitement DECISION_REJETE
        = PII confirmee par une personne). "confirme" l'emporte sur
        "pendant_revision_humaine" quand un identifiant a les deux motifs.
        """
        controle = self._rejouer_controle()
        candidats = candidats_a_revoir(controle, verdicts=frozenset({VERDICT_CONFIRME, VERDICT_REVISION_HUMAINE}))
        cles_acceptees = {d.cle for d in self.decisions.toutes() if d.decision == DECISION_ACCEPTE}

        razons: dict[str, str] = {}
        for c in candidats:
            if c.verdict == VERDICT_REVISION_HUMAINE and c.cle not in cles_acceptees:
                razons[c.cle.identifiant] = RAISON_PENDANT_REVISION_HUMAINE
        for c in candidats:
            if c.verdict == VERDICT_CONFIRME:
                razons[c.cle.identifiant] = RAISON_CONFIRME

        return razons

    # ##########################################################################
    # identifiants_a_exclure_publication_set
    # ##########################################################################
    def identifiants_a_exclure_publication_set(self) -> set[str]:
        """
        Ensemble des `ExemplePivot.identifiant` a exclure d'une
        publication, toutes raisons confondues (CONFIRME ou en attente
        de revision humaine) ; utilise par `DecouperSplitsUseCase`
        (10/09/2026, etendu le 11/09/2026 aux candidats CONFIRMES)
        pour exclure ces exemples du decoupage train/val/test par
        precaution, sans dupliquer `identifiants_a_exclure_publication`.
        """
        return set(self.identifiants_a_exclure_publication().keys())

    # ##########################################################################
    # enregistrer_decision
    # ##########################################################################
    def enregistrer_decision(
        self, candidat: CandidatARevoir, decision: str, horodatage: str, note: str = ""
    ) -> DecisionRevisionHumaine:
        """Construit et persiste IMMEDIATEMENT la decision humaine sur `candidat`."""
        enregistrement = DecisionRevisionHumaine(
            cle=candidat.cle, passage=candidat.passage, decision=decision, horodatage=horodatage, note=note
        )
        self.decisions.enregistrer(enregistrement)
        return enregistrement
