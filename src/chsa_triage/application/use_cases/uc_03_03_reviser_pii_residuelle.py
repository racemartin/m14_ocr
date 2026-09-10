"""
Cas d'usage : fermer reellement l'exigence NF2 du cahier des charges
("anonymisation validee MANUELLEMENT, 0 PII residuelle sur echantillon
de controle") en donnant a une personne le moyen de trancher les
candidats que `controler_qualite_anonymisation.py` marque
`VERDICT_REVISION_HUMAINE` -- ni le regex ni la seconde opinion spaCy
ne les tranchent seuls -- et de PERSISTER sa decision.

Design "replay cumule" (09/09/2026, decision du capitaine) : a cause du
muestreo incremental de `ControlerQualiteAnonymisationUseCase` (cf.
uc_03_02), le rapport d'une execution donnee ne contient QUE les
candidats du lot fraichement echantillonne -- il ne suffit donc pas de
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

from chsa_triage.application.use_cases.uc_03_02_controler_qualite_anonymisation import (
    JETON_MASQUE_DEFAUT,
    STRATUM_PRINCIPAL,
    STRATUM_SANS_ENTITE,
    VERDICT_REVISION_HUMAINE,
    ControleQualiteAnonymisation,
    _est_exemple_sans_entite,
    _paires_champs,
    cle_candidat_faux_positif,
    cle_candidat_pii,
)
from chsa_triage.domain.model import (
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


@dataclass(frozen=True, slots=True)
class CandidatARevoir:
    """Un candidat REVISION_HUMAINE, uniformise entre les 3 sources, pret a etre montre pour decision."""

    cle          : CleCandidatRevision
    source_corpus: str  # ExemplePivot.source (nom du corpus, ex. "MediQAl") -- pas source_liste
    langue       : str  # "" pour SOURCE_CANDIDATS_FAUX_POSITIFS (pas de langue stockee sur ce candidat)
    passage      : str


# ##############################################################################
# candidats_a_revoir
# ##############################################################################
def candidats_a_revoir(controle: ControleQualiteAnonymisation) -> list[CandidatARevoir]:
    """Rassemble, uniformisés, tous les candidats VERDICT_REVISION_HUMAINE des 3 sources de `controle`."""
    resultats: list[CandidatARevoir] = []

    for c in controle.candidats_pii:
        if c.verdict == VERDICT_REVISION_HUMAINE:
            resultats.append(
                CandidatARevoir(cle_candidat_pii(SOURCE_CANDIDATS_PII, c), c.source, c.langue, c.passage)
            )

    for c in controle.candidats_faux_positifs:
        if c.verdict == VERDICT_REVISION_HUMAINE:
            resultats.append(
                CandidatARevoir(cle_candidat_faux_positif(c), c.source, "", c.fragment_masque)
            )

    for c in controle.candidats_pii_sans_entite:
        if c.verdict == VERDICT_REVISION_HUMAINE:
            resultats.append(
                CandidatARevoir(
                    cle_candidat_pii(SOURCE_CANDIDATS_PII_SANS_ENTITE, c), c.source, c.langue, c.passage
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
    verification -- le `passage` de 40 caracteres de contexte de
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
    Orchestre le mode verification de `reviser_pii_residuelle.py` :
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
    # candidats_en_attente
    # ##########################################################################
    def candidats_en_attente(self) -> list[CandidatARevoir]:
        """
        Recalcule (replay deterministe, cf. docstring du module) les
        candidats REVISION_HUMAINE sur TOUS les identifiants deja
        echantillonnes (les deux strates), puis exclut ceux ayant deja
        une decision humaine persistee.
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

        deja_decides = self.decisions.cles_decidees()
        return [c for c in candidats_a_revoir(controle) if c.cle not in deja_decides]

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
