"""
Tests du cas d'usage de revision humaine persistee (NF2). Utilise les
memes faux adaptateurs en memoire que `test_controler_qualite_anonymisation.py`
(FauxVerificateurEntites, FauxRepository) plus un faux registre
d'echantillons et un faux registre de decisions, pour tester le
"replay cumule" sans dependre de spaCy ni du disque.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases.uc_03_02_controler_qualite_anonymisation import (
    STRATUM_PRINCIPAL,
    STRATUM_SANS_ENTITE,
)
from chsa_triage.application.use_cases.uc_03_03_reviser_pii_residuelle import (
    ReviserPiiResiduelleUseCase,
    texte_original_et_anonymise,
)
from chsa_triage.domain.model import (
    DECISION_ACCEPTE,
    SOURCE_CANDIDATS_PII,
    CleCandidatRevision,
    DecisionRevisionHumaine,
    ExemplePivot,
    Langue,
    Message,
    TypeExemple,
)
from chsa_triage.domain.ports.verificateur_entites import VerdictEntiteNommee


class FauxVerificateurEntites:
    def __init__(self, verdicts_par_passage: dict[str, VerdictEntiteNommee]) -> None:
        self._verdicts = verdicts_par_passage

    def verifier(self, texte: str, langue: str, debut: int, fin: int) -> VerdictEntiteNommee:
        return self._verdicts.get(texte[debut:fin], VerdictEntiteNommee.AUCUNE_ENTITE)


class FauxRepository:
    def __init__(self, items: list[ExemplePivot] | None = None) -> None:
        self.items: dict[str, ExemplePivot] = {item.identifiant: item for item in (items or [])}

    def sauvegarder(self, item: ExemplePivot) -> None:
        self.items[item.identifiant] = item

    def sauvegarder_plusieurs(self, items: Iterable[ExemplePivot]) -> None:
        for item in items:
            self.items[item.identifiant] = item

    def trouver_par_id(self, identifiant: str):
        return self.items.get(identifiant)

    def lister(self, filtre: dict | None = None):
        for exemple in self.items.values():
            if filtre is None or all(getattr(exemple, cle) == valeur for cle, valeur in filtre.items()):
                yield exemple

    def compter(self, filtre: dict | None = None) -> int:
        return sum(1 for _ in self.lister(filtre))

    def identifiants_existants(self) -> set[str]:
        return set(self.items.keys())


class FauxRegistreEchantillons:
    def __init__(self, vus: dict[str, set[str]] | None = None) -> None:
        self._vus: dict[str, set[str]] = {k: set(v) for k, v in (vus or {}).items()}

    def identifiants_vus(self, stratum: str) -> set[str]:
        return set(self._vus.get(stratum, set()))

    def marquer_vus(self, stratum: str, identifiants, horodatage: str) -> None:
        self._vus.setdefault(stratum, set()).update(identifiants)


class FauxDecisions:
    def __init__(self) -> None:
        self._decisions: dict[CleCandidatRevision, DecisionRevisionHumaine] = {}

    def toutes(self) -> list[DecisionRevisionHumaine]:
        return list(self._decisions.values())

    def cles_decidees(self) -> set[CleCandidatRevision]:
        return set(self._decisions.keys())

    def trouver(self, cle: CleCandidatRevision):
        return self._decisions.get(cle)

    def enregistrer(self, decision: DecisionRevisionHumaine) -> None:
        self._decisions[decision.cle] = decision


def _exemple(source: str = "MediQAl", symptomes: str = "Fievre") -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(source, cle),
        identifiant_source_brute=cle,
        source=source,
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes=symptomes,
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )


def _anonymiser(exemple: ExemplePivot, symptomes_anon: str) -> ExemplePivot:
    return replace(exemple, symptomes=symptomes_anon, anonymise=True)


def test_candidats_en_attente_replay_sur_les_identifiants_deja_echantillonnes():
    """Le use case doit retrouver un candidat REVISION_HUMAINE meme si sa vague n'est plus le dernier lot."""
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original = _exemple(symptomes="orig")
    anonymise = _anonymiser(original, "Some Product was mentioned.")

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=FauxDecisions(),
    )

    en_attente = cas_usage.candidats_en_attente()

    assert len(en_attente) == 1
    assert en_attente[0].cle.identifiant == original.identifiant
    assert en_attente[0].cle.source_liste == SOURCE_CANDIDATS_PII


def test_candidats_en_attente_exclut_ceux_deja_decides():
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original = _exemple(symptomes="orig")
    anonymise = _anonymiser(original, "Some Product was mentioned.")

    decisions = FauxDecisions()
    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=decisions,
    )

    (candidat,) = cas_usage.candidats_en_attente()
    cas_usage.enregistrer_decision(candidat, DECISION_ACCEPTE, "2026-09-09T10:00:00+00:00")

    assert cas_usage.candidats_en_attente() == []


def test_enregistrer_decision_persiste_immediatement():
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original = _exemple(symptomes="orig")
    anonymise = _anonymiser(original, "Some Product was mentioned.")
    decisions = FauxDecisions()

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=decisions,
    )
    (candidat,) = cas_usage.candidats_en_attente()

    enregistree = cas_usage.enregistrer_decision(candidat, DECISION_ACCEPTE, "2026-09-09T10:00:00+00:00", note="ok")

    relue = decisions.trouver(candidat.cle)
    assert relue is not None
    assert relue.decision == DECISION_ACCEPTE
    assert relue.note == "ok"
    assert enregistree.cle == candidat.cle


def test_candidats_en_attente_couvre_les_deux_strates():
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original_principal = _exemple(symptomes="orig 1")
    anonymise_principal = _anonymiser(original_principal, "Some Product was mentioned.")
    original_sans_entite = _exemple(symptomes="Some Product was mentioned.")
    anonymise_sans_entite = _anonymiser(original_sans_entite, "Some Product was mentioned.")  # texte inchange

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original_principal, original_sans_entite]),
        repository_anonymise=FauxRepository([anonymise_principal, anonymise_sans_entite]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons(
            {
                STRATUM_PRINCIPAL: {original_principal.identifiant},
                STRATUM_SANS_ENTITE: {original_sans_entite.identifiant},
            }
        ),
        decisions=FauxDecisions(),
    )

    en_attente = cas_usage.candidats_en_attente()
    identifiants = {c.cle.identifiant for c in en_attente}

    assert identifiants == {original_principal.identifiant, original_sans_entite.identifiant}


def test_texte_original_et_anonymise_retrouve_le_champ_nomme():
    original = _exemple(symptomes="Jean Dupont est venu.")
    anonymise = _anonymiser(original, "<INFO_MASQUEE> est venu.")

    paire = texte_original_et_anonymise(original, anonymise, "symptomes")

    assert paire == ("Jean Dupont est venu.", "<INFO_MASQUEE> est venu.")


def test_texte_original_et_anonymise_retourne_none_pour_champ_inconnu():
    original = _exemple()
    anonymise = _anonymiser(original, original.symptomes)

    assert texte_original_et_anonymise(original, anonymise, "champ_inexistant") is None


def test_identifiants_a_exclure_publication_inclut_confirme_et_pendant_revision_humaine():
    """Un candidat CONFIRME (email non masque) et un candidat REVISION_HUMAINE portent chacun leur identifiant."""
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original_confirme = _exemple(symptomes="contact: jean@example.com")
    anonymise_confirme = _anonymiser(original_confirme, "contact: jean@example.com")  # email jamais masque
    original_en_attente = _exemple(symptomes="orig")
    anonymise_en_attente = _anonymiser(original_en_attente, "Some Product was mentioned.")

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original_confirme, original_en_attente]),
        repository_anonymise=FauxRepository([anonymise_confirme, anonymise_en_attente]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons(
            {STRATUM_PRINCIPAL: {original_confirme.identifiant, original_en_attente.identifiant}}
        ),
        decisions=FauxDecisions(),
    )

    razons = cas_usage.identifiants_a_exclure_publication()

    assert razons == {
        original_confirme.identifiant: "confirme",
        original_en_attente.identifiant: "pendant_revision_humaine",
    }


def test_identifiants_a_exclure_publication_exclut_ceux_deja_acceptes():
    """Un candidat REVISION_HUMAINE avec decision DECISION_ACCEPTE ne doit plus figurer dans l'export."""
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original = _exemple(symptomes="orig")
    anonymise = _anonymiser(original, "Some Product was mentioned.")

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=FauxDecisions(),
    )
    (candidat,) = cas_usage.candidats_en_attente()
    cas_usage.enregistrer_decision(candidat, DECISION_ACCEPTE, "2026-09-11T10:00:00+00:00")

    assert cas_usage.identifiants_a_exclure_publication() == {}


def test_identifiants_a_exclure_publication_deduplique_plusieurs_candidats_du_meme_identifiant():
    """Deux candidats REVISION_HUMAINE distincts sur le meme identifiant ne donnent qu'UNE entree exportee."""
    verificateur = FauxVerificateurEntites(
        {"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE, "Other Brand": VerdictEntiteNommee.ENTITE_NON_PERTINENTE}
    )
    original = _exemple(symptomes="orig")
    anonymise = _anonymiser(original, "Some Product met Other Brand.")

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=FauxDecisions(),
    )

    en_attente = cas_usage.candidats_en_attente()
    assert len(en_attente) >= 2  # deux candidats distincts, meme identifiant

    razons = cas_usage.identifiants_a_exclure_publication()

    assert razons == {original.identifiant: "pendant_revision_humaine"}


def test_identifiants_a_exclure_publication_confirme_gagne_sur_pendant_revision_humaine():
    """Un identifiant avec a la fois un candidat CONFIRME et un candidat REVISION_HUMAINE garde le motif 'confirme'."""
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    original = _exemple(symptomes="contact: jean@example.com, Some Product was mentioned.")
    anonymise = _anonymiser(original, "contact: jean@example.com, Some Product was mentioned.")

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=FauxDecisions(),
    )

    razons = cas_usage.identifiants_a_exclure_publication()

    assert razons == {original.identifiant: "confirme"}


def test_candidats_en_attente_ignore_les_verdicts_deja_tranches():
    """Un candidat CONFIRME/FAUX_POSITIF_REGEX (regex+spaCy tranchent seuls) ne doit jamais apparaitre ici."""
    verificateur = FauxVerificateurEntites({})  # AUCUNE_ENTITE partout -> jamais REVISION_HUMAINE
    original = _exemple(symptomes="contact: jean@example.com")
    anonymise = _anonymiser(original, "contact: jean@example.com")  # email jamais masque -> VERDICT_CONFIRME

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise]),
        verificateur_entites=verificateur,
        registre_echantillons=FauxRegistreEchantillons({STRATUM_PRINCIPAL: {original.identifiant}}),
        decisions=FauxDecisions(),
    )

    assert cas_usage.candidats_en_attente() == []
