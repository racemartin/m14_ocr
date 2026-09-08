"""
Tests du controle qualite d'anonymisation, design fichier-a-fichier
(compare un ExemplePivot original a sa version anonymisee). Utilise un
FAUX VerificateurEntitesNommees (pas de spaCy reel ici -- deterministe
et rapide). L'integration reelle avec spaCy est couverte separement
par `tests/infrastructure/test_spacy_verificateur_entites.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases.anonymiser_dataset import StatistiquesSource
from chsa_triage.application.use_cases.controler_qualite_anonymisation import (
    VERDICT_CONFIRME,
    VERDICT_FAUX_POSITIF_REGEX,
    VERDICT_REVISION_HUMAINE,
    ControleQualiteAnonymisation,
    ControlerQualiteAnonymisationUseCase,
    controle_vers_dict,
    formater_rapport_markdown,
)
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.domain.ports.verificateur_entites import VerdictEntiteNommee


class FauxVerificateurEntites:
    """Verdict configure par passage exact -- pas de vraie NLP."""

    def __init__(self, verdicts_par_passage: dict[str, VerdictEntiteNommee]) -> None:
        self._verdicts = verdicts_par_passage

    def verifier(self, texte: str, langue: str, debut: int, fin: int) -> VerdictEntiteNommee:
        return self._verdicts.get(texte[debut:fin], VerdictEntiteNommee.AUCUNE_ENTITE)


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire."""

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


def _exemple(source: str = "MediQAl", symptomes: str = "Fievre", identifiant: str | None = None) -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=identifiant or ExemplePivot.nouvel_identifiant(source, cle),
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


# ----------------------------------------------------------------------
# ControleQualiteAnonymisation.observer() -- comparaison d'un couple
# ----------------------------------------------------------------------


def test_observer_conserve_un_exemple_illustratif_par_source():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}), max_exemples_par_source=2)

    for _ in range(3):
        original = _exemple("MediQAl", "original patient X")
        controle.observer(original, _anonymiser(original, "original <INFO_MASQUEE>"))

    assert controle.nombre_exemples_observes == 3
    assert len(controle.exemples_par_source["MediQAl"]) == 2  # plafonne


def test_pii_residuelle_email_est_confirmee_directement_sans_spacy():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple(symptomes="contact: jean@example.com")
    controle.observer(original, _anonymiser(original, "contact: jean@example.com"))  # pas masque

    assert len(controle.candidats_pii) == 1
    assert controle.candidats_pii[0].type_motif == "email"
    assert controle.candidats_pii[0].verdict == VERDICT_CONFIRME


def test_bigramme_confirme_par_spacy_est_marque_confirme():
    verificateur = FauxVerificateurEntites({"Jean Dupont": VerdictEntiteNommee.ENTITE_PERTINENTE})
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur)
    original = _exemple(symptomes="orig")
    controle.observer(original, _anonymiser(original, "Jean Dupont est venu."))

    (candidat,) = [c for c in controle.candidats_pii if c.type_motif == "bigramme_capitalise"]
    assert candidat.verdict == VERDICT_CONFIRME


def test_bigramme_sans_entite_spacy_est_ecarte_comme_faux_positif_regex():
    verificateur = FauxVerificateurEntites({"Chronic Pain": VerdictEntiteNommee.AUCUNE_ENTITE})
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur)
    original = _exemple(symptomes="orig")
    controle.observer(original, _anonymiser(original, "Chronic Pain syndrome diagnosed."))

    (candidat,) = [c for c in controle.candidats_pii if c.type_motif == "bigramme_capitalise"]
    assert candidat.verdict == VERDICT_FAUX_POSITIF_REGEX


def test_bigramme_avec_entite_non_pertinente_est_en_attente_de_revision_humaine():
    verificateur = FauxVerificateurEntites({"Some Product": VerdictEntiteNommee.ENTITE_NON_PERTINENTE})
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur)
    original = _exemple(symptomes="orig")
    controle.observer(original, _anonymiser(original, "Some Product was mentioned."))

    (candidat,) = [c for c in controle.candidats_pii if c.type_motif == "bigramme_capitalise"]
    assert candidat.verdict == VERDICT_REVISION_HUMAINE


def test_masquage_confirme_par_spacy_n_est_pas_un_candidat_faux_positif():
    """Un fragment original masque que spaCy reconnait comme entite nommee = masquage legitime, pas signale."""
    verificateur = FauxVerificateurEntites({"Jean Dupont": VerdictEntiteNommee.ENTITE_PERTINENTE})
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur)
    original = _exemple(symptomes="Contactez Jean Dupont pour un avis.")
    controle.observer(original, _anonymiser(original, "Contactez <INFO_MASQUEE> pour un avis."))

    assert controle.candidats_faux_positifs == []


def test_masquage_sans_entite_spacy_est_signale_comme_candidat_faux_positif():
    """Un fragment original masque que spaCy NE reconnait PAS comme entite = candidat de sur-masquage."""
    verificateur = FauxVerificateurEntites({"Polycystic": VerdictEntiteNommee.AUCUNE_ENTITE})
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur)
    original = _exemple(symptomes="Il souffre de Polycystic ovarian syndrome.")
    controle.observer(original, _anonymiser(original, "Il souffre de <INFO_MASQUEE> ovarian syndrome."))

    assert len(controle.candidats_faux_positifs) == 1
    assert controle.candidats_faux_positifs[0].fragment_masque == "Polycystic"
    assert controle.candidats_faux_positifs[0].verdict == VERDICT_FAUX_POSITIF_REGEX


def test_candidats_faux_positifs_est_plafonne_par_source():
    verificateur = FauxVerificateurEntites({})  # AUCUNE_ENTITE par defaut pour tout
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur, max_faux_positifs_par_source=1)

    for i in range(3):
        original = _exemple(symptomes=f"Terme{i} inhabituel present.")
        controle.observer(original, _anonymiser(original, f"<INFO_MASQUEE> inhabituel present."))

    assert len(controle.candidats_faux_positifs) == 1


def test_controle_vers_dict_serialise_toutes_les_sections():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple("MediQAl", "contact jean@example.com")
    controle.observer(original, _anonymiser(original, "contact jean@example.com"))

    d = controle_vers_dict(
        controle,
        horodatage="2026-09-08T10:00:00+00:00",
        dataset_original="data/processed/dataset_pivot.jsonl",
        dataset_anonymise="data/processed/dataset_pivot_anonymise.jsonl",
    )

    assert d["horodatage"] == "2026-09-08T10:00:00+00:00"
    assert d["nombre_exemples_observes"] == 1
    assert len(d["candidats_pii_residuelle"]) == 1
    assert "MediQAl" in d["exemples_par_source"]


def test_rapport_markdown_indique_la_portee_echantillon():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple("MediQAl", "contact jean@example.com")
    controle.observer(original, _anonymiser(original, "contact jean@example.com"))

    markdown = formater_rapport_markdown(
        controle,
        horodatage="2026-09-08T10:00:00+00:00",
        dataset_original="data/processed/dataset_pivot.jsonl",
        dataset_anonymise="data/processed/dataset_pivot_anonymise.jsonl",
        taille_echantillon_demandee=200,
        total_anonymise_disponible=5000,
        statistiques_cumulees={"MediQAl": StatistiquesSource(registres_traites=5000, registres_avec_entite=4000, entites_par_type={"PERSON": 100})},
    )

    assert "1 exemples compares" in markdown or "1 exemples" in markdown
    assert "5000 exemples disponibles" in markdown
    assert "jean@example.com" in markdown
    assert "PERSON 100" in markdown


# ----------------------------------------------------------------------
# ControlerQualiteAnonymisationUseCase -- orchestration fichier-a-fichier
# ----------------------------------------------------------------------


def test_use_case_croise_par_identifiant_entre_original_et_anonymise():
    original = _exemple("MediQAl", "contact jean@example.com")
    anonymise = _anonymiser(original, "contact jean@example.com")

    repository_original = FauxRepository([original])
    repository_anonymise = FauxRepository([anonymise])

    cas_usage = ControlerQualiteAnonymisationUseCase(
        repository_original=repository_original,
        repository_anonymise=repository_anonymise,
        verificateur_entites=FauxVerificateurEntites({}),
        taille_echantillon=None,
    )
    controle = cas_usage.executer()

    assert controle.nombre_exemples_observes == 1
    assert cas_usage.nombre_introuvables_dans_original == 0


def test_use_case_echantillonne_quand_taille_echantillon_est_plus_petite():
    originaux = [_exemple("MediQAl") for _ in range(50)] + [_exemple("MedQuAD") for _ in range(50)]
    anonymises = [_anonymiser(e, "[ANON]") for e in originaux]

    cas_usage = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        taille_echantillon=10,
        graine_aleatoire=42,
    )
    controle = cas_usage.executer()

    assert controle.nombre_exemples_observes == 10


def test_use_case_signale_les_identifiants_introuvables_dans_l_original():
    """Garde-fou : un exemple present dans --anonymise mais absent de --dataset ne doit pas planter."""
    original = _exemple("MediQAl")
    anonymise_orphelin = _exemple("MediQAl", identifiant="chsa-orphelin-inexistant")

    cas_usage = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository([original]),
        repository_anonymise=FauxRepository([anonymise_orphelin]),
        verificateur_entites=FauxVerificateurEntites({}),
        taille_echantillon=None,
    )
    controle = cas_usage.executer()

    assert controle.nombre_exemples_observes == 0
    assert cas_usage.nombre_introuvables_dans_original == 1
