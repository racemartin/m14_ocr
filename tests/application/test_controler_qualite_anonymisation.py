"""
Tests du controle qualite d'anonymisation, design fichier-a-fichier
(compare un ExemplePivot original a sa version anonymisee). Utilise un
FAUX VerificateurEntitesNommees (pas de spaCy reel ici ; deterministe
et rapide). L'integration reelle avec spaCy est couverte separement
par `tests/infrastructure/test_spacy_verificateur_entites.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases.uc_03_00_anonymiser_dataset import StatistiquesSource
from chsa_triage.application.use_cases.uc_03_02_controler_qualite_anonymisation import (
    STRATUM_PRINCIPAL,
    STRATUM_SANS_ENTITE,
    VERDICT_CONFIRME,
    VERDICT_FAUX_POSITIF_REGEX,
    VERDICT_REVISION_HUMAINE,
    ControleQualiteAnonymisation,
    ControlerQualiteAnonymisationUseCase,
    _est_exemple_sans_entite,
    controle_vers_dict,
    formater_rapport_markdown,
)
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.domain.ports.verificateur_entites import VerdictEntiteNommee


class FauxVerificateurEntites:
    """Verdict configure par passage exact ; pas de vraie NLP."""

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


class FauxRegistreEchantillons:
    """Faux RegistreEchantillonsControleQualite, en memoire ; une instance par test = etat neuf."""

    def __init__(self) -> None:
        self._vus: dict[str, set[str]] = {}

    def identifiants_vus(self, stratum: str) -> set[str]:
        return set(self._vus.get(stratum, set()))

    def marquer_vus(self, stratum: str, identifiants, horodatage: str) -> None:
        self._vus.setdefault(stratum, set()).update(identifiants)


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
# ControleQualiteAnonymisation.observer() : comparaison d'un couple
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
# ControlerQualiteAnonymisationUseCase : orchestration fichier-a-fichier
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
        registre_echantillons=FauxRegistreEchantillons(),
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
        registre_echantillons=FauxRegistreEchantillons(),
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
        registre_echantillons=FauxRegistreEchantillons(),
        taille_echantillon=None,
    )
    controle = cas_usage.executer()

    assert controle.nombre_exemples_observes == 0
    assert cas_usage.nombre_introuvables_dans_original == 1


# ----------------------------------------------------------------------
# Item 3 : stratum dedie "sans entite detectee" ; independant du
# tirage stratifie (type_exemple, source) ci-dessus.
# ----------------------------------------------------------------------


def test_est_exemple_sans_entite_vrai_quand_rien_ne_change():
    original = _exemple(symptomes="Rien de particulier a signaler.")
    anonymise = _anonymiser(original, "Rien de particulier a signaler.")
    assert _est_exemple_sans_entite(original, anonymise) is True


def test_est_exemple_sans_entite_faux_des_qu_un_champ_change():
    original = _exemple(symptomes="Jean Dupont est venu.")
    anonymise = _anonymiser(original, "<INFO_MASQUEE> est venu.")
    assert _est_exemple_sans_entite(original, anonymise) is False


def test_observer_sans_entite_compte_a_part_de_observer():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple(symptomes="contact jean@example.com")
    controle.observer_sans_entite(original, _anonymiser(original, "contact jean@example.com"))

    assert controle.nombre_exemples_sans_entite_observes == 1
    assert controle.nombre_exemples_observes == 0
    assert len(controle.candidats_pii_sans_entite) == 1
    assert controle.candidats_pii_sans_entite[0].type_motif == "email"
    assert controle.candidats_pii == []
    assert len(controle.exemples_sans_entite) == 1


def test_use_case_isole_le_stratum_sans_entite_independamment_du_tirage_principal():
    originaux_sans_entite = [_exemple("MediQAl", symptomes=f"Rien de notable numero {i}.") for i in range(5)]
    anonymises_sans_entite = [_anonymiser(e, e.symptomes) for e in originaux_sans_entite]

    originaux_avec_entite = [_exemple("MediQAl", symptomes=f"Jean Dupont {i} est venu.") for i in range(5)]
    anonymises_avec_entite = [_anonymiser(e, "<INFO_MASQUEE> est venu.") for e in originaux_avec_entite]

    cas_usage = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux_sans_entite + originaux_avec_entite),
        repository_anonymise=FauxRepository(anonymises_sans_entite + anonymises_avec_entite),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=FauxRegistreEchantillons(),
        taille_echantillon=None,
        taille_echantillon_sans_entite=None,
    )
    controle = cas_usage.executer()

    # Le tirage principal (§1-4) porte toujours sur TOUS les exemples,
    # avec ou sans entite ; inchange par l'ajout du nouveau stratum.
    assert controle.nombre_exemples_observes == 10
    # Le nouveau stratum isole EXACTEMENT les 5 "sans entite".
    assert controle.nombre_disponibles_sans_entite == 5
    assert controle.nombre_exemples_sans_entite_observes == 5


def test_use_case_echantillonne_le_stratum_sans_entite_independamment():
    originaux = [_exemple("MediQAl", symptomes=f"Rien de notable numero {i}.") for i in range(20)]
    anonymises = [_anonymiser(e, e.symptomes) for e in originaux]

    cas_usage = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=FauxRegistreEchantillons(),
        taille_echantillon=None,
        taille_echantillon_sans_entite=5,
        graine_aleatoire_sans_entite=7,
    )
    controle = cas_usage.executer()

    assert controle.nombre_disponibles_sans_entite == 20
    assert controle.nombre_exemples_sans_entite_observes == 5


def test_controle_vers_dict_inclut_le_stratum_sans_entite():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple("MediQAl", "contact jean@example.com")
    controle.observer_sans_entite(original, _anonymiser(original, "contact jean@example.com"))

    d = controle_vers_dict(
        controle,
        horodatage="2026-09-08T10:00:00+00:00",
        dataset_original="data/processed/dataset_pivot.jsonl",
        dataset_anonymise="data/processed/dataset_pivot_anonymise.jsonl",
    )

    stratum = d["stratum_sans_entite_detectee"]
    assert stratum["nombre_observes"] == 1
    assert len(stratum["candidats_pii_residuelle"]) == 1
    assert len(stratum["exemples"]) == 1


def test_rapport_markdown_inclut_la_section_stratum_sans_entite():
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple("MediQAl", "contact jean@example.com")
    controle.observer_sans_entite(original, _anonymiser(original, "contact jean@example.com"))
    controle.nombre_disponibles_sans_entite = 42

    markdown = formater_rapport_markdown(
        controle,
        horodatage="2026-09-08T10:00:00+00:00",
        dataset_original="data/processed/dataset_pivot.jsonl",
        dataset_anonymise="data/processed/dataset_pivot_anonymise.jsonl",
        taille_echantillon_demandee=200,
        total_anonymise_disponible=5000,
        statistiques_cumulees={},
    )

    assert "Stratum dedie" in markdown
    assert "**42**" in markdown
    assert "jean@example.com" in markdown


# ----------------------------------------------------------------------
# Muestreo incremental (09/09/2026) ; meme patron que --limite pour
# AnonymiserDatasetUseCase : une deuxieme execution ne doit jamais
# re-echantillonner un identifiant deja vu lors d'une execution
# precedente, sur AUCUN des deux strates (principal / sans_entite).
# ----------------------------------------------------------------------


def test_pii_residuelle_conserve_debut_fin_du_match():
    """`debut`/`fin` du match dans le texte anonymise, necessaires a la cle stable de revision humaine."""
    controle = ControleQualiteAnonymisation(verificateur_entites=FauxVerificateurEntites({}))
    original = _exemple(symptomes="contact: jean@example.com")
    texte_anonymise = "contact: jean@example.com"
    controle.observer(original, _anonymiser(original, texte_anonymise))

    candidat = controle.candidats_pii[0]
    assert texte_anonymise[candidat.debut:candidat.fin] == "jean@example.com"


def test_candidat_faux_positif_conserve_debut_fin_du_fragment():
    verificateur = FauxVerificateurEntites({"Polycystic": VerdictEntiteNommee.AUCUNE_ENTITE})
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur)
    texte_original = "Il souffre de Polycystic ovarian syndrome."
    original = _exemple(symptomes=texte_original)
    controle.observer(original, _anonymiser(original, "Il souffre de <INFO_MASQUEE> ovarian syndrome."))

    candidat = controle.candidats_faux_positifs[0]
    assert texte_original[candidat.debut:candidat.fin] == "Polycystic"


def test_max_faux_positifs_par_source_none_ne_plafonne_pas():
    verificateur = FauxVerificateurEntites({})  # AUCUNE_ENTITE par defaut -> faux positif
    controle = ControleQualiteAnonymisation(verificateur_entites=verificateur, max_faux_positifs_par_source=None)

    for i in range(15):
        original = _exemple(symptomes=f"Terme{i} inhabituel present.")
        controle.observer(original, _anonymiser(original, "<INFO_MASQUEE> inhabituel present."))

    assert len(controle.candidats_faux_positifs) == 15


def test_muestreo_incremental_exclut_les_identifiants_deja_echantillonnes():
    originaux = [_exemple("MediQAl") for _ in range(20)]
    anonymises = [_anonymiser(e, "[ANON]") for e in originaux]
    registre = FauxRegistreEchantillons()

    cas_usage_1 = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=registre,
        taille_echantillon=10,
    )
    controle_1 = cas_usage_1.executer()
    assert controle_1.nombre_exemples_observes == 10
    premiere_vague = registre.identifiants_vus(STRATUM_PRINCIPAL)
    assert len(premiere_vague) == 10

    cas_usage_2 = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=registre,
        taille_echantillon=10,
    )
    controle_2 = cas_usage_2.executer()

    # La deuxieme vague tire EXACTEMENT les 10 identifiants restants,
    # sans jamais recroiser ceux de la premiere.
    assert controle_2.nombre_exemples_observes == 10
    deuxieme_vague = registre.identifiants_vus(STRATUM_PRINCIPAL) - premiere_vague
    assert len(deuxieme_vague) == 10
    assert deuxieme_vague.isdisjoint(premiere_vague)
    assert registre.identifiants_vus(STRATUM_PRINCIPAL) == {e.identifiant for e in originaux}


def test_muestreo_incremental_troisieme_execution_ne_trouve_plus_rien():
    originaux = [_exemple("MediQAl") for _ in range(10)]
    anonymises = [_anonymiser(e, "[ANON]") for e in originaux]
    registre = FauxRegistreEchantillons()

    for _ in range(2):
        ControlerQualiteAnonymisationUseCase(
            repository_original=FauxRepository(originaux),
            repository_anonymise=FauxRepository(anonymises),
            verificateur_entites=FauxVerificateurEntites({}),
            registre_echantillons=registre,
            taille_echantillon=10,
        ).executer()

    cas_usage_3 = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=registre,
        taille_echantillon=10,
    )
    controle_3 = cas_usage_3.executer()

    assert controle_3.nombre_exemples_observes == 0


def test_muestreo_incremental_stratum_sans_entite_exclut_les_deja_vus():
    originaux = [_exemple("MediQAl", symptomes=f"Rien de notable numero {i}.") for i in range(20)]
    anonymises = [_anonymiser(e, e.symptomes) for e in originaux]  # texte inchange -> stratum "sans entite"
    registre = FauxRegistreEchantillons()

    cas_usage_1 = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=registre,
        taille_echantillon_sans_entite=10,
        graine_aleatoire_sans_entite=7,
    )
    controle_1 = cas_usage_1.executer()
    assert controle_1.nombre_exemples_sans_entite_observes == 10

    cas_usage_2 = ControlerQualiteAnonymisationUseCase(
        repository_original=FauxRepository(originaux),
        repository_anonymise=FauxRepository(anonymises),
        verificateur_entites=FauxVerificateurEntites({}),
        registre_echantillons=registre,
        taille_echantillon_sans_entite=10,
        graine_aleatoire_sans_entite=7,
    )
    controle_2 = cas_usage_2.executer()

    assert controle_2.nombre_exemples_sans_entite_observes == 10
    # Le total du stratum reste inchange par le muestreo incremental --
    # seul le TIRAGE est restreint aux identifiants pas encore vus.
    assert controle_2.nombre_disponibles_sans_entite == 20
    assert registre.identifiants_vus(STRATUM_SANS_ENTITE) == {e.identifiant for e in anonymises}
