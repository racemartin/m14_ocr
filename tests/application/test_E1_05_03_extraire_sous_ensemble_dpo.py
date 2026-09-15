"""
Tests de la couche application pour ExtraireSousEnsembleDpoUseCase,
avec un faux adaptateur en memoire (meme patron que
`test_E1_05_02_extraire_sous_ensemble_sft.py`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases import (
    ExtraireSousEnsembleDpoUseCase,
    calculer_repartition_par_strate,
    formater_tableau_repartition,
)
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple, TypeSplit


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire."""

    def __init__(self, items: list[ExemplePivot]) -> None:
        self.items: dict[str, ExemplePivot] = {item.identifiant: item for item in items}

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


def _exemple_dpo(source: str = "UltraMedical-Preference", avec_split: bool = True) -> ExemplePivot:
    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(source, uuid4().hex),
        source=source,
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="Question ?"),),
        chosen=(Message(role="assistant", contenu="Bonne reponse."),),
        rejected=(Message(role="assistant", contenu="Mauvaise reponse."),),
    )
    exemple = replace(exemple, anonymise=True)
    if avec_split:
        exemple = replace(exemple, split=TypeSplit.TRAIN)
    return exemple


def _exemple_sft(source: str = "MediQAl", avec_split: bool = True) -> ExemplePivot:
    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(source, uuid4().hex),
        source=source,
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )
    exemple = replace(exemple, anonymise=True)
    if avec_split:
        exemple = replace(exemple, split=TypeSplit.TRAIN)
    return exemple


def _pool_stratifie_80_10_10(compositions: dict[str, int]) -> list[ExemplePivot]:
    """
    Construit un pool d'exemples DPO avec split deja assigne, ~80/10/10
    par source, pour tester le recoupage proportionnel par strate.
    `compositions` associe une source a son nombre total d'exemples.
    """
    exemples: list[ExemplePivot] = []
    for source, total in compositions.items():
        nombre_train = round(total * 0.8)
        nombre_val = round(total * 0.1)
        nombre_test = total - nombre_train - nombre_val
        repartition_split = (
            (TypeSplit.TRAIN, nombre_train),
            (TypeSplit.VALIDATION, nombre_val),
            (TypeSplit.TEST_CLINIQUE, nombre_test),
        )
        for split, nombre in repartition_split:
            for _ in range(nombre):
                exemple = ExemplePivot(
                    identifiant=ExemplePivot.nouvel_identifiant(source, uuid4().hex),
                    source=source,
                    type_exemple=TypeExemple.DPO,
                    langue=Langue.ANGLAIS,
                    prompt=(Message(role="user", contenu="Question ?"),),
                    chosen=(Message(role="assistant", contenu="Bonne reponse."),),
                    rejected=(Message(role="assistant", contenu="Mauvaise reponse."),),
                )
                exemples.append(replace(exemple, anonymise=True, split=split))
    return exemples


def test_extraction_ignore_les_exemples_sans_split():
    exemples = [_exemple_dpo(avec_split=True) for _ in range(5)] + [_exemple_dpo(avec_split=False) for _ in range(3)]
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleDpoUseCase(repository=repository, taille_cible=5)
    resultat = cas_usage.executer()

    assert len(resultat) == 5
    assert all(e.split is not None for e in resultat)
    assert cas_usage.nombre_avec_split == 5
    assert cas_usage.nombre_exclus == 0
    assert cas_usage.manque == 0


def test_extraction_exclut_les_exemples_sft():
    """Symetrique du trou du 12/09/2026 (cf. test SFT) : ne doit produire que du DPO."""
    exemples_dpo = [_exemple_dpo(avec_split=True) for _ in range(4)]
    exemples_sft = [_exemple_sft(avec_split=True) for _ in range(3)]
    repository = FauxRepository(exemples_dpo + exemples_sft)

    cas_usage = ExtraireSousEnsembleDpoUseCase(repository=repository, taille_cible=10)
    resultat = cas_usage.executer()

    assert len(resultat) == 4
    assert all(e.type_exemple == TypeExemple.DPO for e in resultat)
    assert cas_usage.nombre_avec_split == 4
    assert cas_usage.manque == 6


def test_extraction_exclut_les_identifiants_listes():
    exemples = [_exemple_dpo(avec_split=True) for _ in range(5)]
    identifiant_exclu = exemples[0].identifiant
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleDpoUseCase(
        repository=repository, identifiants_a_exclure=frozenset({identifiant_exclu}), taille_cible=5
    )
    resultat = cas_usage.executer()

    identifiants_resultat = {e.identifiant for e in resultat}
    assert identifiant_exclu not in identifiants_resultat
    assert len(resultat) == 4
    assert cas_usage.nombre_avec_split == 5
    assert cas_usage.nombre_exclus == 1
    assert cas_usage.nombre_disponible_final == 4


def test_extraction_rapporte_le_manque_quand_le_resultat_est_sous_la_taille_cible():
    exemples = [_exemple_dpo(avec_split=True) for _ in range(10)]
    identifiants_exclus = frozenset(e.identifiant for e in exemples[:4])
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleDpoUseCase(
        repository=repository, identifiants_a_exclure=identifiants_exclus, taille_cible=8
    )
    resultat = cas_usage.executer()

    assert len(resultat) == 6
    assert cas_usage.nombre_exclus == 4
    assert cas_usage.manque == 2


def test_extraction_manque_nul_quand_taille_cible_atteinte():
    exemples = [_exemple_dpo(avec_split=True) for _ in range(10)]
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleDpoUseCase(repository=repository, taille_cible=5)
    cas_usage.executer()

    assert cas_usage.manque == 0


def test_extraction_recoupe_a_exactement_taille_cible_quand_il_y_a_un_surplus():
    exemples = _pool_stratifie_80_10_10({"UltraMedical-Preference": 6000, "MediQAl": 2000, "MedQuAD": 2000})
    assert len(exemples) == 10000
    identifiants_exclus = frozenset(e.identifiant for e in exemples[:74])
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleDpoUseCase(
        repository=repository, identifiants_a_exclure=identifiants_exclus, taille_cible=5000
    )
    resultat = cas_usage.executer()

    assert len(resultat) == 5000
    assert cas_usage.nombre_disponible_final == 9926
    assert cas_usage.nombre_tronque == 9926 - 5000
    assert cas_usage.manque == 0
    assert len({e.identifiant for e in resultat}) == 5000


def test_extraction_recoupe_preserve_la_proportion_par_strate():
    compositions = {"UltraMedical-Preference": 6000, "MediQAl": 800, "MedQuAD": 3200}
    exemples = _pool_stratifie_80_10_10(compositions)
    repository = FauxRepository(exemples)
    total = sum(compositions.values())
    taille_cible = 2000

    cas_usage = ExtraireSousEnsembleDpoUseCase(repository=repository, taille_cible=taille_cible)
    resultat = cas_usage.executer()

    assert len(resultat) == taille_cible

    compteur_par_source: dict[str, int] = {}
    for exemple in resultat:
        compteur_par_source[exemple.source] = compteur_par_source.get(exemple.source, 0) + 1

    for source, total_source in compositions.items():
        part_attendue = taille_cible * (total_source / total)
        assert abs(compteur_par_source.get(source, 0) - part_attendue) <= 1


def test_extraction_recoupe_reste_proche_de_80_10_10_dans_chaque_strate():
    compositions = {"UltraMedical-Preference": 6000, "MediQAl": 800, "MedQuAD": 3200}
    exemples = _pool_stratifie_80_10_10(compositions)
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleDpoUseCase(repository=repository, taille_cible=2000)
    resultat = cas_usage.executer()

    repartition = calculer_repartition_par_strate(resultat)
    for cle, compteur_strate in repartition.items():
        total_strate = sum(compteur_strate.values())
        for split, part_attendue in (("train", 0.80), ("val", 0.10), ("test", 0.10)):
            part_observee = compteur_strate.get(split, 0) / total_strate
            assert abs(part_observee - part_attendue) <= 0.05, (
                f"strate {cle} split {split} : {part_observee:.3f} attendu ~{part_attendue}"
            )


def test_calculer_repartition_par_strate_et_formater_tableau_somme_le_total_attendu():
    compositions = {"UltraMedical-Preference": 600, "MediQAl": 80, "MedQuAD": 320}
    exemples = _pool_stratifie_80_10_10(compositions)

    repartition = calculer_repartition_par_strate(exemples)
    total_reparti = sum(sum(compteur.values()) for compteur in repartition.values())
    assert total_reparti == sum(compositions.values())

    tableau = formater_tableau_repartition(repartition)
    for source in compositions:
        assert f"dpo/{source}" in tableau
    lignes_donnees = tableau.splitlines()[1:]
    assert len(lignes_donnees) == len(compositions)


def test_chemin_sortie_defaut_du_cli_depend_de_la_taille_reelle():
    """`--sortie` par defaut doit refleter la `--taille` reellement utilisee, pas une valeur figee."""
    from interfaces.cli.E1_05_03_extraire_sous_ensemble_dpo import chemin_sortie_defaut

    assert chemin_sortie_defaut(5000) == "data/processed/dataset_chsa_triage_dpo_anonymise_5000.jsonl"
    assert chemin_sortie_defaut(8000) == "data/processed/dataset_chsa_triage_dpo_anonymise_8000.jsonl"
