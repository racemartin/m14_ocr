"""
Tests de la couche application pour DecouperSplitsUseCase, avec de
faux adaptateurs en memoire.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases import DecouperSplitsUseCase
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple, TypeSplit


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire, filtre inclus."""

    def __init__(self, items: list[ExemplePivot]) -> None:
        self.items: dict[str, ExemplePivot] = {item.identifiant: item for item in items}
        self.appels_sauvegarder = 0

    def sauvegarder(self, item: ExemplePivot) -> None:
        self.appels_sauvegarder += 1
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


def _exemple_anonymise(source: str, type_exemple: TypeExemple = TypeExemple.SFT) -> ExemplePivot:
    return replace(
        ExemplePivot(
            identifiant=ExemplePivot.nouvel_identifiant(source, uuid4().hex),
            source=source,
            type_exemple=type_exemple,
            langue=Langue.FRANCAIS,
            prompt=(Message(role="user", contenu="Question ?"),),
            completion=(Message(role="assistant", contenu="Reponse."),),
        ),
        anonymise=True,
    )


def test_decouper_splits_ne_touche_pas_les_exemples_non_anonymises():
    exemples = [_exemple_anonymise("MediQAl") for _ in range(10)]
    exemples.append(replace(_exemple_anonymise("MediQAl"), anonymise=False))
    repository = FauxRepository(exemples)

    cas_usage = DecouperSplitsUseCase(repository=repository)
    decompte = cas_usage.executer()

    assert sum(decompte.values()) == 10
    non_anonymise = [e for e in repository.items.values() if not e.anonymise]
    assert len(non_anonymise) == 1
    assert non_anonymise[0].split is None


def test_decouper_splits_appelle_sauvegarder_plusieurs_une_seule_fois():
    exemples = [_exemple_anonymise("MediQAl") for _ in range(50)]
    repository = FauxRepository(exemples)

    cas_usage = DecouperSplitsUseCase(repository=repository)
    cas_usage.executer()

    # sauvegarder() (par item) ne doit jamais etre appele -- seul
    # sauvegarder_plusieurs() doit l'etre (bug O(n^2) corrige).
    assert repository.appels_sauvegarder == 0
    assert all(e.split is not None for e in repository.items.values())


def test_decouper_splits_est_stratifie_par_type_exemple_et_source():
    """
    Une source tres petite (5 exemples) et une source tres grande
    (500 exemples) doivent chacune se retrouver representees dans
    train/val/test -- pas juste la grande a cause d'un shuffle global.
    """
    exemples = (
        [_exemple_anonymise("FrenchMedMCQA", TypeExemple.SFT) for _ in range(5)]
        + [_exemple_anonymise("UltraMedical-Preference", TypeExemple.DPO) for _ in range(500)]
    )
    repository = FauxRepository(exemples)

    cas_usage = DecouperSplitsUseCase(repository=repository, proportion_val=0.10, proportion_test=0.10)
    cas_usage.executer()

    petite_source = [e for e in repository.items.values() if e.source == "FrenchMedMCQA"]
    splits_petite_source = {e.split for e in petite_source}
    # Avec seulement 5 exemples et 10%/10%, train doit au moins etre present.
    assert TypeSplit.TRAIN in splits_petite_source

    grande_source = [e for e in repository.items.values() if e.source == "UltraMedical-Preference"]
    splits_grande_source = {e.split for e in grande_source}
    assert splits_grande_source == {TypeSplit.TRAIN, TypeSplit.VALIDATION, TypeSplit.TEST_CLINIQUE}


def test_decouper_splits_n_sous_echantillonne_avant_repartition():
    """
    --n doit prelever un sous-ensemble stratifie (type_exemple, source)
    AVANT le decoupage train/val/test : seuls les exemples selectionnes
    recoivent un split, le reste demeure `split=None`.
    """
    exemples = (
        [_exemple_anonymise("MediQAl", TypeExemple.SFT) for _ in range(80)]
        + [_exemple_anonymise("MedQuAD", TypeExemple.SFT) for _ in range(15)]
        + [_exemple_anonymise("UltraMedical-Preference", TypeExemple.DPO) for _ in range(5)]
    )
    repository = FauxRepository(exemples)

    cas_usage = DecouperSplitsUseCase(repository=repository, n=20, graine_aleatoire=42)
    decompte = cas_usage.executer()

    assert sum(decompte.values()) == 20
    avec_split = [e for e in repository.items.values() if e.split is not None]
    assert len(avec_split) == 20
    sources_reparties = {e.source for e in avec_split}
    assert sources_reparties == {"MediQAl", "MedQuAD", "UltraMedical-Preference"}


def test_decouper_splits_n_superieur_au_disponible_reparti_tout():
    exemples = [_exemple_anonymise("MediQAl") for _ in range(10)]
    repository = FauxRepository(exemples)

    cas_usage = DecouperSplitsUseCase(repository=repository, n=5000)
    decompte = cas_usage.executer()

    assert sum(decompte.values()) == 10


def test_decouper_splits_n_none_comportement_inchange():
    exemples = [_exemple_anonymise("MediQAl") for _ in range(10)]
    repository = FauxRepository(exemples)

    cas_usage = DecouperSplitsUseCase(repository=repository, n=None)
    decompte = cas_usage.executer()

    assert sum(decompte.values()) == 10


def test_decouper_splits_croissance_stable_ne_reordonne_jamais_les_deja_assignes():
    """
    Point de vigilance du cahier des charges : le jeu de test ne doit
    jamais etre reutilise en entrainement. Agrandir le dataset
    (--n 5000 puis --n 10000, meme repository) ne doit JAMAIS deplacer
    un exemple deja reparti d'un split vers un autre.
    """
    exemples = (
        [_exemple_anonymise("MediQAl", TypeExemple.SFT) for _ in range(80)]
        + [_exemple_anonymise("MedQuAD", TypeExemple.SFT) for _ in range(15)]
        + [_exemple_anonymise("UltraMedical-Preference", TypeExemple.DPO) for _ in range(5)]
        + [_exemple_anonymise("FrenchMedMCQA", TypeExemple.SFT) for _ in range(200)]
    )
    repository = FauxRepository(exemples)

    premiere_cas_usage = DecouperSplitsUseCase(repository=repository, n=100, graine_aleatoire=42)
    premier_decompte = premiere_cas_usage.executer()

    assert sum(premier_decompte.values()) == 100
    assert premiere_cas_usage.nombre_deja_assignes == 0
    assert premiere_cas_usage.nombre_nouveaux == 100

    splits_apres_premiere_execution = {
        identifiant: exemple.split
        for identifiant, exemple in repository.items.items()
        if exemple.split is not None
    }
    assert len(splits_apres_premiere_execution) == 100

    seconde_cas_usage = DecouperSplitsUseCase(repository=repository, n=200, graine_aleatoire=42)
    second_decompte = seconde_cas_usage.executer()

    assert sum(second_decompte.values()) == 200
    assert seconde_cas_usage.nombre_deja_assignes == 100
    assert seconde_cas_usage.nombre_nouveaux == 100

    for identifiant, split_original in splits_apres_premiere_execution.items():
        assert repository.items[identifiant].split == split_original, (
            f"exemple {identifiant} a change de split entre les deux executions -- fuite train/test"
        )

    avec_split_apres_seconde_execution = [e for e in repository.items.values() if e.split is not None]
    assert len(avec_split_apres_seconde_execution) == 200


def test_decouper_splits_n_inferieur_ou_egal_au_deja_assigne_ne_fait_rien_de_nouveau():
    """Reduire un decoupage deja fait n'est pas supporte : --n <= deja assigne ne touche a rien."""
    exemples = [_exemple_anonymise("MediQAl") for _ in range(30)]
    repository = FauxRepository(exemples)

    DecouperSplitsUseCase(repository=repository, n=20, graine_aleatoire=42).executer()
    splits_avant = {identifiant: e.split for identifiant, e in repository.items.items()}

    cas_usage = DecouperSplitsUseCase(repository=repository, n=10, graine_aleatoire=42)
    decompte = cas_usage.executer()

    assert cas_usage.nombre_nouveaux == 0
    assert cas_usage.nombre_deja_assignes == 20
    assert sum(decompte.values()) == 20
    assert {identifiant: e.split for identifiant, e in repository.items.items()} == splits_avant


def test_decouper_splits_sans_n_complete_ce_qui_manque_sans_toucher_au_deja_assigne():
    """Sans --n, le mode "tout" ne recalcule plus depuis zero : il complete seulement ce qui manque."""
    exemples = [_exemple_anonymise("MediQAl") for _ in range(30)]
    repository = FauxRepository(exemples)

    DecouperSplitsUseCase(repository=repository, n=10, graine_aleatoire=42).executer()
    splits_avant = {identifiant: e.split for identifiant, e in repository.items.items() if e.split is not None}
    assert len(splits_avant) == 10

    cas_usage = DecouperSplitsUseCase(repository=repository, n=None, graine_aleatoire=42)
    decompte = cas_usage.executer()

    assert cas_usage.nombre_deja_assignes == 10
    assert cas_usage.nombre_nouveaux == 20
    assert sum(decompte.values()) == 30
    for identifiant, split_original in splits_avant.items():
        assert repository.items[identifiant].split == split_original
    assert all(e.split is not None for e in repository.items.values())
