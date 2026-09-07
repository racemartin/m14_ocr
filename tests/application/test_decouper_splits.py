"""
Tests de la couche application pour DecouperSplitsUseCase, avec de
faux adaptateurs en memoire.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

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
            identifiant=ExemplePivot.nouvel_identifiant(source),
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
