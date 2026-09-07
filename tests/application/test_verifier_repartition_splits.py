"""
Tests de la couche application pour VerifierRepartitionSplitsUseCase,
avec de faux adaptateurs en memoire.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from chsa_triage.application.use_cases import VerifierRepartitionSplitsUseCase
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple, TypeSplit


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire, filtre inclus."""

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


def _exemple_reparti(source: str, split: TypeSplit, type_exemple: TypeExemple = TypeExemple.SFT) -> ExemplePivot:
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
        split=split,
    )


def test_verifier_repartition_splits_regroupe_par_strate():
    exemples = (
        [_exemple_reparti("MediQAl", TypeSplit.TRAIN) for _ in range(8)]
        + [_exemple_reparti("MediQAl", TypeSplit.VALIDATION)]
        + [_exemple_reparti("MediQAl", TypeSplit.TEST_CLINIQUE)]
        + [_exemple_reparti("UltraMedical-Preference", TypeSplit.TRAIN, TypeExemple.DPO) for _ in range(5)]
    )
    repository = FauxRepository(exemples)

    cas_usage = VerifierRepartitionSplitsUseCase(repository=repository)
    repartition = cas_usage.executer()

    assert repartition[("sft", "MediQAl")] == {"train": 8, "val": 1, "test": 1}
    assert repartition[("dpo", "UltraMedical-Preference")] == {"train": 5}


def test_verifier_repartition_splits_ignore_non_repartis():
    exemples = [
        _exemple_reparti("MediQAl", TypeSplit.TRAIN),
        replace(_exemple_reparti("MediQAl", TypeSplit.TRAIN), split=None),
        replace(_exemple_reparti("MediQAl", TypeSplit.TRAIN), anonymise=False, split=None),
    ]
    repository = FauxRepository(exemples)

    cas_usage = VerifierRepartitionSplitsUseCase(repository=repository)
    repartition = cas_usage.executer()

    assert repartition == {("sft", "MediQAl"): {"train": 1}}
