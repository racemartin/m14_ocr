"""
Tests de la couche application pour ExtraireSousEnsembleSftUseCase,
avec un faux adaptateur en memoire (meme patron que
`test_decouper_splits.py`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases import ExtraireSousEnsembleSftUseCase
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


def _exemple(source: str = "MediQAl", avec_split: bool = True) -> ExemplePivot:
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


def test_extraction_ignore_les_exemples_sans_split():
    exemples = [_exemple(avec_split=True) for _ in range(5)] + [_exemple(avec_split=False) for _ in range(3)]
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleSftUseCase(repository=repository, taille_cible=5)
    resultat = cas_usage.executer()

    assert len(resultat) == 5
    assert all(e.split is not None for e in resultat)
    assert cas_usage.nombre_avec_split == 5
    assert cas_usage.nombre_exclus == 0
    assert cas_usage.manque == 0


def test_extraction_exclut_les_identifiants_listes():
    exemples = [_exemple(avec_split=True) for _ in range(5)]
    identifiant_exclu = exemples[0].identifiant
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleSftUseCase(
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
    exemples = [_exemple(avec_split=True) for _ in range(10)]
    identifiants_exclus = frozenset(e.identifiant for e in exemples[:4])
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleSftUseCase(
        repository=repository, identifiants_a_exclure=identifiants_exclus, taille_cible=8
    )
    resultat = cas_usage.executer()

    assert len(resultat) == 6
    assert cas_usage.nombre_exclus == 4
    assert cas_usage.manque == 2


def test_extraction_manque_nul_quand_taille_cible_atteinte():
    exemples = [_exemple(avec_split=True) for _ in range(10)]
    repository = FauxRepository(exemples)

    cas_usage = ExtraireSousEnsembleSftUseCase(repository=repository, taille_cible=5)
    cas_usage.executer()

    assert cas_usage.manque == 0
