"""
Tests de la couche application, avec des faux adaptateurs en memoire
(pas de fichier, pas de reseau) -- c'est tout l'interet du decouplage
hexagonal : l'application se teste sans aucune infrastructure reelle.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from chsa_triage.application.use_cases import ConstruireDatasetPivotUseCase
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple


class FauxLecteurCorpus:
    """Faux adaptateur LecteurCorpus, en memoire, pour les tests."""

    def __init__(self, enregistrements: list[dict]) -> None:
        self._enregistrements = enregistrements

    def lire_enregistrements(self) -> Iterator[dict]:
        yield from self._enregistrements

    def compter_enregistrements(self) -> int:
        return len(self._enregistrements)


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire."""

    def __init__(self) -> None:
        self.items: dict[str, ExemplePivot] = {}

    def sauvegarder(self, item: ExemplePivot) -> None:
        self.items[item.identifiant] = item

    def sauvegarder_plusieurs(self, items: Iterable[ExemplePivot]) -> None:
        for item in items:
            self.sauvegarder(item)

    def trouver_par_id(self, identifiant: str):
        return self.items.get(identifiant)

    def lister(self, filtre: dict | None = None):
        yield from self.items.values()

    def compter(self, filtre: dict | None = None) -> int:
        return len(self.items)


def _mapper_test(enregistrement: dict) -> ExemplePivot | None:
    if "question" not in enregistrement:
        return None
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test", enregistrement["question"]),
        identifiant_source_brute=enregistrement["question"],
        source="test",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu=enregistrement["question"]),),
        completion=(Message(role="assistant", contenu=enregistrement.get("reponse", "")),),
    )


def test_construire_dataset_pivot_ignore_enregistrements_invalides():
    lecteur = FauxLecteurCorpus([
        {"question": "Q1", "reponse": "R1"},
        {"champ_invalide": "sans question"},
        {"question": "Q2", "reponse": "R2"},
    ])
    repository = FauxRepository()

    cas_usage = ConstruireDatasetPivotUseCase(lecteur=lecteur, repository=repository)
    nombre = cas_usage.executer(_mapper_test)

    assert nombre == 2
    assert repository.compter() == 2
    assert cas_usage.doublons == []


def test_construire_dataset_pivot_dedoublonne_les_registres_identiques():
    """
    Deux enregistrements bruts avec la meme cle naturelle (donc le
    meme identifiant deterministe) sont de vrais doublons -- un seul
    doit atterrir dans le pivot, l'autre doit etre expose via
    `.doublons` (jamais silencieusement perdu).
    """
    lecteur = FauxLecteurCorpus([
        {"question": "Q1", "reponse": "R1"},
        {"question": "Q1", "reponse": "R1"},  # doublon exact de la ligne precedente
        {"question": "Q2", "reponse": "R2"},
    ])
    repository = FauxRepository()

    cas_usage = ConstruireDatasetPivotUseCase(lecteur=lecteur, repository=repository)
    nombre = cas_usage.executer(_mapper_test)

    assert nombre == 2
    assert repository.compter() == 2
    assert len(cas_usage.doublons) == 1
    assert cas_usage.doublons[0].prompt[0].contenu == "Q1"


def test_construire_dataset_pivot_conserve_le_premier_exemple_rencontre():
    lecteur = FauxLecteurCorpus([
        {"question": "Q1", "reponse": "R1"},
        {"question": "Q1", "reponse": "R1"},
        {"question": "Q1", "reponse": "R1"},
    ])
    repository = FauxRepository()

    cas_usage = ConstruireDatasetPivotUseCase(lecteur=lecteur, repository=repository)
    nombre = cas_usage.executer(_mapper_test)

    assert nombre == 1
    assert len(cas_usage.doublons) == 2
