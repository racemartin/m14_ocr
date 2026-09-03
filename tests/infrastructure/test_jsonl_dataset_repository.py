"""Test de l'adaptateur JsonlDatasetRepository — round-trip sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository


def test_round_trip_sauvegarde_et_lecture(tmp_path: Path):
    chemin = tmp_path / "dataset_pivot.jsonl"
    repository = JsonlDatasetRepository(chemin)

    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test"),
        source="test",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes="douleur thoracique",
        prompt=(Message(role="user", contenu="J'ai mal a la poitrine"),),
        completion=(Message(role="assistant", contenu="Depuis quand ?"),),
    )

    repository.sauvegarder(exemple)

    exemple_relu = repository.trouver_par_id(exemple.identifiant)
    assert exemple_relu is not None
    assert exemple_relu.symptomes == "douleur thoracique"
    assert exemple_relu.prompt[0].contenu == "J'ai mal a la poitrine"


def test_filtre_par_champ(tmp_path: Path):
    chemin = tmp_path / "dataset_pivot.jsonl"
    repository = JsonlDatasetRepository(chemin)

    for i in range(3):
        repository.sauvegarder(
            ExemplePivot(
                identifiant=f"id-{i}",
                source="test",
                type_exemple=TypeExemple.SFT,
                langue=Langue.FRANCAIS,
                anonymise=(i == 0),
            )
        )

    non_anonymises = list(repository.lister(filtre={"anonymise": False}))
    assert len(non_anonymises) == 2
