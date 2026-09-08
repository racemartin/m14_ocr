"""Test de l'adaptateur JsonlDatasetRepository — round-trip sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository
from chsa_triage.infrastructure.adapters.jsonl_dataset_repository import ajouter_exemples_jsonl


def test_round_trip_sauvegarde_et_lecture(tmp_path: Path):
    chemin = tmp_path / "dataset_pivot.jsonl"
    repository = JsonlDatasetRepository(chemin)

    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test", "1"),
        identifiant_source_brute="1",
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
    assert exemple_relu.identifiant_source_brute == "1"


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


def test_identifiants_existants_sur_fichier_vide(tmp_path: Path):
    chemin = tmp_path / "vide.jsonl"
    repository = JsonlDatasetRepository(chemin)
    assert repository.identifiants_existants() == set()


def test_identifiants_existants_retourne_tous_les_ids(tmp_path: Path):
    chemin = tmp_path / "dataset.jsonl"
    repository = JsonlDatasetRepository(chemin)

    for i in range(3):
        repository.sauvegarder(
            ExemplePivot(
                identifiant=f"id-{i}",
                source="test",
                type_exemple=TypeExemple.SFT,
                langue=Langue.FRANCAIS,
            )
        )

    assert repository.identifiants_existants() == {"id-0", "id-1", "id-2"}


def test_ajouter_exemples_jsonl_ne_fusionne_pas_par_identifiant(tmp_path: Path):
    """
    Contrairement a `sauvegarder_plusieurs` (fusion par id), `ajouter_exemples_jsonl`
    ajoute en fin de fichier SANS dedoublonner -- necessaire pour un
    fichier d'audit ou plusieurs entrees peuvent legitimement partager
    le meme identifiant (doublons ecartes du pivot).
    """
    chemin = tmp_path / "doublons_supprimes.jsonl"
    exemple_1 = ExemplePivot(identifiant="dup-1", source="test", type_exemple=TypeExemple.SFT, langue=Langue.FRANCAIS, symptomes="A")
    exemple_2 = ExemplePivot(identifiant="dup-1", source="test", type_exemple=TypeExemple.SFT, langue=Langue.FRANCAIS, symptomes="B")

    ajouter_exemples_jsonl(chemin, [exemple_1])
    ajouter_exemples_jsonl(chemin, [exemple_2])

    lignes = chemin.read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 2  # les deux entrees sont conservees, meme si meme identifiant
