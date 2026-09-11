"""Test de l'adaptateur JsonlExempleFormateRepository ; round-trip reel sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.infrastructure.adapters.jsonl_exemple_formate_repository import (
    JsonlExempleFormateRepository,
)


def test_fichier_vide_liste_rien(tmp_path: Path):
    repo = JsonlExempleFormateRepository(tmp_path / "formate.jsonl")
    assert list(repo.lister()) == []
    assert repo.compter() == 0
    assert repo.identifiants_existants() == set()


def test_sauvegarder_plusieurs_puis_relire(tmp_path: Path):
    repo = JsonlExempleFormateRepository(tmp_path / "formate.jsonl")
    repo.sauvegarder_plusieurs(
        [ExempleFormate(identifiant="a", texte="hello"), ExempleFormate(identifiant="b", texte="world")]
    )

    assert repo.compter() == 2
    assert repo.identifiants_existants() == {"a", "b"}
    trouve = repo.trouver_par_id("a")
    assert trouve is not None
    assert trouve.texte == "hello"


def test_sauvegarder_remplace_par_identifiant(tmp_path: Path):
    repo = JsonlExempleFormateRepository(tmp_path / "formate.jsonl")
    repo.sauvegarder(ExempleFormate(identifiant="a", texte="premier"))
    repo.sauvegarder(ExempleFormate(identifiant="a", texte="second"))

    assert repo.compter() == 1
    assert repo.trouver_par_id("a").texte == "second"


def test_lister_avec_filtre(tmp_path: Path):
    repo = JsonlExempleFormateRepository(tmp_path / "formate.jsonl")
    repo.sauvegarder_plusieurs(
        [ExempleFormate(identifiant="a", texte="hello"), ExempleFormate(identifiant="b", texte="hello")]
    )

    filtres = list(repo.lister(filtre={"identifiant": "a"}))
    assert len(filtres) == 1
    assert filtres[0].identifiant == "a"


def test_nouvelle_instance_relit_le_meme_fichier(tmp_path: Path):
    chemin = tmp_path / "formate.jsonl"
    JsonlExempleFormateRepository(chemin).sauvegarder(ExempleFormate(identifiant="a", texte="hello"))

    repo_relu = JsonlExempleFormateRepository(chemin)
    assert repo_relu.trouver_par_id("a") is not None


def test_trouver_par_id_absent_retourne_none(tmp_path: Path):
    repo = JsonlExempleFormateRepository(tmp_path / "formate.jsonl")
    assert repo.trouver_par_id("inexistant") is None
