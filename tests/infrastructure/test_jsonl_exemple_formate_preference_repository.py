"""Test de l'adaptateur JsonlExempleFormatePreferenceRepository ; round-trip reel sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.infrastructure.adapters.jsonl_exemple_formate_preference_repository import (
    JsonlExempleFormatePreferenceRepository,
)


def _exemple(identifiant: str = "a") -> ExempleFormatePreference:
    return ExempleFormatePreference(
        identifiant=identifiant,
        texte_prompt="<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n",
        texte_chosen="<|im_start|>assistant\n<think>ok</think>{}<|im_end|>\n",
        texte_rejected="<|im_start|>assistant\ntexte libre<|im_end|>\n",
    )


def test_fichier_vide_liste_rien(tmp_path: Path):
    repo = JsonlExempleFormatePreferenceRepository(tmp_path / "formate.jsonl")
    assert list(repo.lister()) == []
    assert repo.compter() == 0
    assert repo.identifiants_existants() == set()


def test_sauvegarder_plusieurs_puis_relire(tmp_path: Path):
    repo = JsonlExempleFormatePreferenceRepository(tmp_path / "formate.jsonl")
    repo.sauvegarder_plusieurs([_exemple("a"), _exemple("b")])

    assert repo.compter() == 2
    assert repo.identifiants_existants() == {"a", "b"}
    trouve = repo.trouver_par_id("a")
    assert trouve == _exemple("a")


def test_sauvegarder_remplace_par_identifiant(tmp_path: Path):
    repo = JsonlExempleFormatePreferenceRepository(tmp_path / "formate.jsonl")
    repo.sauvegarder(_exemple("a"))
    remplacement = ExempleFormatePreference(
        identifiant="a", texte_prompt="autre", texte_chosen="autre", texte_rejected="autre"
    )
    repo.sauvegarder(remplacement)

    assert repo.compter() == 1
    assert repo.trouver_par_id("a").texte_prompt == "autre"


def test_nouvelle_instance_relit_le_meme_fichier(tmp_path: Path):
    chemin = tmp_path / "formate.jsonl"
    JsonlExempleFormatePreferenceRepository(chemin).sauvegarder(_exemple())

    repo_relu = JsonlExempleFormatePreferenceRepository(chemin)
    assert repo_relu.trouver_par_id("a") is not None


def test_trouver_par_id_absent_retourne_none(tmp_path: Path):
    repo = JsonlExempleFormatePreferenceRepository(tmp_path / "formate.jsonl")
    assert repo.trouver_par_id("inexistant") is None
