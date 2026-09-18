"""Test de l'adaptateur JsonlPreferenceReformuleeRepository ; round-trip reel sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.model.preference_reformulee import ChosenReformule
from chsa_triage.infrastructure.adapters.jsonl_preference_reformulee_repository import (
    JsonlPreferenceReformuleeRepository,
)


def _chosen_reformule(identifiant: str = "a") -> ChosenReformule:
    return ChosenReformule(
        identifiant=identifiant,
        chosen_reformule=(
            Message(role="assistant", contenu='<think>raisonnement</think>{"niveau": 3}'),
        ),
        horodatage="2026-09-19T00:00:00+00:00",
    )


def test_fichier_vide_liste_rien(tmp_path: Path):
    repo = JsonlPreferenceReformuleeRepository(tmp_path / "reformule.jsonl")
    assert list(repo.lister()) == []
    assert repo.compter() == 0
    assert repo.identifiants_existants() == set()


def test_sauvegarder_puis_relire_roundtrip_complet(tmp_path: Path):
    repo = JsonlPreferenceReformuleeRepository(tmp_path / "reformule.jsonl")
    item = _chosen_reformule()

    repo.sauvegarder(item)
    relu = repo.trouver_par_id("a")

    assert relu == item
    assert isinstance(relu.chosen_reformule, tuple)
    assert relu.chosen_reformule[0].role == "assistant"


def test_sauvegarder_remplace_par_identifiant(tmp_path: Path):
    repo = JsonlPreferenceReformuleeRepository(tmp_path / "reformule.jsonl")
    repo.sauvegarder(_chosen_reformule())
    autre = ChosenReformule(
        identifiant="a",
        chosen_reformule=(Message(role="assistant", contenu="nouveau"),),
        horodatage="2026-09-19T01:00:00+00:00",
    )
    repo.sauvegarder(autre)

    assert repo.compter() == 1
    assert repo.trouver_par_id("a").chosen_reformule[0].contenu == "nouveau"


def test_sauvegarder_plusieurs_puis_lister(tmp_path: Path):
    repo = JsonlPreferenceReformuleeRepository(tmp_path / "reformule.jsonl")
    repo.sauvegarder_plusieurs([_chosen_reformule("a"), _chosen_reformule("b")])

    assert repo.compter() == 2
    assert repo.identifiants_existants() == {"a", "b"}


def test_nouvelle_instance_relit_le_meme_fichier(tmp_path: Path):
    chemin = tmp_path / "reformule.jsonl"
    JsonlPreferenceReformuleeRepository(chemin).sauvegarder(_chosen_reformule())

    repo_relu = JsonlPreferenceReformuleeRepository(chemin)
    assert repo_relu.trouver_par_id("a") is not None


def test_trouver_par_id_absent_retourne_none(tmp_path: Path):
    repo = JsonlPreferenceReformuleeRepository(tmp_path / "reformule.jsonl")
    assert repo.trouver_par_id("inexistant") is None
