"""Test de l'adaptateur JsonlCheckpointRepository ; round-trip reel sur disque, dataclasses imbriquees incluses."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model.checkpoint_entraine import (
    CheckpointEntraine,
    VerdictConvergence,
)
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement
from chsa_triage.infrastructure.adapters.jsonl_checkpoint_repository import (
    JsonlCheckpointRepository,
)


def _checkpoint(identifiant: str = "cp-1", verdict: VerdictConvergence = VerdictConvergence.SAINE) -> CheckpointEntraine:
    return CheckpointEntraine(
        identifiant=identifiant,
        chemin="outputs/sft-lora/run-1",
        modele_base="Qwen/Qwen3-1.7B-Base",
        configuration_lora=ConfigurationLora(rang=16, alpha=32, dropout=0.05, modules_cibles=("q_proj", "v_proj")),
        hyperparametres=HyperparametresEntrainement(
            taux_apprentissage=2e-4, nombre_epoques=3, taille_lot=4, packing=True, type_perte="chunked_nll"
        ),
        metriques_finales=MetriquesEntrainement(etape=100, perte_train=0.5, perte_validation=0.6, norme_gradient=1.2),
        verdict_convergence=verdict,
        horodatage="2026-09-11T00:00:00+00:00",
    )


def test_fichier_vide_liste_rien(tmp_path: Path):
    repo = JsonlCheckpointRepository(tmp_path / "checkpoints.jsonl")
    assert list(repo.lister()) == []
    assert repo.compter() == 0


def test_sauvegarder_et_relire_roundtrip_complet(tmp_path: Path):
    repo = JsonlCheckpointRepository(tmp_path / "checkpoints.jsonl")
    checkpoint = _checkpoint()

    repo.sauvegarder(checkpoint)
    relu = repo.trouver_par_id("cp-1")

    assert relu == checkpoint
    assert isinstance(relu.configuration_lora.modules_cibles, tuple)
    assert relu.verdict_convergence == VerdictConvergence.SAINE


def test_sauvegarder_remplace_par_identifiant(tmp_path: Path):
    repo = JsonlCheckpointRepository(tmp_path / "checkpoints.jsonl")
    repo.sauvegarder(_checkpoint(verdict=VerdictConvergence.SOUS_APPRENTISSAGE))
    repo.sauvegarder(_checkpoint(verdict=VerdictConvergence.SAINE))

    assert repo.compter() == 1
    assert repo.trouver_par_id("cp-1").verdict_convergence == VerdictConvergence.SAINE


def test_sauvegarder_plusieurs_puis_lister(tmp_path: Path):
    repo = JsonlCheckpointRepository(tmp_path / "checkpoints.jsonl")
    repo.sauvegarder_plusieurs([_checkpoint("cp-1"), _checkpoint("cp-2")])

    assert repo.compter() == 2
    assert repo.identifiants_existants() == {"cp-1", "cp-2"}


def test_nouvelle_instance_relit_le_meme_fichier(tmp_path: Path):
    chemin = tmp_path / "checkpoints.jsonl"
    JsonlCheckpointRepository(chemin).sauvegarder(_checkpoint())

    repo_relu = JsonlCheckpointRepository(chemin)
    assert repo_relu.trouver_par_id("cp-1") is not None
