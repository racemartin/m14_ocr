"""
Tests de `SauvegarderCheckpointSftUseCase`, avec un faux adaptateur
RepositoryLectureEcriture en memoire (troisieme type d'entite apres
`ExemplePivot` et `ExempleFormate`, meme port generique).
"""

from __future__ import annotations

from collections.abc import Iterable

from chsa_triage.application.use_cases import SauvegarderCheckpointSftUseCase
from chsa_triage.domain.model.checkpoint_entraine import CheckpointEntraine, VerdictConvergence
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire."""

    def __init__(self, items: list[CheckpointEntraine] | None = None) -> None:
        self.items: dict[str, CheckpointEntraine] = {item.identifiant: item for item in (items or [])}

    def sauvegarder(self, item: CheckpointEntraine) -> None:
        self.items[item.identifiant] = item

    def sauvegarder_plusieurs(self, items: Iterable[CheckpointEntraine]) -> None:
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

    def identifiants_existants(self) -> set[str]:
        return set(self.items.keys())


def _config_lora() -> ConfigurationLora:
    return ConfigurationLora(rang=16, alpha=32, dropout=0.05, modules_cibles=("q_proj", "k_proj"))


def _hyperparametres() -> HyperparametresEntrainement:
    return HyperparametresEntrainement(
        taux_apprentissage=2e-4, nombre_epoques=3, taille_lot=4, packing=True, type_perte="chunked_nll"
    )


def _metriques_finales() -> MetriquesEntrainement:
    return MetriquesEntrainement(etape=1200, perte_train=0.83, perte_validation=0.91, norme_gradient=1.4)


def test_sauvegarder_checkpoint_persiste_un_enregistrement_de_metadonnees():
    repository = FauxRepository()
    cas_usage = SauvegarderCheckpointSftUseCase(repository_checkpoints=repository, horloge=lambda: "2026-09-11T00:00:00+00:00")

    checkpoint = cas_usage.executer(
        identifiant="chsa-sft-lora-abc123",
        chemin="checkpoints/sft-lora/2026-09-11/",
        modele_base="Qwen/Qwen3-1.7B-Base",
        configuration_lora=_config_lora(),
        hyperparametres=_hyperparametres(),
        metriques_finales=_metriques_finales(),
        verdict_convergence=VerdictConvergence.SAINE,
    )

    assert repository.items["chsa-sft-lora-abc123"] == checkpoint
    assert checkpoint.chemin == "checkpoints/sft-lora/2026-09-11/"
    assert checkpoint.modele_base == "Qwen/Qwen3-1.7B-Base"
    assert checkpoint.configuration_lora == _config_lora()
    assert checkpoint.hyperparametres == _hyperparametres()
    assert checkpoint.metriques_finales == _metriques_finales()
    assert checkpoint.verdict_convergence == VerdictConvergence.SAINE
    assert checkpoint.horodatage == "2026-09-11T00:00:00+00:00"


def test_sauvegarder_checkpoint_utilise_une_horloge_reelle_par_defaut():
    repository = FauxRepository()
    cas_usage = SauvegarderCheckpointSftUseCase(repository_checkpoints=repository)

    checkpoint = cas_usage.executer(
        identifiant="chsa-sft-lora-def456",
        chemin="checkpoints/sft-lora/essai/",
        modele_base="Qwen/Qwen3-1.7B-Base",
        configuration_lora=_config_lora(),
        hyperparametres=_hyperparametres(),
        metriques_finales=_metriques_finales(),
        verdict_convergence=VerdictConvergence.SURAPPRENTISSAGE,
    )

    assert checkpoint.horodatage  # non vide, format ISO reel, pas fige a la main ici
    assert "T" in checkpoint.horodatage


def test_sauvegarder_checkpoint_ecrase_un_enregistrement_existant_avec_le_meme_identifiant():
    repository = FauxRepository()
    cas_usage = SauvegarderCheckpointSftUseCase(repository_checkpoints=repository, horloge=lambda: "t1")
    cas_usage.executer(
        identifiant="chsa-sft-lora-abc123",
        chemin="checkpoints/premiere-version/",
        modele_base="Qwen/Qwen3-1.7B-Base",
        configuration_lora=_config_lora(),
        hyperparametres=_hyperparametres(),
        metriques_finales=_metriques_finales(),
        verdict_convergence=VerdictConvergence.SOUS_APPRENTISSAGE,
    )

    cas_usage.horloge = lambda: "t2"
    checkpoint_final = cas_usage.executer(
        identifiant="chsa-sft-lora-abc123",
        chemin="checkpoints/version-corrigee/",
        modele_base="Qwen/Qwen3-1.7B-Base",
        configuration_lora=_config_lora(),
        hyperparametres=_hyperparametres(),
        metriques_finales=_metriques_finales(),
        verdict_convergence=VerdictConvergence.SAINE,
    )

    assert len(repository.items) == 1
    assert repository.items["chsa-sft-lora-abc123"] == checkpoint_final
    assert repository.items["chsa-sft-lora-abc123"].chemin == "checkpoints/version-corrigee/"
