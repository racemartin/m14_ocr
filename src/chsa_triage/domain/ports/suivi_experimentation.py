"""
Port du suivi d'experimentation, partage par EntrainerSftUseCase et
AjusterBoucleHyperparametresSftUseCase. Deux adaptateurs concrets
prevus (`MlflowSuiviExperimentation`, `TensorboardSuiviExperimentation`),
injectes par `training/sft_train.py` selon
`recipes/sft_qwen3_lora.yaml::suivi.backend` : ni les cas d'usage ni ce
port ne connaissent le backend choisi.
"""

from __future__ import annotations

from typing import Protocol


class SuiviExperimentation(Protocol):
    """Port generique : demarrer un run, y logger des metriques, le terminer."""

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        """Ouvre un nouveau run de suivi (MLflow/TensorBoard) sous le nom donne."""
        ...

    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None:
        """Enregistre une valeur de metrique pour l'etape (pas) courante du run ouvert."""
        ...

    def terminer_run(self) -> None:
        """Cloture le run de suivi actuellement ouvert."""
        ...
