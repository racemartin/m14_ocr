"""
Adaptateur secondaire : suivi d'experimentation via MLflow.

Implemente le port `SuiviExperimentation`. `uri_tracking` est passe
tel quel a `mlflow.set_tracking_uri` : le choix du backend (fichier
local, `sqlite:///...`, serveur distant) reste une decision de
`training/sft_train.py`/`recipes/sft_qwen3_lora.yaml`, pas de cet
adaptateur. Note pour les tests/usages locaux : MLflow >= 3 refuse le
backend fichier brut par defaut (`MlflowException: ... maintenance
mode ...`) ; utiliser un backend `sqlite:///chemin/mlflow.db` ou
exporter `MLFLOW_ALLOW_FILE_STORE=true`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MlflowSuiviExperimentation:
    """Suivi d'experimentation via MLflow (start_run/log_param/log_metric/end_run)."""

    uri_tracking: str
    _run_actif: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import mlflow

        mlflow.set_tracking_uri(self.uri_tracking)

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        import mlflow

        self._run_actif = mlflow.start_run(run_name=nom)
        if parametres:
            mlflow.log_params(parametres)

    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None:
        import mlflow

        mlflow.log_metric(nom, valeur, step=etape)

    def terminer_run(self) -> None:
        import mlflow

        mlflow.end_run()
        self._run_actif = None
