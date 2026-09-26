"""
Adaptateur secondaire : suivi d'experimentation via un dataset Hugging
Face Hub, pour visualisation EN VIVO depuis un Space Streamlit distinct
pendant un run sur HF Jobs, sans acces direct au disque du job.

Meme patron que `TensorboardSuiviExperimentation` : un repertoire local
par run, synchronise en arriere-plan vers le Hub via
`huggingface_hub.CommitScheduler`. Le scheduler est cree paresseusement
au premier `demarrer_run` et reutilise pour les runs suivants du meme
processus (la boucle d'ajustement d'hyperparametres en demarre
plusieurs). `fabrique_scheduler` est injectable pour les tests, sans
reseau reel.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _creer_scheduler_reel(
    repo_id: str, dossier_local: str, intervalle_minutes: float
) -> Any:
    from huggingface_hub import CommitScheduler

    return CommitScheduler(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dossier_local,
        every=intervalle_minutes,
    )


@dataclass(slots=True)
class HfDatasetSuiviExperimentation:
    """Suivi d'experimentation via un dataset Hugging Face Hub (CommitScheduler)."""

    repo_id: str
    repertoire_local: str
    intervalle_minutes: float = 1.0
    fabrique_scheduler: Callable[[str, str, float], Any] = _creer_scheduler_reel
    _scheduler: Any = field(default=None, init=False, repr=False)
    _chemin_jsonl: Path | None = field(default=None, init=False, repr=False)

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        Path(self.repertoire_local).mkdir(parents=True, exist_ok=True)
        if self._scheduler is None:
            self._scheduler = self.fabrique_scheduler(
                self.repo_id, self.repertoire_local, self.intervalle_minutes
            )

        repertoire_run = Path(self.repertoire_local) / nom
        repertoire_run.mkdir(parents=True, exist_ok=True)
        self._chemin_jsonl = repertoire_run / "metriques.jsonl"
        self._chemin_jsonl.write_text("", encoding="utf-8")
        if parametres:
            (repertoire_run / "parametres.json").write_text(
                json.dumps(parametres, default=str), encoding="utf-8"
            )

    def logger_metrique(
        self,
        nom: str,
        valeur: float,
        etape: int,
        horodatage: float | None = None,
    ) -> None:
        ligne = {
            "etape": etape,
            "nom": nom,
            "valeur": valeur,
            "horodatage": horodatage if horodatage is not None else time.time(),
        }
        with self._chemin_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ligne) + "\n")

    def terminer_run(self) -> None:
        if self._scheduler is not None:
            self._scheduler.push_to_hub()
        self._chemin_jsonl = None
