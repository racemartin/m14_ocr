"""
Adaptateur secondaire : suivi d'experimentation via TensorBoard.

Implemente le port `SuiviExperimentation`. Un `SummaryWriter` distinct
est ouvert par run, sous `repertoire_logs/<nom>` (meme convention que
les runs MLflow nommes) : c'est ce sous-repertoire qu'un
`EventAccumulator`/`tensorboard --logdir` doit ensuite pointer pour
relire les metriques.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TensorboardSuiviExperimentation:
    """Suivi d'experimentation via TensorBoard (SummaryWriter.add_scalar/add_text)."""

    repertoire_logs: str
    _writer: Any = field(default=None, init=False, repr=False)

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=os.path.join(self.repertoire_logs, nom))
        if parametres:
            self._writer.add_text("parametres", str(parametres))

    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None:
        self._writer.add_scalar(nom, valeur, etape)

    def terminer_run(self) -> None:
        self._writer.close()
        self._writer = None
