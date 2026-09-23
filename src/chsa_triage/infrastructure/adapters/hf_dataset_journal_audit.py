"""
Adaptateur secondaire : journal d'audit (F6, tracabilite) persiste via
un dataset Hugging Face Hub, pour survivre aux redemarrages du Space
Docker/GPU (filesystem ephemere, pas de stockage persistant payant
active, cf. AGENTS.md). Implemente le port `JournalAudit`, meme patron
exact que `HfDatasetSuiviExperimentation`
(`hf_dataset_suivi_experimentation.py`) : un dossier de travail local
surveille en arriere-plan par un `huggingface_hub.CommitScheduler`, qui
pousse vers un repo dataset du Hub toutes les N minutes, gratuit,
aucun stockage persistant HF payant necessaire.

Contrairement au suivi d'experimentation (un fichier JSONL PAR RUN,
remis a zero a chaque `demarrer_run`), le journal d'audit n'a pas de
notion de run : un seul fichier JSONL, append-only, jamais tronque
(meme discipline que `JsonlJournalAudit`), qui grandit pour toute la
duree de vie du processus API.

`CommitScheduler` demarre un thread d'arriere-plan des sa construction
et appelle `HfApi.create_repo` (reseau reel) : un seul scheduler est
cree paresseusement, au premier `consigner()`. `fabrique_scheduler` est
injectable (meme principe que dans `HfDatasetSuiviExperimentation`)
pour que les tests passent un double sans reseau reel.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chsa_triage.domain.model.entree_audit import EntreeAudit

NOM_FICHIER_JOURNAL_AUDIT = "journal_audit.jsonl"


def _creer_scheduler_reel(repo_id: str, dossier_local: str, intervalle_minutes: float) -> Any:
    from huggingface_hub import CommitScheduler

    return CommitScheduler(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dossier_local,
        every=intervalle_minutes,
    )


@dataclass(slots=True)
class HfDatasetJournalAudit:
    """Implemente `JournalAudit` via un dataset Hugging Face Hub (CommitScheduler)."""

    repo_id              : str
    repertoire_local       : str
    intervalle_minutes      : float = 1.0
    fabrique_scheduler        : Callable[[str, str, float], Any] = _creer_scheduler_reel
    _scheduler                  : Any = field(default=None, init=False, repr=False)
    _verrou                      : threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def consigner(self, entree: EntreeAudit) -> None:
        Path(self.repertoire_local).mkdir(parents=True, exist_ok=True)
        chemin_jsonl = Path(self.repertoire_local) / NOM_FICHIER_JOURNAL_AUDIT
        ligne = json.dumps(asdict(entree), ensure_ascii=False)
        with self._verrou:
            if self._scheduler is None:
                self._scheduler = self.fabrique_scheduler(
                    self.repo_id, self.repertoire_local, self.intervalle_minutes
                )
            with chemin_jsonl.open("a", encoding="utf-8") as f:
                f.write(ligne + "\n")
