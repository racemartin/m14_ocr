"""
Adaptateur secondaire : journal d'audit (F6) persiste via un dataset
Hugging Face Hub, pour survivre aux redemarrages du Space Docker/GPU
(filesystem ephemere). Meme patron que `HfDatasetSuiviExperimentation` :
un dossier local surveille par `huggingface_hub.CommitScheduler`, qui
pousse vers le Hub toutes les N minutes.

Contrairement au suivi d'experimentation (un fichier JSONL PAR RUN),
le journal d'audit n'a pas de notion de run : un seul fichier
append-only qui grandit pour toute la duree de vie du processus API.

Bug reel trouve en deploiement (29/09/2026) : `CommitScheduler` ne fait
que POUSSER l'etat du dossier local, il ne le fait jamais redescendre
depuis le Hub au demarrage. Le dossier local etant sur le filesystem
ephemere du Space, chaque redemarrage du conteneur repartait d'un
fichier local VIDE ; le premier push suivant ecrasait alors, sur le
Hub, l'historique complet des sessions precedentes par le seul contenu
de la session en cours. Corrige en rehydratant le fichier local depuis
le Hub (s'il existe deja) AVANT le tout premier `consigner()` de
chaque processus, pour que l'ajout reparte bien de l'historique complet.
`fabrique_scheduler`/`fabrique_telechargement` sont injectables pour
les tests, sans reseau reel.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chsa_triage.domain.model.entree_audit import EntreeAudit

NOM_FICHIER_JOURNAL_AUDIT = "journal_audit.jsonl"


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


def _telecharger_journal_existant_reel(
    repo_id: str, chemin_local: Path
) -> None:
    """Rehydrate `chemin_local` depuis le Hub si un journal y existe deja.
    Silencieux (aucune action) si le depot ou le fichier n'existe pas
    encore : cas normal du tout premier `consigner()` du projet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        chemin_telecharge = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=NOM_FICHIER_JOURNAL_AUDIT,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return
    shutil.copy(chemin_telecharge, chemin_local)


@dataclass(slots=True)
class HfDatasetJournalAudit:
    """Implemente `JournalAudit` via un dataset Hugging Face Hub (CommitScheduler)."""

    repo_id: str
    repertoire_local: str
    intervalle_minutes: float = 1.0
    fabrique_scheduler: Callable[[str, str, float], Any] = _creer_scheduler_reel
    fabrique_telechargement: Callable[[str, Path], None] = (
        _telecharger_journal_existant_reel
    )
    _scheduler: Any = field(default=None, init=False, repr=False)
    _verrou: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def consigner(self, entree: EntreeAudit) -> None:
        Path(self.repertoire_local).mkdir(parents=True, exist_ok=True)
        chemin_jsonl = Path(self.repertoire_local) / NOM_FICHIER_JOURNAL_AUDIT
        ligne = json.dumps(asdict(entree), ensure_ascii=False)
        with self._verrou:
            if self._scheduler is None:
                if not chemin_jsonl.exists():
                    self.fabrique_telechargement(self.repo_id, chemin_jsonl)
                self._scheduler = self.fabrique_scheduler(
                    self.repo_id, self.repertoire_local, self.intervalle_minutes
                )
            with chemin_jsonl.open("a", encoding="utf-8") as f:
                f.write(ligne + "\n")
