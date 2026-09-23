"""
Adaptateur du port `JournalAudit` (F6, tracabilite) : append-only sur
un fichier JSONL. Contrairement a `JsonlDatasetRepository`
(dedoublonnage par identifiant, reecriture complete a chaque sauvegarde,
cf. AGENTS.md), une entree d'audit n'a pas d'identite a fusionner :
chaque tour de conversation ou appel de diagnostic est un fait
immuable, jamais reecrit ni supprime. Meme famille que
`ajouter_exemples_jsonl` (`jsonl_dataset_repository.py`), mais dedie a
`EntreeAudit` plutot qu'a `ExemplePivot`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

from chsa_triage.domain.model.entree_audit import EntreeAudit


class JsonlJournalAudit:
    """Implemente `JournalAudit` : une ligne JSON ajoutee par `consigner()`.

    `_verrou` serialise les ecritures concurrentes (plusieurs requetes
    FastAPI simultanees peuvent consigner en meme temps sur le meme
    fichier)."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self.chemin_fichier = Path(chemin_fichier)
        self._verrou = threading.Lock()

    def consigner(self, entree: EntreeAudit) -> None:
        self.chemin_fichier.parent.mkdir(parents=True, exist_ok=True)
        ligne = json.dumps(asdict(entree), ensure_ascii=False)
        with self._verrou:
            with self.chemin_fichier.open("a", encoding="utf-8") as f:
                f.write(ligne + "\n")
