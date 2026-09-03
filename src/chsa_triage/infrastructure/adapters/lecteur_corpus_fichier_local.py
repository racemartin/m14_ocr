"""
Adaptateur secondaire : lecture d'un corpus brut depuis un fichier
local (CSV ou JSONL), via pandas.

Implemente le port `LecteurCorpus`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


class LecteurCorpusFichierLocal:
    """Adaptateur local implementant LecteurCorpus (CSV ou JSONL)."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        if not self._chemin.exists():
            raise FileNotFoundError(f"Corpus introuvable : {self._chemin}")

    def lire_enregistrements(self) -> Iterator[dict]:
        dataframe = self._charger_dataframe()
        yield from dataframe.to_dict(orient="records")

    def compter_enregistrements(self) -> int:
        return len(self._charger_dataframe())

    def _charger_dataframe(self) -> pd.DataFrame:
        if self._chemin.suffix == ".csv":
            return pd.read_csv(self._chemin)
        if self._chemin.suffix in (".jsonl", ".json"):
            return pd.read_json(self._chemin, lines=self._chemin.suffix == ".jsonl")
        raise ValueError(f"Format de corpus non supporte : {self._chemin.suffix}")
