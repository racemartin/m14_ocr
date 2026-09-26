"""
Adaptateur secondaire : lecture d'un corpus brut depuis un fichier
local (CSV ou JSONL), via pandas. Implemente le port `LecteurCorpus`.

Bug reel rencontre sur un corpus de 966 Mo : sans `taille_bloc`, tout
le fichier charge d'un coup peut depasser la RAM (OOM killer Linux,
sans traceback). `taille_bloc` lit par morceaux (`pandas(chunksize=...)`),
memoire de pointe bornee par le bloc. Supporte uniquement CSV/JSONL
(pas le JSON tableau unique, que pandas ne decoupe pas par blocs).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


class LecteurCorpusFichierLocal:
    """Adaptateur local implementant LecteurCorpus (CSV ou JSONL)."""

    def __init__(
        self, chemin_fichier: str | Path, taille_bloc: int | None = None
    ) -> None:
        self._chemin = Path(chemin_fichier)
        self._taille_bloc = taille_bloc
        if not self._chemin.exists():
            raise FileNotFoundError(f"Corpus introuvable : {self._chemin}")

    def lire_enregistrements(self) -> Iterator[dict]:
        if self._taille_bloc and self._supporte_lecture_par_blocs():
            for bloc in self._iterer_blocs():
                yield from bloc.to_dict(orient="records")
            return
        dataframe = self._charger_dataframe()
        yield from dataframe.to_dict(orient="records")

    def compter_enregistrements(self) -> int:
        if self._taille_bloc and self._supporte_lecture_par_blocs():
            return sum(len(bloc) for bloc in self._iterer_blocs())
        return len(self._charger_dataframe())

    def _supporte_lecture_par_blocs(self) -> bool:
        if self._chemin.suffix == ".csv":
            return True
        return self._chemin.suffix == ".jsonl"

    def _iterer_blocs(self):
        """Iterateur de DataFrames, un par bloc de `taille_bloc` lignes."""
        if self._chemin.suffix == ".csv":
            return pd.read_csv(self._chemin, chunksize=self._taille_bloc)
        return pd.read_json(
            self._chemin, lines=True, chunksize=self._taille_bloc
        )

    def _charger_dataframe(self) -> pd.DataFrame:
        if self._chemin.suffix == ".csv":
            return pd.read_csv(self._chemin)
        if self._chemin.suffix in (".jsonl", ".json"):
            return pd.read_json(
                self._chemin, lines=self._chemin.suffix == ".jsonl"
            )
        raise ValueError(
            f"Format de corpus non supporte : {self._chemin.suffix}"
        )
