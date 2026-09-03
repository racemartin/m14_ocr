"""
Adaptateur secondaire : lecture d'un corpus brut depuis un fichier
local (CSV ou JSONL), via pandas.

Implemente le port `LecteurCorpus`.

NOTE (03/09/2026, decouvert sur ultramedical_preference.jsonl, 966 Mo) :
sans `taille_bloc`, tout le fichier est charge d'un coup en DataFrame
puis converti en liste de dictionnaires -- sur un gros corpus, ca peut
depasser la RAM disponible (OOM killer Linux, processus tue sans
traceback Python). Passer `taille_bloc` fait lire le fichier par
morceaux via `pandas(chunksize=...)` : chaque bloc est charge, cede,
puis libere avant de lire le suivant -- la memoire de pointe reste
bornee par la taille du bloc, pas par la taille du fichier entier.
Uniquement supporte pour CSV et JSONL (pas le JSON tableau unique,
`lines=False`, que pandas ne sait pas decouper par blocs) ; sans objet
JSONL/CSV a decouper, `taille_bloc` est silencieusement ignore et la
lecture reste la lecture complete habituelle.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


class LecteurCorpusFichierLocal:
    """Adaptateur local implementant LecteurCorpus (CSV ou JSONL)."""

    def __init__(self, chemin_fichier: str | Path, taille_bloc: int | None = None) -> None:
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
        return pd.read_json(self._chemin, lines=True, chunksize=self._taille_bloc)

    def _charger_dataframe(self) -> pd.DataFrame:
        if self._chemin.suffix == ".csv":
            return pd.read_csv(self._chemin)
        if self._chemin.suffix in (".jsonl", ".json"):
            return pd.read_json(self._chemin, lines=self._chemin.suffix == ".jsonl")
        raise ValueError(f"Format de corpus non supporte : {self._chemin.suffix}")
