"""
Adaptateur secondaire : profilage via ydata-profiling.

Implemente le port `Profileur`. Produit un rapport HTML detaille sur
disque, et une synthese structuree (`RapportProfilage`) exploitable
par la couche application.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

from chsa_triage.domain.ports.profileur import RapportProfilage


class YdataProfileur:
    """Adaptateur ydata-profiling implementant le port Profileur."""

    def __init__(self, dossier_rapports: str | Path = "data/processed/rapports_profilage") -> None:
        self._dossier_rapports = Path(dossier_rapports)
        self._dossier_rapports.mkdir(parents=True, exist_ok=True)

    def profiler(self, enregistrements: Iterable[dict], nom_corpus: str) -> RapportProfilage:
        dataframe = pd.DataFrame(list(enregistrements))

        rapport = ProfileReport(
            dataframe,
            title=f"Profilage — {nom_corpus}",
            minimal=True,  # rapport allege : suffisant pour decider nettoyer ou non
        )

        chemin_rapport = self._dossier_rapports / f"{nom_corpus}.html"
        rapport.to_file(chemin_rapport)

        return RapportProfilage(
            nombre_enregistrements=len(dataframe),
            taux_valeurs_manquantes=dataframe.isna().mean().to_dict(),
            taux_doublons=dataframe.duplicated().mean(),
            longueur_texte_moyenne=self._longueurs_moyennes(dataframe),
            chemin_rapport_detaille=str(chemin_rapport),
        )

    @staticmethod
    def _longueurs_moyennes(dataframe: pd.DataFrame) -> dict[str, float]:
        longueurs = {}
        for colonne in dataframe.select_dtypes(include="object").columns:
            longueurs[colonne] = dataframe[colonne].astype(str).str.len().mean()
        return longueurs
