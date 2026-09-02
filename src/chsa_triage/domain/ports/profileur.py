"""
Port generique de profilage d'un ensemble de donnees.

N'importe quel outil de profiling (ydata-profiling aujourd'hui,
un autre demain) peut implementer ce port tant qu'il retourne un
rapport structure minimal exploitable par la couche application pour
decider "faut-il nettoyer ce corpus ?".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RapportProfilage:
    """Synthese minimale d'un profilage, exploitable par l'application."""

    nombre_enregistrements   : int
    taux_valeurs_manquantes   : dict[str, float]
    taux_doublons             : float
    longueur_texte_moyenne    : dict[str, float]
    chemin_rapport_detaille   : str | None = None   # ex. fichier HTML genere


class Profileur(Protocol):
    """Port generique de profilage d'enregistrements bruts."""

    def profiler(self, enregistrements: Iterable[dict], nom_corpus: str) -> RapportProfilage:
        """Produit un rapport de profilage sur un ensemble d'enregistrements."""
        ...
