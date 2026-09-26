"""
Port de persistance de l'echantillonnage incremental du controle
qualite d'anonymisation : quels `identifiant` ont deja ete tires dans
un echantillon (par stratum), lors d'executions PRECEDENTES de
`E1_04_02_controler_qualite_anonymisation.py`. Rend l'echantillonnage
repetable sur plusieurs executions sans jamais retirer deux fois le
meme identifiant. Separe du registre des decisions humaines : ce port
ne sait rien des candidats de PII ni de leurs decisions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class RegistreEchantillonsControleQualite(Protocol):
    """Port de persistance des identifiants deja echantillonnes pour le controle qualite, par stratum."""

    def identifiants_vus(self, stratum: str) -> set[str]:
        """Retourne l'ensemble des `identifiant` deja echantillonnes pour ce stratum."""
        ...

    def marquer_vus(
        self, stratum: str, identifiants: Iterable[str], horodatage: str
    ) -> None:
        """Enregistre `identifiants` comme deja echantillonnes pour ce stratum."""
        ...
