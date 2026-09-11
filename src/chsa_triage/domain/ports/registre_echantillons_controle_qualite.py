"""
Port de persistance du muestreo incremental du controle qualite
d'anonymisation : quels `identifiant` ont deja ete tires dans un
echantillon de controle qualite (par stratum), lors d'executions
PRECEDENTES de `E1_04_02_controler_qualite_anonymisation.py`.

Meme role, pour ce cas d'usage, que `RepositoryLectureEcriture.identifiants_existants()`
pour `AnonymiserDatasetUseCase` : determiner "qu'est-ce qui a deja ete
traite" sans dependre d'un champ mute sur l'item source, pour rendre le
muestreo repetable sur plusieurs executions sans jamais retirer deux
fois le meme identifiant. Volontairement separe du registre des
decisions humaines (`RegistreDecisionsRevisionHumaine`) : ce port ne
sait rien des candidats de PII residuelle ni de leurs decisions, il
sait seulement "quels exemples ont deja ete soumis a une comparaison
original/anonymise".
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class RegistreEchantillonsControleQualite(Protocol):
    """Port de persistance des identifiants deja echantillonnes pour le controle qualite, par stratum."""

    def identifiants_vus(self, stratum: str) -> set[str]:
        """Retourne l'ensemble des `identifiant` deja echantillonnes pour ce stratum."""
        ...

    def marquer_vus(self, stratum: str, identifiants: Iterable[str], horodatage: str) -> None:
        """Enregistre `identifiants` comme deja echantillonnes pour ce stratum."""
        ...
