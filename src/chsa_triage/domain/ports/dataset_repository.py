"""
Port generique de persistance : contrat que doit respecter tout
adaptateur de stockage (fichiers JSONL locaux, Hugging Face Datasets,
base de donnees, etc.).

IMPORTANT : les noms de methode sont volontairement GENERIQUES
(sauvegarder / trouver_par_id / lister), jamais lies au vocabulaire
medical. Le meme port pourrait servir a persister n'importe quel
type d'entite du domaine, pas seulement ExemplePivot. C'est ce qui
permet de changer de technologie de stockage sans modifier une seule
ligne de la couche application.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, TypeVar

T = TypeVar("T")


class RepositoryLectureEcriture(Protocol[T]):
    """Port generique de lecture/ecriture, parametre par le type T."""

    def sauvegarder(self, item: T) -> None:
        """Persiste un item unique (ecrase si l'identifiant existe deja)."""
        ...

    def sauvegarder_plusieurs(self, items: Iterable[T]) -> None:
        """Persiste plusieurs items en une seule operation."""
        ...

    def trouver_par_id(self, identifiant: str) -> T | None:
        """Retourne l'item correspondant a l'identifiant, ou None."""
        ...

    def lister(self, filtre: dict | None = None) -> Iterator[T]:
        """
        Retourne un iterateur sur les items, filtre optionnellement
        par un dictionnaire de correspondances champ -> valeur.
        """
        ...

    def compter(self, filtre: dict | None = None) -> int:
        """Retourne le nombre d'items correspondant au filtre."""
        ...
