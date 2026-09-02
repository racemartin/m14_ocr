"""
Port generique de lecture de corpus brut.

Encore une fois, la signature reste generique : `lire_enregistrements`
retourne des dictionnaires bruts (structure non normalisee), pas des
`ExemplePivot`. C'est la couche application (use case
`construire_dataset_pivot`) qui sait comment transformer un
enregistrement brut d'une source donnee en `ExemplePivot` -- via une
fonction de mapping injectee, pas via une methode specifique au
domaine medical sur ce port.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class LecteurCorpus(Protocol):
    """Port generique pour lire les enregistrements bruts d'une source."""

    def lire_enregistrements(self) -> Iterator[dict]:
        """
        Retourne un iterateur de dictionnaires bruts, un par
        enregistrement source, sans transformation.
        """
        ...

    def compter_enregistrements(self) -> int:
        """Retourne le nombre total d'enregistrements disponibles."""
        ...
