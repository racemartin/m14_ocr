"""
Port generique d'anonymisation de texte.

Le port ne sait rien de Presidio, de spaCy ni des modeles linguistiques
utilises : il expose seulement un contrat "texte en entree -> texte
anonymise en sortie", plus un rapport structure des entites trouvees
(pour le controle qualite RGPD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EntiteDetectee:
    """Une entite sensible detectee dans un texte (nom, lieu, etc.)."""

    type_entite : str    # ex. "PERSON", "LOCATION"
    debut        : int
    fin          : int
    score        : float


@dataclass(frozen=True, slots=True)
class ResultatAnonymisation:
    """Resultat d'une passe d'anonymisation sur un texte."""

    texte_original     : str
    texte_anonymise     : str
    entites_detectees    : tuple[EntiteDetectee, ...]


class Anonymiseur(Protocol):
    """Port generique d'anonymisation de texte libre."""

    def anonymiser(self, texte: str, langue: str) -> ResultatAnonymisation:
        """Detecte et masque les entites sensibles d'un texte."""
        ...
