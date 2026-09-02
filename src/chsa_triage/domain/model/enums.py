"""
Enumerations du domaine.

Aucune dependance externe : uniquement la bibliotheque standard.
Ce module ne doit JAMAIS importer pandas, presidio, datasets, etc.
"""

from __future__ import annotations

from enum import Enum


class Langue(str, Enum):
    """Langue d'un exemple du corpus."""

    FRANCAIS = "fr"
    ANGLAIS = "en"


class TypeExemple(str, Enum):
    """Nature de l'exemple : instruction-reponse (SFT) ou preference (DPO)."""

    SFT = "sft"
    DPO = "dpo"


class NiveauConfiance(str, Enum):
    """Confiance accordee a la source ou a l'annotation d'un exemple."""

    HAUTE = "haute"
    MOYENNE = "moyenne"
    BASSE = "basse"


class TypeSplit(str, Enum):
    """Repartition d'un exemple dans les jeux d'entrainement/evaluation."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST_CLINIQUE = "test"
