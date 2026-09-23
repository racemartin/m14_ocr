"""
Entree du journal d'audit (F6, tracabilite) : un tour d'entretien ou un
appel de diagnostic, avec horodatage, entree, sortie et version du
modele utilise. Zero dependance externe, meme discipline que le reste
de `domain/model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EntreeAudit:
    """Une ligne du journal d'audit, independante du support de persistance."""

    horodatage        : str
    type_evenement     : str   # "tour_entretien" | "diagnostic"
    conversation_id    : str
    entree                : str
    sortie                : str
    version_modele      : str
    metadonnees          : dict = field(default_factory=dict)
