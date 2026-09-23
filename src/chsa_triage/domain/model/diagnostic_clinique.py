"""
Resultat structure de l'appel "obtenir diagnostic" (F2/F3/F4) : niveau
ESI, categorie, ressources estimees, et le raisonnement clinique
explicite (bloc `<think>`). Distinct de `ChosenReformule` (Etape 3,
donnee d'entrainement DPO) : ceci est une reponse d'inference vivante,
jamais persistee comme exemple d'entrainement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiagnosticClinique:
    """Sortie structuree du modele au format cible F3/F4 (`<think>` + JSON)."""

    raisonnement          : str
    niveau                  : int
    categorie                : str
    ressources_estimees   : str
