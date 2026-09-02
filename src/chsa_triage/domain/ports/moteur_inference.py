"""
Port generique du moteur d'inference — prepare des l'Etape 1 pour
que le futur chat (Etape 4) puisse brancher indifferemment un
adaptateur local (llama.cpp / GGUF) ou distant (endpoint vLLM payant)
sans que l'application ni l'interface ne connaissent la difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ReponseModele:
    """Reponse generique retournee par n'importe quel moteur d'inference."""

    texte              : str
    nombre_tokens_entree : int = 0
    nombre_tokens_sortie : int = 0
    latence_ms           : float = 0.0
    metadonnees           : dict = field(default_factory=dict)


class MoteurInference(Protocol):
    """Port generique : envoyer des messages, recevoir une reponse."""

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        """
        Envoie une liste de messages (format {"role": ..., "content": ...})
        au moteur d'inference et retourne une reponse generique.
        L'adaptateur concret sait si c'est un appel local (llama.cpp) ou
        une requete HTTP vers un endpoint vLLM distant.
        """
        ...
