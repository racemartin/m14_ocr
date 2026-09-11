"""
Entite ExempleFormate : rendu ChatML complet d'un ExemplePivot, produit
par un FormateurConversation (cf. domain.ports.formateur_conversation)
et consomme par EntraineurSupervise.entrainer().

Dataclass pure, aucune dependance externe (pas de transformers ici :
l'appel a `apply_chat_template` est un souci d'infrastructure), meme
regle que le reste de `domain.model`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExempleFormate:
    """Rendu ChatML d'un ExemplePivot, pret a etre tokenize pour l'entrainement SFT."""

    identifiant : str   # repris de ExemplePivot.identifiant, jamais regenere
    texte        : str   # rendu ChatML complet (system+user+assistant)
