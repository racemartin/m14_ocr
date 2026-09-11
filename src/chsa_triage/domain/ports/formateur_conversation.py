"""
Port du formatage ChatML : rendre un ExemplePivot en texte pret a etre
tokenize pour l'entrainement SFT. L'adaptateur concret
(`ChatMLFormateurAdapter`, infrastructure) enveloppe
`AutoTokenizer.apply_chat_template` ; ce port ne connait ni tokenizer
ni transformers, meme principe que `domain.ports.moteur_inference`.
"""

from __future__ import annotations

from typing import Protocol

from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.model.exemple_pivot import ExemplePivot


class FormateurConversation(Protocol):
    """Port generique : rendre un ExemplePivot en ChatML."""

    def formater(self, exemple: ExemplePivot) -> ExempleFormate:
        """Rend un ExemplePivot en ChatML via apply_chat_template
        (add_generation_prompt=False : exemple d'entrainement complet,
        pas une invite a generer)."""
        ...
