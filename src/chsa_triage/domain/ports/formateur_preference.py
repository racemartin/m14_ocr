"""
Port du formatage d'un triplet de preference : rendre un ExemplePivot
DPO en `ExempleFormatePreference` (prompt/chosen/rejected distincts),
pret a etre tokenize pour l'entrainement DPO. L'adaptateur concret
(`ChatMLFormateurAdapter`, infrastructure, deja utilise pour
`FormateurConversation`/`FormateurInviteZeroShot`) enveloppe
`AutoTokenizer.apply_chat_template` ; ce port ne connait ni tokenizer
ni transformers, meme principe que les deux autres ports de formatage.

Port SEPARE de `FormateurConversation` (pas une methode supplementaire
ajoutee dessus) : meme precedent deja pose entre `FormateurConversation`
et `FormateurInviteZeroShot` (contrats structurellement differents,
meme adaptateur concret peut implementer plusieurs ports), cf.
docs/04_etape3_dpo/00_introduction_concepts.md §4.3.
"""

from __future__ import annotations

from typing import Protocol

from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.domain.model.exemple_pivot import ExemplePivot


class FormateurPreference(Protocol):
    """Port generique : rendre un ExemplePivot DPO en triplet prompt/chosen/rejected."""

    def formater(self, exemple: ExemplePivot) -> ExempleFormatePreference:
        """Rend `exemple.prompt`/`chosen`/`rejected` en triplet texte distinct, jamais concatene."""
        ...
