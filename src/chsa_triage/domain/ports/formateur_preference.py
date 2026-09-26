"""
Port du formatage d'un triplet de preference : rendre un ExemplePivot
DPO en `ExempleFormatePreference` (prompt/chosen/rejected distincts).
Meme `ChatMLFormateurAdapter` que `FormateurConversation`/
`FormateurInviteZeroShot`, mais port SEPARE (pas une methode de plus
sur `FormateurConversation`) car le contrat differe structurellement :
trois textes distincts, jamais concatenes.
"""

from __future__ import annotations

from typing import Protocol

from chsa_triage.domain.model.exemple_formate_preference import (
    ExempleFormatePreference,
)
from chsa_triage.domain.model.exemple_pivot import ExemplePivot


class FormateurPreference(Protocol):
    """Port generique : rendre un ExemplePivot DPO en triplet prompt/chosen/rejected."""

    def formater_preference(
        self, exemple: ExemplePivot
    ) -> ExempleFormatePreference:
        """Rend `exemple.prompt`/`chosen`/`rejected` en triplet texte distinct, jamais concatene."""
        ...
