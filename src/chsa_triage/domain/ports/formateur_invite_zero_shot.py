"""
Port du formatage d'une invite d'inference zero-shot : rendre
UNIQUEMENT le `prompt` d'un `ExemplePivot` en texte pret a etre envoye
a un `MoteurInference`, sans montrer de reponse. Distinct de
`domain.ports.formateur_conversation.FormateurConversation` (qui rend
prompt+completion pour l'ENTRAINEMENT, `add_generation_prompt=False`) :
ici c'est l'oppose exact, `add_generation_prompt=True`, utilise par
`E1_06_00_evaluer_baseline_zero_shot.py` (Etape 1bis). Le meme adaptateur
concret (`ChatMLFormateurAdapter`) implemente les deux ports.
"""

from __future__ import annotations

from typing import Protocol

from chsa_triage.domain.model.exemple_pivot import ExemplePivot


class FormateurInviteZeroShot(Protocol):
    """Port generique : rendre le prompt seul d'un ExemplePivot en invite d'inference."""

    def formater_invite_zero_shot(self, exemple: ExemplePivot) -> str:
        """Rend `exemple.prompt` (jamais `completion`) via apply_chat_template, `add_generation_prompt=True`."""
        ...
