"""
Adaptateur secondaire : rendu ChatML reel d'un ExemplePivot via
`AutoTokenizer.apply_chat_template` (le chat template natif du
modele, pas un template maison), conformement a
docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md §7.

Implemente le port `FormateurConversation`. Le tokenizer est charge
paresseusement (au premier `formater()`, pas a la construction) : la
plupart des tests unitaires du domaine/de l'application n'ont pas
besoin de charger un vrai tokenizer, seuls les tests d'integration de
cet adaptateur (`tests/infrastructure/test_chatml_formateur_adapter.py`)
le font.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.model.exemple_pivot import ExemplePivot


@dataclass(slots=True)
class ChatMLFormateurAdapter:
    """Rend un ExemplePivot en texte ChatML via le tokenizer reel du modele cible."""

    nom_modele: str = "Qwen/Qwen3-1.7B-Base"
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _obtenir_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.nom_modele, trust_remote_code=True
            )
        return self._tokenizer

    def formater(self, exemple: ExemplePivot) -> ExempleFormate:
        """
        Concatene prompt + completion (dans cet ordre : un exemple
        d'entrainement complet, pas une invite a generer) et rend le
        tout via le chat template natif du tokenizer,
        `add_generation_prompt=False`.
        """
        tokenizer = self._obtenir_tokenizer()
        messages = [
            {"role": message.role, "content": message.contenu}
            for message in (*exemple.prompt, *exemple.completion)
        ]
        texte = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return ExempleFormate(identifiant=exemple.identifiant, texte=texte)
