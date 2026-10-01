"""
Adaptateur secondaire : rendu ChatML reel d'un ExemplePivot via
`AutoTokenizer.apply_chat_template` (le chat template natif du modele,
pas un template maison). Implemente `FormateurConversation`,
`FormateurInviteZeroShot` et `FormateurPreference`. Le tokenizer est
charge paresseusement (au premier appel), pour que la plupart des tests
n'aient pas besoin d'un vrai tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.model.exemple_formate import ExempleFormate, LimiteTour
from chsa_triage.domain.model.exemple_formate_preference import (
    ExempleFormatePreference,
)
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
        `add_generation_prompt=False`. Calcule aussi `tours` (bornes
        caracteres par tour, cf. `LimiteTour`) pour que
        `assistant_only_loss` puisse masquer la perte aux tours
        assistant uniquement.

        Les bornes des tours `prompt` sont obtenues par prefixes
        successifs de `apply_chat_template(messages_prompt[:i], ...)` :
        sur, car aucun de ces prefixes ne contient de tour assistant
        (le bug de stripping `<think>` de Qwen3 ne cible QUE les tours
        assistant non-finaux, cf. `LimiteTour`). La borne du tour
        `completion` n'a besoin d'aucun appel supplementaire : c'est le
        tour final du rendu complet par construction, donc jamais
        strippe, et elle couvre tout le reste de `texte` apres les
        tours prompt.
        """
        tokenizer = self._obtenir_tokenizer()
        messages_prompt = [
            {"role": message.role, "content": message.contenu}
            for message in exemple.prompt
        ]
        messages_completion = [
            {"role": message.role, "content": message.contenu}
            for message in exemple.completion
        ]
        texte = tokenizer.apply_chat_template(
            messages_prompt + messages_completion,
            tokenize=False,
            add_generation_prompt=False,
        )

        tours: list[LimiteTour] = []
        debut = 0
        for i in range(1, len(messages_prompt) + 1):
            rendu_prefixe = tokenizer.apply_chat_template(
                messages_prompt[:i], tokenize=False, add_generation_prompt=False
            )
            fin = len(rendu_prefixe)
            tours.append(
                LimiteTour(role=messages_prompt[i - 1]["role"], debut=debut, fin=fin)
            )
            debut = fin
        if messages_completion:
            tours.append(LimiteTour(role="assistant", debut=debut, fin=len(texte)))

        return ExempleFormate(
            identifiant=exemple.identifiant, texte=texte, tours=tuple(tours)
        )

    def formater_invite_zero_shot(self, exemple: ExemplePivot) -> str:
        """
        Rend UNIQUEMENT `exemple.prompt` (jamais `completion`) via le
        meme chat template natif, `add_generation_prompt=True` :
        l'oppose exact de `formater()`, pour une invite d'inference
        (montrer le prompt, laisser le modele generer la suite) plutot
        qu'un exemple d'entrainement deja complet. Utilise par
        l'evaluation baseline zero-shot (Etape 1bis,
        `E1_06_00_evaluer_baseline.py`) ; hors du port
        `FormateurConversation`, dont le contrat documente explicitement
        `add_generation_prompt=False`.
        """
        tokenizer = self._obtenir_tokenizer()
        messages = [
            {"role": message.role, "content": message.contenu}
            for message in exemple.prompt
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def formater_preference(
        self, exemple: ExemplePivot
    ) -> ExempleFormatePreference:
        """
        Rend `exemple.prompt`/`chosen`/`rejected` en triplet texte
        DISTINCT. `texte_prompt` suit le meme rendu que
        `formater_invite_zero_shot()` ; `texte_chosen`/`texte_rejected`
        rendent chacun le tour assistant seul via les tokens de controle
        reels du template plutot que `apply_chat_template()` (cf.
        `_rendre_tour_assistant_seul` pour le bug reel que ce
        contournement corrige).
        """
        tokenizer = self._obtenir_tokenizer()
        messages_prompt = [
            {"role": message.role, "content": message.contenu}
            for message in exemple.prompt
        ]
        texte_prompt = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        texte_chosen = self._rendre_tour_assistant_seul(exemple.chosen)
        texte_rejected = self._rendre_tour_assistant_seul(exemple.rejected)
        return ExempleFormatePreference(
            identifiant=exemple.identifiant,
            texte_prompt=texte_prompt,
            texte_chosen=texte_chosen,
            texte_rejected=texte_rejected,
        )

    @staticmethod
    def _rendre_tour_assistant_seul(tour) -> str:
        """Rend un tour assistant seul via les tokens de controle reels
        du template Qwen3, PAS via `apply_chat_template()` : bug reel
        trouve ou `apply_chat_template` sur un message assistant isole
        (sans tour `user` precedent) strippait le bloc `<think>`.
        `contenu` est utilise verbatim, jamais reparse."""
        return "".join(
            f"<|im_start|>{message.role}\n{message.contenu}<|im_end|>\n"
            for message in tour
        )
