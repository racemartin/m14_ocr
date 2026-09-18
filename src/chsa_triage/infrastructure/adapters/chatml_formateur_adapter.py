"""
Adaptateur secondaire : rendu ChatML reel d'un ExemplePivot via
`AutoTokenizer.apply_chat_template` (le chat template natif du
modele, pas un template maison), conformement a
docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md §7.

Implemente les ports `FormateurConversation`, `FormateurInviteZeroShot`
et `FormateurPreference` (methode `formater_preference()`, ajoutee pour
l'Etape 3/DPO, docs/04_etape3_dpo/03_guide_implementation_pas_a_pas.md
etape 10). Le tokenizer est charge paresseusement (au premier appel,
pas a la construction) : la plupart des tests unitaires du domaine/de
l'application n'ont pas besoin de charger un vrai tokenizer, seuls les
tests d'integration de cet adaptateur
(`tests/infrastructure/test_chatml_formateur_adapter.py`) le font.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
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
        messages = [{"role": message.role, "content": message.contenu} for message in exemple.prompt]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def formater_preference(self, exemple: ExemplePivot) -> ExempleFormatePreference:
        """
        Rend `exemple.prompt`/`chosen`/`rejected` en triplet texte
        DISTINCT (implemente `FormateurPreference`, Etape 3/DPO) :
        `texte_prompt` suit le meme rendu que `formater_invite_zero_shot()`
        (`add_generation_prompt=True`, prompt seul) ; `texte_chosen`/
        `texte_rejected` rendent chacun le tour assistant correspondant
        SEUL (sans `system` ni `user`), via les tokens de controle reels
        du template plutot que `apply_chat_template()` (cf.
        `_rendre_tour_assistant_seul` ci-dessous pour le pourquoi).

        CORRECTION reelle (19/09/2026, cf. AGENTS.md) d'un bug confirme
        empiriquement par le test dedie
        (`tests/infrastructure/test_chatml_formateur_adapter.py::test_formater_preference_rend_un_triplet_texte_distinct`) :
        appeler `apply_chat_template([message_assistant_seul], ...)`
        (l'ancienne implementation) faisait considerer au template Qwen3
        ce tour assistant comme "non final" (aucun message `user` dans
        la liste passee => `last_query_index` retombe a l'index du seul
        message present, `loop.index0 > last_query_index` devient faux),
        ce qui declenche la branche du template qui NE reinjecte PAS le
        bloc `<think>...</think>` (uniquement la partie post-`</think>`,
        cf. `reasoning_content`/`content` dans le jinja source). Verifie
        directement par inspection du jinja reel (`tokenizer.chat_template`,
        installation temporaire, meme methode deja utilisee dans ce
        projet pour `peft.LoraConfig`/`trl.SFTConfig`/`trl.DPOTrainer`) :
        c'est exactement cette branche qui strippe. La correction
        contourne `apply_chat_template()` pour ce champ precis et
        construit le texte directement avec les tokens de controle reels
        du modele (`<|im_start|>{role}\\n{contenu}<|im_end|>\\n`,
        confirmes atomiques a la tokenisation, meme verification que
        `scripts/check_env_gpu.py::verifier_chat_template`) : `contenu`
        est utilise VERBATIM (jamais scinde sur `</think>`), donc
        `<think>` survit pour `chosen`. Confirme sans regression pour
        `rejected` (jamais de `<think>` dans son contenu source) : la
        sortie est BYTE-IDENTIQUE a l'ancien rendu via
        `apply_chat_template`, verifie par appel reel compare aux deux
        methodes sur le meme texte.
        """
        tokenizer = self._obtenir_tokenizer()
        messages_prompt = [{"role": message.role, "content": message.contenu} for message in exemple.prompt]
        texte_prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
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
        """
        Rend un tour assistant seul via les tokens de controle reels du
        template Qwen3 (`<|im_start|>{role}\\n{contenu}<|im_end|>\\n`),
        PAS via `apply_chat_template()` : cf. docstring de
        `formater_preference()` pour le bug reel que ce contournement
        corrige (`apply_chat_template` sur un message assistant isole,
        sans tour `user` precedent dans la liste, strippe `<think>`).
        `contenu` est utilise tel quel, jamais reparse : c'est le meme
        texte que celui produit par `parser_reformulation_stricte()`
        pour un `chosen` reformule, deja dans le format cible exact.
        """
        return "".join(f"<|im_start|>{message.role}\n{message.contenu}<|im_end|>\n" for message in tour)
