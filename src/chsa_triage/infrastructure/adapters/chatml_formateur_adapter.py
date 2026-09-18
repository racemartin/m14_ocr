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
        SEUL (sans `system` ni `user`), via le meme chat template natif.
        Mapping esquisse par analogie, jamais verifie contre un vrai
        appel `DPOTrainer.train()` (point de vigilance explicitement
        laisse ouvert par docs/04_etape3_dpo/00_introduction_concepts.md,
        "Point de vigilance").

        Trouvaille reelle (verifiee par appel direct, pas supposee, cf.
        AGENTS.md) : le chat template natif de Qwen3-1.7B-Base RETIRE le
        bloc `<think>...</think>` d'un tour assistant qui n'est pas le
        dernier tour genere. `texte_chosen` ne contient donc PLUS le
        raisonnement `<think>` d'un `ChosenReformule` (§3.2 du document
        d'introduction), seul le JSON cible survit au rendu. Signale ici
        comme point de vigilance pour l'implementation reelle du DPO
        (etape 12, hors perimetre), pas corrige : corriger cela
        supposerait soit un template different, soit de ne plus passer
        par `apply_chat_template` pour ce champ precis.
        """
        tokenizer = self._obtenir_tokenizer()
        messages_prompt = [{"role": message.role, "content": message.contenu} for message in exemple.prompt]
        texte_prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        texte_chosen = self._rendre_tour_assistant_seul(tokenizer, exemple.chosen)
        texte_rejected = self._rendre_tour_assistant_seul(tokenizer, exemple.rejected)
        return ExempleFormatePreference(
            identifiant=exemple.identifiant,
            texte_prompt=texte_prompt,
            texte_chosen=texte_chosen,
            texte_rejected=texte_rejected,
        )

    @staticmethod
    def _rendre_tour_assistant_seul(tokenizer: Any, tour) -> str:
        messages = [{"role": message.role, "content": message.contenu} for message in tour]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
