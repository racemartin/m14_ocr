"""
Tests du calcul des bornes `ExempleFormate.tours` par
`ChatMLFormateurAdapter.formater`, avec un FAUX tokenizer (pas de
reseau/HF requis, contrairement a
`tests/infrastructure/test_chatml_formateur_adapter.py`) : injecte
directement dans `_tokenizer` pour court-circuiter
`_obtenir_tokenizer()`. Le faux tokenizer reproduit fidelement le
format ChatML reel (`<|im_start|>{role}\\n{content}<|im_end|>\\n`, pas
de prefixe de generation hors `add_generation_prompt=True`), ce qui
suffit a verifier que les bornes calculees par prefixes successifs
pointent bien sur les bons tours dans `texte`.
"""

from __future__ import annotations

from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.infrastructure.adapters.chatml_formateur_adapter import (
    ChatMLFormateurAdapter,
)


class _FauxTokenizerChatML:
    """Reproduit le format ChatML de Qwen3 (cf.
    `ChatMLFormateurAdapter._rendre_tour_assistant_seul`), y compris le
    comportement reel documente du bug de stripping `<think>` sur un
    tour assistant NON-FINAL de la liste passee : sert a prouver que
    `formater()` n'expose jamais ce cas (seuls des prefixes SANS tour
    assistant sont rendus via ce faux `apply_chat_template`)."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ) -> str:
        assert tokenize is False
        rendu = "".join(
            self._rendre_message(message, est_final=(i == len(messages) - 1))
            for i, message in enumerate(messages)
        )
        if add_generation_prompt:
            rendu += "<|im_start|>assistant\n"
        return rendu

    @staticmethod
    def _rendre_message(message: dict, est_final: bool) -> str:
        contenu = message["content"]
        if message["role"] == "assistant" and not est_final:
            # reproduit le bug reel : <think> strippe sur un tour
            # assistant non final
            contenu = contenu.split("</think>")[-1]
        return f"<|im_start|>{message['role']}\n{contenu}<|im_end|>\n"


def _adaptateur_avec_faux_tokenizer() -> ChatMLFormateurAdapter:
    adaptateur = ChatMLFormateurAdapter()
    adaptateur._tokenizer = _FauxTokenizerChatML()
    return adaptateur


def _exemple(prompt: tuple[Message, ...], completion: tuple[Message, ...]) -> ExemplePivot:
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test", "cas-01"),
        source="Test",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=prompt,
        completion=completion,
    )


def test_formater_calcule_les_bornes_user_puis_assistant():
    exemple = _exemple(
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )
    adaptateur = _adaptateur_avec_faux_tokenizer()

    resultat = adaptateur.formater(exemple)

    assert len(resultat.tours) == 2
    assert resultat.tours[0].role == "user"
    assert resultat.tours[1].role == "assistant"
    assert resultat.tours[0].debut == 0
    assert resultat.tours[0].fin == resultat.tours[1].debut
    assert resultat.tours[1].fin == len(resultat.texte)
    assert resultat.texte[resultat.tours[0].debut : resultat.tours[0].fin] == (
        "<|im_start|>user\nQuestion ?<|im_end|>\n"
    )
    assert resultat.texte[resultat.tours[1].debut : resultat.tours[1].fin] == (
        "<|im_start|>assistant\nReponse.<|im_end|>\n"
    )
    assert resultat.textes_assistant() == ("<|im_start|>assistant\nReponse.<|im_end|>\n",)


def test_formater_gere_un_prompt_multi_tours_system_et_user():
    exemple = _exemple(
        prompt=(
            Message(role="system", contenu="Tu es un assistant medical."),
            Message(role="user", contenu="Question ?"),
        ),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )
    adaptateur = _adaptateur_avec_faux_tokenizer()

    resultat = adaptateur.formater(exemple)

    assert [tour.role for tour in resultat.tours] == ["system", "user", "assistant"]
    # chaque tour est contigu : la fin de l'un est le debut du suivant
    for precedent, suivant in zip(resultat.tours, resultat.tours[1:]):
        assert precedent.fin == suivant.debut
    assert resultat.tours[-1].fin == len(resultat.texte)


def test_formater_preserve_think_sur_le_tour_assistant_final():
    """Le tour assistant du rendu complet est TOUJOURS le dernier tour
    passe a `apply_chat_template` (jamais non-final) : le `<think>` du
    faux tokenizer (qui reproduit le bug reel de stripping sur un tour
    assistant non-final) doit donc survivre dans `texte`, et le tour
    `assistant` calcule doit couvrir tout le bloc, `<think>` inclus."""
    exemple = _exemple(
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(
            Message(
                role="assistant",
                contenu="<think>raisonnement</think>Reponse.",
            ),
        ),
    )
    adaptateur = _adaptateur_avec_faux_tokenizer()

    resultat = adaptateur.formater(exemple)

    assert "<think>raisonnement</think>" in resultat.texte
    (texte_assistant,) = resultat.textes_assistant()
    assert "<think>raisonnement</think>" in texte_assistant
