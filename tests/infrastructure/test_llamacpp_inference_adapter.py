"""
Tests de LlamaCppInferenceAdapter.

Deux familles :
- Unitaires (toujours executes, sans reseau) : un faux client HTTP
  injecte via `_client`, verifie l'URL/le corps envoyes et le mapping
  de la reponse JSON vers `ReponseModele`, pour les deux modes
  (`/v1/chat/completions` et `/completion`).
- Integration REELLE (opt-in, PAS executee par defaut) : contre un
  `llama-server` deja lance en local, servant un GGUF reel de
  `Qwen/Qwen3-1.7B-Base`. Saute automatiquement si la variable
  d'environnement `LLAMACPP_URL_SERVEUR_TEST` n'est pas definie (ex.
  cette suite tourne dans un environnement CI sans serveur/modele
  provisionne) : voir AGENTS.md pour comment demarrer un tel serveur
  et la commande exacte utilisee pour la verification manuelle
  effectuee lors de l'implementation de cet adaptateur.
"""

from __future__ import annotations

import os

import pytest

from chsa_triage.infrastructure.adapters.llamacpp_inference_adapter import (
    LlamaCppInferenceAdapter,
)


class FauxReponseHttp:
    def __init__(self, donnees: dict) -> None:
        self._donnees = donnees

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._donnees


class FauxClientHttp:
    """Faux client HTTP, en memoire : capture le dernier appel `post()`."""

    def __init__(self, donnees_reponse: dict) -> None:
        self._donnees_reponse = donnees_reponse
        self.dernier_url: str | None = None
        self.dernier_corps: dict | None = None

    def post(self, url: str, json: dict) -> FauxReponseHttp:
        self.dernier_url = url
        self.dernier_corps = json
        return FauxReponseHttp(self._donnees_reponse)


def test_generer_mode_chat_completions_par_defaut():
    """Sans `invite_deja_rendue`, POST `/v1/chat/completions`, `messages` transmis tel quel."""
    faux_client = FauxClientHttp(
        {
            "choices": [{"message": {"content": "Bonjour, comment puis-je vous aider ?"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            "model": "qwen3-1.7b-base",
        }
    )
    adaptateur = LlamaCppInferenceAdapter(url_serveur_local="http://127.0.0.1:9999", _client=faux_client)
    messages = [{"role": "user", "content": "Bonjour"}]

    reponse = adaptateur.generer(messages, {"temperature": 0.0})

    assert faux_client.dernier_url == "http://127.0.0.1:9999/v1/chat/completions"
    assert faux_client.dernier_corps == {"messages": messages, "temperature": 0.0}
    assert reponse.texte == "Bonjour, comment puis-je vous aider ?"
    assert reponse.nombre_tokens_entree == 12
    assert reponse.nombre_tokens_sortie == 8
    assert reponse.latence_ms >= 0.0
    assert reponse.metadonnees["modele"] == "qwen3-1.7b-base"


def test_generer_mode_completion_brute_avec_invite_deja_rendue():
    """`invite_deja_rendue=True` : POST `/completion` (natif), `prompt` = content du dernier message."""
    faux_client = FauxClientHttp({"content": "Reponse brute.", "tokens_evaluated": 17, "tokens_predicted": 5})
    adaptateur = LlamaCppInferenceAdapter(url_serveur_local="http://127.0.0.1:9999", _client=faux_client)
    invite = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"

    reponse = adaptateur.generer([{"role": "user", "content": invite}], {"invite_deja_rendue": True, "n_predict": 5})

    assert faux_client.dernier_url == "http://127.0.0.1:9999/completion"
    assert faux_client.dernier_corps == {"prompt": invite, "n_predict": 5}
    assert reponse.texte == "Reponse brute."
    assert reponse.nombre_tokens_entree == 17
    assert reponse.nombre_tokens_sortie == 5


def test_generer_mode_completion_brute_leve_si_aucun_message():
    adaptateur = LlamaCppInferenceAdapter(_client=FauxClientHttp({}))
    with pytest.raises(ValueError):
        adaptateur.generer([], {"invite_deja_rendue": True})


def test_parametre_invite_deja_rendue_jamais_transmis_au_serveur():
    """`invite_deja_rendue` est un signal interne a l'adaptateur, jamais un parametre llama.cpp reel."""
    faux_client = FauxClientHttp({"content": "ok", "tokens_evaluated": 1, "tokens_predicted": 1})
    adaptateur = LlamaCppInferenceAdapter(_client=faux_client)

    adaptateur.generer([{"role": "user", "content": "x"}], {"invite_deja_rendue": True, "temperature": 0.5})

    assert "invite_deja_rendue" not in faux_client.dernier_corps
    assert faux_client.dernier_corps["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Integration reelle (opt-in), contre un llama-server deja lance en local.
# ---------------------------------------------------------------------------

URL_SERVEUR_TEST = os.environ.get("LLAMACPP_URL_SERVEUR_TEST")

pytestmark_integration = pytest.mark.skipif(
    not URL_SERVEUR_TEST,
    reason="LLAMACPP_URL_SERVEUR_TEST non definie : pas de llama-server reel disponible pour ce run "
    "(voir AGENTS.md pour demarrer un serveur local et rejouer ce test contre un GGUF reel).",
)


@pytestmark_integration
def test_integration_reelle_chat_completions():
    adaptateur = LlamaCppInferenceAdapter(url_serveur_local=URL_SERVEUR_TEST, timeout_secondes=180.0)

    reponse = adaptateur.generer(
        [{"role": "user", "content": "Dis bonjour."}], {"temperature": 0.0, "max_tokens": 6}
    )

    assert isinstance(reponse.texte, str) and len(reponse.texte) > 0
    assert reponse.latence_ms > 0.0
    assert reponse.nombre_tokens_sortie > 0


@pytestmark_integration
def test_integration_reelle_completion_brute_avec_invite_pre_rendue():
    adaptateur = LlamaCppInferenceAdapter(url_serveur_local=URL_SERVEUR_TEST, timeout_secondes=180.0)
    invite = "<|im_start|>user\nDis bonjour.<|im_end|>\n<|im_start|>assistant\n"

    reponse = adaptateur.generer(
        [{"role": "user", "content": invite}], {"invite_deja_rendue": True, "n_predict": 6, "temperature": 0.0}
    )

    assert isinstance(reponse.texte, str) and len(reponse.texte) > 0
    assert reponse.latence_ms > 0.0
    assert reponse.nombre_tokens_sortie == 6
