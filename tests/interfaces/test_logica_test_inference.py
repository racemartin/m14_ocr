"""
Tests de la logique pure du frontend de test
(`interfaces/web/logica_test_inference.py`) : aucun Streamlit, aucun
reseau reel. `httpx.MockTransport` simule les reponses de l'API
FastAPI (`interfaces/api/`), meme discipline que les tests avec faux
client HTTP en memoire de `VllmEndpointInferenceAdapter`.
"""

from __future__ import annotations

import json

import httpx

from interfaces.web.logica_test_inference import (
    demarrer_conversation,
    interroger_sante,
    obtenir_diagnostic,
    poursuivre_conversation,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://api-test", transport=httpx.MockTransport(handler))


def test_interroger_sante_relaie_le_corps_json():
    def handler(requete: httpx.Request) -> httpx.Response:
        assert requete.url.path == "/sante"
        return httpx.Response(200, json={"disponible": True, "detail": "serveur vLLM disponible"})

    resultat = interroger_sante(_client(handler))

    assert resultat == {"disponible": True, "detail": "serveur vLLM disponible"}


def test_interroger_sante_ne_laisse_jamais_remonter_une_erreur_brute_si_api_injoignable():
    def handler(requete: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusee", request=requete)

    resultat = interroger_sante(_client(handler))

    assert resultat["disponible"] is False
    assert "injoignable" in resultat["detail"]


def test_demarrer_conversation_retourne_l_identifiant():
    def handler(requete: httpx.Request) -> httpx.Response:
        assert requete.method == "POST"
        assert requete.url.path == "/conversations"
        return httpx.Response(200, json={"conversation_id": "abc-123"})

    assert demarrer_conversation(_client(handler)) == "abc-123"


def test_poursuivre_conversation_envoie_le_message_et_retourne_la_reponse():
    def handler(requete: httpx.Request) -> httpx.Response:
        assert requete.url.path == "/conversations/abc-123/messages"
        assert json.loads(requete.content) == {"message": "Douleur au ventre."}
        return httpx.Response(
            200, json={"conversation_id": "abc-123", "message_assistant": "Depuis quand ?"}
        )

    reponse = poursuivre_conversation(_client(handler), "abc-123", "Douleur au ventre.")

    assert reponse == "Depuis quand ?"


def test_obtenir_diagnostic_retourne_le_corps_complet():
    def handler(requete: httpx.Request) -> httpx.Response:
        assert requete.method == "POST"
        assert requete.url.path == "/conversations/abc-123/diagnostic"
        return httpx.Response(
            200,
            json={
                "conversation_id": "abc-123",
                "format_respecte": True,
                "niveau": 3,
                "categorie": "respiratoire",
                "ressources_estimees": "oxygenotherapie",
                "raisonnement": "raisonnement clinique",
                "texte_brut": "<think>...</think>{...}",
            },
        )

    resultat = obtenir_diagnostic(_client(handler), "abc-123")

    assert resultat["format_respecte"] is True
    assert resultat["niveau"] == 3
    assert resultat["categorie"] == "respiratoire"
