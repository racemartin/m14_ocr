"""
Tests de VllmEndpointInferenceAdapter.

Unitaires uniquement (pas d'integration reelle, aucun serveur vLLM
reel n'est provisionne dans cet environnement) : un faux client HTTP
injecte via `_client`, verifie l'URL/le corps/les en-tetes envoyes et
le mapping de la reponse JSON vers `ReponseModele`. Meme patron que
`tests/infrastructure/test_llamacpp_inference_adapter.py`.
"""

from __future__ import annotations

from chsa_triage.infrastructure.adapters.vllm_endpoint_inference_adapter import (
    VllmEndpointInferenceAdapter,
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


def test_generer_poste_vers_chat_completions_avec_le_modele_par_defaut():
    faux_client = FauxClientHttp(
        {
            "choices": [{"message": {"content": "Niveau ESI 3, orientation vers..."}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17},
            "model": "dpo",
        }
    )
    adaptateur = VllmEndpointInferenceAdapter(url_endpoint="http://127.0.0.1:8000", _client=faux_client)
    messages = [{"role": "user", "content": "Patient de 40 ans, douleur thoracique."}]

    reponse = adaptateur.generer(messages, {"temperature": 0.0})

    assert faux_client.dernier_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert faux_client.dernier_corps == {"model": "dpo", "messages": messages, "temperature": 0.0}
    assert reponse.texte == "Niveau ESI 3, orientation vers..."
    assert reponse.nombre_tokens_entree == 42
    assert reponse.nombre_tokens_sortie == 17
    assert reponse.latence_ms >= 0.0
    assert reponse.metadonnees["modele"] == "dpo"


def test_generer_permet_de_cibler_un_autre_modele_via_parametres():
    faux_client = FauxClientHttp(
        {"choices": [{"message": {"content": "ok"}}], "usage": {}, "model": "base"}
    )
    adaptateur = VllmEndpointInferenceAdapter(url_endpoint="http://127.0.0.1:8000", _client=faux_client)

    adaptateur.generer([{"role": "user", "content": "x"}], {"model": "base"})

    assert faux_client.dernier_corps["model"] == "base"


def test_parametre_model_jamais_duplique_dans_les_parametres_transmis():
    faux_client = FauxClientHttp({"choices": [{"message": {"content": "ok"}}], "usage": {}})
    adaptateur = VllmEndpointInferenceAdapter(url_endpoint="http://127.0.0.1:8000", _client=faux_client)

    adaptateur.generer([{"role": "user", "content": "x"}], {"temperature": 0.2})

    assert faux_client.dernier_corps == {
        "model": "dpo",
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.2,
    }


def test_cle_api_envoyee_en_en_tete_authorization():
    """`cle_api` doit produire un en-tete `Authorization: Bearer <cle>` sur le client HTTP reel."""
    adaptateur = VllmEndpointInferenceAdapter(url_endpoint="http://127.0.0.1:8000", cle_api="secret-123")

    client = adaptateur._obtenir_client()

    assert client.headers["Authorization"] == "Bearer secret-123"


def test_sans_cle_api_aucun_en_tete_authorization():
    adaptateur = VllmEndpointInferenceAdapter(url_endpoint="http://127.0.0.1:8000")

    client = adaptateur._obtenir_client()

    assert "Authorization" not in client.headers
