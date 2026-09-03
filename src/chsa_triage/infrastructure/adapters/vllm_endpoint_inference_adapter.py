"""
Adaptateur secondaire (SQUELETTE — branche a l'Etape 4) : inference
distante via l'endpoint vLLM deploye sur le cloud (livrable mission).

Implemente le meme port `MoteurInference` que l'adaptateur local :
le backend FastAPI (interfaces/api) choisit l'un ou l'autre par
injection de dependance, sans que l'application ne le sache.
"""

from __future__ import annotations

from chsa_triage.domain.ports.moteur_inference import ReponseModele


class VllmEndpointInferenceAdapter:
    """
    Adaptateur distant implementant MoteurInference via une requete
    HTTP vers l'endpoint vLLM deploye sur le cloud.

    TODO (Etape 4) : client HTTP vers l'endpoint vLLM (API compatible
    OpenAI /v1/chat/completions), avec gestion des secrets/cles
    (point de vigilance mission : "proteger les cles/secrets et
    l'acces aux endpoints").
    """

    def __init__(self, url_endpoint: str, cle_api: str | None = None) -> None:
        self._url_endpoint = url_endpoint
        self._cle_api = cle_api

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        raise NotImplementedError(
            "VllmEndpointInferenceAdapter sera implemente a l'Etape 4 "
            "(endpoint de demonstration vLLM)."
        )
