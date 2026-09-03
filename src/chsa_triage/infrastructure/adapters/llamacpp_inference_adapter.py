"""
Adaptateur secondaire (SQUELETTE — branche a l'Etape 4) : inference
locale via llama.cpp / GGUF, conformement a la strategie de cout
minimal (cf. echange sur l'inference locale).

Implemente le port `MoteurInference`.
"""

from __future__ import annotations

from chsa_triage.domain.ports.moteur_inference import ReponseModele


class LlamaCppInferenceAdapter:
    """
    Adaptateur local implementant MoteurInference via un serveur
    llama.cpp deja lance en local (ex. `llama-server -m modele.gguf`).

    TODO (Etape 4) : brancher un client HTTP vers le serveur
    llama-server local (API compatible OpenAI), ou utiliser le
    binding python `llama-cpp-python` directement.
    """

    def __init__(self, url_serveur_local: str = "http://127.0.0.1:8080") -> None:
        self._url_serveur_local = url_serveur_local

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        # TODO (Etape 4) : mesurer time.perf_counter() autour de l'appel
        # HTTP reel vers llama-server (POST /v1/chat/completions)
        raise NotImplementedError(
            "LlamaCppInferenceAdapter sera implemente a l'Etape 4 "
            "(chat local via llama.cpp)."
        )
