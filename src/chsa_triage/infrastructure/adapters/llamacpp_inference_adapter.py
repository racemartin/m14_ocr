"""
Adaptateur secondaire : inference locale via llama.cpp / GGUF, contre
un serveur `llama-server` DEJA LANCE (ex. `llama-server -m
Qwen3-1.7B-Base.Q4_K_M.gguf --port 8080 -c 2048`). Ne demarre pas le
serveur lui-meme.

Deux modes :
- Normal (defaut) : POST `/v1/chat/completions`, le serveur applique
  son propre chat template (embarque dans le GGUF).
- `invite_deja_rendue=True` : POST `/completion` (API native, prompt
  brut) ; `messages` contient un seul message deja rendu par le VRAI
  tokenizer HF (`ChatMLFormateurAdapter`), pour eviter toute derive
  entre le chat template embarque dans le GGUF et celui du tokenizer
  d'origine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.ports.moteur_inference import ReponseModele

TIMEOUT_SECONDES_DEFAUT = 120.0


@dataclass(slots=True)
class LlamaCppInferenceAdapter:
    """
    Adaptateur local implementant MoteurInference via un serveur
    llama.cpp deja lance en local (ex. `llama-server -m modele.gguf`).

    `_client` est injectable (utilise par les tests avec un faux client
    httpx, sans reseau ni serveur reel) ; laisse a None en usage
    normal, un `httpx.Client` reel est cree paresseusement au premier
    `generer()`.
    """

    url_serveur_local: str = "http://127.0.0.1:8080"
    timeout_secondes: float = TIMEOUT_SECONDES_DEFAUT
    _client: Any = field(default=None, repr=False)

    def _obtenir_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_secondes)
        return self._client

    def generer(
        self, messages: list[dict], parametres: dict | None = None
    ) -> ReponseModele:
        """
        `latence_ms` mesure exactement le temps de l'appel HTTP reel
        (`time.perf_counter()` autour de `client.post`), pas de temps
        de construction/parsing autour. `parametres["invite_deja_rendue"]`
        (defaut `False`) bascule entre les deux modes decrits dans le
        docstring du module ; retiree de `parametres` avant transmission
        au serveur (ce n'est pas un parametre de generation llama.cpp).
        """
        parametres = dict(parametres or {})
        invite_deja_rendue = parametres.pop("invite_deja_rendue", False)

        if invite_deja_rendue:
            return self._generer_completion_brute(messages, parametres)
        return self._generer_chat_completion(messages, parametres)

    def _generer_chat_completion(
        self, messages: list[dict], parametres: dict
    ) -> ReponseModele:
        client = self._obtenir_client()
        corps = {"messages": messages, **parametres}

        debut = time.perf_counter()
        reponse = client.post(
            f"{self.url_serveur_local}/v1/chat/completions", json=corps
        )
        latence_ms = (time.perf_counter() - debut) * 1000

        reponse.raise_for_status()
        donnees = reponse.json()

        texte = donnees["choices"][0]["message"]["content"]
        usage = donnees.get("usage", {})

        return ReponseModele(
            texte=texte,
            nombre_tokens_entree=usage.get("prompt_tokens", 0),
            nombre_tokens_sortie=usage.get("completion_tokens", 0),
            latence_ms=latence_ms,
            metadonnees={
                "modele": donnees.get("model", ""),
                "reponse_brute": donnees,
            },
        )

    def _generer_completion_brute(
        self, messages: list[dict], parametres: dict
    ) -> ReponseModele:
        """
        `/completion` (API native llama.cpp) : `prompt` est le
        `content` du DERNIER message de `messages` (un seul message
        attendu dans ce mode, deja rendu en texte ChatML complet).
        """
        if not messages:
            raise ValueError(
                "invite_deja_rendue=True necessite au moins un message (l'invite deja rendue)"
            )

        client = self._obtenir_client()
        corps = {"prompt": messages[-1]["content"], **parametres}

        debut = time.perf_counter()
        reponse = client.post(
            f"{self.url_serveur_local}/completion", json=corps
        )
        latence_ms = (time.perf_counter() - debut) * 1000

        reponse.raise_for_status()
        donnees = reponse.json()

        return ReponseModele(
            texte=donnees["content"],
            nombre_tokens_entree=donnees.get("tokens_evaluated", 0),
            nombre_tokens_sortie=donnees.get("tokens_predicted", 0),
            latence_ms=latence_ms,
            metadonnees={"reponse_brute": donnees},
        )
