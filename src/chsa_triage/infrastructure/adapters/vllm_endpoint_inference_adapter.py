"""
Adaptateur secondaire : inference distante via un serveur vLLM
(`vllm serve`, API compatible OpenAI), branche a l'Etape 4.

Implemente le meme port `MoteurInference` que `LlamaCppInferenceAdapter`
(Etape 1bis) et `TransformersLoraInferenceAdapter` (Etape 2/3) : le
backend FastAPI (`interfaces/api/`) choisit l'adaptateur par injection
de dependance, sans que l'application ne connaisse la difference.

Sert nativement l'adaptateur LoRA DPO SANS fusion prealable avec la
base (decision Etape 4, cf. AGENTS.md/roadmap) :

    vllm serve Qwen/Qwen3-1.7B-Base \\
        --enable-lora --lora-modules dpo=mombasstic/chsa-triage-dpo-lora

Un seul mode (contrairement a `LlamaCppInferenceAdapter`, qui en a
deux) : POST `/v1/chat/completions`, `messages` transmis tel quel, le
serveur vLLM applique son propre chat template. `parametres["model"]`
permet de cibler l'adaptateur LoRA charge (ex. `"dpo"`, le nom donne a
`--lora-modules` ci-dessus) plutot que la base ; a defaut, vLLM sert le
modele de base.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.ports.moteur_inference import ReponseModele

TIMEOUT_SECONDES_DEFAUT = 120.0
MODELE_PAR_DEFAUT = "dpo"


@dataclass(slots=True)
class VllmEndpointInferenceAdapter:
    """
    Adaptateur distant implementant `MoteurInference` via une requete
    HTTP vers un serveur vLLM (local ou deploye sur le cloud).

    `_client` est injectable (utilise par les tests avec un faux client
    httpx, sans reseau ni serveur reel) ; laisse a `None` en usage
    normal, un `httpx.Client` reel est cree paresseusement au premier
    `generer()`. `cle_api`, si fournie, est envoyee en en-tete
    `Authorization: Bearer <cle>` (vLLM peut etre lance avec
    `--api-key`) ; jamais loguee ni incluse dans `metadonnees`.
    """

    url_endpoint    : str
    cle_api           : str | None = None
    nom_modele        : str = MODELE_PAR_DEFAUT
    timeout_secondes  : float = TIMEOUT_SECONDES_DEFAUT
    _client             : Any = field(default=None, repr=False)

    def _obtenir_client(self) -> Any:
        if self._client is None:
            import httpx

            entetes = {}
            if self.cle_api:
                entetes["Authorization"] = f"Bearer {self.cle_api}"
            self._client = httpx.Client(timeout=self.timeout_secondes, headers=entetes)
        return self._client

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        """
        `latence_ms` mesure exactement le temps de l'appel HTTP reel
        (`time.perf_counter()` autour de `client.post`), pas de temps
        de construction/parsing autour. `parametres["model"]` (defaut
        `self.nom_modele`) selectionne l'adaptateur LoRA vLLM cible ;
        retire de `parametres` avant transmission (place explicitement
        au niveau superieur du corps de requete, comme l'exige l'API
        compatible OpenAI de vLLM).
        """
        parametres = dict(parametres or {})
        modele = parametres.pop("model", self.nom_modele)

        client = self._obtenir_client()
        corps = {"model": modele, "messages": messages, **parametres}

        debut = time.perf_counter()
        reponse = client.post(f"{self.url_endpoint}/v1/chat/completions", json=corps)
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
            metadonnees={"modele": donnees.get("model", modele), "reponse_brute": donnees},
        )
