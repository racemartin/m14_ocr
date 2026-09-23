"""
Logique pure (aucun Streamlit, aucune variable d'environnement) du
frontend de test de l'entretien clinique. Separee de
`app_test_inference.py` pour rester testable sans Streamlit installe
(meme discipline que `monitoring/logica_suivi_entrainement.py` pour le
dashboard de suivi d'entrainement, un but different : courbes
d'entrainement, pas un chat).

Chaque fonction prend un `httpx.Client` deja construit par l'appelant
(URL de base + en-tete `X-API-Key` portes par le client) : testable
avec `httpx.MockTransport`, sans reseau ni API/vLLM reels
(`tests/interfaces/test_logica_test_inference.py`).
"""

from __future__ import annotations

import httpx


def interroger_sante(client: httpx.Client) -> dict:
    """
    Ne laisse jamais une erreur de connexion brute remonter a
    l'appelant : l'API elle-meme (Space Docker/GPU) peut etre
    injoignable (pas encore allumee), pas seulement le moteur vLLM
    qu'elle sonde en interne via `/sante` (cf.
    `interfaces/api/app.py`, qui retourne toujours HTTP 200 pour la
    partie vLLM, mais ne protege pas contre une API totalement
    injoignable).
    """
    try:
        reponse = client.get("/sante")
        reponse.raise_for_status()
        return reponse.json()
    except httpx.HTTPError as erreur:
        return {"disponible": False, "detail": f"API injoignable : {erreur}"}


def demarrer_conversation(client: httpx.Client) -> str:
    reponse = client.post("/conversations")
    reponse.raise_for_status()
    return reponse.json()["conversation_id"]


def poursuivre_conversation(client: httpx.Client, conversation_id: str, message: str) -> str:
    reponse = client.post(f"/conversations/{conversation_id}/messages", json={"message": message})
    reponse.raise_for_status()
    return reponse.json()["message_assistant"]


def obtenir_diagnostic(client: httpx.Client, conversation_id: str) -> dict:
    reponse = client.post(f"/conversations/{conversation_id}/diagnostic")
    reponse.raise_for_status()
    return reponse.json()
