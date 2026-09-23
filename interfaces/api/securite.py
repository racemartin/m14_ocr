"""
Authentification minimale de l'API de demonstration, par cle statique
en en-tete `X-API-Key` : proportionnee a un POC (cahier des charges,
point de vigilance "proteger les cles/secrets et l'acces aux
endpoints"), jamais une clef en dur dans le code (comparee a
`cle_attendue`, injectee depuis une variable d'environnement par
`interfaces/api/main.py`, jamais un defaut).
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status


def creer_dependance_verification_cle_api(cle_attendue: str):
    """Fabrique une dependance FastAPI verifiant l'en-tete `X-API-Key` contre `cle_attendue`."""

    def verifier_cle_api(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key != cle_attendue:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Cle API invalide ou absente")

    return verifier_cle_api
