"""Schemas Pydantic des requetes/reponses de l'API FastAPI (Etape 4)."""

from __future__ import annotations

from pydantic import BaseModel


class DemarrerConversationReponse(BaseModel):
    conversation_id: str


class MessageEntretienRequete(BaseModel):
    message: str


class MessageEntretienReponse(BaseModel):
    conversation_id: str
    message_assistant: str


class TourHistorique(BaseModel):
    role: str
    contenu: str


class ConversationReponse(BaseModel):
    conversation_id: str
    historique: list[TourHistorique]


class DiagnosticReponse(BaseModel):
    conversation_id: str
    format_respecte: bool
    niveau: int | None = None
    categorie: str | None = None
    ressources_estimees: str | None = None
    raisonnement: str | None = None
    texte_brut: str
