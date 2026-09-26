"""
Stockage en memoire de l'historique des conversations d'entretien.
Deliberement hors de l'architecture hexagonale ports/adapters : une
conversation vivante est un etat de session ephemere (perdu au
redemarrage), pas une entite metier persistee. Les cas d'usage restent
purs (historique pris en parametre) ; `interfaces/api/app.py` possede
ce magasin. Un deploiement multi-instance voudrait un stockage partage
(Redis, base) : hors perimetre de ce POC.
"""

from __future__ import annotations

import threading
import uuid

from chsa_triage.domain.model.exemple_pivot import Message


class MagasinConversationsMemoire:
    """Associe un `conversation_id` a son historique `list[Message]`, thread-safe."""

    def __init__(self) -> None:
        self._conversations: dict[str, list[Message]] = {}
        self._verrou = threading.Lock()

    def creer(self) -> str:
        conversation_id = str(uuid.uuid4())
        with self._verrou:
            self._conversations[conversation_id] = []
        return conversation_id

    def obtenir(self, conversation_id: str) -> list[Message] | None:
        with self._verrou:
            historique = self._conversations.get(conversation_id)
            return list(historique) if historique is not None else None

    def ajouter(self, conversation_id: str, *messages: Message) -> None:
        with self._verrou:
            self._conversations[conversation_id].extend(messages)
