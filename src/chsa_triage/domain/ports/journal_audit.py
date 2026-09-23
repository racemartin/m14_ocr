"""
Port de tracabilite (F6) : consigner un tour d'entretien ou un appel de
diagnostic. Mecanisme leger vise (JSONL append ou SQLite), jamais une
stack d'observabilite lourde (hors perimetre du cahier des charges,
cf. §1 "Hors perimetre").
"""

from __future__ import annotations

from typing import Protocol

from chsa_triage.domain.model.entree_audit import EntreeAudit


class JournalAudit(Protocol):
    """Port generique : consigner une entree d'audit, sans savoir ou elle atterrit."""

    def consigner(self, entree: EntreeAudit) -> None:
        """Persiste `entree` de facon durable et append-only (jamais reecrite/mutee)."""
        ...
