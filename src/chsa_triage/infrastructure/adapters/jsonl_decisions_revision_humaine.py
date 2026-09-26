"""
Adaptateur secondaire : persistance JSONL des decisions humaines sur
les candidats de PII residuelle. A la difference du registre
d'echantillons (append-only), une decision peut etre CORRIGEE : relit
tout le fichier, fusionne par cle stable, reecrit tout — un cout
O(n^2) acceptable ici (volume borne par la vitesse de lecture d'une
personne, pas par le corpus complet).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from chsa_triage.domain.model import (
    CleCandidatRevision,
    DecisionRevisionHumaine,
)


def decision_vers_dict(decision: DecisionRevisionHumaine) -> dict:
    return {
        "source_liste": decision.cle.source_liste,
        "identifiant": decision.cle.identifiant,
        "champ": decision.cle.champ,
        "type_motif": decision.cle.type_motif,
        "debut": decision.cle.debut,
        "fin": decision.cle.fin,
        "passage": decision.passage,
        "decision": decision.decision,
        "horodatage": decision.horodatage,
        "note": decision.note,
    }


def decision_depuis_dict(d: dict) -> DecisionRevisionHumaine:
    cle = CleCandidatRevision(
        source_liste=d["source_liste"],
        identifiant=d["identifiant"],
        champ=d["champ"],
        type_motif=d["type_motif"],
        debut=d["debut"],
        fin=d["fin"],
    )
    return DecisionRevisionHumaine(
        cle=cle,
        passage=d.get("passage", ""),
        decision=d["decision"],
        horodatage=d["horodatage"],
        note=d.get("note", ""),
    )


class JsonlDecisionsRevisionHumaine:
    """Adaptateur JSONL local implementant RegistreDecisionsRevisionHumaine."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    def toutes(self) -> list[DecisionRevisionHumaine]:
        return list(self._lire_tous().values())

    def cles_decidees(self) -> set[CleCandidatRevision]:
        return set(self._lire_tous().keys())

    def trouver(
        self, cle: CleCandidatRevision
    ) -> DecisionRevisionHumaine | None:
        return self._lire_tous().get(cle)

    def enregistrer(self, decision: DecisionRevisionHumaine) -> None:
        """Persiste `decision` IMMEDIATEMENT (remplace toute decision anterieure de meme cle)."""
        decisions = self._lire_tous()
        decisions[decision.cle] = decision
        self._ecrire_tous(decisions.values())

    def _lire_tous(self) -> dict[CleCandidatRevision, DecisionRevisionHumaine]:
        if self._chemin.stat().st_size == 0:
            return {}
        decisions: dict[CleCandidatRevision, DecisionRevisionHumaine] = {}
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    decision = decision_depuis_dict(json.loads(ligne))
                    decisions[decision.cle] = decision
        return decisions

    def _ecrire_tous(
        self, decisions: Iterable[DecisionRevisionHumaine]
    ) -> None:
        with self._chemin.open("w", encoding="utf-8") as f:
            for decision in decisions:
                f.write(
                    json.dumps(decision_vers_dict(decision), ensure_ascii=False)
                    + "\n"
                )
