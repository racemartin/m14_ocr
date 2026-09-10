"""Test de l'adaptateur JsonlDecisionsRevisionHumaine ; round-trip et correction sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.domain.model import (
    DECISION_ACCEPTE,
    DECISION_REJETE,
    SOURCE_CANDIDATS_PII,
    CleCandidatRevision,
    DecisionRevisionHumaine,
)
from chsa_triage.infrastructure.adapters.jsonl_decisions_revision_humaine import (
    JsonlDecisionsRevisionHumaine,
)


def _decision(identifiant: str = "id-1", decision: str = DECISION_ACCEPTE) -> DecisionRevisionHumaine:
    cle = CleCandidatRevision(SOURCE_CANDIDATS_PII, identifiant, "symptomes", "bigramme_capitalise", 10, 21)
    return DecisionRevisionHumaine(
        cle=cle, passage="...Jean Dupont...", decision=decision, horodatage="2026-09-09T10:00:00+00:00"
    )


def test_cles_decidees_sur_fichier_vide(tmp_path: Path):
    registre = JsonlDecisionsRevisionHumaine(tmp_path / "decisions.jsonl")
    assert registre.cles_decidees() == set()
    assert registre.toutes() == []


def test_enregistrer_persiste_et_se_relit(tmp_path: Path):
    chemin = tmp_path / "decisions.jsonl"
    registre = JsonlDecisionsRevisionHumaine(chemin)
    decision = _decision()

    registre.enregistrer(decision)

    assert registre.cles_decidees() == {decision.cle}
    relue = registre.trouver(decision.cle)
    assert relue is not None
    assert relue.decision == DECISION_ACCEPTE
    assert relue.passage == "...Jean Dupont..."


def test_enregistrer_deux_fois_la_meme_cle_remplace_la_decision(tmp_path: Path):
    """Mode --modify : corriger une decision anterieure sans dupliquer l'enregistrement."""
    chemin = tmp_path / "decisions.jsonl"
    registre = JsonlDecisionsRevisionHumaine(chemin)
    decision_initiale = _decision(decision=DECISION_ACCEPTE)

    registre.enregistrer(decision_initiale)
    registre.enregistrer(_decision(decision=DECISION_REJETE))

    assert len(registre.toutes()) == 1
    assert registre.trouver(decision_initiale.cle).decision == DECISION_REJETE


def test_deux_cles_differentes_restent_distinctes(tmp_path: Path):
    chemin = tmp_path / "decisions.jsonl"
    registre = JsonlDecisionsRevisionHumaine(chemin)

    registre.enregistrer(_decision("id-1", DECISION_ACCEPTE))
    registre.enregistrer(_decision("id-2", DECISION_REJETE))

    assert len(registre.toutes()) == 2
    cles = registre.cles_decidees()
    assert len(cles) == 2


def test_nouvelle_instance_relit_le_meme_fichier(tmp_path: Path):
    chemin = tmp_path / "decisions.jsonl"
    decision = _decision()
    JsonlDecisionsRevisionHumaine(chemin).enregistrer(decision)

    registre_relu = JsonlDecisionsRevisionHumaine(chemin)
    assert registre_relu.trouver(decision.cle) is not None


def test_trouver_cle_inexistante_retourne_none(tmp_path: Path):
    registre = JsonlDecisionsRevisionHumaine(tmp_path / "decisions.jsonl")
    cle_absente = CleCandidatRevision(SOURCE_CANDIDATS_PII, "id-x", "champ", "email", 0, 5)
    assert registre.trouver(cle_absente) is None
