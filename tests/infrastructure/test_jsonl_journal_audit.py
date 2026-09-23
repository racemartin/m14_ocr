"""Tests de JsonlJournalAudit : append-only, jamais de reecriture/troncature."""

from __future__ import annotations

import json

from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.infrastructure.adapters.jsonl_journal_audit import JsonlJournalAudit


def _entree(conversation_id: str = "conv-1", sortie: str = "reponse") -> EntreeAudit:
    return EntreeAudit(
        horodatage="2026-09-23T10:00:00+00:00",
        type_evenement="tour_entretien",
        conversation_id=conversation_id,
        entree="Bonjour, j'ai mal a la tete.",
        sortie=sortie,
        version_modele="mombasstic/chsa-triage-dpo-lora",
    )


def test_consigner_ecrit_une_ligne_jsonl(tmp_path):
    chemin = tmp_path / "audit.jsonl"
    journal = JsonlJournalAudit(chemin)

    journal.consigner(_entree())

    lignes = chemin.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 1
    donnees = json.loads(lignes[0])
    assert donnees["conversation_id"] == "conv-1"
    assert donnees["type_evenement"] == "tour_entretien"
    assert donnees["version_modele"] == "mombasstic/chsa-triage-dpo-lora"


def test_consigner_ajoute_sans_ecraser_les_entrees_precedentes(tmp_path):
    chemin = tmp_path / "audit.jsonl"
    journal = JsonlJournalAudit(chemin)

    journal.consigner(_entree(sortie="premiere reponse"))
    journal.consigner(_entree(sortie="deuxieme reponse"))

    lignes = chemin.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 2
    assert json.loads(lignes[0])["sortie"] == "premiere reponse"
    assert json.loads(lignes[1])["sortie"] == "deuxieme reponse"


def test_consigner_cree_le_dossier_parent_si_absent(tmp_path):
    chemin = tmp_path / "sous_dossier" / "audit.jsonl"
    journal = JsonlJournalAudit(chemin)

    journal.consigner(_entree())

    assert chemin.exists()
