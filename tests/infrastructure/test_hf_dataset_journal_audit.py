"""
Test d'integration reel de HfDatasetJournalAudit ; pas un mock sur le
fichier JSONL local, mais un double de test pour le scheduler
(`fabrique_scheduler`) : Environnement A n'a pas de reseau reel, et le
vrai `huggingface_hub.CommitScheduler` appelle `HfApi.create_repo` des
sa construction. Meme patron que `test_hf_dataset_suivi_experimentation.py`.
"""

from __future__ import annotations

import json

from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.infrastructure.adapters.hf_dataset_journal_audit import (
    HfDatasetJournalAudit,
)


class _SchedulerFactice:
    """Double de test : compte les appels au lieu de parler au Hub."""

    def __init__(self) -> None:
        self.nombre_push_to_hub = 0

    def push_to_hub(self) -> None:
        self.nombre_push_to_hub += 1


def _fabrique_scheduler_factice_espionne(appels: list[tuple[str, str, float]]):
    scheduler = _SchedulerFactice()

    def fabrique(repo_id: str, dossier_local: str, intervalle_minutes: float):
        appels.append((repo_id, dossier_local, intervalle_minutes))
        return scheduler

    return fabrique, scheduler


def _entree(conversation_id: str = "conv-1", sortie: str = "reponse") -> EntreeAudit:
    return EntreeAudit(
        horodatage="2026-09-23T10:00:00+00:00",
        type_evenement="tour_entretien",
        conversation_id=conversation_id,
        entree="Bonjour, j'ai mal a la tete.",
        sortie=sortie,
        version_modele="mombasstic/chsa-triage-dpo-lora",
    )


def _lire_jsonl(chemin) -> list[dict]:
    with chemin.open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def test_consigner_ecrit_une_ligne_jsonl_dans_le_dossier_surveille(tmp_path):
    appels: list[tuple[str, str, float]] = []
    fabrique, scheduler = _fabrique_scheduler_factice_espionne(appels)
    journal = HfDatasetJournalAudit(
        repo_id="mombasstic/chsa-triage-audit-journal",
        repertoire_local=str(tmp_path),
        fabrique_scheduler=fabrique,
    )

    journal.consigner(_entree())

    lignes = _lire_jsonl(tmp_path / "journal_audit.jsonl")
    assert len(lignes) == 1
    assert lignes[0]["conversation_id"] == "conv-1"
    assert lignes[0]["type_evenement"] == "tour_entretien"
    assert lignes[0]["version_modele"] == "mombasstic/chsa-triage-dpo-lora"

    # Un seul scheduler cree, paresseusement, au premier `consigner()`.
    assert appels == [("mombasstic/chsa-triage-audit-journal", str(tmp_path), 1.0)]
    # `consigner()` n'attend jamais de synchronisation Hub : le
    # CommitScheduler pousse en arriere-plan selon `intervalle_minutes`.
    assert scheduler.nombre_push_to_hub == 0


def test_consigner_ajoute_sans_ecraser_les_entrees_precedentes(tmp_path):
    appels: list[tuple[str, str, float]] = []
    fabrique, _ = _fabrique_scheduler_factice_espionne(appels)
    journal = HfDatasetJournalAudit(
        repo_id="mombasstic/chsa-triage-audit-journal",
        repertoire_local=str(tmp_path),
        fabrique_scheduler=fabrique,
    )

    journal.consigner(_entree(sortie="premiere reponse"))
    journal.consigner(_entree(sortie="deuxieme reponse"))

    lignes = _lire_jsonl(tmp_path / "journal_audit.jsonl")
    assert len(lignes) == 2
    assert lignes[0]["sortie"] == "premiere reponse"
    assert lignes[1]["sortie"] == "deuxieme reponse"

    # Un seul scheduler cree pour tous les appels (pas un par entree).
    assert len(appels) == 1


def test_consigner_cree_le_dossier_local_si_absent(tmp_path):
    appels: list[tuple[str, str, float]] = []
    fabrique, _ = _fabrique_scheduler_factice_espionne(appels)
    dossier = tmp_path / "sous_dossier_absent"
    journal = HfDatasetJournalAudit(
        repo_id="mombasstic/chsa-triage-audit-journal",
        repertoire_local=str(dossier),
        fabrique_scheduler=fabrique,
    )

    journal.consigner(_entree())

    assert (dossier / "journal_audit.jsonl").exists()
