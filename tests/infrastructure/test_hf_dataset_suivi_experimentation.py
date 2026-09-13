"""
Test d'integration reel de HfDatasetSuiviExperimentation ; pas un mock
sur le fichier JSONL local, mais un double de test pour le scheduler
(`fabrique_scheduler`) : Environnement A n'a pas de reseau reel, et le
vrai `huggingface_hub.CommitScheduler` appelle `HfApi.create_repo` des
sa construction.
"""

from __future__ import annotations

import json

from chsa_triage.infrastructure.adapters.hf_dataset_suivi_experimentation import (
    HfDatasetSuiviExperimentation,
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


def _lire_jsonl(chemin) -> list[dict]:
    with chemin.open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def test_run_demarre_metriques_ecrites_en_jsonl_format_long(tmp_path):
    appels: list[tuple[str, str, float]] = []
    fabrique, scheduler = _fabrique_scheduler_factice_espionne(appels)
    suivi = HfDatasetSuiviExperimentation(
        repo_id="mombasstic/chsa-triage-sft-metrics",
        repertoire_local=str(tmp_path),
        fabrique_scheduler=fabrique,
    )

    suivi.demarrer_run("run-test", {"rang_lora": 8})
    suivi.logger_metrique("perte_train", 1.2, 0)
    suivi.logger_metrique("perte_validation", 1.5, 0)
    suivi.logger_metrique("perte_train", 0.8, 1)
    suivi.terminer_run()

    chemin_jsonl = tmp_path / "run-test" / "metriques.jsonl"
    lignes = _lire_jsonl(chemin_jsonl)

    assert len(lignes) == 3
    assert lignes[0] == {"etape": 0, "nom": "perte_train", "valeur": 1.2, "horodatage": lignes[0]["horodatage"]}
    assert lignes[1]["nom"] == "perte_validation"
    assert lignes[1]["valeur"] == 1.5
    assert lignes[2]["etape"] == 1

    parametres = json.loads((tmp_path / "run-test" / "parametres.json").read_text(encoding="utf-8"))
    assert parametres == {"rang_lora": 8}

    # Un seul scheduler cree pour tout le run (pas un par appel).
    assert appels == [("mombasstic/chsa-triage-sft-metrics", str(tmp_path), 1.0)]
    # `terminer_run` force une synchronisation finale, sans attendre le
    # prochain tick en arriere-plan du scheduler.
    assert scheduler.nombre_push_to_hub == 1


def test_plusieurs_runs_successifs_reutilisent_le_meme_scheduler(tmp_path):
    appels: list[tuple[str, str, float]] = []
    fabrique, scheduler = _fabrique_scheduler_factice_espionne(appels)
    suivi = HfDatasetSuiviExperimentation(
        repo_id="mombasstic/chsa-triage-sft-metrics",
        repertoire_local=str(tmp_path),
        fabrique_scheduler=fabrique,
    )

    suivi.demarrer_run("essai-1", {})
    suivi.logger_metrique("perte_train", 1.0, 0)
    suivi.terminer_run()

    suivi.demarrer_run("essai-2", {})
    suivi.logger_metrique("perte_train", 0.9, 0)
    suivi.terminer_run()

    assert len(appels) == 1  # un seul scheduler cree, reutilise pour essai-2
    assert scheduler.nombre_push_to_hub == 2  # une synchronisation finale par run
    assert (tmp_path / "essai-1" / "metriques.jsonl").exists()
    assert (tmp_path / "essai-2" / "metriques.jsonl").exists()


def test_nouveau_run_meme_nom_repart_d_un_fichier_jsonl_vide(tmp_path):
    appels: list[tuple[str, str, float]] = []
    fabrique, _ = _fabrique_scheduler_factice_espionne(appels)
    suivi = HfDatasetSuiviExperimentation(
        repo_id="mombasstic/chsa-triage-sft-metrics",
        repertoire_local=str(tmp_path),
        fabrique_scheduler=fabrique,
    )

    suivi.demarrer_run("run-test", {})
    suivi.logger_metrique("perte_train", 1.2, 0)
    suivi.terminer_run()

    suivi.demarrer_run("run-test", {})  # relance sous le meme nom : fichier remis a zero
    suivi.logger_metrique("perte_train", 0.5, 0)
    suivi.terminer_run()

    lignes = _lire_jsonl(tmp_path / "run-test" / "metriques.jsonl")
    assert len(lignes) == 1
    assert lignes[0]["valeur"] == 0.5
