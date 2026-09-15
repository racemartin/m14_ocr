"""
Test d'integration reel de MlflowSuiviExperimentation ; pas un mock.

Pointe `mlflow.set_tracking_uri` vers une base sqlite dans un
repertoire temporaire (`tmp_path`) et confirme que les metriques
logguees sont bien relisibles ensuite via `MlflowClient`.

Se saute automatiquement si `mlflow` n'est pas installe, meme
principe que `tests/infrastructure/test_presidio_anonymiseur.py`.
"""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

from chsa_triage.infrastructure.adapters.mlflow_suivi_experimentation import (
    MlflowSuiviExperimentation,
)


def test_run_demarre_metriques_logguees_relisibles_apres_coup(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)

    suivi.demarrer_run("run-test", {"rang_lora": 8, "taux_apprentissage": 0.0002})
    suivi.logger_metrique("perte", 1.2, 0)
    suivi.logger_metrique("perte", 0.8, 1)
    suivi.terminer_run()

    client = mlflow.MlflowClient(tracking_uri=uri)
    experience = client.get_experiment_by_name("Default")
    runs = client.search_runs(experiment_ids=[experience.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["rang_lora"] == "8"

    historique = client.get_metric_history(run.info.run_id, "perte")
    valeurs = sorted((m.step, m.value) for m in historique)
    assert valeurs == [(0, 1.2), (1, 0.8)]


def test_horodatage_explicite_est_respecte(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)

    suivi.demarrer_run("run-test", {})
    suivi.logger_metrique("perte", 1.2, 0, horodatage=1_700_000_000.5)
    suivi.terminer_run()

    client = mlflow.MlflowClient(tracking_uri=uri)
    experience = client.get_experiment_by_name("Default")
    run = client.search_runs(experiment_ids=[experience.experiment_id])[0]

    historique = client.get_metric_history(run.info.run_id, "perte")
    assert historique[0].timestamp == int(1_700_000_000.5 * 1000)
