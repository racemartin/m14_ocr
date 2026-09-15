"""
Tests de `monitoring/importer_mlflow_local.py`.

`runs_a_importer` est pure (aucune dependance). `runs_deja_importes`,
`reproduire_run` et `importer_runs` sont testes en integration REELLE
contre un MLflow local (`sqlite:///` dans `tmp_path`, Environnement A
valide, cf. `test_mlflow_suivi_experimentation.py`), mais avec la
source de telechargement HF INJECTEE (`obtenir_runs_disponibles`/
`obtenir_texte_metriques`/`obtenir_parametres`) : aucun reseau reel,
meme principe que `fabrique_scheduler` de
`HfDatasetSuiviExperimentation`.
"""

from __future__ import annotations

import json

import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient

from chsa_triage.infrastructure.adapters.mlflow_suivi_experimentation import (
    MlflowSuiviExperimentation,
)
from monitoring.importer_mlflow_local import (
    importer_runs,
    reproduire_run,
    runs_a_importer,
    runs_deja_importes,
)


def _uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


def _jsonl(*lignes: tuple[int, str, float], horodatage: float = 1.0) -> str:
    return "\n".join(
        json.dumps({"etape": etape, "nom": nom, "valeur": valeur, "horodatage": horodatage})
        for etape, nom, valeur in lignes
    )


# ---------------------------------------------------------------------------
# runs_a_importer (pure)
# ---------------------------------------------------------------------------


def test_runs_a_importer_exclut_les_runs_deja_presents():
    resultat = runs_a_importer(["essai-1", "essai-2"], {"essai-1"}, forcer=False)
    assert resultat == ["essai-2"]


def test_runs_a_importer_forcer_reimporte_tout():
    resultat = runs_a_importer(["essai-1", "essai-2"], {"essai-1"}, forcer=True)
    assert resultat == ["essai-1", "essai-2"]


def test_runs_a_importer_rien_de_nouveau():
    assert runs_a_importer(["essai-1"], {"essai-1"}, forcer=False) == []


# ---------------------------------------------------------------------------
# runs_deja_importes / reproduire_run (integration MLflow reelle, sqlite local)
# ---------------------------------------------------------------------------


def test_runs_deja_importes_vide_sur_mlflow_local_neuf(tmp_path):
    client = MlflowClient(tracking_uri=_uri(tmp_path))
    assert runs_deja_importes(client) == set()


def test_reproduire_run_ecrit_parametres_et_metriques_relisibles(tmp_path):
    uri = _uri(tmp_path)
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)

    nombre = reproduire_run(
        suivi,
        "essai-1",
        _jsonl((0, "perte_train", 1.2), (0, "norme_gradient", 0.9), (1, "perte_train", 0.8)),
        {"rang_lora": 8},
    )
    assert nombre == 3

    client = MlflowClient(tracking_uri=uri)
    assert runs_deja_importes(client) == {"essai-1"}

    experience = client.get_experiment_by_name("Default")
    run = client.search_runs(experiment_ids=[experience.experiment_id])[0]
    assert run.data.params["rang_lora"] == "8"
    historique = client.get_metric_history(run.info.run_id, "perte_train")
    assert sorted((m.step, m.value) for m in historique) == [(0, 1.2), (1, 0.8)]


def test_reproduire_run_reporte_l_horodatage_original(tmp_path):
    uri = _uri(tmp_path)
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)

    reproduire_run(
        suivi,
        "essai-1",
        _jsonl((0, "perte_train", 1.2), horodatage=1_700_000_000.5),
        {},
    )

    client = MlflowClient(tracking_uri=uri)
    experience = client.get_experiment_by_name("Default")
    run = client.search_runs(experiment_ids=[experience.experiment_id])[0]
    historique = client.get_metric_history(run.info.run_id, "perte_train")
    assert historique[0].timestamp == int(1_700_000_000.5 * 1000)


# ---------------------------------------------------------------------------
# importer_runs (orchestration complete, source HF injectee)
# ---------------------------------------------------------------------------


def _sources_factices(contenus: dict[str, tuple[str, dict]]):
    def obtenir_runs_disponibles(repo_id: str) -> list[str]:
        return sorted(contenus)

    def obtenir_texte_metriques(repo_id: str, nom: str) -> str:
        return contenus[nom][0]

    def obtenir_parametres(repo_id: str, nom: str) -> dict:
        return contenus[nom][1]

    return obtenir_runs_disponibles, obtenir_texte_metriques, obtenir_parametres


def test_importer_runs_importe_tout_au_premier_passage(tmp_path):
    uri = _uri(tmp_path)
    client = MlflowClient(tracking_uri=uri)
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)
    obtenir_runs, obtenir_texte, obtenir_params = _sources_factices(
        {
            "essai-1": (_jsonl((0, "perte_train", 1.0)), {"rang_lora": 8}),
            "essai-2": (_jsonl((0, "perte_train", 2.0)), {}),
        }
    )

    importes = importer_runs(
        client, suivi, "depot-factice", forcer=False,
        obtenir_runs_disponibles=obtenir_runs, obtenir_texte_metriques=obtenir_texte, obtenir_parametres=obtenir_params,
    )

    assert importes == ["essai-1", "essai-2"]
    assert runs_deja_importes(client) == {"essai-1", "essai-2"}


def test_importer_runs_est_idempotent_sans_forcer(tmp_path):
    uri = _uri(tmp_path)
    client = MlflowClient(tracking_uri=uri)
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)
    obtenir_runs, obtenir_texte, obtenir_params = _sources_factices(
        {"essai-1": (_jsonl((0, "perte_train", 1.0)), {})}
    )

    premier = importer_runs(
        client, suivi, "depot-factice", forcer=False,
        obtenir_runs_disponibles=obtenir_runs, obtenir_texte_metriques=obtenir_texte, obtenir_parametres=obtenir_params,
    )
    second = importer_runs(
        client, suivi, "depot-factice", forcer=False,
        obtenir_runs_disponibles=obtenir_runs, obtenir_texte_metriques=obtenir_texte, obtenir_parametres=obtenir_params,
    )

    assert premier == ["essai-1"]
    assert second == []  # deja importe, pas de doublon


def test_importer_runs_forcer_reimporte(tmp_path):
    uri = _uri(tmp_path)
    client = MlflowClient(tracking_uri=uri)
    suivi = MlflowSuiviExperimentation(uri_tracking=uri)
    obtenir_runs, obtenir_texte, obtenir_params = _sources_factices(
        {"essai-1": (_jsonl((0, "perte_train", 1.0)), {})}
    )

    importer_runs(
        client, suivi, "depot-factice", forcer=False,
        obtenir_runs_disponibles=obtenir_runs, obtenir_texte_metriques=obtenir_texte, obtenir_parametres=obtenir_params,
    )
    second = importer_runs(
        client, suivi, "depot-factice", forcer=True,
        obtenir_runs_disponibles=obtenir_runs, obtenir_texte_metriques=obtenir_texte, obtenir_parametres=obtenir_params,
    )

    assert second == ["essai-1"]
