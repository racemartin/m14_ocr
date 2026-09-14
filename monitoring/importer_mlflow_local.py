"""
Importateur : reproduit les runs du depot dataset HF de metriques
d'entrainement (`hf_dataset_runs.REPO_ID_PAR_DEFAUT`, alimente par
`HfDatasetSuiviExperimentation`) dans un MLflow LOCAL (SQLite), pour
pouvoir ouvrir `mlflow ui` en local et parcourir l'historique complet
des runs SFT (et DPO, meme mecanisme generique, quand ce backend
existera) sans monter de serveur MLflow distant persistant.

Decision : pas de serveur MLflow distant
(aurait exige Postgres + Docker + authentification d'un Space prive,
trop d'infrastructure pour ce POC). A la place, ce script tourne SUR
LA MACHINE LOCALE de l'utilisateur : il telecharge chaque
`<nom_run>/metriques.jsonl` (et `parametres.json` si present) via
`hf_dataset_runs.py`, puis les reproduit contre
`MlflowSuiviExperimentation` (meme adaptateur, meme sequence
demarrer_run -> logger_metrique -> terminer_run, que
`training/E2_04_sft_train.py` utilise deja pour le backend
`suivi.backend: mlflow`), construit avec `uri_tracking="sqlite:///..."`.

Idempotence : un run deja present dans le MLflow local (nom de run
retrouve via `MlflowClient().search_runs` sur l'experience "Default",
celle que `MlflowSuiviExperimentation.demarrer_run` utilise
implicitement puisqu'il n'appelle jamais `mlflow.set_experiment`)
n'est pas reimporte, sauf `--forcer`. Pas de fichier de marques a
part : c'est MLflow lui-meme, deja la source de verite, qui repond a
"qu'est-ce qui est deja importe ?".

Vit dans `monitoring/` (pas `interfaces/cli/` ni `training/`) pour la
meme raison que `app_suivi_entrainement.py` (cf. son docstring et
AGENTS.md) : visualisation/outillage passif autour du suivi
d'entrainement, jamais un pas execute par le job d'entrainement
lui-meme ni une CLI de la sequence de cas d'usage E1/E2.

Usage :
    uv run python monitoring/importer_mlflow_local.py
    uv run python monitoring/importer_mlflow_local.py --repo-id mombasstic/chsa-triage-sft-metrics --forcer
    uv run mlflow ui --backend-store-uri sqlite:///~/.chsa-triage/mlflow.db   # (developper le ~ au prealable, cf. README)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlflow.tracking import MlflowClient

from chsa_triage.infrastructure.adapters.mlflow_suivi_experimentation import (
    MlflowSuiviExperimentation,
)
from monitoring.hf_dataset_runs import (
    REPO_ID_PAR_DEFAUT,
    lister_runs,
    telecharger_parametres,
    telecharger_texte_metriques,
)
from monitoring.logica_suivi_entrainement import analyser_jsonl_metriques

CHEMIN_SQLITE_PAR_DEFAUT = "~/.chsa-triage/mlflow.db"

# `MlflowSuiviExperimentation.demarrer_run` ne fixe jamais d'experience
# explicitement (`mlflow.start_run` sans `experiment_id`/`experiment_name`) :
# tous les runs qu'il cree atterrissent dans l'experience "Default".
NOM_EXPERIENCE_MLFLOW_DEFAUT = "Default"


def runs_deja_importes(client: MlflowClient, nom_experience: str = NOM_EXPERIENCE_MLFLOW_DEFAUT) -> set[str]:
    """Noms des runs deja presents dans le MLflow local (experience `nom_experience`)."""
    experience = client.get_experiment_by_name(nom_experience)
    if experience is None:
        return set()
    runs = client.search_runs(experiment_ids=[experience.experiment_id], max_results=50_000)
    return {run.info.run_name for run in runs}


def runs_a_importer(noms_disponibles: list[str], noms_deja_importes: set[str], forcer: bool) -> list[str]:
    """Runs distants a effectivement importer : tous si `forcer`, sinon ceux absents du MLflow local."""
    if forcer:
        return list(noms_disponibles)
    return [nom for nom in noms_disponibles if nom not in noms_deja_importes]


def reproduire_run(suivi: MlflowSuiviExperimentation, nom: str, texte_metriques: str, parametres: dict) -> int:
    """
    Rejoue un run telecharge (JSONL format LONG) contre `suivi` :
    demarrer_run -> une ligne de log par metrique -> terminer_run.
    Retourne le nombre de metriques rejouees.
    """
    lignes = analyser_jsonl_metriques(texte_metriques)
    suivi.demarrer_run(nom, parametres)
    for ligne in lignes:
        suivi.logger_metrique(ligne.nom, ligne.valeur, ligne.etape)
    suivi.terminer_run()
    return len(lignes)


def importer_runs(
    client: MlflowClient,
    suivi: MlflowSuiviExperimentation,
    repo_id: str,
    forcer: bool,
    obtenir_runs_disponibles=lister_runs,
    obtenir_texte_metriques=telecharger_texte_metriques,
    obtenir_parametres=telecharger_parametres,
) -> list[str]:
    """
    Orchestre l'import complet : decide quoi importer, telecharge,
    rejoue. Les trois `obtenir_*` sont injectables (meme principe que
    `fabrique_scheduler` de `HfDatasetSuiviExperimentation`) pour
    tester cette fonction sans reseau reel. Retourne les noms
    effectivement importes.
    """
    disponibles = obtenir_runs_disponibles(repo_id)
    deja_importes = runs_deja_importes(client)
    a_importer = runs_a_importer(disponibles, deja_importes, forcer)

    importes = []
    for nom in a_importer:
        texte_metriques = obtenir_texte_metriques(repo_id, nom)
        parametres = obtenir_parametres(repo_id, nom)
        nombre_metriques = reproduire_run(suivi, nom, texte_metriques, parametres)
        print(f"importe : {nom} ({nombre_metriques} metriques)")
        importes.append(nom)
    return importes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=REPO_ID_PAR_DEFAUT, help=f"Depot dataset HF de metriques (defaut {REPO_ID_PAR_DEFAUT})")
    parser.add_argument(
        "--base-sqlite",
        default=CHEMIN_SQLITE_PAR_DEFAUT,
        help=f"Chemin du fichier SQLite MLflow local (defaut {CHEMIN_SQLITE_PAR_DEFAUT})",
    )
    parser.add_argument("--forcer", action="store_true", help="Reimporte meme les runs deja presents dans le MLflow local")
    arguments = parser.parse_args()

    chemin_sqlite = Path(arguments.base_sqlite).expanduser().resolve()
    chemin_sqlite.parent.mkdir(parents=True, exist_ok=True)
    uri_tracking = f"sqlite:///{chemin_sqlite}"

    print(f"Depot dataset HF : {arguments.repo_id}")
    print(f"MLflow local : {uri_tracking}")

    client = MlflowClient(tracking_uri=uri_tracking)
    suivi = MlflowSuiviExperimentation(uri_tracking=uri_tracking)
    importes = importer_runs(client, suivi, arguments.repo_id, arguments.forcer)

    if not importes:
        print("Rien a importer : tous les runs distants sont deja dans le MLflow local (utiliser --forcer pour reimporter).")
    else:
        print(f"{len(importes)} run(s) importe(s) : {', '.join(importes)}")
    print(f"Ouvrir l'interface : uv run mlflow ui --backend-store-uri {uri_tracking}")


if __name__ == "__main__":
    main()
