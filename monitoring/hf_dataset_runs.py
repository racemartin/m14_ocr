"""
Frontiere reseau partagee vers le depot dataset HF de metriques
d'entrainement (`HfApi.list_repo_files`/`hf_hub_download`), reutilisee
par le dashboard Streamlit EN VIVO (`app_suivi_entrainement.py`) et
l'importateur MLflow local (`importer_mlflow_local.py`) : evite de
dupliquer les memes deux appels HF Hub dans les deux modules.

Aucune logique de parsing/pivot ici (`logica_suivi_entrainement.py`,
pure, testee sans reseau) : ce module se limite a lister les runs et
telecharger le contenu brut de leurs fichiers.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from monitoring.logica_suivi_entrainement import (
    NOM_FICHIER_METRIQUES,
    extraire_noms_runs,
)

REPO_ID_PAR_DEFAUT = "mombasstic/chsa-triage-sft-metrics"
NOM_FICHIER_PARAMETRES = "parametres.json"


def lister_runs(repo_id: str) -> list[str]:
    """Runs disponibles dans le depot dataset (sous-repertoires portant `metriques.jsonl`)."""
    chemins = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    return extraire_noms_runs(chemins)


def telecharger_texte_metriques(repo_id: str, nom_run: str) -> str:
    """Contenu brut (JSONL format LONG) de `<nom_run>/metriques.jsonl`."""
    chemin_local = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=f"{nom_run}/{NOM_FICHIER_METRIQUES}"
    )
    return Path(chemin_local).read_text(encoding="utf-8")


def telecharger_parametres(repo_id: str, nom_run: str) -> dict:
    """
    Contenu de `<nom_run>/parametres.json`, ou `{}` s'il n'existe pas
    (fichier optionnel, ecrit par `HfDatasetSuiviExperimentation.demarrer_run`
    seulement quand `parametres` est non vide).
    """
    try:
        chemin_local = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=f"{nom_run}/{NOM_FICHIER_PARAMETRES}"
        )
    except EntryNotFoundError:
        return {}
    return json.loads(Path(chemin_local).read_text(encoding="utf-8"))
