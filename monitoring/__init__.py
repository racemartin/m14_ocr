"""
Paquet du suivi d'entrainement : dashboard Streamlit EN VIVO
(`app_suivi_entrainement.py`) et importateur MLflow local
(`importer_mlflow_local.py`), tous deux lecteurs passifs du meme
depot dataset HF (`hf_dataset_runs.py`, frontiere reseau partagee).

Vit hors `training/` (aucun de ces fichiers n'est un pas execute par
le job d'entrainement, ce sont des outils passifs deployes/executes a
part) et hors `interfaces/cli/` (pas des CLI de la sequence de cas
d'usage E1/E2) : meme critere que
`domain/`/`ports/`/`infrastructure/adapters/`, d'ou l'absence de
prefixe `E1_`/`E2_` sur les fichiers de ce paquet.
"""

from __future__ import annotations
