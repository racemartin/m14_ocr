"""
Point d'entree CLI — Etape 1, action "anonymiser".

Usage :
    uv run python interfaces/cli/anonymiser_dataset.py \
        --dataset data/processed/dataset_pivot.jsonl \
        --strategie replace
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import AnonymiserDatasetUseCase
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    PresidioAnonymiseur,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="anonymiser_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL")
    parser.add_argument("--strategie", default="replace", choices=["replace", "mask", "redact"])
    arguments = parser.parse_args()

    log.START_ACTION("anonymiser_dataset", "main", "anonymisation Presidio du dataset pivot")
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("strategie", arguments.strategie)

    repository  = JsonlDatasetRepository(arguments.dataset)
    anonymiseur = PresidioAnonymiseur(strategie=arguments.strategie)

    log.STEP(1, "Anonymisation Presidio", "peut prendre du temps selon le nombre d'exemples")
    try:
        cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur)
        nombre_traites = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR("anonymiser_dataset", f"echec de l'anonymisation de {arguments.dataset} : {erreur}")
        raise

    log.PARAMETER_VALUE("exemples anonymises", nombre_traites)
    log.FINISH_ACTION("anonymiser_dataset", "main", f"{nombre_traites} exemples anonymises (strategie={arguments.strategie})")

    print(f"{nombre_traites} exemples anonymises (strategie={arguments.strategie}).")


if __name__ == "__main__":
    main()
