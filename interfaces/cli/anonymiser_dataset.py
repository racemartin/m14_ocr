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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL")
    parser.add_argument("--strategie", default="replace", choices=["replace", "mask", "redact"])
    arguments = parser.parse_args()

    repository  = JsonlDatasetRepository(arguments.dataset)
    anonymiseur = PresidioAnonymiseur(strategie=arguments.strategie)

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur)
    nombre_traites = cas_usage.executer()

    print(f"{nombre_traites} exemples anonymises (strategie={arguments.strategie}).")


if __name__ == "__main__":
    main()
