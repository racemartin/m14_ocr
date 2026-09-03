"""
Point d'entree CLI — Etape 1, action "decouper en splits".

Usage :
    uv run python interfaces/cli/decouper_splits.py \
        --dataset data/processed/dataset_pivot.jsonl
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import DecouperSplitsUseCase
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository
from tools.rafael.log_tool import LogTool

log = LogTool(origin="decouper_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL")
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--proportion-val", type=float, default=0.10)
    parser.add_argument("--proportion-test", type=float, default=0.10)
    arguments = parser.parse_args()

    log.START_ACTION("decouper_splits", "main", "decoupage train/val/test")
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("graine", arguments.graine)
    log.PARAMETER_VALUE("proportion-val", arguments.proportion_val)
    log.PARAMETER_VALUE("proportion-test", arguments.proportion_test)

    repository = JsonlDatasetRepository(arguments.dataset)

    log.STEP(1, "Decoupage aleatoire des splits")
    try:
        cas_usage = DecouperSplitsUseCase(
            repository=repository,
            graine_aleatoire=arguments.graine,
            proportion_val=arguments.proportion_val,
            proportion_test=arguments.proportion_test,
        )
        decompte = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR("decouper_splits", f"echec du decoupage de {arguments.dataset} : {erreur}")
        raise

    for split, nombre in decompte.items():
        log.PARAMETER_VALUE(f"split {split}", nombre)
    log.FINISH_ACTION("decouper_splits", "main", f"splits ecrits pour {arguments.dataset}")

    print("Repartition des splits :")
    for split, nombre in decompte.items():
        print(f"  {split:<10}: {nombre}")


if __name__ == "__main__":
    main()
