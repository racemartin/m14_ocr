"""
Point d'entree CLI — Etape 1, action "construire le dataset pivot".

Usage :
    uv run python interfaces/cli/construire_dataset_pivot.py \
        --source data/raw/mediqal.jsonl \
        --corpus mediqal \
        --sortie data/processed/dataset_pivot.jsonl
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import ConstruireDatasetPivotUseCase
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    LecteurCorpusFichierLocal,
)
from interfaces.cli.mappers_corpus import MAPPERS_PAR_CORPUS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Chemin du fichier corpus brut")
    parser.add_argument("--corpus", required=True, choices=sorted(MAPPERS_PAR_CORPUS.keys()))
    parser.add_argument("--sortie", required=True, help="Chemin du fichier pivot JSONL de sortie")
    arguments = parser.parse_args()

    lecteur    = LecteurCorpusFichierLocal(arguments.source)
    repository = JsonlDatasetRepository(arguments.sortie)
    mapper     = MAPPERS_PAR_CORPUS[arguments.corpus]

    cas_usage = ConstruireDatasetPivotUseCase(lecteur=lecteur, repository=repository)
    nombre_exemples = cas_usage.executer(mapper)

    print(f"{nombre_exemples} exemples pivot ecrits dans {arguments.sortie}")


if __name__ == "__main__":
    main()
