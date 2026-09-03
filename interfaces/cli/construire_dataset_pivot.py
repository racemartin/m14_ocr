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
from tools.rafael.log_tool import LogTool

log = LogTool(origin="construire_dataset_pivot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Chemin du fichier corpus brut")
    parser.add_argument("--corpus", required=True, choices=sorted(MAPPERS_PAR_CORPUS.keys()))
    parser.add_argument("--sortie", required=True, help="Chemin du fichier pivot JSONL de sortie")
    arguments = parser.parse_args()

    log.START_ACTION("construire_dataset_pivot", "main", "mapping vers le schema pivot")
    log.PARAMETER_VALUE("source", arguments.source)
    log.PARAMETER_VALUE("corpus", arguments.corpus)
    log.PARAMETER_VALUE("sortie", arguments.sortie)

    lecteur    = LecteurCorpusFichierLocal(arguments.source)
    repository = JsonlDatasetRepository(arguments.sortie)
    mapper     = MAPPERS_PAR_CORPUS[arguments.corpus]

    log.STEP(1, "Mapping enregistrement -> ExemplePivot", f"mapper={mapper.__name__}")
    try:
        cas_usage = ConstruireDatasetPivotUseCase(lecteur=lecteur, repository=repository)
        nombre_exemples = cas_usage.executer(mapper)
    except Exception as erreur:
        log.LEVEL_4_ERROR("construire_dataset_pivot", f"echec du mapping pour {arguments.corpus} : {erreur}")
        raise

    if nombre_exemples == 0:
        log.LEVEL_5_WARNING(
            "construire_dataset_pivot",
            f"0 exemple pivot produit depuis {arguments.source} -- le mapper '{mapper.__name__}' "
            "n'a reconnu aucun enregistrement (schema incompatible ?), verifier le mapper avant de continuer",
        )
    log.PARAMETER_VALUE("exemples pivot ecrits", nombre_exemples)
    log.FINISH_ACTION("construire_dataset_pivot", "main", f"{nombre_exemples} exemples ecrits dans {arguments.sortie}")

    print(f"{nombre_exemples} exemples pivot ecrits dans {arguments.sortie}")


if __name__ == "__main__":
    main()
