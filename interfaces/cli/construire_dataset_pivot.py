"""
Point d'entree CLI — Etape 1, action "construire le dataset pivot".

Usage :
    uv run python interfaces/cli/construire_dataset_pivot.py \
        --source data/raw/mediqal.jsonl \
        --corpus mediqal \
        --sortie data/processed/dataset_pivot.jsonl

Option --taille-bloc (07/09/2026, ajoutee suite a un OOM reel sur
ultramedical_preference.jsonl, 966 Mo, 5.8 Go de RAM disponibles) :
sans lecture par blocs, `LecteurCorpusFichierLocal` charge tout le
fichier source en DataFrame pandas d'un coup avant de le convertir en
dicts -- meme probleme deja documente et corrige cote
`profiler_corpus.py --bloque`. Avec --taille-bloc N, la lecture du
fichier source se fait par blocs de N lignes (pandas chunksize) :
chaque bloc est converti puis mappe avant que le suivant soit charge,
la memoire de pointe reste bornee par la taille du bloc. Les exemples
pivot mappes restent, eux, accumules en memoire jusqu'a l'ecriture
finale (comportement inchange de `ConstruireDatasetPivotUseCase`) --
--taille-bloc borne la lecture du fichier brut, pas la taille du
dataset pivot en sortie.
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
    parser.add_argument(
        "--taille-bloc",
        type=int,
        default=None,
        help="Lire le fichier source par blocs de N lignes (evite l'OOM sur un gros corpus, ex. ultramedical_preference.jsonl)",
    )
    arguments = parser.parse_args()

    log.START_ACTION("construire_dataset_pivot", "main", "mapping vers le schema pivot")
    log.PARAMETER_VALUE("source", arguments.source)
    log.PARAMETER_VALUE("corpus", arguments.corpus)
    log.PARAMETER_VALUE("sortie", arguments.sortie)
    log.PARAMETER_VALUE("taille-bloc", arguments.taille_bloc or "(desactive -- lecture complete)")

    lecteur    = LecteurCorpusFichierLocal(arguments.source, taille_bloc=arguments.taille_bloc)
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
