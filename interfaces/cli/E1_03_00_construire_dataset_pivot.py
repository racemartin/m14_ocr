"""
STEP 02
Point d'entree CLI, Etape 1, action "construire le dataset pivot".

Usage :
    uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py \
        --source data/raw/mediqal.jsonl \
        --corpus mediqal \
        --sortie data/processed/dataset_pivot.jsonl

`--taille-bloc N` lit le fichier source par blocs de N lignes (memoire
bornee, meme correctif OOM que `E1_02_profiler_corpus.py --bloque`) ;
les exemples pivot mappes restent accumules en memoire jusqu'a
l'ecriture finale.

Grace a l'identifiant deterministe (`ExemplePivot.nouvel_identifiant`),
les vrais doublons (FrenchMedMCQA 1, MedQuAD 48, UltraMedical-Preference
12272) sont ecartes du pivot et archives, jamais silencieusement
perdus, dans --doublons.
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import ConstruireDatasetPivotUseCase
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    LecteurCorpusFichierLocal,
)
from chsa_triage.infrastructure.adapters.jsonl_dataset_repository import (
    ajouter_exemples_jsonl,
)
from interfaces.cli.E1_03_01_mappers_corpus import MAPPERS_PAR_CORPUS
from tools.rafael.log_tool import LogTool

log = LogTool(origin="construire_dataset_pivot")

CHEMIN_DOUBLONS_DEFAUT = "data/processed/doublons_supprimes.jsonl"


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True, help="Chemin du fichier corpus brut"
    )
    parser.add_argument(
        "--corpus", required=True, choices=sorted(MAPPERS_PAR_CORPUS.keys())
    )
    parser.add_argument(
        "--sortie",
        required=True,
        help="Chemin du fichier pivot JSONL de sortie",
    )
    parser.add_argument(
        "--taille-bloc",
        type=int,
        default=None,
        help="Lire le fichier source par blocs de N lignes (evite l'OOM sur un gros corpus, ex. ultramedical_preference.jsonl)",
    )
    parser.add_argument(
        "--doublons",
        default=CHEMIN_DOUBLONS_DEFAUT,
        help=f"Fichier d'archive des enregistrements bruts dedoublonnes, JSONL en mode ajout "
        f"(defaut {CHEMIN_DOUBLONS_DEFAUT})",
    )
    arguments = parser.parse_args()

    log.START_ACTION(
        "construire_dataset_pivot", "main", "mapping vers le schema pivot"
    )
    log.PARAMETER_VALUE("source", arguments.source)
    log.PARAMETER_VALUE("corpus", arguments.corpus)
    log.PARAMETER_VALUE("sortie", arguments.sortie)
    log.PARAMETER_VALUE(
        "taille-bloc", arguments.taille_bloc or "(desactive, lecture complete)"
    )

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    lecteur = LecteurCorpusFichierLocal(
        arguments.source, taille_bloc=arguments.taille_bloc
    )
    repository = JsonlDatasetRepository(arguments.sortie)
    mapper = MAPPERS_PAR_CORPUS[arguments.corpus]

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(
        1, "Mapping enregistrement -> ExemplePivot", f"mapper={mapper.__name__}"
    )
    try:
        cas_usage = ConstruireDatasetPivotUseCase(
            lecteur=lecteur, repository=repository
        )
        nombre_exemples = cas_usage.executer(mapper)
    except Exception as erreur:
        log.LEVEL_4_ERROR(
            "construire_dataset_pivot",
            f"echec du mapping pour {arguments.corpus} : {erreur}",
        )
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    if nombre_exemples == 0:
        log.LEVEL_5_WARNING(
            "construire_dataset_pivot",
            f"0 exemple pivot produit depuis {arguments.source} ; le mapper '{mapper.__name__}' "
            "n'a reconnu aucun enregistrement (schema incompatible ?), verifier le mapper avant de continuer",
        )

    if cas_usage.doublons:
        ajouter_exemples_jsonl(arguments.doublons, cas_usage.doublons)
        log.LEVEL_5_WARNING(
            "construire_dataset_pivot",
            f"{len(cas_usage.doublons)} enregistrement(s) strictement identique(s) ecarte(s) du pivot "
            f"(meme identifiant deterministe), archives dans {arguments.doublons}",
        )

    log.PARAMETER_VALUE("exemples pivot ecrits", nombre_exemples)
    log.PARAMETER_VALUE("doublons ecartes", len(cas_usage.doublons))
    log.FINISH_ACTION(
        "construire_dataset_pivot",
        "main",
        f"{nombre_exemples} exemples ecrits dans {arguments.sortie}",
    )

    print(f"{nombre_exemples} exemples pivot ecrits dans {arguments.sortie}")
    if cas_usage.doublons:
        print(
            f"{len(cas_usage.doublons)} doublons ecartes, archives dans {arguments.doublons}"
        )


if __name__ == "__main__":
    main()
