"""
STEP 02
Point d'entree CLI, Etape 1, action "construire le dataset pivot".

Usage :
    uv run python interfaces/cli/construire_dataset_pivot.py \
        --source data/raw/mediqal.jsonl \
        --corpus mediqal \
        --sortie data/processed/dataset_pivot.jsonl

Option --taille-bloc (07/09/2026, ajoutee suite a un OOM reel sur
ultramedical_preference.jsonl, 966 Mo, 5.8 Go de RAM disponibles) :
sans lecture par blocs, `LecteurCorpusFichierLocal` charge tout le
fichier source en DataFrame pandas d'un coup avant de le convertir en
dicts, meme probleme deja documente et corrige cote
`profiler_corpus.py --bloque`. Avec --taille-bloc N, la lecture du
fichier source se fait par blocs de N lignes (pandas chunksize) :
chaque bloc est converti puis mappe avant que le suivant soit charge,
la memoire de pointe reste bornee par la taille du bloc. Les exemples
pivot mappes restent, eux, accumules en memoire jusqu'a l'ecriture
finale (comportement inchange de `ConstruireDatasetPivotUseCase`) ;
--taille-bloc borne la lecture du fichier brut, pas la taille du
dataset pivot en sortie.

Dedoublonnage reel (08/09/2026, identifiant deterministe) : depuis que
`ExemplePivot.nouvel_identifiant` est
deterministe, deux enregistrements bruts strictement identiques sur
les champs qui alimentent le pivot produisent desormais le MEME
identifiant ; de vrais doublons trouves dans les donnees reelles
(FrenchMedMCQA 1, MedQuAD 48, UltraMedical-Preference 12272, cf.
`docs/02_etape1_donnees/00_couverture_exigences_officielles.md`).
`ConstruireDatasetPivotUseCase` n'en garde qu'un seul dans le pivot ;
les doublons ecartes sont archives (jamais silencieusement perdus)
dans --doublons (defaut `data/processed/doublons_supprimes.jsonl`,
append), un fichier partage entre les 6 executions de ce script.
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import ConstruireDatasetPivotUseCase
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    LecteurCorpusFichierLocal,
)
from chsa_triage.infrastructure.adapters.jsonl_dataset_repository import ajouter_exemples_jsonl
from interfaces.cli.mappers_corpus import MAPPERS_PAR_CORPUS
from tools.rafael.log_tool import LogTool

log = LogTool(origin="construire_dataset_pivot")

CHEMIN_DOUBLONS_DEFAUT = "data/processed/doublons_supprimes.jsonl"


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Chemin du fichier corpus brut")
    parser.add_argument("--corpus", required=True, choices=sorted(MAPPERS_PAR_CORPUS.keys()))
    parser.add_argument("--sortie", required=True, help="Chemin du fichier pivot JSONL de sortie")
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

    log.START_ACTION("construire_dataset_pivot", "main", "mapping vers le schema pivot")
    log.PARAMETER_VALUE("source", arguments.source)
    log.PARAMETER_VALUE("corpus", arguments.corpus)
    log.PARAMETER_VALUE("sortie", arguments.sortie)
    log.PARAMETER_VALUE("taille-bloc", arguments.taille_bloc or "(desactive, lecture complete)")

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    lecteur    = LecteurCorpusFichierLocal(arguments.source, taille_bloc=arguments.taille_bloc)
    repository = JsonlDatasetRepository(arguments.sortie)
    mapper     = MAPPERS_PAR_CORPUS[arguments.corpus]

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Mapping enregistrement -> ExemplePivot", f"mapper={mapper.__name__}")
    try:
        cas_usage = ConstruireDatasetPivotUseCase(lecteur=lecteur, repository=repository)
        nombre_exemples = cas_usage.executer(mapper)
    except Exception as erreur:
        log.LEVEL_4_ERROR("construire_dataset_pivot", f"echec du mapping pour {arguments.corpus} : {erreur}")
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
    log.FINISH_ACTION("construire_dataset_pivot", "main", f"{nombre_exemples} exemples ecrits dans {arguments.sortie}")

    print(f"{nombre_exemples} exemples pivot ecrits dans {arguments.sortie}")
    if cas_usage.doublons:
        print(f"{len(cas_usage.doublons)} doublons ecartes, archives dans {arguments.doublons}")


if __name__ == "__main__":
    main()
