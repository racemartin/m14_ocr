"""
STEP 04.1
Point d'entree CLI — Etape 1, action "verifier la repartition des
splits".

`decouper_splits.py` n'affiche que le total global par split
(train/val/test). Ce script relit le dataset pivot deja reparti et
affiche, pour chaque strate `(type_exemple, source)`, le decompte ET
le pourcentage par split, pour verifier visuellement que
l'echantillonnage stratifie est bien reste representatif dans CHAQUE
split, pas seulement au global (ex. une petite source comme
FrenchMedMCQA ne doit pas se retrouver absente de train ou de test).

Usage :
    uv run python interfaces/cli/verifier_repartition_splits.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl

IMPORTANT (08/09/2026, design source/sortie separes) : --dataset doit
pointer vers le fichier ANONYMISE (`dataset_pivot_anonymise.jsonl`),
PAS vers le pivot original ; celui-ci n'est jamais anonymise/reparti
en place.
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import VerifierRepartitionSplitsUseCase
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository
from tools.rafael.log_tool import LogTool

log = LogTool(origin="verifier_repartition_splits")

ORDRE_SPLITS = ("train", "val", "test")


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL")
    arguments = parser.parse_args()

    log.START_ACTION("verifier_repartition_splits", "main", "verification de la repartition des splits par strate")
    log.PARAMETER_VALUE("dataset", arguments.dataset)

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository = JsonlDatasetRepository(arguments.dataset)

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Lecture des exemples deja anonymises et repartis")
    try:
        cas_usage = VerifierRepartitionSplitsUseCase(repository=repository)
        repartition = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR("verifier_repartition_splits", f"echec de la verification pour {arguments.dataset} : {erreur}")
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    if not repartition:
        log.LEVEL_5_WARNING(
            "verifier_repartition_splits",
            f"aucun exemple anonymise+reparti trouve dans {arguments.dataset} ; "
            "lancer anonymiser_dataset.py puis decouper_splits.py d'abord",
        )

    for cle, compteur_strate in sorted(repartition.items()):
        total_strate = sum(compteur_strate.values())
        log.PARAMETER_VALUE(f"strate {cle}", f"{total_strate} exemples")

    print("Repartition des splits par strate (type_exemple, source) :")
    print(f"{'Strate':<45} {'Total':>7}  " + "  ".join(f"{s:>14}" for s in ORDRE_SPLITS))
    for cle, compteur_strate in sorted(repartition.items()):
        total_strate = sum(compteur_strate.values())
        colonnes = []
        for split in ORDRE_SPLITS:
            n = compteur_strate.get(split, 0)
            pourcentage = (n / total_strate * 100) if total_strate else 0.0
            colonnes.append(f"{n:>6} ({pourcentage:4.1f}%)")
        nom_strate = f"{cle[0]}/{cle[1]}"
        print(f"{nom_strate:<45} {total_strate:>7}  " + "  ".join(colonnes))

    log.FINISH_ACTION("verifier_repartition_splits", "main", f"{len(repartition)} strates verifiees pour {arguments.dataset}")


if __name__ == "__main__":
    main()
