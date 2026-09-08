"""
STEP 04
Point d'entree CLI — Etape 1, action "decouper en splits".

Usage :
    uv run python interfaces/cli/decouper_splits.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --n 5000

IMPORTANT (08/09/2026, design source/sortie separes) : --dataset doit
pointer vers le fichier ANONYMISE (`dataset_pivot_anonymise.jsonl`,
sortie de `anonymiser_dataset.py`), PAS vers le pivot original
`dataset_pivot.jsonl` -- celui-ci n'est jamais anonymise en place et
ne contient donc jamais d'exemple avec `anonymise=True`. Ce script
lit ET ecrit sur le meme fichier (le champ `split` est ajoute en
place sur le fichier anonymise).

Option --n (sous-echantillonnage avant repartition, meme logique
produit que `--limite` sur `anonymiser_dataset.py`) : pour obtenir un
dataset d'entrainement de taille N plutot que repartir TOUT ce qui a
deja ete anonymise, `--n N` preleve d'abord un echantillon stratifie
(type_exemple, source) de taille N parmi les exemples `anonymise=True`
disponibles (methode du plus grand reste, cf.
`chsa_triage.application.echantillonnage.echantillon_stratifie`), puis
repartit train/val/test sur ce sous-ensemble. Si N est omis, ou si N
est superieur ou egal au nombre d'exemples anonymises disponibles, le
comportement est inchange : tout ce qui est anonymise est reparti (un
avertissement est alors trace via LogTool, comme pour les autres
scripts Etape 1).
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import DecouperSplitsUseCase
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository
from tools.rafael.log_tool import LogTool

log = LogTool(origin="decouper_splits")


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL")
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--proportion-val", type=float, default=0.10)
    parser.add_argument("--proportion-test", type=float, default=0.10)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Taille de l'echantillon stratifie (type_exemple, source) a repartir parmi les exemples "
             "deja anonymises (defaut : tout repartir) ; si N depasse le nombre disponible, tout est "
             "reparti et un avertissement est trace",
    )
    arguments = parser.parse_args()

    log.START_ACTION("decouper_splits", "main", "decoupage train/val/test")
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("graine", arguments.graine)
    log.PARAMETER_VALUE("proportion-val", arguments.proportion_val)
    log.PARAMETER_VALUE("proportion-test", arguments.proportion_test)
    log.PARAMETER_VALUE("n", arguments.n if arguments.n is not None else "(aucun -- tout repartir)")

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS
    # -------------------------------------------------------------------------
    repository = JsonlDatasetRepository(arguments.dataset)

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    if arguments.n is not None:
        disponible = repository.compter(filtre={"anonymise": True})
        if arguments.n >= disponible:
            log.LEVEL_5_WARNING(
                "decouper_splits",
                f"--n {arguments.n} demande mais seulement {disponible} exemples anonymises disponibles -- "
                "tous les exemples anonymises seront repartis (aucune erreur, --n ignore pour cette execution)",
            )

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Decoupage aleatoire des splits")
    try:
        cas_usage = DecouperSplitsUseCase(
            repository=repository,
            graine_aleatoire=arguments.graine,
            proportion_val=arguments.proportion_val,
            proportion_test=arguments.proportion_test,
            n=arguments.n,
        )
        decompte = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR("decouper_splits", f"echec du decoupage de {arguments.dataset} : {erreur}")
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    for split, nombre in decompte.items():
        log.PARAMETER_VALUE(f"split {split}", nombre)
    log.FINISH_ACTION("decouper_splits", "main", f"splits ecrits pour {arguments.dataset}")

    print("Repartition des splits :")
    for split, nombre in decompte.items():
        print(f"  {split:<10}: {nombre}")


if __name__ == "__main__":
    main()
