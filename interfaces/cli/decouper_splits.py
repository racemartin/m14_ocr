"""
STEP 04
Point d'entree CLI, Etape 1, action "decouper en splits".

Usage :
    uv run python interfaces/cli/decouper_splits.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --n 5000

IMPORTANT (08/09/2026, design source/sortie separes) : --dataset doit
pointer vers le fichier ANONYMISE (`dataset_pivot_anonymise.jsonl`,
sortie de `anonymiser_dataset.py`), PAS vers le pivot original
`dataset_pivot.jsonl` ; celui-ci n'est jamais anonymise en place et
ne contient donc jamais d'exemple avec `anonymise=True`. Ce script
lit ET ecrit sur le meme fichier (le champ `split` est ajoute en
place sur le fichier anonymise).

CROISSANCE STABLE, JAMAIS DE REORDONNANCEMENT (10/09/2026,
CHANGEMENT DE COMPORTEMENT reel par rapport a avant).
Un exemple qui a deja un `split` (execution anterieure) n'est JAMAIS
reassigne, quel que soit le `--n` demande ensuite : agrandir le jeu de
donnees en relancant avec un `--n` plus grand (ou sans `--n`) ne fait
QUE completer ce qui manque, il ne recalcule plus jamais le decoupage
en entier. Avant ce changement, relancer avec un `--n` different (ou
sans `--n`) pouvait deplacer un exemple deja vu de `train` vers `test`
(ou l'inverse) ; une fuite silencieuse d'exemples d'entrainement dans
le jeu de test, ce que le cahier des charges interdit explicitement.
Exemple concret : `--n 5000` puis, plus tard, `--n 10000` ; les 5000
premiers exemples GARDENT exactement le split qui leur a ete assigne
la premiere fois ; seuls 5000 exemples NOUVEAUX (jamais vus) recoivent
un split lors de la seconde execution.

Option --n (taille CIBLE cumulee, pas taille de cette seule execution)
: `--n N` preleve `N - (nombre deja assigne)` nouveaux exemples,
echantillon stratifie (type_exemple, source), methode du plus grand
reste, cf. `chsa_triage.application.echantillonnage.echantillon_stratifie`,
parmi les exemples `anonymise=True` qui n'ont PAS encore de split,
puis leur assigne train/val/test. Si `N` est <= au nombre d'exemples
deja assignes, il n'y a rien de nouveau a faire : REDUIRE un decoupage
deja fait n'est PAS supporte (le jeu ne peut que grandir), un
avertissement est trace via LogTool (pas une erreur). Si `--n` est
omis, tous les exemples anonymises qui n'ont pas encore de split
recoivent un split (mode "completer ce qui manque").
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
        help="Taille CIBLE cumulee (deja assignes + nouveaux) ; seuls les exemples sans split "
             "encore sont candidats aux N - (deja assignes) nouveaux prelevements (defaut : "
             "completer, tout exemple anonymise sans split recoit un split). N <= au nombre "
             "deja assigne : rien de nouveau, non supporte de reduire un decoupage existant",
    )
    arguments = parser.parse_args()

    log.START_ACTION("decouper_splits", "main", "decoupage train/val/test (croissance stable)")
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("graine", arguments.graine)
    log.PARAMETER_VALUE("proportion-val", arguments.proportion_val)
    log.PARAMETER_VALUE("proportion-test", arguments.proportion_test)
    log.PARAMETER_VALUE("n (taille cible cumulee)", arguments.n if arguments.n is not None else "(aucun, completer tout)")

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository = JsonlDatasetRepository(arguments.dataset)

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Decoupage des nouveaux exemples (ceux deja assignes ne sont jamais touches)")
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
    log.PARAMETER_VALUE("deja assignes (executions anterieures, non touches)", cas_usage.nombre_deja_assignes)
    log.PARAMETER_VALUE("nouveaux repartis cette execution", cas_usage.nombre_nouveaux)
    if arguments.n is not None and cas_usage.nombre_nouveaux == 0:
        log.LEVEL_5_WARNING(
            "decouper_splits",
            f"--n {arguments.n} <= {cas_usage.nombre_deja_assignes} exemples deja assignes ; rien de "
            "nouveau a repartir (reduire un decoupage deja fait n'est pas supporte, aucune erreur)",
        )
    for split, nombre in decompte.items():
        log.PARAMETER_VALUE(f"split {split} (total cumule)", nombre)
    log.FINISH_ACTION("decouper_splits", "main", f"splits ecrits pour {arguments.dataset}")

    print(f"Nouveaux exemples repartis lors de cette execution : {cas_usage.nombre_nouveaux}")
    print(f"Exemples deja assignes (executions anterieures, non touches) : {cas_usage.nombre_deja_assignes}")
    print("Repartition totale actuelle des splits (deja assignes + nouveaux) :")
    for split, nombre in decompte.items():
        print(f"  {split:<10}: {nombre}")


if __name__ == "__main__":
    main()
