"""
STEP 03
Point d'entree CLI — Etape 1, action "anonymiser".

Usage :
    uv run python interfaces/cli/anonymiser_dataset.py \
        --dataset data/processed/dataset_pivot.jsonl \
        --strategie replace \
        --limite 5000

Option --limite (08/09/2026, decision produit suite a la mesure reelle
d'un temps d'anonymisation complet de ~19h sur les 147204 exemples du
dataset pivot -- cf. docs/02_etape1_donnees/00_couverture_exigences_officielles.md) :
plutot que de trancher une fois pour toutes entre "echantillon" et
"dataset complet", le champ `anonymise` deja present sur ExemplePivot
rend le processus nativement incremental et reprenable
(`AnonymiserDatasetUseCase` ne traite jamais un exemple deja
`anonymise=True`). `--limite N` (defaut 5000, l'objectif de la
mission) anonymise une SOUS-ECHANTILLON stratifie par
(type_exemple, source) de taille N parmi les exemples encore
`anonymise=False` -- le reste du dataset n'est pas touche et pourra
etre traite plus tard en relancant cette commande avec un N plus
grand (les exemples deja anonymises ne sont jamais retraites).
`--limite full` (ou toute valeur >= au nombre d'exemples restants)
traite tout ce qui reste en une seule fois.
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from chsa_triage.application.use_cases import AnonymiserDatasetUseCase
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    PresidioAnonymiseur,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="anonymiser_dataset")


def _parser_limite(valeur: str) -> int | None:
    """Parse --limite : un entier positif, ou "full"/"illimite" pour tout traiter."""
    if valeur.strip().lower() in ("full", "illimite", "none"):
        return None
    nombre = int(valeur)
    if nombre <= 0:
        raise argparse.ArgumentTypeError("--limite doit etre un entier positif, ou 'full'")
    return nombre


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS 
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",   required=True,     help="Chemin du fichier pivot JSONL")
    parser.add_argument("--strategie", default="replace", choices=["replace", "mask", "redact"])
    parser.add_argument(
        "--limite",
        type=_parser_limite,
        default=5000,
        help="Taille de l'echantillon stratifie (type_exemple, source) a anonymiser parmi les "
             "exemples restants (defaut 5000) ; 'full' pour tout traiter d'un coup",
    )
    arguments = parser.parse_args()

    log.START_ACTION("anonymiser_dataset", "main", "anonymisation Presidio du dataset pivot")
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("strategie", arguments.strategie)
    log.PARAMETER_VALUE("limite", arguments.limite if arguments.limite is not None else "full (aucune limite)")

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS 
    # -------------------------------------------------------------------------
    repository  = JsonlDatasetRepository(arguments.dataset)
    anonymiseur = PresidioAnonymiseur(strategie=arguments.strategie)

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE 
    # -------------------------------------------------------------------------
    log.STEP(1, "Anonymisation Presidio", "peut prendre du temps selon le nombre d'exemples")
    try:
        cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=arguments.limite)
        nombre_traites = cas_usage.executer(
            envelopper_iterable=lambda a_traiter: tqdm(a_traiter, desc="Anonymisation Presidio", total=len(a_traiter))
        )
    except Exception as erreur:
        log.LEVEL_4_ERROR("anonymiser_dataset", f"echec de l'anonymisation de {arguments.dataset} : {erreur}")
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    log.PARAMETER_VALUE("exemples anonymises", nombre_traites)
    log.FINISH_ACTION("anonymiser_dataset", "main", f"{nombre_traites} exemples anonymises (strategie={arguments.strategie})")

    print(f"{nombre_traites} exemples anonymises (strategie={arguments.strategie}, limite={arguments.limite or 'full'}).")


if __name__ == "__main__":
    main()
