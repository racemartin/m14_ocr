"""
STEP 04.2
Point d'entree CLI, Etape 1, action "extraire le sous-ensemble SFT
destine a la publication" (§9 du README).

Filtre le pivot ANONYMISE (`--dataset`) sur les exemples deja repartis
en split (`split is not None`, cf. `E1_05_00_decouper_splits.py`), retire ceux
listes dans `--exclusions` (produit par
`E1_04_01_reviser_pii_residuelle.py exporter`, cf. §6 et §9 du README :
identifiants portant un candidat VERDICT_CONFIRME ou
VERDICT_REVISION_HUMAINE sans decision DECISION_ACCEPTE), et ecrit le
resultat dans `--sortie`.

C'est une SOUSTRACTION, pas un nouveau muestreo : si le resultat, apres
exclusion, est plus petit que `--taille`, ce script ne tente jamais de
completer automatiquement le manque (ce role reste a
`E1_05_00_decouper_splits.py --n`, a relancer separement avec un `--n` plus
grand avant de reessayer l'extraction) ; il se contente d'indiquer
clairement combien d'exemples restent et combien manquent.

Si le resultat, apres exclusion, contient PLUS d'exemples que
`--taille`, il est recoupe a exactement `--taille` par echantillonnage
stratifie (type_exemple, source) : la taille publiee correspond
toujours a `--taille` demandee (jamais au surplus disponible).

Usage :
    uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py exporter \
        --dataset data/processed/dataset_pivot.jsonl \
        --anonymise data/processed/dataset_pivot_anonymise.jsonl

    uv run python interfaces/cli/E1_05_03_extraire_sous_ensemble_sft.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --exclusions data/processed/identifiants_a_exclure_publication.jsonl \
        --taille 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chsa_triage.application.use_cases import (
    ExtraireSousEnsembleSftUseCase,
    calculer_repartition_par_strate,
    formater_tableau_repartition,
)
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository
from chsa_triage.infrastructure.adapters.jsonl_dataset_repository import exemple_pivot_vers_dict
from tools.rafael.log_tool import LogTool

log = LogTool(origin="extraire_sous_ensemble_sft")

CHEMIN_DATASET_DEFAUT    = "data/processed/dataset_pivot_anonymise.jsonl"
CHEMIN_EXCLUSIONS_DEFAUT = "data/processed/identifiants_a_exclure_publication.jsonl"


def chemin_sortie_defaut(taille: int) -> str:
    return f"data/processed/dataset_chsa_triage_sft_anonymise_{taille}.jsonl"


def _lire_identifiants_a_exclure(chemin: str) -> frozenset[str]:
    fichier = Path(chemin)
    if not fichier.exists() or fichier.stat().st_size == 0:
        return frozenset()
    with fichier.open("r", encoding="utf-8") as f:
        return frozenset(json.loads(ligne)["identifiant"] for ligne in f if ligne.strip())


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=CHEMIN_DATASET_DEFAUT, help="Chemin du pivot ANONYMISE deja reparti")
    parser.add_argument("--exclusions", default=CHEMIN_EXCLUSIONS_DEFAUT)
    parser.add_argument("--taille", type=int, default=5000, help="Taille cible du sous-ensemble a publier")
    parser.add_argument(
        "--sortie",
        default=None,
        help="Chemin de sortie (defaut : data/processed/dataset_chsa_triage_sft_anonymise_<taille>.jsonl)",
    )
    arguments = parser.parse_args()
    if arguments.sortie is None:
        arguments.sortie = chemin_sortie_defaut(arguments.taille)

    log.START_ACTION("extraire_sous_ensemble_sft", "main", "extraction du sous-ensemble SFT a publier")
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("exclusions", arguments.exclusions)
    log.PARAMETER_VALUE("taille cible", arguments.taille)

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository = JsonlDatasetRepository(arguments.dataset)
    identifiants_a_exclure = _lire_identifiants_a_exclure(arguments.exclusions)
    log.PARAMETER_VALUE("identifiants a exclure lus", len(identifiants_a_exclure))

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Filtrage split != null puis soustraction des exclusions")
    cas_usage = ExtraireSousEnsembleSftUseCase(
        repository=repository,
        identifiants_a_exclure=identifiants_a_exclure,
        taille_cible=arguments.taille,
    )
    resultat = cas_usage.executer()

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    log.PARAMETER_VALUE("exemples avec split (avant exclusion)", cas_usage.nombre_avec_split)
    log.PARAMETER_VALUE("exclus (PII confirmee ou en attente)", cas_usage.nombre_exclus)
    log.PARAMETER_VALUE("disponibles apres exclusion", cas_usage.nombre_disponible_final)
    log.PARAMETER_VALUE("tronques (surplus au-dela de taille cible)", cas_usage.nombre_tronque)

    chemin_sortie = Path(arguments.sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    with chemin_sortie.open("w", encoding="utf-8") as f:
        for exemple in resultat:
            f.write(json.dumps(exemple_pivot_vers_dict(exemple), ensure_ascii=False) + "\n")

    print(f"Exemples avec split (avant exclusion) : {cas_usage.nombre_avec_split}")
    print(f"Exclus (PII confirmee ou en attente de revision humaine) : {cas_usage.nombre_exclus}")
    print(f"Disponibles apres exclusion : {cas_usage.nombre_disponible_final}")
    if cas_usage.nombre_tronque > 0:
        print(f"Recoupes par echantillonnage stratifie : -{cas_usage.nombre_tronque} (surplus au-dela de {arguments.taille})")
    print(f"Ecrits dans {arguments.sortie} : {len(resultat)} exemple(s).")
    print()
    print("Repartition du sous-ensemble ecrit par strate (type_exemple, source) :")
    print(formater_tableau_repartition(calculer_repartition_par_strate(resultat)))

    if cas_usage.manque > 0:
        log.LEVEL_5_WARNING(
            "extraire_sous_ensemble_sft",
            f"{cas_usage.nombre_disponible_final} < taille cible {arguments.taille} ; "
            f"il manque {cas_usage.manque} exemple(s)",
        )
        print()
        print(f"ATTENTION : {cas_usage.nombre_disponible_final} exemple(s) disponibles pour {arguments.taille} demandes.")
        print(f"Il manque {cas_usage.manque} exemple(s) apres exclusion des PII confirmees/en attente.")
        print(
            "Ce script ne complete jamais automatiquement (ce n'est qu'un filtre/une soustraction) : "
            "elargir le decoupage des splits d'abord, ex. :"
        )
        print(
            f"  uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset {arguments.dataset} "
            f"--n {arguments.taille + cas_usage.nombre_exclus}"
        )
        print("puis relancer cette extraction.")
    else:
        print(f"Taille cible atteinte ({len(resultat)} == {arguments.taille}).")

    log.FINISH_ACTION(
        "extraire_sous_ensemble_sft", "main", f"{len(resultat)} exemple(s) ecrits dans {arguments.sortie}"
    )


if __name__ == "__main__":
    main()
