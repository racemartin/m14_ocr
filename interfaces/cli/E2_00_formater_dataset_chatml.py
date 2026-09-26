"""
Point d'entree CLI, Etape 2, action "rendre en ChatML un split du
dataset pivot" (`FormaterDatasetChatMLUseCase`), auparavant invoque
seulement depuis l'interieur de `training/E2_04_sft_train.py`. Un CLI
dedie plutot qu'un flag sur ce dernier : ce point d'entree GPU lourd ne
doit pas se melanger avec un outil d'inspection bon marche.

Deux usages : (1) rendre un split en ChatML sans lancer l'entrainement ;
(2) mode didactique optionnel (`--exemples N`, cape a 2) qui logue en
plus, sans ecrire nulle part, l'`ExemplePivot` brut puis son rendu
ChatML. Purement additif : le chemin d'ecriture normal est identique
avec ou sans `--exemples`.

Usage :
    uv run python interfaces/cli/E2_00_formater_dataset_chatml.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --split train \
        --exemples 2
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import FormaterDatasetChatMLUseCase
from chsa_triage.domain.model.enums import TypeExemple, TypeSplit
from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.model.exemple_pivot import ExemplePivot
from chsa_triage.infrastructure.adapters import (
    ChatMLFormateurAdapter,
    JsonlDatasetRepository,
    JsonlExempleFormateRepository,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="E2_00_formater_dataset_chatml")

NOMBRE_MAX_EXEMPLES_DIDACTIQUES = 2

CHEMIN_DATASET_DEFAUT = "data/processed/dataset_pivot_anonymise.jsonl"
CHEMIN_DATASET_FORMATE_DEFAUT = "data/processed/dataset_formate.jsonl"
NOM_MODELE_DEFAUT = "Qwen/Qwen3-1.7B-Base"


def nombre_exemples_didactiques_cape(valeur: int) -> int:
    """Cape --exemples a NOMBRE_MAX_EXEMPLES_DIDACTIQUES (jamais negatif)."""
    return max(0, min(valeur, NOMBRE_MAX_EXEMPLES_DIDACTIQUES))


def afficher_exemple_didactique(
    log_tool: LogTool,
    index: int,
    total: int,
    exemple: ExemplePivot,
    exemple_formate: ExempleFormate,
) -> None:
    """Affiche par LogTool, en lecture seule, un `ExemplePivot` brut
    puis son rendu ChatML final `exemple_formate` (deja produit par
    `ChatMLFormateurAdapter.formater()`, jamais recalcule ici)."""
    log_tool.STEP(1, f"Exemple didactique {index}/{total}", exemple.identifiant)
    log_tool.PARAMETER_VALUE("source", exemple.source)
    log_tool.PARAMETER_VALUE("langue", exemple.langue.value)

    log_tool.STEP(2, "ExemplePivot brut : prompt")
    for message in exemple.prompt:
        log_tool.PARAMETER_VALUE(f"  role={message.role}", message.contenu)

    log_tool.STEP(2, "ExemplePivot brut : completion")
    for message in exemple.completion:
        log_tool.PARAMETER_VALUE(f"  role={message.role}", message.contenu)

    log_tool.STEP(2, "Texte ChatML rendu (ChatMLFormateurAdapter.formater)")
    log_tool.PARAMETER_VALUE(
        "ChatML", "──────────────────────────────────────┐"
    )
    for ligne in exemple_formate.texte.splitlines():
        log_tool.PARAMETER_VALUE("  |", ligne)
    log_tool.PARAMETER_VALUE(
        "ChatML", "──────────────────────────────────────┘"
    )


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=CHEMIN_DATASET_DEFAUT,
        help="Chemin du pivot ANONYMISE deja reparti",
    )
    parser.add_argument(
        "--dataset-formate",
        default=CHEMIN_DATASET_FORMATE_DEFAUT,
        help=f"Chemin de sortie du rendu ChatML (defaut {CHEMIN_DATASET_FORMATE_DEFAUT})",
    )
    parser.add_argument(
        "--split",
        choices=[s.value for s in TypeSplit],
        default=TypeSplit.TRAIN.value,
    )
    parser.add_argument(
        "--modele",
        default=NOM_MODELE_DEFAUT,
        help="Modele dont le chat template natif est utilise",
    )
    parser.add_argument(
        "--exemples",
        type=int,
        default=0,
        help=f"Logue par LogTool, en plus de l'ecriture normale, jusqu'a {NOMBRE_MAX_EXEMPLES_DIDACTIQUES} "
        "exemples (ExemplePivot brut + rendu ChatML). 0 (defaut) : aucun log didactique, comportement "
        "d'ecriture inchange.",
    )
    arguments = parser.parse_args()
    split = TypeSplit(arguments.split)
    nombre_didactiques = nombre_exemples_didactiques_cape(arguments.exemples)

    log.START_ACTION(
        "E2_00_formater_dataset_chatml",
        "main",
        f"rendu ChatML du split {split.value}",
    )
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("dataset-formate", arguments.dataset_formate)
    log.PARAMETER_VALUE("split", split.value)
    log.PARAMETER_VALUE("modele", arguments.modele)
    log.PARAMETER_VALUE("exemples didactiques demandes", arguments.exemples)
    log.PARAMETER_VALUE("exemples didactiques (capes)", nombre_didactiques)

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection) : memes adaptateurs reels que
    # l'etape 1 de training/E2_04_sft_train.py
    # -------------------------------------------------------------------------
    repository_pivot = JsonlDatasetRepository(arguments.dataset)
    repository_formate = JsonlExempleFormateRepository(
        arguments.dataset_formate
    )
    formateur = ChatMLFormateurAdapter(nom_modele=arguments.modele)

    # Lecture seule, AVANT l'ecriture : n'affecte rien, sert uniquement
    # a alimenter le mode didactique ci-dessous. Meme filtre
    # (split, type_exemple=SFT) que celui applique en interne par
    # FormaterDatasetChatMLUseCase.executer().
    candidats_didactiques: list[ExemplePivot] = []
    if nombre_didactiques > 0:
        candidats_didactiques = list(
            repository_pivot.lister(
                filtre={"split": split, "type_exemple": TypeExemple.SFT}
            )
        )[:nombre_didactiques]

    # -------------------------------------------------------------------------
    # USE CASE (chemin d'ecriture normal, IDENTIQUE avec ou sans mode didactique)
    # -------------------------------------------------------------------------
    log.STEP(1, "Rendu ChatML", "FormaterDatasetChatMLUseCase.executer")
    cas_usage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre = cas_usage.executer(split)
    log.PARAMETER_VALUE("exemples formates", nombre)

    # -------------------------------------------------------------------------
    # MODE DIDACTIQUE (lecture seule / console uniquement) : rappelle
    # ChatMLFormateurAdapter.formater() sur les memes candidats, pour
    # AFFICHAGE SEUL, ne reecrit rien, n'affecte pas repository_formate
    # ni aucun fichier.
    # -------------------------------------------------------------------------
    if candidats_didactiques:
        log.STEP(
            1, "Mode didactique", f"{len(candidats_didactiques)} exemple(s)"
        )
        for index, exemple in enumerate(candidats_didactiques, start=1):
            exemple_formate = formateur.formater(exemple)
            afficher_exemple_didactique(
                log, index, len(candidats_didactiques), exemple, exemple_formate
            )

    log.FINISH_ACTION(
        "E2_00_formater_dataset_chatml",
        "main",
        f"{nombre} exemple(s) rendus dans {arguments.dataset_formate}",
    )
    print(f"Exemples formates ({split.value}) : {nombre}")
    print(f"Ecrits dans {arguments.dataset_formate}")
    if candidats_didactiques:
        print(
            f"Mode didactique : {len(candidats_didactiques)} exemple(s) affiches (voir logs LogTool ci-dessus)."
        )


if __name__ == "__main__":
    main()
