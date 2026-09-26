"""
STEP 04
Point d'entree CLI, Etape 1, action "decouper en splits".

Usage :
    uv run python interfaces/cli/E1_05_00_decouper_splits.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --n 5000

--dataset doit pointer vers le fichier ANONYMISE, pas le pivot original
(qui n'est jamais anonymise en place et n'a donc jamais `anonymise=True`).
Le champ `split` est ajoute en place sur le fichier anonymise.

Croissance stable : un exemple deja `split` n'est jamais reassigne,
quel que soit `--n` ensuite — agrandir le jeu (`--n` plus grand, ou
sans `--n`) complete seulement ce qui manque, jamais un recalcul
complet (evite la fuite silencieuse d'exemples d'entrainement dans le
jeu de test que le cahier des charges interdit).

`--n N` (taille CIBLE cumulee) preleve `N - deja_assignes` nouveaux
exemples par echantillonnage stratifie ; `N` trop petit ne fait rien
(reduire un decoupage existant n'est pas supporte, avertissement pas
erreur) ; omis, tout ce qui manque recoit un split.

Par precaution, un exemple avec une PII residuelle CONFIRMEE ou sans
decision humaine persistee est exclu des candidats de cette execution
(--original/--registre-echantillons/--decisions/--jeton-masque,
memes adaptateurs que `E1_04_01_reviser_pii_residuelle.py verify`).
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import DecouperSplitsUseCase
from chsa_triage.application.use_cases.E1_04_02_controler_qualite_anonymisation import (
    JETON_MASQUE_DEFAUT,
)
from chsa_triage.application.use_cases.E1_04_01_reviser_pii_residuelle import (
    ReviserPiiResiduelleUseCase,
)
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    JsonlDecisionsRevisionHumaine,
    JsonlRegistreEchantillonsControleQualite,
    SpacyVerificateurEntitesNommees,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="decouper_splits")

CHEMIN_ORIGINAL_DEFAUT = "data/processed/dataset_pivot.jsonl"
CHEMIN_REGISTRE_ECHANTILLONS_DEFAUT = (
    "data/processed/controle_qualite_identifiants_echantillonnes.jsonl"
)
CHEMIN_DECISIONS_DEFAUT = "data/processed/decisions_revision_humaine.jsonl"


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="Chemin du fichier pivot JSONL"
    )
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
    parser.add_argument(
        "--original",
        default=CHEMIN_ORIGINAL_DEFAUT,
        help="Chemin du pivot ORIGINAL (pour recalculer les candidats de PII en attente de decision)",
    )
    parser.add_argument(
        "--registre-echantillons", default=CHEMIN_REGISTRE_ECHANTILLONS_DEFAUT
    )
    parser.add_argument("--decisions", default=CHEMIN_DECISIONS_DEFAUT)
    parser.add_argument("--jeton-masque", default=JETON_MASQUE_DEFAUT)
    arguments = parser.parse_args()

    log.START_ACTION(
        "decouper_splits",
        "main",
        "decoupage train/val/test (croissance stable)",
    )
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("graine", arguments.graine)
    log.PARAMETER_VALUE("proportion-val", arguments.proportion_val)
    log.PARAMETER_VALUE("proportion-test", arguments.proportion_test)
    log.PARAMETER_VALUE(
        "n (taille cible cumulee)",
        arguments.n if arguments.n is not None else "(aucun, completer tout)",
    )

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository = JsonlDatasetRepository(arguments.dataset)

    # Meme adaptateurs que `E1_04_01_reviser_pii_residuelle.py verify` :
    # recalcule les identifiants a exclure du decoupage par precaution
    # (PII confirmee ou sans decision humaine).
    revision_pii = ReviserPiiResiduelleUseCase(
        repository_original=JsonlDatasetRepository(arguments.original),
        repository_anonymise=repository,
        verificateur_entites=SpacyVerificateurEntitesNommees(),
        registre_echantillons=JsonlRegistreEchantillonsControleQualite(
            arguments.registre_echantillons
        ),
        decisions=JsonlDecisionsRevisionHumaine(arguments.decisions),
        jeton_masque=arguments.jeton_masque,
    )

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(
        1,
        "Decoupage des nouveaux exemples (ceux deja assignes ne sont jamais touches)",
    )
    try:
        cas_usage = DecouperSplitsUseCase(
            repository=repository,
            graine_aleatoire=arguments.graine,
            proportion_val=arguments.proportion_val,
            proportion_test=arguments.proportion_test,
            n=arguments.n,
            obtenir_identifiants_pii_en_attente=revision_pii.identifiants_a_exclure_publication_set,
        )
        decompte = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR(
            "decouper_splits",
            f"echec du decoupage de {arguments.dataset} : {erreur}",
        )
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    log.PARAMETER_VALUE(
        "deja assignes (executions anterieures, non touches)",
        cas_usage.nombre_deja_assignes,
    )
    log.PARAMETER_VALUE(
        "nouveaux repartis cette execution", cas_usage.nombre_nouveaux
    )
    log.PARAMETER_VALUE(
        "exclus cette execution (PII en attente de decision humaine)",
        cas_usage.nombre_exclus_pii_en_attente,
    )
    if arguments.n is not None and cas_usage.nombre_nouveaux == 0:
        log.LEVEL_5_WARNING(
            "decouper_splits",
            f"--n {arguments.n} <= {cas_usage.nombre_deja_assignes} exemples deja assignes ; rien de "
            "nouveau a repartir (reduire un decoupage deja fait n'est pas supporte, aucune erreur)",
        )
    for split, nombre in decompte.items():
        log.PARAMETER_VALUE(f"split {split} (total cumule)", nombre)
    log.FINISH_ACTION(
        "decouper_splits", "main", f"splits ecrits pour {arguments.dataset}"
    )

    print(
        f"Nouveaux exemples repartis lors de cette execution : {cas_usage.nombre_nouveaux}"
    )
    print(
        f"Exemples deja assignes (executions anterieures, non touches) : {cas_usage.nombre_deja_assignes}"
    )
    print(
        "Exemples exclus cette execution (PII residuelle en attente de decision humaine) : "
        f"{cas_usage.nombre_exclus_pii_en_attente}"
    )
    print("Repartition totale actuelle des splits (deja assignes + nouveaux) :")
    for split, nombre in decompte.items():
        print(f"  {split:<10}: {nombre}")


if __name__ == "__main__":
    main()
