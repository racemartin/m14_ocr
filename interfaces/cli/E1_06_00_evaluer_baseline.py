"""
Point d'entree CLI, Etape 1bis, action "evaluer la baseline zero-shot"
(cahier des charges §9 : "l'accuracy... depasse la baseline zero-shot
de facon mesurable"). Environnement A (local, sans GPU) : l'inference
passe par un `llama-server` (llama.cpp) DEJA LANCE, jamais par
transformers/torch en pleine precision.

PREREQUIS : un `llama-server` doit deja tourner en local sur
`--url-serveur`, servant un GGUF de `Qwen/Qwen3-1.7B-Base` (voir
AGENTS.md pour la commande de demarrage exacte et le GGUF utilise lors
de la verification manuelle de ce script). Ce script ne demarre PAS le
serveur lui-meme.

Usage :
    uv run python interfaces/cli/E1_06_00_evaluer_baseline.py \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --url-serveur http://127.0.0.1:8080

Le run est journalise dans MLflow (`SuiviExperimentation`, meme
mecanisme que `training/E2_04_sft_train.py`, pas un nouveau systeme de
tracking) sous le nom `baseline-zero-shot`. Execute LOCALEMENT (pas sur
un job HF), le run atterrit directement dans le MLflow local
(`--suivi-uri`, defaut `sqlite:///data/processed/mlflow.db`) : PAS
besoin de `monitoring/importer_mlflow_local.py` pour le retrouver (ce
script sert a rapatrier des runs distants publies sur un depot HF, ce
qui n'est pas le cas ici).
"""

from __future__ import annotations  # Annotations de type differees

# Bibliotheque standard
import argparse  # Parsing des arguments CLI

# Bibliotheques du projet (cas d'usage, adaptateurs, logging)
from chsa_triage.application.use_cases   import EvaluerBaselineZeroShotUseCase  # Cas d'usage d'evaluation baseline
from chsa_triage.infrastructure.adapters import (  # Adaptateurs concrets (dataset, LLM, MLflow)
    ChatMLFormateurAdapter,
    JsonlDatasetRepository,
    LlamaCppInferenceAdapter,
    MlflowSuiviExperimentation,
)
from tools.rafael.log_tool               import LogTool  # Utilitaire de logging du projet

log = LogTool(origin="evaluer_baseline")

CHEMIN_DATASET_DEFAUT   = "data/processed/dataset_pivot_anonymise.jsonl"
URL_SERVEUR_DEFAUT      = "http://127.0.0.1:8080"
MODELE_DEFAUT           = "Qwen/Qwen3-1.7B-Base"
URI_SUIVI_MLFLOW_DEFAUT = "sqlite:///data/processed/mlflow.db"
NOM_RUN_DEFAUT          = "baseline-zero-shot"


# ##############################################################################
def main() -> None:
    # ----- PARSE ARGUMENTS ----------------------------------------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=CHEMIN_DATASET_DEFAUT,
        help="Chemin du pivot ANONYMISE (champ split renseigne)",
    )
    parser.add_argument(
        "--url-serveur",
        default=URL_SERVEUR_DEFAUT,
        help="URL d'un llama-server DEJA LANCE",
    )
    parser.add_argument(
        "--modele",
        default=MODELE_DEFAUT,
        help="Nom HF du modele (tokenizer/chat template reel)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = generation deterministe (reproductible)",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=256,
        help="Nombre max de tokens generes par exemple",
    )
    parser.add_argument(
        "--timeout-secondes",
        type=float,
        default=180.0,
        help="Timeout HTTP par requete d'inference",
    )
    parser.add_argument(
        "--suivi-uri",
        default=URI_SUIVI_MLFLOW_DEFAUT,
        help="URI MLflow (meme defaut que training/E2_04_sft_train.py)",
    )
    parser.add_argument("--nom-run", default=NOM_RUN_DEFAUT)
    arguments = parser.parse_args()

    log.START_ACTION(
        "evaluer_baseline", "main", "evaluation baseline zero-shot (Etape 1bis)"
    )
    log.PARAMETER_VALUE("dataset", arguments.dataset)
    log.PARAMETER_VALUE("url serveur llama.cpp", arguments.url_serveur)
    log.PARAMETER_VALUE("modele (tokenizer/chat template)", arguments.modele)
    log.PARAMETER_VALUE("suivi-uri", arguments.suivi_uri)

    # ----- PREPARE ADAPTERS (Dependency Injection) ----------------------------
    repository = JsonlDatasetRepository(arguments.dataset)
    formateur  = ChatMLFormateurAdapter(nom_modele=arguments.modele)
    moteur = LlamaCppInferenceAdapter(
        url_serveur_local=arguments.url_serveur,
        timeout_secondes=arguments.timeout_secondes,
    )
    suivi = MlflowSuiviExperimentation(uri_tracking=arguments.suivi_uri)

    # ----- USE CASE EXECUTE ---------------------------------------------------
    log.STEP(
        1,
        "Generation zero-shot + comparaison sur le split test "
        "(peut prendre du temps sur CPU)",
    )
    try:
        cas_usage = EvaluerBaselineZeroShotUseCase(
            repository=repository,
            formateur=formateur,
            moteur=moteur,
            suivi=suivi,
            parametres_generation={
                "temperature": arguments.temperature,
                "n_predict": arguments.n_predict,
            },
            nom_run=arguments.nom_run,
        )
        resultat = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR(
            "evaluer_baseline", f"echec de l'evaluation baseline : {erreur}"
        )
        raise

    # ----- LOG FINAL INFO -----------------------------------------------------
    log.PARAMETER_VALUE("nombre d'exemples evalues", resultat.nombre_exemples)
    log.PARAMETER_VALUE("exact match", resultat.exact_match)
    log.PARAMETER_VALUE("F1 moyen", resultat.f1_moyen)
    log.PARAMETER_VALUE("latence moyenne (ms)", resultat.latence_ms_moyenne)
    log.FINISH_ACTION(
        "evaluer_baseline",
        "main",
        f"run '{arguments.nom_run}' journalise dans {arguments.suivi_uri}",
    )

    print(
        f"  Exemples evalues (split=test, type_exemple=SFT)....: "
        f"{resultat.nombre_exemples}"
    )
    print(
        f"  Exact match........................................: "
        f"{resultat.exact_match:.3f}"
    )
    print(
        f"  F1 moyen (token)...................................: "
        f"{resultat.f1_moyen:.3f}"
    )
    if resultat.exactitude_niveau.exactitude is not None:
        print(
            f"  Exactitude classification niveau ESI...............: "
            f"{resultat.exactitude_niveau.exactitude:.3f} "
            f"({resultat.exactitude_niveau.nombre_comparables}/"
            f"{resultat.exactitude_niveau.nombre_paires} paires comparables)"
        )
    else:
        print(
            "  Exactitude classification niveau ESI...............: "
            "non calculable (0 paire comparable sur "
            f"{resultat.exactitude_niveau.nombre_paires} ; le dataset actuel "
            "n'a pas de completion au format JSON {niveau, categorie, "
            "ressources_estimees}, cf. "
            "docs/03_etape2_sft/00_introduction_concepts.md)"
        )
    print(
        f"  Latence moyenne par generation.....................: "
        f"{resultat.latence_ms_moyenne:.1f} ms"
    )
    print(
        f"Run MLflow '{arguments.nom_run}' journalise dans "
        f"{arguments.suivi_uri}"
    )


if __name__ == "__main__":
    main()
