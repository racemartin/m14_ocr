"""
Point d'entree CLI, Etape 2, evaluation "post-SFT" (README §2.4) :
mesure la performance du modele reellement entraine par SFT-LoRA sur le
meme sous-ensemble de 278 exemples deja evalue par les deux baselines
zero-shot, avec les memes metriques, pour une comparaison directe.

`EvaluerBaselineZeroShotUseCase` est reutilise sans modification une
troisieme fois : seul l'adaptateur d'inference change (base+LoRA au
lieu de base seule). Le dataset est telecharge depuis le meme depot HF
prive que les baselines (`--dataset-hf-repo`), pour garantir que les
trois runs portent exactement sur le meme sous-ensemble.

Usage (sur un job HF Jobs GPU, cf. README §2.4) :
    uv run python interfaces/cli/E2_05_evaluer_post_sft.py \
        --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
        --depot-lora mombasstic/chsa-triage-sft-lora \
        --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics

Suivi : MLflow local par defaut (`--suivi-uri`) ; `--suivi-hf-repo`
publie le run vers un depot dataset HF a la place.
"""

from __future__ import annotations

import argparse
import tempfile

from chsa_triage.application.use_cases import EvaluerBaselineZeroShotUseCase
from chsa_triage.infrastructure.adapters import (
    ChatMLFormateurAdapter,
    HfDatasetSuiviExperimentation,
    JsonlDatasetRepository,
    MlflowSuiviExperimentation,
    TransformersLoraInferenceAdapter,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="evaluer_post_sft")

DEPOT_DATASET_HF_DEFAUT = "mombasstic/chsa-triage-baseline-test"
NOM_FICHIER_DATASET_HF = "dataset_pivot_test_sft.jsonl"
MODELE_BASE_DEFAUT = "Qwen/Qwen3-1.7B-Base"
DEPOT_LORA_DEFAUT = "mombasstic/chsa-triage-sft-lora"
URI_SUIVI_MLFLOW_DEFAUT = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT = "data/processed/suivi_hf_dataset_post_sft"
NOM_RUN_DEFAUT = "evaluation-post-sft"


# ##############################################################################
def _telecharger_dataset(depot_hf: str, repertoire_local: str) -> str:
    """Telecharge `NOM_FICHIER_DATASET_HF` depuis `depot_hf`. Duplique de
    `E1_06_01_evaluer_baseline_gpu.py` (4 lignes identiques prefere a
    une abstraction pour une fonction aussi triviale)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=depot_hf,
        repo_type="dataset",
        filename=NOM_FICHIER_DATASET_HF,
        local_dir=repertoire_local,
    )


def _construire_suivi(arguments: argparse.Namespace):
    if arguments.suivi_hf_repo:
        return HfDatasetSuiviExperimentation(
            repo_id=arguments.suivi_hf_repo,
            repertoire_local=arguments.suivi_hf_repertoire_local,
        )
    return MlflowSuiviExperimentation(uri_tracking=arguments.suivi_uri)


# ##############################################################################
def main() -> None:
    # ----- PARSE ARGUMENTS ----------------------------------------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-hf-repo",
        default=DEPOT_DATASET_HF_DEFAUT,
        help=f"Depot dataset HF contenant {NOM_FICHIER_DATASET_HF} (split=test, type_exemple=sft), "
        "MEME sous-ensemble que les baselines zero-shot (§2.2)",
    )
    parser.add_argument(
        "--modele-base",
        default=MODELE_BASE_DEFAUT,
        help="Nom HF du modele de base (tokenizer/chat template ET poids, pleine precision bf16)",
    )
    parser.add_argument(
        "--depot-lora",
        default=DEPOT_LORA_DEFAUT,
        help="Depot HF (type modele) contenant les poids LoRA entraines a charger par-dessus --modele-base",
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
        help="Nombre max de tokens generes par exemple (max_new_tokens)",
    )
    parser.add_argument(
        "--suivi-uri",
        default=URI_SUIVI_MLFLOW_DEFAUT,
        help="URI MLflow, utilise seulement si --suivi-hf-repo est absent",
    )
    parser.add_argument(
        "--suivi-hf-repo",
        default=None,
        help="Depot dataset HF pour publier le run via HfDatasetSuiviExperimentation au lieu de "
        "MLflow local (pertinent sur un job distant dont le disque ne survit pas au job) ; "
        "cf. README §2.4",
    )
    parser.add_argument(
        "--suivi-hf-repertoire-local",
        default=REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT,
        help=f"Repertoire de travail local synchronise vers --suivi-hf-repo (defaut {REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT})",
    )
    parser.add_argument("--nom-run", default=NOM_RUN_DEFAUT)
    arguments = parser.parse_args()

    log.START_ACTION(
        "evaluer_post_sft",
        "main",
        "evaluation post-SFT (Etape 2, base+LoRA, transformers/bf16)",
    )
    log.PARAMETER_VALUE("depot dataset HF", arguments.dataset_hf_repo)
    log.PARAMETER_VALUE(
        "modele de base (tokenizer + poids, bf16)", arguments.modele_base
    )
    log.PARAMETER_VALUE("depot LoRA", arguments.depot_lora)
    log.PARAMETER_VALUE("suivi", arguments.suivi_hf_repo or arguments.suivi_uri)

    # ----- TELECHARGEMENT DU DATASET DEPUIS LE HUB -----------------------------
    log.STEP(
        1, "Telechargement du dataset depuis le Hub", arguments.dataset_hf_repo
    )
    with tempfile.TemporaryDirectory() as repertoire_temporaire:
        chemin_dataset = _telecharger_dataset(
            arguments.dataset_hf_repo, repertoire_temporaire
        )
        log.PARAMETER_VALUE("dataset telecharge", chemin_dataset)

        # ----- PREPARE ADAPTERS (Dependency Injection) -------------------------
        repository = JsonlDatasetRepository(chemin_dataset)
        formateur = ChatMLFormateurAdapter(nom_modele=arguments.modele_base)
        moteur = TransformersLoraInferenceAdapter(
            depot_lora=arguments.depot_lora,
            nom_modele_base=arguments.modele_base,
        )
        suivi = _construire_suivi(arguments)

        # ----- USE CASE EXECUTE --------------------------------------------------
        log.STEP(
            2,
            "Generation post-SFT + comparaison sur le split test (GPU, bf16, base+LoRA)",
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
                "evaluer_post_sft", f"echec de l'evaluation post-SFT : {erreur}"
            )
            raise

    # ----- LOG FINAL INFO -----------------------------------------------------
    log.PARAMETER_VALUE("nombre d'exemples evalues", resultat.nombre_exemples)
    log.PARAMETER_VALUE("exact match", resultat.exact_match)
    log.PARAMETER_VALUE("F1 moyen", resultat.f1_moyen)
    log.PARAMETER_VALUE("latence moyenne (ms)", resultat.latence_ms_moyenne)
    log.PARAMETER_VALUE("echecs d'inference", resultat.nombre_echecs_inference)
    log.FINISH_ACTION(
        "evaluer_post_sft",
        "main",
        f"run '{arguments.nom_run}' journalise ({arguments.suivi_hf_repo or arguments.suivi_uri})",
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
        f"  Echecs d'inference (exemples ignores)..............: "
        f"{resultat.nombre_echecs_inference}"
    )
    print(
        f"Run '{arguments.nom_run}' journalise dans "
        f"{arguments.suivi_hf_repo or arguments.suivi_uri}"
    )


if __name__ == "__main__":
    main()
