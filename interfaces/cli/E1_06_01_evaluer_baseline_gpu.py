"""
Point d'entree CLI, Etape 1bis, variante GPU de l'evaluation baseline
zero-shot (cahier des charges §9 : "l'accuracy... depasse la baseline
zero-shot de facon mesurable"). Contrairement a
`E1_06_00_evaluer_baseline.py` (Environnement A, local, llama.cpp/GGUF
quantifie Q4_K_M), ce script cible un job HF Jobs avec GPU reel
(Environnement B) : `transformers.AutoModelForCausalLM` charge en
pleine precision bf16, SANS quantification
(`TransformersInferenceAdapter`), pour comparer plus tard au modele
SFT/DPO SANS melanger l'effet de la quantification avec l'effet reel de
l'entrainement (cf. AGENTS.md).

`EvaluerBaselineZeroShotUseCase` (application) est REUTILISE SANS
MODIFICATION : seul l'adaptateur d'inference injecte change entre les
deux scripts (`LlamaCppInferenceAdapter` vs `TransformersInferenceAdapter`).

Le dataset a evaluer N'EST PAS lu depuis `data/processed/` (gitignore,
absent d'un job distant qui ne clone pas le depot avec ses donnees) :
il est telecharge depuis un depot dataset HF PRIVE deja publie,
`--dataset-hf-repo` (defaut `mombasstic/chsa-triage-baseline-test`,
fichier `dataset_pivot_test_sft.jsonl`, 278 exemples `split=test`/
`type_exemple=sft`, EXACTEMENT le meme sous-ensemble que celui evalue
par le baseline local, pour que les deux resultats soient comparables).
Ce meme fichier vit aussi, versionne, dans
`data/splits/dataset_pivot_test_sft.jsonl` de ce depot.

Usage (sur un job HF Jobs GPU, cf. README §12 pour la commande
`hf jobs uv run` complete et son statut de verification) :
    uv run python interfaces/cli/E1_06_01_evaluer_baseline_gpu.py \
        --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
        --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics

Suivi : par defaut MLflow local (`--suivi-uri`, meme defaut que
`E1_06_00_evaluer_baseline.py`) ; si `--suivi-hf-repo` est fourni, le
run est publie a la place vers un depot dataset HF via
`HfDatasetSuiviExperimentation` (meme mecanisme que
`training/E2_04_sft_train.py --suivi-hf-repo`), pertinent sur un job
distant dont le disque local/SQLite ne survit pas au job.
"""

from __future__ import annotations  # Annotations de type differees

# Bibliotheque standard
import argparse  # Parsing des arguments CLI
import tempfile  # Fichier temporaire pour le JSONL telecharge depuis le Hub

# Bibliotheques du projet (cas d'usage, adaptateurs, logging)
from chsa_triage.application.use_cases   import EvaluerBaselineZeroShotUseCase  # Cas d'usage d'evaluation baseline (reutilise sans modification)
from chsa_triage.infrastructure.adapters import (  # Adaptateurs concrets (dataset, LLM, suivi)
    ChatMLFormateurAdapter,
    HfDatasetSuiviExperimentation,
    JsonlDatasetRepository,
    MlflowSuiviExperimentation,
    TransformersInferenceAdapter,
)
from tools.rafael.log_tool               import LogTool  # Utilitaire de logging du projet

log = LogTool(origin="evaluer_baseline_gpu")

DEPOT_DATASET_HF_DEFAUT      = "mombasstic/chsa-triage-baseline-test"
NOM_FICHIER_DATASET_HF       = "dataset_pivot_test_sft.jsonl"
MODELE_DEFAUT                = "Qwen/Qwen3-1.7B-Base"
URI_SUIVI_MLFLOW_DEFAUT      = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT = "data/processed/suivi_hf_dataset_baseline_gpu"
NOM_RUN_DEFAUT                = "baseline-zero-shot-gpu"


# ##############################################################################
def _telecharger_dataset(depot_hf: str, repertoire_local: str) -> str:
    """
    Telecharge `NOM_FICHIER_DATASET_HF` depuis `depot_hf` (depot
    dataset HF, prive ou public) vers `repertoire_local`, puis retourne
    le chemin local : `JsonlDatasetRepository` n'est PAS modifie pour
    lire depuis le Hub directement, le fichier telecharge a exactement
    le meme format pivot deja filtre (split=test, type_exemple=sft).
    """
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
        help=f"Depot dataset HF contenant {NOM_FICHIER_DATASET_HF} (split=test, type_exemple=sft)",
    )
    parser.add_argument(
        "--modele",
        default=MODELE_DEFAUT,
        help="Nom HF du modele (tokenizer/chat template ET poids, pleine precision bf16)",
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
        help="Depot dataset HF (ex. mombasstic/chsa-triage-baseline-metrics) pour publier le run "
             "via HfDatasetSuiviExperimentation au lieu de MLflow local (pertinent sur un job "
             "distant dont le disque ne survit pas au job) ; cf. README §12",
    )
    parser.add_argument(
        "--suivi-hf-repertoire-local",
        default=REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT,
        help=f"Repertoire de travail local synchronise vers --suivi-hf-repo (defaut {REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT})",
    )
    parser.add_argument("--nom-run", default=NOM_RUN_DEFAUT)
    arguments = parser.parse_args()

    log.START_ACTION(
        "evaluer_baseline_gpu", "main", "evaluation baseline zero-shot GPU (Etape 1bis, transformers/bf16)"
    )
    log.PARAMETER_VALUE("depot dataset HF", arguments.dataset_hf_repo)
    log.PARAMETER_VALUE("modele (tokenizer + poids, bf16)", arguments.modele)
    log.PARAMETER_VALUE("suivi", arguments.suivi_hf_repo or arguments.suivi_uri)

    # ----- TELECHARGEMENT DU DATASET DEPUIS LE HUB -----------------------------
    log.STEP(1, "Telechargement du dataset depuis le Hub", arguments.dataset_hf_repo)
    with tempfile.TemporaryDirectory() as repertoire_temporaire:
        chemin_dataset = _telecharger_dataset(arguments.dataset_hf_repo, repertoire_temporaire)
        log.PARAMETER_VALUE("dataset telecharge", chemin_dataset)

        # ----- PREPARE ADAPTERS (Dependency Injection) -------------------------
        repository = JsonlDatasetRepository(chemin_dataset)
        formateur  = ChatMLFormateurAdapter(nom_modele=arguments.modele)
        moteur     = TransformersInferenceAdapter(nom_modele=arguments.modele)
        suivi      = _construire_suivi(arguments)

        # ----- USE CASE EXECUTE --------------------------------------------------
        log.STEP(2, "Generation zero-shot + comparaison sur le split test (GPU, bf16)")
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
                "evaluer_baseline_gpu", f"echec de l'evaluation baseline GPU : {erreur}"
            )
            raise

    # ----- LOG FINAL INFO -----------------------------------------------------
    log.PARAMETER_VALUE("nombre d'exemples evalues", resultat.nombre_exemples)
    log.PARAMETER_VALUE("exact match", resultat.exact_match)
    log.PARAMETER_VALUE("F1 moyen", resultat.f1_moyen)
    log.PARAMETER_VALUE("latence moyenne (ms)", resultat.latence_ms_moyenne)
    log.PARAMETER_VALUE("echecs d'inference", resultat.nombre_echecs_inference)
    log.FINISH_ACTION(
        "evaluer_baseline_gpu",
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
