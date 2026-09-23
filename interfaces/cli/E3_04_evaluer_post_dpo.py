"""
Point d'entree CLI, Etape 3, evaluation "post-DPO" (README §3 DPO) :
mesure la performance du modele REELLEMENT aligne par DPO (poids a
publier sur `mombasstic/chsa-triage-dpo-lora`, cf. README §3,
continuant le checkpoint SFT-LoRA de §2.3/AGENTS.md), sur le MEME
sous-ensemble de 278 exemples `split=test`/`type_exemple=sft` deja
evalue par les deux baselines zero-shot (`mombasstic/chsa-triage-baseline-test`,
§2.2) ET par l'evaluation post-SFT (§2.5), avec les MEMES metriques
(`application/metriques_evaluation_baseline.py`), pour une comparaison
numero-contre-numero directe entre les quatre runs.

Decision de conception (meme raisonnement que `E2_05_evaluer_post_sft.py`,
README §2.4/§2.5 pour le detail complet) :
`EvaluerBaselineZeroShotUseCase` (application) est REUTILISE SANS
MODIFICATION une QUATRIEME fois (apres CPU/GGUF, GPU/bf16 zero-shot,
puis post-SFT) : c'est un cas d'usage generique "generer une reponse
par exemple du split test via un `MoteurInference`, comparer a la
reference, agreger les metriques", agnostique au fait que le modele
injecte soit entraine ou non, et agnostique au type d'entrainement
(SFT ou DPO) : seul l'adaptateur d'inference change (meme
`TransformersLoraInferenceAdapter` que le post-SFT, pointe vers le
depot LoRA DPO au lieu du depot LoRA SFT). Le nom "Baseline"/"ZeroShot"
du cas d'usage garde son vocabulaire d'origine plutot que d'etre
renomme : renommer aurait touche 4 scripts CLI deja publies (celui-ci,
le post-SFT, et les deux baselines) et leurs eventuels tests pour un
gain cosmetique seul, sans changer le comportement ; ce script
documente explicitement ce reemploi plutot que de le masquer.

Le dataset a evaluer N'EST PAS lu depuis `data/processed/` (gitignore,
absent d'un job distant) : il est telecharge depuis le MEME depot HF
prive deja utilise par les trois runs precedents,
`--dataset-hf-repo` (defaut `mombasstic/chsa-triage-baseline-test`,
fichier `dataset_pivot_test_sft.jsonl`), pour garantir que les quatre
runs (CPU zero-shot, GPU zero-shot, post-SFT, post-DPO) portent
EXACTEMENT sur le meme sous-ensemble.

Usage (sur un job HF Jobs GPU, cf. README §3 pour la commande
`hf jobs uv run` complete et son statut de verification) :
    uv run python interfaces/cli/E3_04_evaluer_post_dpo.py \
        --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
        --depot-lora mombasstic/chsa-triage-dpo-lora \
        --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics

Suivi : par defaut MLflow local (`--suivi-uri`, meme defaut que les
deux baselines et le post-SFT) ; si `--suivi-hf-repo` est fourni, le
run est publie a la place vers un depot dataset HF via
`HfDatasetSuiviExperimentation` (meme mecanisme que
`E1_06_01_evaluer_baseline_gpu.py`/`E2_05_evaluer_post_sft.py`/
`training/E3_03_dpo_train.py --suivi-hf-repo`).
"""

from __future__ import annotations  # Annotations de type differees

# Bibliotheque standard
import argparse  # Parsing des arguments CLI
import tempfile  # Fichier temporaire pour le JSONL telecharge depuis le Hub

# Bibliotheques du projet (cas d'usage, adaptateurs, logging)
from chsa_triage.application.use_cases   import EvaluerBaselineZeroShotUseCase  # Cas d'usage d'evaluation (reutilise sans modification)
from chsa_triage.infrastructure.adapters import (  # Adaptateurs concrets (dataset, LLM, suivi)
    ChatMLFormateurAdapter,
    HfDatasetSuiviExperimentation,
    JsonlDatasetRepository,
    MlflowSuiviExperimentation,
    TransformersLoraInferenceAdapter,
)
from tools.rafael.log_tool               import LogTool  # Utilitaire de logging du projet

log = LogTool(origin="evaluer_post_dpo")

DEPOT_DATASET_HF_DEFAUT      = "mombasstic/chsa-triage-baseline-test"
NOM_FICHIER_DATASET_HF       = "dataset_pivot_test_sft.jsonl"
MODELE_BASE_DEFAUT           = "Qwen/Qwen3-1.7B-Base"
DEPOT_LORA_DEFAUT            = "mombasstic/chsa-triage-dpo-lora"
URI_SUIVI_MLFLOW_DEFAUT      = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT = "data/processed/suivi_hf_dataset_post_dpo"
NOM_RUN_DEFAUT                = "evaluation-post-dpo"


# ##############################################################################
def _telecharger_dataset(depot_hf: str, repertoire_local: str) -> str:
    """
    Telecharge `NOM_FICHIER_DATASET_HF` depuis `depot_hf` vers
    `repertoire_local`, meme fonction que `E1_06_01_evaluer_baseline_gpu.py`/
    `E2_05_evaluer_post_sft.py` (non partagee via un module commun : des
    fichiers de 4 lignes identiques ont ete juges preferables a une
    abstraction pour une fonction aussi triviale, cf. principe de
    non-sur-abstraction du projet).
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
        help=f"Depot dataset HF contenant {NOM_FICHIER_DATASET_HF} (split=test, type_exemple=sft), "
             "MEME sous-ensemble que les baselines zero-shot (§2.2) et le post-SFT (§2.5)",
    )
    parser.add_argument(
        "--modele-base",
        default=MODELE_BASE_DEFAUT,
        help="Nom HF du modele de base (tokenizer/chat template ET poids, pleine precision bf16)",
    )
    parser.add_argument(
        "--depot-lora",
        default=DEPOT_LORA_DEFAUT,
        help="Depot HF (type modele) contenant les poids LoRA DPO entraines a charger par-dessus --modele-base",
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
        "--repetition-penalty",
        type=float,
        default=None,
        help="Parametre repetition_penalty de transformers.generate() (ex. 1.2), transmis tel "
             "quel (cf. _parametres_generation_transformers) ; absent par defaut (non transmis, "
             "comportement inchange). A tester contre les boucles de repetition observees en "
             "generation deterministe (temperature=0.0), cf. README §3 (23/09/2026).",
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
             "cf. README §3",
    )
    parser.add_argument(
        "--suivi-hf-repertoire-local",
        default=REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT,
        help=f"Repertoire de travail local synchronise vers --suivi-hf-repo (defaut {REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT})",
    )
    parser.add_argument("--nom-run", default=NOM_RUN_DEFAUT)
    arguments = parser.parse_args()

    log.START_ACTION(
        "evaluer_post_dpo", "main", "evaluation post-DPO (Etape 3, base+LoRA, transformers/bf16)"
    )
    log.PARAMETER_VALUE("depot dataset HF", arguments.dataset_hf_repo)
    log.PARAMETER_VALUE("modele de base (tokenizer + poids, bf16)", arguments.modele_base)
    log.PARAMETER_VALUE("depot LoRA", arguments.depot_lora)
    log.PARAMETER_VALUE("suivi", arguments.suivi_hf_repo or arguments.suivi_uri)

    # ----- TELECHARGEMENT DU DATASET DEPUIS LE HUB -----------------------------
    log.STEP(1, "Telechargement du dataset depuis le Hub", arguments.dataset_hf_repo)
    with tempfile.TemporaryDirectory() as repertoire_temporaire:
        chemin_dataset = _telecharger_dataset(arguments.dataset_hf_repo, repertoire_temporaire)
        log.PARAMETER_VALUE("dataset telecharge", chemin_dataset)

        # ----- PREPARE ADAPTERS (Dependency Injection) -------------------------
        repository = JsonlDatasetRepository(chemin_dataset)
        formateur  = ChatMLFormateurAdapter(nom_modele=arguments.modele_base)
        moteur     = TransformersLoraInferenceAdapter(
            depot_lora=arguments.depot_lora, nom_modele_base=arguments.modele_base
        )
        suivi      = _construire_suivi(arguments)

        parametres_generation = {
            "temperature": arguments.temperature,
            "n_predict": arguments.n_predict,
        }
        if arguments.repetition_penalty is not None:
            parametres_generation["repetition_penalty"] = arguments.repetition_penalty
        log.PARAMETER_VALUE("parametres de generation", parametres_generation)

        # ----- USE CASE EXECUTE --------------------------------------------------
        log.STEP(2, "Generation post-DPO + comparaison sur le split test (GPU, bf16, base+LoRA)")
        try:
            cas_usage = EvaluerBaselineZeroShotUseCase(
                repository=repository,
                formateur=formateur,
                moteur=moteur,
                suivi=suivi,
                parametres_generation=parametres_generation,
                nom_run=arguments.nom_run,
            )
            resultat = cas_usage.executer()
        except Exception as erreur:
            log.LEVEL_4_ERROR(
                "evaluer_post_dpo", f"echec de l'evaluation post-DPO : {erreur}"
            )
            raise

        # ----- ECHANTILLON DE GENERATIONS (diagnostic) -------------------------
        # Inspecter reellement ce que le modele genere (verbosite, format JSON
        # ou non, etc.) plutot que deviner a partir des seules metriques
        # agregees, meme principe que l'echantillon de diagnostic de
        # ReformulerPreferenceDpoUseCase (E3_00).
        for index, (genere, reference) in enumerate(cas_usage.echantillon_generations):
            log.PARAMETER_VALUE(f"  genere [{index}]", genere)
            log.PARAMETER_VALUE(f"  reference [{index}]", reference)

    # ----- LOG FINAL INFO -----------------------------------------------------
    log.PARAMETER_VALUE("nombre d'exemples evalues", resultat.nombre_exemples)
    log.PARAMETER_VALUE("exact match", resultat.exact_match)
    log.PARAMETER_VALUE("F1 moyen", resultat.f1_moyen)
    log.PARAMETER_VALUE("latence moyenne (ms)", resultat.latence_ms_moyenne)
    log.PARAMETER_VALUE("echecs d'inference", resultat.nombre_echecs_inference)
    log.FINISH_ACTION(
        "evaluer_post_dpo",
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
