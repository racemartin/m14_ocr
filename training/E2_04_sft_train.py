"""
Point d'entree, Etape 2 : entrainement SFT-LoRA reel (Environnement B,
GPU requis). Vit dans `training/` (execute via HF Jobs), pas
`interfaces/cli/`, mais garde le prefixe numerote `E2_04_` car l'ordre
d'execution compte (orchestre les cas d'usage E2_00 a E2_03).

Statut reel : le premier entrainement complet a tourne sur HF Jobs
(GPU L4, ~20 min, 342 pas, verdict SAINE, poids publies sur
`mombasstic/chsa-triage-sft-lora`).

Enchaine, dans l'ordre : (1) rendu ChatML train+validation, (2) boucle
d'entrainement/ajustement d'hyperparametres, (3) sauvegarde des
metadonnees du meilleur checkpoint, (4) si `--checkpoint-hf-repo` est
fourni, publication des POIDS (pas seulement les metadonnees) vers un
depot modele HF prive.

Usage (Environnement B avec GPU) :
    uv run python training/E2_04_sft_train.py \
        --recette recipes/sft_qwen3_lora.yaml \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --dataset-formate data/processed/dataset_formate.jsonl \
        --checkpoints data/processed/checkpoints_sft.jsonl \
        --suivi-hf-repo mombasstic/chsa-triage-sft-metrics \
        --checkpoint-hf-repo mombasstic/chsa-triage-sft-lora \
        --assistant-only-loss false

Porte d'entree recommandee avant de lancer ce script pour de vrai :
    uv run python scripts/check_env_gpu.py --model Qwen/Qwen3-1.7B-Base

Trois garde-fous, tous verifies AVANT de charger le modele (evite de
facturer un GPU pour decouvrir le probleme trop tard) :
1. `assistant_only_loss=true` (valeur du YAML) est incompatible avec
   la forme actuelle d'`ExempleFormate` (texte deja rendu, pas
   conversationnel) et fait echouer `trl.SFTTrainer` — refuse de
   demarrer sauf `--assistant-only-loss false` explicite (perte pleine
   sequence).
2. `type_perte=chunked_nll` (valeur du YAML) n'est pas supporte par la
   version de `trl` que resout HF Jobs (l'extra `remote` liste
   `unsloth` sans borne, ce qui plafonne `trl` a une version pre-1.0) —
   la recette est corrigee a `nll`, verifie avant chargement.
3. Sans `--checkpoint-hf-repo`, les poids restent uniquement dans
   `--repertoire-sortie-checkpoints` local, perdus sur un job distant
   (disque ephemere). Bug reel deja rencontre sur ce point precis :
   `--suivi-hf-repo` etait passe en ligne de commande mais la recette
   avait encore `suivi.backend: mlflow`, donc toute la courbe
   d'entrainement du premier run reel a ete perdue (ecrite dans un
   SQLite local detruit avec le conteneur). Corrige par un defaut de
   recette (`hf_dataset`) et un garde-fou qui refuse de demarrer si
   `--suivi-hf-repo` est fourni sans le backend coherent.
"""

from __future__ import annotations

import argparse

import yaml
from huggingface_hub import HfApi

from chsa_triage.application.use_cases import (
    AjusterBoucleHyperparametresSftUseCase,
    EntrainerSftUseCase,
    FormaterDatasetChatMLUseCase,
    SauvegarderCheckpointSftUseCase,
)
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainement,
)
from chsa_triage.domain.model.enums import TypeSplit
from chsa_triage.infrastructure.adapters import (
    ChatMLFormateurAdapter,
    HfDatasetSuiviExperimentation,
    JsonlCheckpointRepository,
    JsonlDatasetRepository,
    JsonlExempleFormateRepository,
    MlflowSuiviExperimentation,
    TensorboardSuiviExperimentation,
    TrlSftEntraineurAdapter,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="E2_04_sft_train")

VALEURS_TYPE_PERTE_VALIDES = frozenset({"nll", "dft"})

CHEMIN_DATASET_FORMATE_DEFAUT = "data/processed/dataset_formate.jsonl"
CHEMIN_CHECKPOINTS_DEFAUT = "data/processed/checkpoints_sft.jsonl"
URI_SUIVI_MLFLOW_DEFAUT = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT = "data/processed/tensorboard_logs"
REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT = "data/processed/suivi_hf_dataset"
REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT = "outputs/sft-lora"


def _parser_bool(valeur: str) -> bool:
    if valeur.strip().lower() in ("true", "1", "oui", "yes"):
        return True
    if valeur.strip().lower() in ("false", "0", "non", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"valeur booleenne attendue (true/false), recu {valeur!r}"
    )


def _charger_recette(chemin: str) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _construire_grille(
    recette_grille: dict, hyperparametres_initiaux: HyperparametresEntrainement
) -> tuple[HyperparametresEntrainement, ...]:
    """Etale l'axe `taux_apprentissage` en une sequence de
    `HyperparametresEntrainement`, les autres champs fixes. L'axe
    `rang` n'est pas etale ici (pilote `ConfigurationLora`, garde fixe
    par `AjusterBoucleHyperparametresSftUseCase` sur toute la boucle)."""
    return tuple(
        HyperparametresEntrainement(
            taux_apprentissage=taux,
            nombre_epoques=hyperparametres_initiaux.nombre_epoques,
            taille_lot=hyperparametres_initiaux.taille_lot,
            packing=hyperparametres_initiaux.packing,
            type_perte=hyperparametres_initiaux.type_perte,
        )
        for taux in recette_grille.get("taux_apprentissage", [])
    )


def _identifiant_checkpoint(modele_base: str, horodatage: str) -> str:
    import hashlib

    return hashlib.sha256(f"{modele_base}:{horodatage}".encode()).hexdigest()[
        :16
    ]


def _publier_checkpoint_hf(chemin_local: str, depot_hf: str) -> None:
    """Publie les poids LoRA du MEILLEUR essai vers un depot modele HF
    prive. Necessaire sur un job HF Jobs distant (disque ephemere,
    sinon le modele entraine serait perdu). Appele une seule fois,
    apres la boucle d'ajustement, sur le checkpoint retenu."""
    api = HfApi()
    api.create_repo(
        repo_id=depot_hf, repo_type="model", private=True, exist_ok=True
    )
    api.upload_folder(
        repo_id=depot_hf, folder_path=chemin_local, repo_type="model"
    )


def _verifier_type_perte_valide(type_perte: str) -> None:
    """Garde-fou avant tout chargement de modele/GPU : `type_perte` doit
    etre une valeur sure (cf. docstring du module, point 2)."""
    if type_perte not in VALEURS_TYPE_PERTE_VALIDES:
        raise SystemExit(
            f"entrainement.type_perte={type_perte!r} n'est pas garanti disponible : seules "
            f"{sorted(VALEURS_TYPE_PERTE_VALIDES)} sont sures avec la resolution de dependances reelle de "
            "ce projet (trl est plafonne par la dependance 'unsloth', non cablee, de l'extra remote, "
            "des qu'une resolution fraiche a lieu comme sur HF Jobs)."
        )


def _verifier_suivi_hf_repo_coherent(
    backend: str, suivi_hf_repo: str | None
) -> None:
    """Garde-fou avant tout chargement de modele/GPU (cf. docstring du
    module, point 3) : si `--suivi-hf-repo` est fourni, la recette doit
    avoir `suivi.backend: hf_dataset`, sinon les metriques sont ecrites
    en local et perdues sur un job distant."""
    if suivi_hf_repo and backend != "hf_dataset":
        raise SystemExit(
            f"--suivi-hf-repo={suivi_hf_repo!r} est fourni mais recette suivi.backend={backend!r} "
            "(pas hf_dataset) : ce depot serait ignore en silence et les metriques ecrites dans un "
            "MLflow/TensorBoard LOCAL, perdu sur tout job HF Jobs distant (le disque du conteneur ne "
            "survit pas au job). Mettez suivi.backend: hf_dataset dans la recette, ou retirez "
            "--suivi-hf-repo pour un run genuinement local."
        )


def _construire_suivi(recette_suivi: dict, arguments: argparse.Namespace):
    backend = recette_suivi.get("backend", "mlflow")
    _verifier_suivi_hf_repo_coherent(backend, arguments.suivi_hf_repo)
    if backend == "mlflow":
        return MlflowSuiviExperimentation(uri_tracking=arguments.suivi_uri)
    if backend == "tensorboard":
        return TensorboardSuiviExperimentation(
            repertoire_logs=arguments.suivi_repertoire
        )
    if backend == "hf_dataset":
        if not arguments.suivi_hf_repo:
            raise SystemExit(
                "suivi.backend=hf_dataset necessite --suivi-hf-repo (ex. mombasstic/chsa-triage-sft-metrics)"
            )
        return HfDatasetSuiviExperimentation(
            repo_id=arguments.suivi_hf_repo,
            repertoire_local=arguments.suivi_hf_repertoire_local,
        )
    raise ValueError(
        f"backend de suivi inconnu dans la recette : {backend!r} (attendu mlflow|tensorboard|hf_dataset)"
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
        "--recette",
        required=True,
        help="Chemin de la recette YAML (ex. recipes/sft_qwen3_lora.yaml)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Chemin du pivot ANONYMISE deja reparti (champ split renseigne)",
    )
    parser.add_argument(
        "--dataset-formate",
        default=CHEMIN_DATASET_FORMATE_DEFAUT,
        help=f"Chemin de sortie du rendu ChatML (defaut {CHEMIN_DATASET_FORMATE_DEFAUT})",
    )
    parser.add_argument(
        "--checkpoints",
        default=CHEMIN_CHECKPOINTS_DEFAUT,
        help=f"Chemin JSONL des metadonnees de checkpoints (defaut {CHEMIN_CHECKPOINTS_DEFAUT})",
    )
    parser.add_argument(
        "--repertoire-sortie-checkpoints",
        default=REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT,
        help=f"Repertoire ou trl/peft ecrivent les poids LoRA (defaut {REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT})",
    )
    parser.add_argument(
        "--suivi-uri",
        default=URI_SUIVI_MLFLOW_DEFAUT,
        help="URI MLflow si suivi.backend=mlflow",
    )
    parser.add_argument(
        "--suivi-repertoire",
        default=REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT,
        help="Repertoire de logs TensorBoard si suivi.backend=tensorboard",
    )
    parser.add_argument(
        "--suivi-hf-repo",
        default=None,
        help="Repo dataset HF (ex. mombasstic/chsa-triage-sft-metrics) si suivi.backend=hf_dataset, "
        "lu en vivo par monitoring/app_suivi_entrainement.py (cf. README)",
    )
    parser.add_argument(
        "--suivi-hf-repertoire-local",
        default=REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT,
        help=f"Repertoire de travail local synchronise vers --suivi-hf-repo (defaut {REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT})",
    )
    parser.add_argument(
        "--assistant-only-loss",
        type=_parser_bool,
        default=None,
        help="Surcharge entrainement.assistant_only_loss de la recette (true/false). "
        "Par defaut : la valeur de la recette, cf. AVERTISSEMENT dans ce module.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="cf. AVERTISSEMENT non-valide, TrlSftEntraineurAdapter",
    )
    parser.add_argument(
        "--liger-kernel",
        action="store_true",
        help="cf. AVERTISSEMENT non-valide, TrlSftEntraineurAdapter",
    )
    parser.add_argument(
        "--checkpoint-hf-repo",
        default=None,
        help="Depot modele HF prive (ex. mombasstic/chsa-triage-sft-lora) ou publier les poids du "
        "MEILLEUR checkpoint LoRA une fois l'entrainement termine (huggingface_hub.upload_folder). "
        "Sans cet argument, les poids restent UNIQUEMENT dans --repertoire-sortie-checkpoints, local : "
        "sur un job HF Jobs distant, dont le disque ne survit pas au job, le modele entraine serait "
        "alors perdu (seules les metriques de suivi survivraient). Cf. README §2.3.",
    )
    arguments = parser.parse_args()

    log.START_ACTION(
        "E2_04_sft_train",
        "main",
        "entrainement SFT-LoRA reel (Environnement B, GPU)",
    )
    log.PARAMETER_VALUE("recette", arguments.recette)
    log.PARAMETER_VALUE("dataset", arguments.dataset)

    recette = _charger_recette(arguments.recette)
    modele_base = recette["modele_base"]

    assistant_only_loss = arguments.assistant_only_loss
    if assistant_only_loss is None:
        assistant_only_loss = recette["entrainement"].get(
            "assistant_only_loss", False
        )

    if assistant_only_loss:
        log.LEVEL_4_ERROR(
            "E2_04_sft_train",
            "assistant_only_loss=true est garanti d'echouer avec la forme actuelle "
            "d'ExempleFormate (texte ChatML deja rendu, pas conversationnel). "
            "Relancez avec --assistant-only-loss false pour un run reel (perte pleine sequence).",
        )
        raise SystemExit(
            "assistant_only_loss=true : incompatibilite connue, abandon avant de charger le "
            "modele (pas de cout GPU inutile)."
        )

    _verifier_type_perte_valide(recette["entrainement"]["type_perte"])
    _verifier_suivi_hf_repo_coherent(
        recette.get("suivi", {}).get("backend", "mlflow"),
        arguments.suivi_hf_repo,
    )

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository_pivot = JsonlDatasetRepository(arguments.dataset)
    repository_formate = JsonlExempleFormateRepository(
        arguments.dataset_formate
    )
    formateur = ChatMLFormateurAdapter(nom_modele=modele_base)

    # -------------------------------------------------------------------------
    # E2_00 : formater ChatML (train, puis validation)
    # -------------------------------------------------------------------------
    log.STEP(
        1,
        "STEP 1 Rendu ChatML",
        "FormaterDatasetChatMLUseCase, train + validation",
    )
    cas_formatage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre_train = cas_formatage.executer(TypeSplit.TRAIN)
    nombre_val = cas_formatage.executer(TypeSplit.VALIDATION)
    log.PARAMETER_VALUE("exemples train formates", nombre_train)
    log.PARAMETER_VALUE("exemples validation formates", nombre_val)

    identifiants_train = {
        e.identifiant
        for e in repository_pivot.lister(filtre={"split": TypeSplit.TRAIN})
    }
    identifiants_val = {
        e.identifiant
        for e in repository_pivot.lister(filtre={"split": TypeSplit.VALIDATION})
    }
    dataset_train = [
        e
        for e in repository_formate.lister()
        if e.identifiant in identifiants_train
    ]
    dataset_validation = [
        e
        for e in repository_formate.lister()
        if e.identifiant in identifiants_val
    ]

    # -------------------------------------------------------------------------
    # Construire config_lora / hyperparametres / grille depuis la recette
    # -------------------------------------------------------------------------
    config_lora = ConfigurationLora(
        rang=recette["lora"]["rang"],
        alpha=recette["lora"]["alpha"],
        dropout=recette["lora"]["dropout"],
        modules_cibles=tuple(recette["lora"]["modules_cibles"]),
    )
    hyperparametres_initiaux = HyperparametresEntrainement(
        taux_apprentissage=recette["entrainement"]["taux_apprentissage"],
        nombre_epoques=recette["entrainement"]["nombre_epoques"],
        taille_lot=recette["entrainement"]["taille_lot"],
        packing=recette["entrainement"]["packing"],
        type_perte=recette["entrainement"]["type_perte"],
    )
    grille = _construire_grille(
        recette.get("grille_hyperparametres", {}), hyperparametres_initiaux
    )

    # -------------------------------------------------------------------------
    # E2_01 (via E2_02) : entrainer, ajuster si necessaire
    # -------------------------------------------------------------------------
    log.STEP(
        2,
        "Chargement modele + construction grille",
        f"{modele_base}, cf. AVERTISSEMENT non-valide dans TrlSftEntraineurAdapter",
    )
    entraineur = TrlSftEntraineurAdapter(
        identifiant_modele_base=modele_base,
        configuration_quantification=ConfigurationQuantification(
            **recette["quantification"]
        ),
        repertoire_sortie=arguments.repertoire_sortie_checkpoints,
        assistant_only_loss=assistant_only_loss,
        attn_implementation=arguments.attn_implementation,
        utiliser_liger_kernel=arguments.liger_kernel,
    )
    suivi = _construire_suivi(recette.get("suivi", {}), arguments)
    cas_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=suivi)

    log.STEP(
        3,
        "STEP 3 Entraînement SFT-LoRA + boucle d'ajustement",
        "AjusterBoucleHyperparametresSftUseCase",
    )
    cas_boucle = AjusterBoucleHyperparametresSftUseCase(
        cas_usage_entrainement=cas_entrainement, grille=grille
    )
    resultat_boucle = cas_boucle.executer(
        dataset_train, dataset_validation, config_lora, hyperparametres_initiaux
    )
    meilleur = resultat_boucle.meilleur_essai
    log.PARAMETER_VALUE("nombre d'essais", len(resultat_boucle.essais))
    log.PARAMETER_VALUE("verdict retenu", meilleur.verdict.value)
    log.PARAMETER_VALUE(
        "checkpoint retenu", meilleur.resultat.chemin_checkpoint
    )

    # -------------------------------------------------------------------------
    # E2_03 : sauvegarder les metadonnees du meilleur checkpoint
    # -------------------------------------------------------------------------
    log.STEP(
        4,
        "STEP 4 Sauvegarde métadonnées checkpoint",
        "SauvegarderCheckpointSftUseCase",
    )
    repository_checkpoints = JsonlCheckpointRepository(arguments.checkpoints)
    cas_checkpoint = SauvegarderCheckpointSftUseCase(
        repository_checkpoints=repository_checkpoints
    )
    checkpoint = cas_checkpoint.executer(
        identifiant=_identifiant_checkpoint(
            modele_base, meilleur.resultat.chemin_checkpoint
        ),
        chemin=meilleur.resultat.chemin_checkpoint,
        modele_base=modele_base,
        configuration_lora=config_lora,
        hyperparametres=meilleur.hyperparametres,
        metriques_finales=meilleur.resultat.courbe_metriques[-1],
        verdict_convergence=meilleur.verdict,
    )

    # -------------------------------------------------------------------------
    # Publication optionnelle des poids sur HF Hub (indispensable sur un job
    # HF Jobs distant, cf. AVERTISSEMENT dans _publier_checkpoint_hf)
    # -------------------------------------------------------------------------
    if arguments.checkpoint_hf_repo:
        log.STEP(
            5,
            "STEP 5 Publication poids checkpoint HF Hub",
            arguments.checkpoint_hf_repo,
        )
        _publier_checkpoint_hf(
            meilleur.resultat.chemin_checkpoint, arguments.checkpoint_hf_repo
        )
        log.PARAMETER_VALUE("poids publies vers", arguments.checkpoint_hf_repo)

    log.FINISH_ACTION(
        "E2_04_sft_train",
        "main",
        f"checkpoint {checkpoint.identifiant} sauvegarde ({checkpoint.verdict_convergence.value})",
    )
    print(f"Checkpoint SFT-LoRA : {checkpoint.chemin}")
    print(f"Verdict de convergence : {checkpoint.verdict_convergence.value}")
    print(f"Metadonnees persistees dans {arguments.checkpoints}")
    if arguments.checkpoint_hf_repo:
        print(f"Poids LoRA publies dans {arguments.checkpoint_hf_repo}")


if __name__ == "__main__":
    main()
