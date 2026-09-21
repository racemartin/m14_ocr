"""
Point d'entree, Etape 3 : entrainement DPO reel (Environnement B, GPU
requis), continuant le checkpoint SFT-LoRA deja entraine. Conformement
a `docs/04_etape3_dpo/03_guide_implementation_pas_a_pas.md` etape 13.

Vit hors `interfaces/cli/` (dans `training/`, meme precedent que
`E2_04_sft_train.py`), numerote `E3_03` (pas un `E3_NN_uc_*`, meme
raisonnement de nommage : le prochain numero apres le dernier cas
d'usage reellement ecrit, `E3_02_uc_entrainer_dpo.py`, cf. AGENTS.md
convention `E1_NN`/`E2_NN`/`E3_NN`). Meme patron argparse + `LogTool` +
resume console que `training/E2_04_sft_train.py`. Enchaine, DANS
L'ORDRE :

  1. `ReformulerPreferenceDpoUseCase.executer(...)` (E3_00) : reformule
     `chosen` -> `<think>`+JSON sur le sous-ensemble cible (incremental
     et resumable, no-op si deja complet, cf. AGENTS.md sur le patron
     deja etabli en Etape 1). Candidats limites aux splits
     train+validation (jamais test, cf. cahier des charges §9 "ne
     jamais melanger donnees d'entrainement et d'evaluation").
  2. `FormaterDatasetChatMLPreferenceUseCase.executer(split)` (E3_01),
     train puis validation.
  3. `EntrainerDpoUseCase.entrainer(...)` (E3_02) : delegue a
     `TrlDpoEntraineurAdapter`.
  4. `SauvegarderCheckpointSftUseCase.executer(...)` (E2_03, reutilise
     tel quel pour un checkpoint DPO, cf.
     docs/04_etape3_dpo/02_etapes_cas_usage.md §7).
  5. Si `--checkpoint-hf-repo` est fourni : publie les POIDS du
     checkpoint DPO vers un depot HF prive (reutilise
     `_publier_checkpoint_hf` de `E2_04_sft_train.py` tel quel, meme
     avertissement sur la persistance des poids sur un job HF Jobs
     distant, cf. AGENTS.md).

**ENTRAINEMENT REEL JAMAIS LANCE** (aucun GPU disponible a l'ecriture,
meme statut que `training/E2_04_sft_train.py` avant son premier run
reel) : voir `infrastructure/adapters/trl_dpo_entraineur.py` pour le
detail de ce qui est VERIFIE sans GPU (signatures reelles, lecture du
code source de `trl`/`peft`) contre ce qui reste NON VERIFIE.

Usage (local, Environnement B avec GPU) :
    uv run python training/E3_03_dpo_train.py \\
        --recette recipes/dpo_qwen3_lora.yaml \\
        --dataset data/processed/dataset_pivot_anonymise.jsonl \\
        --dataset-reformule data/processed/dataset_dpo_chosen_reformule.jsonl \\
        --dataset-formate data/processed/dataset_formate_preference.jsonl \\
        --checkpoints data/processed/checkpoints_sft.jsonl \\
        --suivi-hf-repo mombasstic/chsa-triage-dpo-metrics \\
        --checkpoint-hf-repo mombasstic/chsa-triage-dpo-lora

Porte d'entree recommandee AVANT de lancer ce script pour de vrai
(meme discipline que `E2_04_sft_train.py`) :
    uv run python scripts/check_env_gpu.py --model Qwen/Qwen3-1.7B-Base

DEVIATION documentee par rapport au guide d'implementation (etape 13,
"Reutilise telles quelles les deux gardes de demarrage deja reelles
dans training/E2_04_sft_train.py") : `_verifier_suivi_hf_repo_coherent`
EST reutilisee TELLE QUELLE (import direct, aucune modification,
logique 100% generique SFT/DPO). `_verifier_type_perte_valide` en
revanche N'EST PAS reutilisee telle quelle : cette fonction verifie
`entrainement.type_perte` contre `VALEURS_TYPE_PERTE_VALIDES =
{"nll", "dft"}`, le vocabulaire REEL de `trl.SFTConfig.loss_type`
(Etape 2), totalement DIFFERENT du vocabulaire reel de
`trl.DPOConfig.loss_type` (Etape 3 : `"sigmoid"`/`"hinge"`/`"ipo"`,
etc., cf. docs/04_etape3_dpo/00_introduction_concepts.md §5). Appliquer
`_verifier_type_perte_valide` telle quelle rejetterait a tort
`type_perte: sigmoid` (absent de `{"nll", "dft"}`), la valeur CORRECTE
et deliberement choisie par `recipes/dpo_qwen3_lora.yaml`. Ce module
definit donc `_verifier_type_perte_dpo_valide`, meme PATRON de garde
(refuse de demarrer avant tout chargement GPU, meme style de message),
mais un ensemble de valeurs sures propre au DPO : verifie reellement
(installation temporaire de trl==1.13.0 ET trl==0.24.0, meme methode
que le reste du projet, desinstalle ensuite) que `"sigmoid"` est
accepte par `trl.DPOConfig` sur LES DEUX versions en jeu (1.13.0 en
local via `uv.lock`, 0.24.0 sur une resolution fraiche HF Jobs a cause
de la meme contrainte `unsloth` non cablee documentee pour le SFT, cf.
AGENTS.md) : le meme risque de plafonnement existe pour le DPO, mais la
valeur `sigmoid` s'avere sure sur les deux versions, contrairement a
`chunked_nll` pour le SFT. Seule `"sigmoid"` est admise ici (la seule
valeur reellement utilisee par ce projet, cf. la recette) : pas
`"hinge"`/`"ipo"` (jamais verifies, jamais utilises).
"""

from __future__ import annotations

import argparse
import itertools

import yaml

from chsa_triage.application.use_cases import (
    EntrainerDpoUseCase,
    FormaterDatasetChatMLPreferenceUseCase,
    ReformulerPreferenceDpoUseCase,
    SauvegarderCheckpointSftUseCase,
)
from chsa_triage.application.verdict_convergence import evaluer_convergence
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.enums import TypeExemple, TypeSplit
from chsa_triage.infrastructure.adapters import (
    ChatMLFormateurAdapter,
    HfDatasetSuiviExperimentation,
    JsonlCheckpointRepository,
    JsonlDatasetRepository,
    JsonlExempleFormatePreferenceRepository,
    JsonlPreferenceReformuleeRepository,
    MlflowSuiviExperimentation,
    TensorboardSuiviExperimentation,
    TransformersLoraInferenceAdapter,
    TrlDpoEntraineurAdapter,
)
from tools.rafael.log_tool import LogTool
from training.E2_04_sft_train import (
    _publier_checkpoint_hf,
    _verifier_suivi_hf_repo_coherent,
)

log = LogTool(origin="E3_03_dpo_train")

# cf. DEVIATION documentee en tete de module : vocabulaire DPO, distinct
# de VALEURS_TYPE_PERTE_VALIDES (SFT) de training/E2_04_sft_train.py.
VALEURS_TYPE_PERTE_DPO_VALIDES = frozenset({"sigmoid"})

CHEMIN_DATASET_REFORMULE_DEFAUT      = "data/processed/dataset_dpo_chosen_reformule.jsonl"
CHEMIN_DATASET_FORMATE_DEFAUT        = "data/processed/dataset_formate_preference.jsonl"
CHEMIN_CHECKPOINTS_DEFAUT            = "data/processed/checkpoints_sft.jsonl"
URI_SUIVI_MLFLOW_DEFAUT              = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT  = "data/processed/tensorboard_logs"
REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT     = "data/processed/suivi_hf_dataset"
REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT = "outputs/dpo-lora"


def _charger_recette(chemin: str) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _identifiant_checkpoint(modele_base: str, horodatage: str) -> str:
    import hashlib

    return hashlib.sha256(f"dpo:{modele_base}:{horodatage}".encode()).hexdigest()[:16]


def _verifier_type_perte_dpo_valide(type_perte: str) -> None:
    """
    Garde-fou AVANT tout chargement de modele/GPU, meme PATRON que
    `training.E2_04_sft_train._verifier_type_perte_valide`, mais un
    ensemble de valeurs sures PROPRE AU DPO (cf. DEVIATION documentee
    en tete de ce module : le vocabulaire de `trl.DPOConfig.loss_type`
    n'a aucun rapport avec celui de `trl.SFTConfig.loss_type`).
    """
    if type_perte not in VALEURS_TYPE_PERTE_DPO_VALIDES:
        raise SystemExit(
            f"entrainement.type_perte={type_perte!r} n'est pas garanti disponible/valide pour le DPO : "
            f"seules {sorted(VALEURS_TYPE_PERTE_DPO_VALIDES)} sont verifiees sures par ce projet "
            "(trl.DPOConfig.loss_type, pas le meme vocabulaire que trl.SFTConfig.loss_type de l'Etape 2). "
            "Voir infrastructure/adapters/trl_dpo_entraineur.py et l'en-tete de ce module pour le detail."
        )


def _construire_suivi(recette_suivi: dict, arguments: argparse.Namespace):
    backend = recette_suivi.get("backend", "mlflow")
    _verifier_suivi_hf_repo_coherent(backend, arguments.suivi_hf_repo)
    if backend == "mlflow":
        return MlflowSuiviExperimentation(uri_tracking=arguments.suivi_uri)
    if backend == "tensorboard":
        return TensorboardSuiviExperimentation(repertoire_logs=arguments.suivi_repertoire)
    if backend == "hf_dataset":
        if not arguments.suivi_hf_repo:
            raise SystemExit("suivi.backend=hf_dataset necessite --suivi-hf-repo (ex. mombasstic/chsa-triage-dpo-metrics)")
        return HfDatasetSuiviExperimentation(
            repo_id=arguments.suivi_hf_repo,
            repertoire_local=arguments.suivi_hf_repertoire_local,
        )
    raise ValueError(f"backend de suivi inconnu dans la recette : {backend!r} (attendu mlflow|tensorboard|hf_dataset)")


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recette", required=True, help="Chemin de la recette YAML (ex. recipes/dpo_qwen3_lora.yaml)")
    parser.add_argument("--dataset", required=True, help="Chemin du pivot ANONYMISE deja reparti (champ split renseigne)")
    parser.add_argument(
        "--dataset-reformule",
        default=CHEMIN_DATASET_REFORMULE_DEFAUT,
        help=f"Chemin de sortie des ChosenReformule (defaut {CHEMIN_DATASET_REFORMULE_DEFAUT})",
    )
    parser.add_argument(
        "--dataset-formate",
        default=CHEMIN_DATASET_FORMATE_DEFAUT,
        help=f"Chemin de sortie du rendu triplet prompt/chosen/rejected (defaut {CHEMIN_DATASET_FORMATE_DEFAUT})",
    )
    parser.add_argument(
        "--checkpoints",
        default=CHEMIN_CHECKPOINTS_DEFAUT,
        help=f"Chemin JSONL des metadonnees de checkpoints, PARTAGE avec le SFT (defaut {CHEMIN_CHECKPOINTS_DEFAUT})",
    )
    parser.add_argument(
        "--repertoire-sortie-checkpoints",
        default=REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT,
        help=f"Repertoire ou trl/peft ecrivent les poids LoRA (defaut {REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT})",
    )
    parser.add_argument("--suivi-uri", default=URI_SUIVI_MLFLOW_DEFAUT, help="URI MLflow si suivi.backend=mlflow")
    parser.add_argument(
        "--suivi-repertoire",
        default=REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT,
        help="Repertoire de logs TensorBoard si suivi.backend=tensorboard",
    )
    parser.add_argument(
        "--suivi-hf-repo",
        default=None,
        help="Repo dataset HF (ex. mombasstic/chsa-triage-dpo-metrics) si suivi.backend=hf_dataset, "
             "lu en vivo par monitoring/app_suivi_entrainement.py (cf. README)",
    )
    parser.add_argument(
        "--suivi-hf-repertoire-local",
        default=REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT,
        help=f"Repertoire de travail local synchronise vers --suivi-hf-repo (defaut {REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT})",
    )
    parser.add_argument(
        "--checkpoint-hf-repo",
        default=None,
        help="Depot modele HF prive (ex. mombasstic/chsa-triage-dpo-lora) ou publier les poids du "
             "checkpoint DPO une fois l'entrainement termine (huggingface_hub.upload_folder). Sans cet "
             "argument, les poids restent UNIQUEMENT dans --repertoire-sortie-checkpoints, local : sur un "
             "job HF Jobs distant, dont le disque ne survit pas au job, le modele entraine serait alors "
             "perdu. Cf. AGENTS.md (meme risque deja documente pour le SFT).",
    )
    arguments = parser.parse_args()

    log.START_ACTION("E3_03_dpo_train", "main", "entrainement DPO reel (Environnement B, GPU)")
    log.PARAMETER_VALUE("recette", arguments.recette)
    log.PARAMETER_VALUE("dataset", arguments.dataset)

    recette = _charger_recette(arguments.recette)
    modele_base = recette["modele_base"]
    checkpoint_politique_depart = recette["checkpoint_politique_depart"]

    _verifier_type_perte_dpo_valide(recette["entrainement"]["type_perte"])
    _verifier_suivi_hf_repo_coherent(recette.get("suivi", {}).get("backend", "mlflow"), arguments.suivi_hf_repo)

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository_pivot     = JsonlDatasetRepository(arguments.dataset)
    repository_reformule = JsonlPreferenceReformuleeRepository(arguments.dataset_reformule)
    repository_formate   = JsonlExempleFormatePreferenceRepository(arguments.dataset_formate)
    formateur              = ChatMLFormateurAdapter(nom_modele=modele_base)

    # -------------------------------------------------------------------------
    # E3_00 : reformuler chosen -> <think>+JSON, sous-ensemble train+validation
    # UNIQUEMENT (jamais test, cf. cahier des charges §9). No-op si le
    # sous-ensemble cible (recette reformulation.taille_cible) est deja complet.
    # -------------------------------------------------------------------------
    log.STEP(1, "Reformulation chosen -> <think>+JSON", "ReformulerPreferenceDpoUseCase")
    moteur_reformulation = TransformersLoraInferenceAdapter(
        depot_lora=checkpoint_politique_depart, nom_modele_base=modele_base
    )
    cas_reformulation = ReformulerPreferenceDpoUseCase(
        moteur=moteur_reformulation,
        repository_reformule=repository_reformule,
        taille_cible=recette.get("reformulation", {}).get("taille_cible", 5000),
    )
    candidats_reformulation = itertools.chain(
        repository_pivot.lister(filtre={"split": TypeSplit.TRAIN, "type_exemple": TypeExemple.DPO}),
        repository_pivot.lister(filtre={"split": TypeSplit.VALIDATION, "type_exemple": TypeExemple.DPO}),
    )
    nombre_reformules = cas_reformulation.executer(candidats_reformulation)
    log.PARAMETER_VALUE("exemples reformules (cette execution)", nombre_reformules)
    log.PARAMETER_VALUE("echecs de reformulation (cette execution)", cas_reformulation.nombre_echecs_reformulation)
    for index, texte_echec in enumerate(cas_reformulation.echantillon_echecs_reformulation):
        log.PARAMETER_VALUE(f"echantillon echec reformulation [{index}]", texte_echec)

    # -------------------------------------------------------------------------
    # E3_01 : fusionner pivot + ChosenReformule, rendre le triplet texte
    # -------------------------------------------------------------------------
    log.STEP(2, "Rendu triplet prompt/chosen/rejected", "FormaterDatasetChatMLPreferenceUseCase, train puis validation")
    cas_formatage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre_train = cas_formatage.executer(TypeSplit.TRAIN)
    nombre_val   = cas_formatage.executer(TypeSplit.VALIDATION)
    log.PARAMETER_VALUE("exemples train formates", nombre_train)
    log.PARAMETER_VALUE("exemples validation formates", nombre_val)

    identifiants_train = {
        e.identifiant
        for e in repository_pivot.lister(filtre={"split": TypeSplit.TRAIN, "type_exemple": TypeExemple.DPO})
    }
    identifiants_val = {
        e.identifiant
        for e in repository_pivot.lister(filtre={"split": TypeSplit.VALIDATION, "type_exemple": TypeExemple.DPO})
    }
    dataset_train      = [e for e in repository_formate.lister() if e.identifiant in identifiants_train]
    dataset_validation = [e for e in repository_formate.lister() if e.identifiant in identifiants_val]

    if not dataset_train:
        raise SystemExit(
            "aucun exemple train forme (dataset_train vide) : verifier que la reformulation (etape 1) a "
            "produit des ChosenReformule pour au moins un identifiant du split train, avant de charger le "
            "modele GPU."
        )

    # -------------------------------------------------------------------------
    # Construire config_lora / hyperparametres depuis la recette
    # -------------------------------------------------------------------------
    config_lora = ConfigurationLora(
        rang=recette["lora"]["rang"],
        alpha=recette["lora"]["alpha"],
        dropout=recette["lora"]["dropout"],
        modules_cibles=tuple(recette["lora"]["modules_cibles"]),
    )
    hyperparametres = HyperparametresEntrainementDpo(**recette["entrainement"])

    # -------------------------------------------------------------------------
    # E3_02 : entrainer (continuation du checkpoint SFT-LoRA)
    # -------------------------------------------------------------------------
    log.STEP(3, "Chargement du modele quantifie + checkpoint SFT-LoRA", f"{modele_base} + {checkpoint_politique_depart}")
    entraineur = TrlDpoEntraineurAdapter(
        identifiant_modele_base=modele_base,
        configuration_quantification=ConfigurationQuantification(**recette["quantification"]),
        chemin_checkpoint_politique_depart=checkpoint_politique_depart,
        repertoire_sortie=arguments.repertoire_sortie_checkpoints,
    )
    suivi = _construire_suivi(recette.get("suivi", {}), arguments)
    cas_entrainement = EntrainerDpoUseCase(entraineur=entraineur, suivi=suivi)

    log.STEP(4, "Entrainement DPO", "EntrainerDpoUseCase")
    resultat = cas_entrainement.entrainer(
        dataset_train, dataset_validation, config_lora, hyperparametres, checkpoint_politique_depart
    )
    verdict = evaluer_convergence(resultat.courbe_metriques)
    log.PARAMETER_VALUE("verdict de convergence", verdict.value)
    log.PARAMETER_VALUE("checkpoint DPO", resultat.chemin_checkpoint)

    # -------------------------------------------------------------------------
    # E2_03 (reutilise tel quel) : sauvegarder les metadonnees du checkpoint DPO
    # -------------------------------------------------------------------------
    log.STEP(5, "Sauvegarde des metadonnees du checkpoint", "SauvegarderCheckpointSftUseCase")
    repository_checkpoints = JsonlCheckpointRepository(arguments.checkpoints)
    cas_checkpoint = SauvegarderCheckpointSftUseCase(repository_checkpoints=repository_checkpoints)
    checkpoint = cas_checkpoint.executer(
        identifiant=_identifiant_checkpoint(modele_base, resultat.chemin_checkpoint),
        chemin=resultat.chemin_checkpoint,
        modele_base=modele_base,
        configuration_lora=config_lora,
        hyperparametres=hyperparametres,
        metriques_finales=resultat.courbe_metriques[-1],
        verdict_convergence=verdict,
    )

    # -------------------------------------------------------------------------
    # Publication optionnelle des poids sur HF Hub
    # -------------------------------------------------------------------------
    if arguments.checkpoint_hf_repo:
        log.STEP(6, "Publication des poids du checkpoint DPO sur HF Hub", arguments.checkpoint_hf_repo)
        _publier_checkpoint_hf(resultat.chemin_checkpoint, arguments.checkpoint_hf_repo)
        log.PARAMETER_VALUE("poids publies vers", arguments.checkpoint_hf_repo)

    log.FINISH_ACTION("E3_03_dpo_train", "main", f"checkpoint {checkpoint.identifiant} sauvegarde ({checkpoint.verdict_convergence.value})")
    print(f"Checkpoint DPO-LoRA : {checkpoint.chemin}")
    print(f"Verdict de convergence : {checkpoint.verdict_convergence.value}")
    print(f"Metriques de recompense : {resultat.metriques_recompense}")
    print(f"Metadonnees persistees dans {arguments.checkpoints}")
    if arguments.checkpoint_hf_repo:
        print(f"Poids LoRA publies dans {arguments.checkpoint_hf_repo}")


if __name__ == "__main__":
    main()
