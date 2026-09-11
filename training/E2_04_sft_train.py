"""
Point d'entree, Etape 2 : entrainement SFT-LoRA reel (Environnement B,
GPU requis). Conformement a
`docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md` etape 10.

Vit hors `interfaces/cli/` (dans `training/`, cf. README "scripts
exécutés via HF Jobs" : Etapes 2-3), mais garde le prefixe numerote
`E2_04_` : l'ordre d'execution compte (ce script orchestre les 4 cas
d'usage E2_00 a E2_03), meme si ce n'est pas un script `interfaces/cli/`
au sens strict. Meme patron argparse + `LogTool` + resume console que
`interfaces/cli/E1_04_00_anonymiser_dataset.py`.

**JAMAIS EXECUTE REELLEMENT** (aucun GPU disponible a l'ecriture, cf.
`infrastructure/adapters/trl_sft_entraineur.py` pour le detail de ce
qui a ete verifie sans GPU vs. ce qui reste a confirmer sur une vraie
session GPU). Ce script encadene, DANS L'ORDRE DU DIAGRAMME
(`docs/diagrams/03_etape2_sft/activite/pipeline_sft_lora.puml`) :

  1. `FormaterDatasetChatMLUseCase.executer(split)` (train, puis
     validation) : persiste le rendu ChatML dans `--dataset-formate`.
  2. `AjusterBoucleHyperparametresSftUseCase.executer(...)` : englobe
     le premier essai (`EntrainerSftUseCase`, cas d'usage E2_01) et,
     si non `SAINE`, la boucle d'ajustement d'hyperparametres (E2_02).
  3. `SauvegarderCheckpointSftUseCase.executer(...)` (E2_03) : persiste
     les metadonnees du meilleur essai dans `--checkpoints`.

Usage :
    uv run python training/E2_04_sft_train.py \
        --recette recipes/sft_qwen3_lora.yaml \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --dataset-formate data/processed/dataset_formate.jsonl \
        --checkpoints data/processed/checkpoints_sft.jsonl

Porte d'entree recommandee AVANT de lancer ce script pour de vrai
(`docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md` etape 10,
`01_installation_configuration.md` §5) :
    uv run python scripts/check_env_gpu.py --model Qwen/Qwen3-1.7B-Base

TROUVAILLE REELLE, VERIFIEE SANS GPU (cf. le meme avertissement,
detaille, dans `infrastructure/adapters/trl_sft_entraineur.py`) :
`recipes/sft_qwen3_lora.yaml::entrainement.assistant_only_loss` vaut
`true`, mais cette valeur est GARANTIE DE FAIRE ECHOUER `trl.SFTTrainer`
(`ValueError`, "dataset is not conversational") avec la forme actuelle
d'`ExempleFormate` (texte ChatML deja rendu, pas une liste de
messages structuree). Pour eviter de charger (et facturer) le modele
quantifie avant de decouvrir ce probleme, ce script REFUSE de
demarrer si `--assistant-only-loss` (par defaut : la valeur du YAML)
resout a `True`, avec un message explicite pointant vers l'analyse
complete. Utiliser `--assistant-only-loss false` pour lancer un
premier run reel malgre cette limite connue (perte pleine sequence,
pas seulement sur les tokens assistant).
"""

from __future__ import annotations

import argparse

import yaml

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
    JsonlCheckpointRepository,
    JsonlDatasetRepository,
    JsonlExempleFormateRepository,
    MlflowSuiviExperimentation,
    TensorboardSuiviExperimentation,
    TrlSftEntraineurAdapter,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="E2_04_sft_train")

CHEMIN_DATASET_FORMATE_DEFAUT = "data/processed/dataset_formate.jsonl"
CHEMIN_CHECKPOINTS_DEFAUT     = "data/processed/checkpoints_sft.jsonl"
URI_SUIVI_MLFLOW_DEFAUT       = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT = "data/processed/tensorboard_logs"
REPERTOIRE_CHECKPOINTS_SORTIE_DEFAUT = "outputs/sft-lora"


def _parser_bool(valeur: str) -> bool:
    if valeur.strip().lower() in ("true", "1", "oui", "yes"):
        return True
    if valeur.strip().lower() in ("false", "0", "non", "no"):
        return False
    raise argparse.ArgumentTypeError(f"valeur booleenne attendue (true/false), recu {valeur!r}")


def _charger_recette(chemin: str) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _construire_grille(
    recette_grille: dict, hyperparametres_initiaux: HyperparametresEntrainement
) -> tuple[HyperparametresEntrainement, ...]:
    """
    Etale l'axe `taux_apprentissage` de
    `recette_grille_hyperparametres` en une sequence de
    `HyperparametresEntrainement`, les autres champs restant fixes aux
    valeurs initiales. L'axe `rang` du YAML N'EST PAS etale ici (point
    ouvert deja documente dans `application/grille_hyperparametres.py`
    : `rang` pilote `ConfigurationLora`, pas `HyperparametresEntrainement`,
    et `AjusterBoucleHyperparametresSftUseCase` garde `config_lora` fixe
    sur toute la boucle) : le cablage du rang dans la boucle reste a
    faire, pas invente ici.
    """
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

    return hashlib.sha256(f"{modele_base}:{horodatage}".encode()).hexdigest()[:16]


def _construire_suivi(recette_suivi: dict, arguments: argparse.Namespace):
    backend = recette_suivi.get("backend", "mlflow")
    if backend == "mlflow":
        return MlflowSuiviExperimentation(uri_tracking=arguments.suivi_uri)
    if backend == "tensorboard":
        return TensorboardSuiviExperimentation(repertoire_logs=arguments.suivi_repertoire)
    raise ValueError(f"backend de suivi inconnu dans la recette : {backend!r} (attendu mlflow|tensorboard)")


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recette", required=True, help="Chemin de la recette YAML (ex. recipes/sft_qwen3_lora.yaml)")
    parser.add_argument("--dataset", required=True, help="Chemin du pivot ANONYMISE deja reparti (champ split renseigne)")
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
    parser.add_argument("--suivi-uri", default=URI_SUIVI_MLFLOW_DEFAUT, help="URI MLflow si suivi.backend=mlflow")
    parser.add_argument(
        "--suivi-repertoire",
        default=REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT,
        help="Repertoire de logs TensorBoard si suivi.backend=tensorboard",
    )
    parser.add_argument(
        "--assistant-only-loss",
        type=_parser_bool,
        default=None,
        help="Surcharge entrainement.assistant_only_loss de la recette (true/false). "
             "Par defaut : la valeur de la recette, cf. AVERTISSEMENT dans ce module.",
    )
    parser.add_argument("--attn-implementation", default="sdpa", help="cf. AVERTISSEMENT non-valide, TrlSftEntraineurAdapter")
    parser.add_argument("--liger-kernel", action="store_true", help="cf. AVERTISSEMENT non-valide, TrlSftEntraineurAdapter")
    arguments = parser.parse_args()

    log.START_ACTION("E2_04_sft_train", "main", "entrainement SFT-LoRA reel (Environnement B, GPU)")
    log.PARAMETER_VALUE("recette", arguments.recette)
    log.PARAMETER_VALUE("dataset", arguments.dataset)

    recette = _charger_recette(arguments.recette)
    modele_base = recette["modele_base"]

    assistant_only_loss = arguments.assistant_only_loss
    if assistant_only_loss is None:
        assistant_only_loss = recette["entrainement"].get("assistant_only_loss", False)

    if assistant_only_loss:
        log.LEVEL_4_ERROR(
            "E2_04_sft_train",
            "assistant_only_loss=true est garanti d'echouer avec la forme actuelle "
            "d'ExempleFormate (texte ChatML deja rendu, pas conversationnel) : "
            "voir l'avertissement en tete de infrastructure/adapters/trl_sft_entraineur.py. "
            "Relancez avec --assistant-only-loss false pour un run reel (perte pleine sequence), "
            "ou corrigez d'abord la forme des donnees.",
        )
        raise SystemExit(
            "assistant_only_loss=true : incompatibilite connue et verifiee sans GPU, "
            "abandon avant de charger le modele (pas de cout GPU inutile). "
            "Voir infrastructure/adapters/trl_sft_entraineur.py pour le detail."
        )

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository_pivot   = JsonlDatasetRepository(arguments.dataset)
    repository_formate = JsonlExempleFormateRepository(arguments.dataset_formate)
    formateur            = ChatMLFormateurAdapter(nom_modele=modele_base)

    # -------------------------------------------------------------------------
    # E2_00 : formater ChatML (train, puis validation)
    # -------------------------------------------------------------------------
    log.STEP(1, "Rendu ChatML", "FormaterDatasetChatMLUseCase, train puis validation")
    cas_formatage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot, repository_formate=repository_formate, formateur=formateur
    )
    nombre_train = cas_formatage.executer(TypeSplit.TRAIN)
    nombre_val   = cas_formatage.executer(TypeSplit.VALIDATION)
    log.PARAMETER_VALUE("exemples train formates", nombre_train)
    log.PARAMETER_VALUE("exemples validation formates", nombre_val)

    identifiants_train = {e.identifiant for e in repository_pivot.lister(filtre={"split": TypeSplit.TRAIN})}
    identifiants_val   = {e.identifiant for e in repository_pivot.lister(filtre={"split": TypeSplit.VALIDATION})}
    dataset_train      = [e for e in repository_formate.lister() if e.identifiant in identifiants_train]
    dataset_validation = [e for e in repository_formate.lister() if e.identifiant in identifiants_val]

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
    grille = _construire_grille(recette.get("grille_hyperparametres", {}), hyperparametres_initiaux)

    # -------------------------------------------------------------------------
    # E2_01 (via E2_02) : entrainer, ajuster si necessaire
    # -------------------------------------------------------------------------
    log.STEP(2, "Chargement du modele quantifie", f"{modele_base}, cf. AVERTISSEMENT non-valide dans TrlSftEntraineurAdapter")
    entraineur = TrlSftEntraineurAdapter(
        identifiant_modele_base=modele_base,
        configuration_quantification=ConfigurationQuantification(**recette["quantification"]),
        repertoire_sortie=arguments.repertoire_sortie_checkpoints,
        assistant_only_loss=assistant_only_loss,
        attn_implementation=arguments.attn_implementation,
        utiliser_liger_kernel=arguments.liger_kernel,
    )
    suivi = _construire_suivi(recette.get("suivi", {}), arguments)
    cas_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=suivi)

    log.STEP(3, "Entrainement SFT-LoRA + boucle d'ajustement", "AjusterBoucleHyperparametresSftUseCase")
    cas_boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_entrainement, grille=grille)
    resultat_boucle = cas_boucle.executer(dataset_train, dataset_validation, config_lora, hyperparametres_initiaux)
    meilleur = resultat_boucle.meilleur_essai
    log.PARAMETER_VALUE("nombre d'essais", len(resultat_boucle.essais))
    log.PARAMETER_VALUE("verdict retenu", meilleur.verdict.value)
    log.PARAMETER_VALUE("checkpoint retenu", meilleur.resultat.chemin_checkpoint)

    # -------------------------------------------------------------------------
    # E2_03 : sauvegarder les metadonnees du meilleur checkpoint
    # -------------------------------------------------------------------------
    log.STEP(4, "Sauvegarde des metadonnees du checkpoint", "SauvegarderCheckpointSftUseCase")
    repository_checkpoints = JsonlCheckpointRepository(arguments.checkpoints)
    cas_checkpoint = SauvegarderCheckpointSftUseCase(repository_checkpoints=repository_checkpoints)
    checkpoint = cas_checkpoint.executer(
        identifiant=_identifiant_checkpoint(modele_base, meilleur.resultat.chemin_checkpoint),
        chemin=meilleur.resultat.chemin_checkpoint,
        modele_base=modele_base,
        configuration_lora=config_lora,
        hyperparametres=meilleur.hyperparametres,
        metriques_finales=meilleur.resultat.courbe_metriques[-1],
        verdict_convergence=meilleur.verdict,
    )

    log.FINISH_ACTION("E2_04_sft_train", "main", f"checkpoint {checkpoint.identifiant} sauvegarde ({checkpoint.verdict_convergence.value})")
    print(f"Checkpoint SFT-LoRA : {checkpoint.chemin}")
    print(f"Verdict de convergence : {checkpoint.verdict_convergence.value}")
    print(f"Metadonnees persistees dans {arguments.checkpoints}")


if __name__ == "__main__":
    main()
