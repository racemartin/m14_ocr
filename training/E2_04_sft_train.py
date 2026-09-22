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

**ENTRAINEMENT REEL COMPLET JAMAIS LANCE** (couterait une session GPU
payante pour un resultat deja partiellement connu par construction, cf.
README §2.3). Le CABLAGE de bout en bout (guard `assistant_only_loss`,
resolution/installation du paquet `chsa-triage[remote]`, acces au
`--recette`/`--dataset` sur un job distant) A ETE verifie pour de vrai
sur HF Jobs reel (`--flavor cpu-basic`, guard atteint et declenche,
cf. README §2.3) ; cf. `infrastructure/adapters/trl_sft_entraineur.py`
pour le detail de ce qui reste NON verifie (le training loop lui-meme,
faute de GPU). Ce script encadene, DANS L'ORDRE DU DIAGRAMME
(`docs/diagrams/03_etape2_sft/activite/pipeline_sft_lora.puml`) :

  1. `FormaterDatasetChatMLUseCase.executer(split)` (train, puis
     validation) : persiste le rendu ChatML dans `--dataset-formate`.
  2. `AjusterBoucleHyperparametresSftUseCase.executer(...)` : englobe
     le premier essai (`EntrainerSftUseCase`, cas d'usage E2_01) et,
     si non `SAINE`, la boucle d'ajustement d'hyperparametres (E2_02).
  3. `SauvegarderCheckpointSftUseCase.executer(...)` (E2_03) : persiste
     les metadonnees du meilleur essai dans `--checkpoints`.
  4. Si `--checkpoint-hf-repo` est fourni : publie les POIDS (pas
     seulement les metadonnees) du meilleur essai vers un depot modele
     HF prive, cf. AVERTISSEMENT ci-dessous.

Usage (local, Environnement B avec GPU) :
    uv run python training/E2_04_sft_train.py \
        --recette recipes/sft_qwen3_lora.yaml \
        --dataset data/processed/dataset_pivot_anonymise.jsonl \
        --dataset-formate data/processed/dataset_formate.jsonl \
        --checkpoints data/processed/checkpoints_sft.jsonl \
        --suivi-hf-repo mombasstic/chsa-triage-sft-metrics \
        --checkpoint-hf-repo mombasstic/chsa-triage-sft-lora \
        --assistant-only-loss false

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

AVERTISSEMENT CRITIQUE, `type_perte` (trouve sur un vrai job GPU L4
facture, echoue le 16/09/2026, cf. AGENTS.md) : `recipes/
sft_qwen3_lora.yaml::entrainement.type_perte` valait `chunked_nll`,
qui EXISTE bien dans trl>=1.12, mais la dependance `unsloth` de
l'extra `remote` (non cablee dans le code) plafonne `trl` a 0.24.0 des
qu'une resolution fraiche a lieu comme sur HF Jobs (`hf jobs uv run`
n'utilise pas `uv.lock`), et ce trl 0.24.0 ne connait pas
`chunked_nll`. La recette est corrigee a `nll` ; ce script verifie en
plus `type_perte` (`_verifier_type_perte_valide`) AVANT de charger le
modele, meme patron que le guard `assistant_only_loss` ci-dessus. Voir
l'AVERTISSEMENT complet dans `infrastructure/adapters/
trl_sft_entraineur.py`.

AVERTISSEMENT CRITIQUE, PERSISTANCE DES POIDS (trouve et corrige le
16/09/2026, cf. AGENTS.md) : `TrlSftEntraineurAdapter.entrainer()`
ecrit les poids LoRA UNIQUEMENT en local (`trainer.save_model()` sous
`--repertoire-sortie-checkpoints`, defaut `outputs/sft-lora`) ;
`SauvegarderCheckpointSftUseCase` (E2_03) ne persiste que des
METADONNEES (chemin, hyperparametres, verdict), jamais les poids
eux-memes. Sur un job HF Jobs distant, dont le disque ne survit PAS au
job (meme fait deja documente pour la baseline GPU,
`interfaces/cli/E1_06_01_evaluer_baseline_gpu.py`), lancer ce script
SANS `--checkpoint-hf-repo` termine un entrainement reel facture en
perdant le modele entraine lui-meme : seules les metriques de suivi
(`--suivi-hf-repo`) survivraient. `--checkpoint-hf-repo` (ex.
`mombasstic/chsa-triage-sft-lora`) publie les poids du MEILLEUR essai
(jamais les essais intermediaires rejetes de la grille) vers un depot
modele HF prive via `huggingface_hub.upload_folder`, une fois la
boucle d'ajustement terminee. `trl.SFTConfig`/`transformers.
TrainingArguments` supportent nativement `push_to_hub`/`hub_model_id`
(verifie reellement, sans GPU, par inspection de signature), mais
brancher ce flag directement dans `TrlSftEntraineurAdapter.entrainer()`
publierait CHAQUE essai de la grille, pas seulement le meilleur : d'ou
le choix de publier explicitement ICI, apres selection du meilleur
essai, plutot que dans l'adaptateur. NON VERIFIE : la publication d'un
checkpoint REEL (poids produits par un vrai entrainement), faute de
GPU ici ; VERIFIE : la resolution des chemins/arguments et que
`HfApi.create_repo`/`upload_folder` sont les bons appels (signatures
reelles inspectees), cf. `tests/training/test_E2_04_sft_train.py`.

AVERTISSEMENT CRITIQUE, PERSISTANCE DU SUIVI (trouve et corrige le
16/09/2026, cf. AGENTS.md) : l'affirmation ci-dessus ("seules les
metriques de suivi survivraient" sans `--checkpoint-hf-repo`) etait
FAUSSE en pratique jusqu'a ce correctif. Le premier entrainement
SFT-LoRA complet reellement lance sur HF Jobs (GPU L4, ~20 min, 342
pas, verdict "saine", poids publies avec succes sur
`mombasstic/chsa-triage-sft-lora`) a malgre tout perdu TOUTE sa courbe
d'entrainement (perte train/validation, norme gradient) : `--suivi-hf-repo`
etait bien passe sur la ligne de commande, mais `recipes/
sft_qwen3_lora.yaml::suivi.backend` valait encore `mlflow` a ce
moment-la, et `_construire_suivi()` ci-dessous ignorait SILENCIEUSEMENT
`--suivi-hf-repo` tant que `backend != hf_dataset` : les metriques ont
ete ecrites dans un SQLite LOCAL (`data/processed/mlflow.db` par
defaut) a l'interieur du conteneur ephemere du job, detruit avec lui.
Corrige par deux moyens complementaires : (1) `recipes/
sft_qwen3_lora.yaml::suivi.backend` vaut maintenant `hf_dataset` par
defaut (`--suivi-hf-repo` fonctionne desormais sans edition manuelle de
la recette) ; (2) `_verifier_suivi_hf_repo_coherent()` ci-dessous REFUSE
de demarrer (avant tout chargement de modele/GPU, meme patron que le
guard `assistant_only_loss`/`type_perte`) si `--suivi-hf-repo` est
fourni alors que `suivi.backend != hf_dataset`, pour que ce flag ne
soit plus JAMAIS ignore en silence, meme si la recette est de nouveau
modifiee a l'avenir. Voir `tests/domain/test_configuration_entrainement.py`
(regression sur la valeur par defaut de la recette) et
`tests/training/test_E2_04_sft_train.py` (le guard lui-meme), tous deux
sans GPU ni reseau reel.
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
CHEMIN_CHECKPOINTS_DEFAUT     = "data/processed/checkpoints_sft.jsonl"
URI_SUIVI_MLFLOW_DEFAUT       = "sqlite:///data/processed/mlflow.db"
REPERTOIRE_SUIVI_TENSORBOARD_DEFAUT = "data/processed/tensorboard_logs"
REPERTOIRE_SUIVI_HF_LOCAL_DEFAUT    = "data/processed/suivi_hf_dataset"
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


def _publier_checkpoint_hf(chemin_local: str, depot_hf: str) -> None:
    """
    Publie les poids LoRA du MEILLEUR essai (deja ecrits localement par
    `trainer.save_model()`, cf. `TrlSftEntraineurAdapter.entrainer`) vers
    un depot modele HF prive. Necessaire sur un job HF Jobs distant : le
    disque de ces jobs ne survit pas au job (meme raison que
    `--suivi-hf-repo`, cf. `HfDatasetSuiviExperimentation`), donc sans
    cette etape un entrainement reel facture perdrait le modele
    entraine lui-meme, seules les metriques survivraient. Appele UNE
    SEULE FOIS, apres la boucle d'ajustement d'hyperparametres, sur le
    checkpoint retenu (`resultat_boucle.meilleur_essai`) : les essais
    intermediaires rejetes ne sont jamais publies.
    """
    api = HfApi()
    api.create_repo(repo_id=depot_hf, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=depot_hf, folder_path=chemin_local, repo_type="model")


def _verifier_type_perte_valide(type_perte: str) -> None:
    """
    Garde-fou AVANT tout chargement de modele/GPU : `type_perte` doit
    etre une valeur de `loss_type` sure pour ce projet. `trl` accepte
    bien `'chunked_nll'` depuis trl>=1.12 (verifie source, cf.
    AVERTISSEMENT dans `infrastructure/adapters/trl_sft_entraineur.py`),
    mais la dependance `unsloth` de l'extra `remote` (non cablee dans
    le code, jamais utilisee) impose `trl<=0.24.0` des qu'une
    resolution FRAICHE a lieu (`hf jobs uv run --with "chsa-triage[remote]
    @ git+..."` ne reutilise PAS `uv.lock`, contrairement a `uv sync
    --extra remote` en local) : ce trl 0.24.0 reellement installe par
    le job ne connait pas encore `'chunked_nll'`. Trouvaille reelle, sur
    un job GPU L4 de pay reel qui a echoue avec exactement cette erreur
    le 16/09/2026 (cf. AGENTS.md, meme AVERTISSEMENT).
    """
    if type_perte not in VALEURS_TYPE_PERTE_VALIDES:
        raise SystemExit(
            f"entrainement.type_perte={type_perte!r} n'est pas garanti disponible : seules "
            f"{sorted(VALEURS_TYPE_PERTE_VALIDES)} sont sures avec la resolution de dependances reelle de "
            "ce projet (trl est plafonne a 0.24.0 par la dependance 'unsloth', non cablee, de l'extra "
            "remote, des qu'une resolution fraiche a lieu comme sur HF Jobs). Voir "
            "infrastructure/adapters/trl_sft_entraineur.py et AGENTS.md pour le detail reel, verifie sur "
            "un job GPU facture qui a echoue avec cette erreur exacte."
        )


def _verifier_suivi_hf_repo_coherent(backend: str, suivi_hf_repo: str | None) -> None:
    """
    Garde-fou AVANT tout chargement de modele/GPU : si `--suivi-hf-repo`
    est fourni (signal explicite que l'appelant attend une persistance
    DISTANTE et durable de la courbe d'entrainement), la recette DOIT
    avoir `suivi.backend: hf_dataset`, sinon `_construire_suivi` ignore
    silencieusement ce depot et ecrit dans un MLflow/TensorBoard LOCAL a
    la place. Trouvaille reelle : c'est exactement ce qui s'est produit
    sur le premier entrainement SFT-LoRA complet lance pour de vrai sur
    HF Jobs (GPU L4, ~20 min, 342 pas, verdict "saine", poids publies
    avec succes) : `--suivi-hf-repo` etait bien passe sur la ligne de
    commande, mais `suivi.backend` valait encore `mlflow` dans la
    recette, donc les metriques (perte train/validation, norme
    gradient) ont ete ecrites dans le SQLite local du conteneur HF
    Jobs, qui ne survit pas au job : la courbe entiere est perdue,
    irrecuperable (cf. AGENTS.md). Meme patron que le guard
    `_verifier_type_perte_valide` ci-dessus.
    """
    if suivi_hf_repo and backend != "hf_dataset":
        raise SystemExit(
            f"--suivi-hf-repo={suivi_hf_repo!r} est fourni mais recette suivi.backend={backend!r} "
            "(pas hf_dataset) : ce depot serait ignore en silence et les metriques ecrites dans un "
            "MLflow/TensorBoard LOCAL, perdu sur tout job HF Jobs distant (le disque du conteneur ne "
            "survit pas au job). Mettez suivi.backend: hf_dataset dans la recette, ou retirez "
            "--suivi-hf-repo pour un run genuinement local. Voir AGENTS.md : un entrainement reel facture "
            "a deja perdu sa courbe complete exactement de cette facon."
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
            raise SystemExit("suivi.backend=hf_dataset necessite --suivi-hf-repo (ex. mombasstic/chsa-triage-sft-metrics)")
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
    parser.add_argument("--attn-implementation", default="sdpa", help="cf. AVERTISSEMENT non-valide, TrlSftEntraineurAdapter")
    parser.add_argument("--liger-kernel", action="store_true", help="cf. AVERTISSEMENT non-valide, TrlSftEntraineurAdapter")
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

    _verifier_type_perte_valide(recette["entrainement"]["type_perte"])
    _verifier_suivi_hf_repo_coherent(recette.get("suivi", {}).get("backend", "mlflow"), arguments.suivi_hf_repo)

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS (Dependency Injection)
    # -------------------------------------------------------------------------
    repository_pivot   = JsonlDatasetRepository(arguments.dataset)
    repository_formate = JsonlExempleFormateRepository(arguments.dataset_formate)
    formateur            = ChatMLFormateurAdapter(nom_modele=modele_base)

    # -------------------------------------------------------------------------
    # E2_00 : formater ChatML (train, puis validation)
    # -------------------------------------------------------------------------
    log.STEP(1, "STEP 1 Rendu ChatML", "FormaterDatasetChatMLUseCase, train + validation")
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
    log.STEP(2, "Chargement modele + construction grille", f"{modele_base}, cf. AVERTISSEMENT non-valide dans TrlSftEntraineurAdapter")
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

    log.STEP(3, "STEP 3 Entraînement SFT-LoRA + boucle d'ajustement", "AjusterBoucleHyperparametresSftUseCase")
    cas_boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_entrainement, grille=grille)
    resultat_boucle = cas_boucle.executer(dataset_train, dataset_validation, config_lora, hyperparametres_initiaux)
    meilleur = resultat_boucle.meilleur_essai
    log.PARAMETER_VALUE("nombre d'essais", len(resultat_boucle.essais))
    log.PARAMETER_VALUE("verdict retenu", meilleur.verdict.value)
    log.PARAMETER_VALUE("checkpoint retenu", meilleur.resultat.chemin_checkpoint)

    # -------------------------------------------------------------------------
    # E2_03 : sauvegarder les metadonnees du meilleur checkpoint
    # -------------------------------------------------------------------------
    log.STEP(4, "STEP 4 Sauvegarde métadonnées checkpoint", "SauvegarderCheckpointSftUseCase")
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

    # -------------------------------------------------------------------------
    # Publication optionnelle des poids sur HF Hub (indispensable sur un job
    # HF Jobs distant, cf. AVERTISSEMENT dans _publier_checkpoint_hf)
    # -------------------------------------------------------------------------
    if arguments.checkpoint_hf_repo:
        log.STEP(5, "STEP 5 Publication poids checkpoint HF Hub", arguments.checkpoint_hf_repo)
        _publier_checkpoint_hf(meilleur.resultat.chemin_checkpoint, arguments.checkpoint_hf_repo)
        log.PARAMETER_VALUE("poids publies vers", arguments.checkpoint_hf_repo)

    log.FINISH_ACTION("E2_04_sft_train", "main", f"checkpoint {checkpoint.identifiant} sauvegarde ({checkpoint.verdict_convergence.value})")
    print(f"Checkpoint SFT-LoRA : {checkpoint.chemin}")
    print(f"Verdict de convergence : {checkpoint.verdict_convergence.value}")
    print(f"Metadonnees persistees dans {arguments.checkpoints}")
    if arguments.checkpoint_hf_repo:
        print(f"Poids LoRA publies dans {arguments.checkpoint_hf_repo}")


if __name__ == "__main__":
    main()
