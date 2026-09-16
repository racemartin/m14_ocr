"""
Adaptateur secondaire (Environnement B, GPU requis) : entrainement
SFT-LoRA reel via `trl.SFTTrainer` + `peft.LoraConfig`, conformement a
`docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md` etape 9.

Implemente le port `EntraineurSupervise`. Seul module du projet qui
importe `trl`/`peft`/`bitsandbytes` (regle "seule
infrastructure/adapters importe des bibliotheques externes",
`docs/01_environnement/01_architecture_hexagonale.md` §2) ; ces
imports sont fonction-scoped (dans `__init__`/`entrainer`), pas au
niveau du module, meme discipline que `ChatMLFormateurAdapter`/
`MlflowSuiviExperimentation` : ce fichier reste importable en
Environnement A (`peft`/`trl`/`bitsandbytes` absents) tant qu'on
n'instancie pas `TrlSftEntraineurAdapter` pour de vrai.

AVERTISSEMENT CRITIQUE (ecrit le 11/09/2026, jamais execute reellement) :
ce module n'a JAMAIS tourne sur un GPU, faute d'Environnement B au
moment de l'ecriture. Tout ce qui suit distingue explicitement ce qui
a ete VERIFIE sans GPU (signatures reelles, comportement de code lu,
appels de fonctions pures) de ce qui reste NON VERIFIE et attend une
vraie session GPU reelle.

VERIFIE SANS GPU (peft==0.20.0, trl==1.13.0, installes temporairement
dans le venv local via `uv pip install`, PAS ajoutes en dependance
permanente d'Environnement A ; desinstalles apres verification) :
  - `peft.LoraConfig(r=, lora_alpha=, lora_dropout=, target_modules=,
    task_type=peft.TaskType.CAUSAL_LM)` : signature reelle inspectee
    (`inspect.signature`), tous les noms de champs utilises ci-dessous
    existent bien.
  - `trl.SFTConfig(...)` accepte bien `output_dir`, `learning_rate`,
    `num_train_epochs`, `per_device_train_batch_size`, `packing`,
    `loss_type`, `assistant_only_loss`, `use_liger_kernel`,
    `eval_strategy`, `report_to` : champs reels verifies (pas
    supposes) sur `trl.SFTConfig.__init__` (trl 1.13.0).
  - `trl.SFTTrainer(model=, args=, train_dataset=, eval_dataset=,
    processing_class=, peft_config=)` : signature reelle verifiee ;
    `SFTTrainer` applique lui-meme `peft.get_peft_model` en interne
    quand `peft_config` est fourni (pas d'appel manuel necessaire, pas
    besoin d'appeler `prepare_model_for_kbit_training` explicitement
    non plus, `SFTTrainer` le fait aussi en interne pour un modele
    quantifie).
  - `train_dataset`/`eval_dataset` attendent un `datasets.Dataset`
    (ou `IterableDataset`), pas une liste de dict brute : d'ou la
    conversion `Dataset.from_list(...)` ci-dessous.
  - `transformers.BitsAndBytesConfig(...)` : deja verifie en phase
    domaine (cf. `domain/model/configuration_entrainement.py`).

  - **TROUVAILLE REELLE, VERIFIEE SANS GPU (bloquante)** :
    `assistant_only_loss=True` (valeur du YAML,
    `recipes/sft_qwen3_lora.yaml::entrainement.assistant_only_loss`)
    est INCOMPATIBLE avec la forme actuelle du pipeline. Preuve lue
    directement dans `trl/trainer/sft_trainer.py` (1.13.0, lignes
    ~1245-1248) : `SFTTrainer` leve `ValueError` des la preparation du
    dataset si `args.assistant_only_loss and not
    is_conversational(dataset_sample)`. Confirme par appel reel (sans
    GPU, sans modele) : `trl.data_utils.is_conversational({"text":
    "..."})` retourne `False`. Or `ExempleFormate.texte` (produit par
    `ChatMLFormateurAdapter`, deja [FAIT] et teste) est un texte ChatML
    DEJA RENDU (`apply_chat_template(..., tokenize=False)`), pas une
    liste de messages structuree ("messages"/"prompt"+"completion") :
    la seule forme que `is_conversational` reconnait. Ce module passe
    donc la colonne `"text"` a `SFTTrainer` (`dataset_text_field` par
    defaut de `SFTConfig`), ce qui rend `assistant_only_loss=True`
    garanti-en-echec, AVANT tout entrainement reel, mais APRES le
    chargement (couteux, facture) du modele quantifie dans le
    constructeur de cet adaptateur. Consequence de conception :
    `assistant_only_loss` par defaut ICI est `False` (utilisable avec
    la forme de donnees actuelle), PAS `True` comme le YAML le
    souhaite : eviter de faire echouer une session GPU facturee sur un
    bug garanti plutot que de respecter aveuglement le YAML. Options
    si le masquage assistant-only est requis :
    (a) faire porter par `ExempleFormate` les messages structures
    (`prompt`/`completion` bruts) en plus/a la place du texte
    pre-rendu, pour que `SFTTrainer` applique lui-meme la chat
    template (avec ses marqueurs `{% generation %}` patches en
    interne, cf. `trl/chat_templates/README.md`) ; ou (b) accepter la
    perte pleine sequence (prompt+completion) pour ce premier run et
    ne pas activer ce flag. `training/E2_04_sft_train.py` fait une
    verification prealable (avant de construire cet adaptateur, donc
    sans cout GPU) si `--assistant-only-loss` resout a `True`.

  - **TROUVAILLE REELLE, sur un vrai job GPU L4 facture (echoue le
    16/09/2026)** : `recipes/sft_qwen3_lora.yaml::entrainement.type_perte`
    valait `chunked_nll`, passe tel quel a `SFTConfig(loss_type=...)`.
    Le job a echoue a la construction de `SFTTrainer` : ValueError,
    message exact "Invalid loss_type chunked_nll passed. Supported
    values are 'nll' and 'dft'." A NE PAS CONFONDRE avec "trl ne supporte pas
    chunked_nll" : verifie par lecture reelle du code source de
    `trl.trainer.sft_trainer`/`sft_config` (installe temporairement,
    plusieurs versions), `chunked_nll` EXISTE bien depuis trl>=1.12 (en
    fait la valeur PAR DEFAUT de `SFTConfig.loss_type` quand
    `use_liger_kernel=False`, cf. le "VERIFIE SANS GPU" ci-dessus qui
    date du 11/09/2026 avec trl==1.13.0 reellement installe). La cause
    reelle est une INCOMPATIBILITE DE DEPENDANCES, pas une erreur de
    frappe : l'extra `remote` de `pyproject.toml` liste `"unsloth"` sans
    borne de version, alors qu'`unsloth` n'est PAS cable dans le code
    (cf. AVERTISSEMENT ci-dessus, "Unsloth : PAS integre ici du tout").
    Or la derniere version publiee d'`unsloth` (verifie reellement via
    l'API JSON PyPI, `unsloth==2026.9.4`) exige, SANS extra, `trl>=0.18.2,
    !=0.19.0,<=0.24.0` : une contrainte non conditionnelle qui force
    N'IMPORTE QUELLE resolution fraiche a plafonner `trl` a 0.24.0 (un
    trl pre-1.0, ecrit AVANT que `chunked_nll` existe : verifie en
    installant `trl==0.24.0` pour de vrai, message d'erreur
    `ValueError` IDENTIQUE mot pour mot a celui du job reel). Le
    `uv.lock` commite de ce depot masque le probleme en local (il a
    fige un `unsloth==2024.8` tres ancien, sans contrainte sur `trl`,
    laissant `trl` remonter a 1.13.0), mais `hf jobs uv run --with
    "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main"`
    (cf. README §12) n'utilise PAS `uv.lock` : il resout les
    dependances a neuf a chaque lancement, contre l'etat REEL de PyPI
    au moment du job, pas l'etat fige localement. Reproduit pour de vrai
    (`uv pip install "trl>=0.9" "unsloth"` dans un venv jetable, sans
    lockfile) : le resolveur choisit bien `trl==0.24.0` +
    `unsloth==2026.9.4`. Decision : PAS de remplacement direct pour
    l'optimisation memoire que `chunked_nll` visait (chunker le calcul
    de la projection `lm_head` pour eviter de materialiser le tenseur de
    logits complet) : ce mecanisme n'existe simplement pas dans trl
    0.24.0, et tant qu'`unsloth` (non cable) reste dans l'extra `remote`
    sans borne, AUCUNE resolution fraiche ne pourra jamais obtenir un
    trl>=1.12. `recipes/sft_qwen3_lora.yaml::entrainement.type_perte`
    est donc corrige a `nll` (valeur standard, supportee par TOUTES les
    versions de trl observees ici, de 0.24.0 a 1.13.0), et
    `training/E2_04_sft_train.py` verifie desormais `type_perte` contre
    l'ensemble sur des valeurs sures AVANT de charger le modele (meme
    patron que le guard `assistant_only_loss` ci-dessus). Marge memoire
    pour ce run precis (LoRA rang 16, 4 modules cibles, sur un modele
    1.7B quantifie NF4, `l4x1` 24 Go de VRAM) : le raisonnement deja
    documente en README §2.3 (marge large) reste valide SANS
    `chunked_nll`, puisqu'il ne reposait pas sur cette optimisation
    specifique (le calcul avait deja ete fait pour `loss_type="nll"`
    standard) ; `chunked_nll` aurait ete un coussin de securite
    supplementaire, pas une condition de faisabilite. Si le pic memoire
    s'avere reellement trop juste sur un vrai run, retirer `unsloth`
    (non utilise) de l'extra `remote` de `pyproject.toml` permettrait a
    `trl>=1.12` de se resoudre et de recuperer `chunked_nll` pour de
    vrai : option NON appliquee ici (changement de dependances plus
    large que ce correctif, laisse a une decision explicite future).

NON VERIFIE, a confirmer sur une vraie session GPU
avant de faire confiance a ce module :
  - Que le modele charge en 4-bit (QLoRA NF4) + LoRA s'entraine
    reellement sans erreur CUDA/memoire sur `Qwen3-1.7B-Base`.
  - Le mapping `Trainer.state.log_history` -> `MetriquesEntrainement`
    ci-dessous (`_courbe_depuis_log_history`) : ecrit d'apres le
    comportement DOCUMENTE de `transformers.Trainer` (cles
    "loss"/"grad_norm"/"step" aux pas de `logging_steps`, "eval_loss"
    aux pas de `eval_steps`), jamais confronte a un vrai
    `log_history`.
  - Que `chemin_checkpoint` (repertoire ecrit par
    `trainer.save_model()`) est bien rechargeable ensuite par
    `peft`/l'Etape 3 (DPO).
  - Optimisations Unsloth / Liger Kernel / FlashAttention-2: AUCUNE
    mesure empirique possible sans GPU (temps par epoque, VRAM de
    pic). Choix par defaut ICI, NON VALIDES :
      * `attn_implementation="sdpa"` par defaut (pas "flash_attention_2") :
        SDPA est integre a PyTorch (pas d'installation supplementaire,
        pas de compilation CUDA a part), donc c'est le choix qui a le
        moins de chances d'echouer a l'installation sur un
        environnement cloud pas encore verifie ; FlashAttention-2 est
        probablement plus rapide mais necessite le paquet `flash-attn`
        (compilation longue, echoue souvent sans le bon toolchain
        CUDA) : expose via le parametre `attn_implementation` du
        constructeur pour permettre de le mesurer et de basculer si
        souhaite.
      * `utiliser_liger_kernel=False` par defaut (champ reel de
        `trl.SFTConfig`, verifie ci-dessus) : active seulement via le
        parametre du constructeur, jamais mesure.
      * Unsloth : PAS integre ici du tout. `FastLanguageModel.from_pretrained`
        remplace entierement le chemin de chargement
        `AutoModelForCausalLM` + `BitsAndBytesConfig` choisi ci-dessous
        (chargement/quantification geres differemment par Unsloth) :
        un changement d'architecture de ce module, pas un flag a
        cocher, a trancher avec de vraies mesures
        GPU, pas devine ici.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainement,
)
from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.ports.entraineur_supervise import (
    MetriquesEntrainement,
    ResultatEntrainementSFT,
)


def _horodatage_nom_run() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _courbe_depuis_log_history(log_history: list[dict]) -> tuple[MetriquesEntrainement, ...]:
    """
    Reconstruit la courbe de metriques a partir de
    `transformers.Trainer.state.log_history` (dont `SFTTrainer`
    herite le comportement, pas surcharge) : les entrees
    d'entrainement portent les cles "loss"/"grad_norm"/"step" aux pas
    multiples de `logging_steps`, les entrees d'evaluation portent
    "eval_loss"/"step" aux pas multiples de `eval_steps`. JAMAIS
    confronte a un vrai `log_history` (aucun GPU disponible a
    l'ecriture) : a verifier explicitement sur le premier run reel du
    avant de faire confiance a la courbe produite.
    """
    pertes_validation_par_etape: dict[int, float] = {
        int(entree["step"]): entree["eval_loss"]
        for entree in log_history
        if "eval_loss" in entree and "step" in entree
    }

    points: list[MetriquesEntrainement] = []
    for entree in log_history:
        if "loss" not in entree or "step" not in entree:
            continue
        etape = int(entree["step"])
        points.append(
            MetriquesEntrainement(
                etape=etape,
                perte_train=entree["loss"],
                perte_validation=pertes_validation_par_etape.get(etape),
                norme_gradient=entree.get("grad_norm", 0.0),
            )
        )
    return tuple(points)


@dataclass(slots=True)
class TrlSftEntraineurAdapter:
    """
    Adaptateur GPU implementant `EntraineurSupervise` via
    `trl.SFTTrainer` + `peft.LoraConfig`. Charge le modele de base
    quantifie et le tokenizer UNE SEULE FOIS dans `__init__` (meme
    patron que `PresidioAnonymiseur`) : jamais expose au-dela de la
    frontiere domain/application (le domaine ne voit que
    `ExempleFormate`/`ConfigurationLora`/`HyperparametresEntrainement`/
    `ResultatEntrainementSFT`).
    """

    identifiant_modele_base    : str
    configuration_quantification : ConfigurationQuantification
    repertoire_sortie            : str = "outputs/sft-lora"
    assistant_only_loss           : bool = False   # cf. AVERTISSEMENT en tete de module
    attn_implementation            : str = "sdpa"   # cf. AVERTISSEMENT : NON VALIDE empiriquement
    utiliser_liger_kernel           : bool = False   # cf. AVERTISSEMENT : NON VALIDE empiriquement
    _modele                          : Any = field(default=None, init=False, repr=False)
    _tokenizer                        : Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        conf = self.configuration_quantification
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=conf.bits == 4,
            load_in_8bit=conf.bits == 8,
            bnb_4bit_quant_type=conf.type_quantification,
            bnb_4bit_use_double_quant=conf.double_quantification,
            bnb_4bit_compute_dtype=getattr(torch, conf.dtype_calcul),
        )

        self._modele = AutoModelForCausalLM.from_pretrained(
            self.identifiant_modele_base,
            quantization_config=bnb_config,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.identifiant_modele_base, trust_remote_code=True
        )

    def entrainer(
        self,
        dataset_train      : Iterable[ExempleFormate],
        dataset_validation  : Iterable[ExempleFormate],
        config_lora          : ConfigurationLora,
        hyperparametres       : HyperparametresEntrainement,
    ) -> ResultatEntrainementSFT:
        """
        Construit `peft.LoraConfig` + `trl.SFTConfig` + `trl.SFTTrainer`
        a partir des dataclasses pures du domaine, lance
        `trainer.train()`, sauvegarde le checkpoint LoRA, et reconstruit
        la courbe de metriques depuis `trainer.state.log_history` (cf.
        `_courbe_depuis_log_history`, NON VERIFIE contre un vrai run).
        """
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from trl import SFTConfig, SFTTrainer

        dataset_train_hf = Dataset.from_list([{"text": exemple.texte} for exemple in dataset_train])
        dataset_validation_hf = Dataset.from_list([{"text": exemple.texte} for exemple in dataset_validation])

        lora_config = LoraConfig(
            r=config_lora.rang,
            lora_alpha=config_lora.alpha,
            lora_dropout=config_lora.dropout,
            target_modules=list(config_lora.modules_cibles),
            task_type=TaskType.CAUSAL_LM,
        )

        chemin_checkpoint = str(Path(self.repertoire_sortie) / f"run-{_horodatage_nom_run()}")

        sft_config = SFTConfig(
            output_dir=chemin_checkpoint,
            learning_rate=hyperparametres.taux_apprentissage,
            num_train_epochs=hyperparametres.nombre_epoques,
            per_device_train_batch_size=hyperparametres.taille_lot,
            packing=hyperparametres.packing,
            loss_type=hyperparametres.type_perte,
            assistant_only_loss=self.assistant_only_loss,
            use_liger_kernel=self.utiliser_liger_kernel,
            eval_strategy="epoch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self._modele,
            args=sft_config,
            train_dataset=dataset_train_hf,
            eval_dataset=dataset_validation_hf,
            processing_class=self._tokenizer,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(chemin_checkpoint)

        courbe = _courbe_depuis_log_history(trainer.state.log_history)
        return ResultatEntrainementSFT(chemin_checkpoint=chemin_checkpoint, courbe_metriques=courbe)
