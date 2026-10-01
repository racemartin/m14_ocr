"""
Adaptateur secondaire (Environnement B, GPU) : entrainement SFT-LoRA
reel via `trl.SFTTrainer` + `peft.LoraConfig`. Implemente le port
`EntraineurSupervise`. Seul module qui importe `trl`/`peft`/
`bitsandbytes`, imports fonction-scoped (`__init__`/`entrainer`) pour
que ce fichier reste importable en Environnement A sans ces paquets.

Statut reel : le premier run SFT-LoRA complet a tourne sur GPU cloud
L4 et a atteint le verdict de convergence SAINE (poids publies sur
`mombasstic/chsa-triage-sft-lora`, cf. README) — ce module a donc bien
ete valide de bout en bout, pas seulement par lecture de code. Ce
premier run utilisait `assistant_only_loss=False` (cf. point 1
ci-dessous, resolu depuis) : perte pleine sequence, pas encore masquee
aux tours assistant.

Deux bugs reels trouves et corriges avant/pendant ce run :
1. `assistant_only_loss=True` (valeur du YAML) etait INCOMPATIBLE avec
   la forme d'alors des donnees : `ExempleFormate.texte` etait un texte
   ChatML deja rendu (colonne `"text"`), pas la forme "messages"
   structuree qu'exige `is_conversational()` pour ce flag cote
   `trl.SFTTrainer`. RESOLU (`m14-ocr-assistant-only-loss-estructurado`) :
   plutot que de se fier a `is_conversational()`/
   `apply_chat_template(..., return_assistant_tokens_mask=True)` de
   `trl` (qui exige que le chat template du tokenizer definisse les
   balises jinja `{% generation %}`, jamais verifie sur le template
   Qwen3 reel, et de toute facon fragile face au bug de stripping
   `<think>` documente au point 2bis ci-dessous), ce module tokenize
   desormais chaque exemple tour par tour via `ExempleFormate.tours`
   (bornes calculees par `ChatMLFormateurAdapter.formater`) et
   construit lui-meme des `labels` avec `-100` (ignore_index de
   `torch.nn.CrossEntropyLoss`) sur les tours non-assistant quand
   `hyperparametres.assistant_only_loss=True` (cf.
   `_tokeniser_exemple_masque`). Le dataset HF passe alors des colonnes
   `input_ids`/`attention_mask`/`labels` deja construites a
   `SFTTrainer`, qui saute sa propre pipeline de tokenisation/masquage
   (comportement standard pour un dataset deja tokenize).
2. `type_perte=chunked_nll` (valeur du YAML) a fait echouer un job GPU
   facture reel : la cause n'etait pas "trl ne supporte pas
   chunked_nll" mais une incompatibilite de dependances — l'extra
   `remote` liste `unsloth` sans borne de version, qui force toute
   resolution fraiche (HF Jobs ne fige pas `uv.lock`) a plafonner `trl`
   a une version pre-1.0 sans `chunked_nll`. Corrige en repassant
   `type_perte` a `nll` (supporte par toutes les versions observees) ;
   `unsloth` reste dans l'extra sans etre cable au code, a trancher
   plus tard si le pic memoire l'exige.
3. `_courbe_depuis_log_history` perdait silencieusement les 3 mesures
   `eval_loss` d'un run reel (`sft-lora-essai-1`, job
   `6abec439404719ba3761a42c`, confirme via `hf jobs logs`) : indexait
   les pertes de validation par leur `step` exact, or `eval_strategy=
   "epoch"` les declenche aux bornes d'epoque (113/226/339 sur ce run,
   339 pas sur 3 epoques), jamais un multiple de `logging_steps=10`
   (meme phenomene deja documente sur le tout premier run SFT, cf. le
   fichier `AGENTS.md` du projet, bornes 114/228/342). RESOLU : associe
   chaque `eval_loss` au pas d'entrainement le PLUS PROCHE plutot
   qu'exiger une egalite exacte.

Note sur le bug Qwen3 `<think>` (point 2bis) : `apply_chat_template()`
strippe silencieusement les blocs `<think>` d'un tour assistant
NON-FINAL (deja trouve et corrige cote DPO, cf.
`ChatMLFormateurAdapter._rendre_tour_assistant_seul`). `ExempleFormate.tours`
evite cette exposition cote SFT en ne calculant les bornes des tours
`prompt` que par prefixes `apply_chat_template` SANS tour assistant
(toujours surs), et en deduisant la borne du tour `completion` par
simple arithmetique (il est toujours le tour final du rendu complet,
jamais strippe) plutot que par un appel `apply_chat_template`
supplementaire sur un sous-ensemble contenant ce tour. Cf. docstring de
`LimiteTour` (domain/model/exemple_formate.py) pour le detail.

Choix non mesures empiriquement (marge suffisante sur ce run, jamais
comparee a une alternative) : `attn_implementation="sdpa"` plutot que
FlashAttention-2 (evite une compilation CUDA fragile) ; Liger Kernel
desactive par defaut ; Unsloth non integre du tout (changerait le
chemin de chargement du modele, pas un simple flag).
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


def _courbe_depuis_log_history(
    log_history: list[dict],
) -> tuple[MetriquesEntrainement, ...]:
    """Reconstruit la courbe de metriques depuis
    `transformers.Trainer.state.log_history` : les entrees
    d'entrainement portent "loss"/"grad_norm"/"step", les entrees
    d'evaluation "eval_loss"/"step".

    `eval_strategy="epoch"` declenche une evaluation au pas global ou
    l'epoque se termine, pas necessairement un multiple de
    `logging_steps` (confirme reel, pas hypothetique : job
    `6abec439404719ba3761a42c`, 339 pas sur 3 epoques -> bornes
    113/226/339, aucune multiple de `logging_steps=10`). Associer
    chaque `eval_loss` au pas d'ENTRAINEMENT le plus proche (au lieu
    d'exiger une egalite exacte de `step`) evite de perdre
    silencieusement les 3 mesures de validation reellement calculees
    par ce job mais jusqu'ici jetees avant d'atteindre le journal
    (aucun pas d'entrainement ne partageait exactement 113/226/339)."""
    pertes_validation_par_etape_eval: dict[int, float] = {
        int(entree["step"]): entree["eval_loss"]
        for entree in log_history
        if "eval_loss" in entree and "step" in entree
    }
    etapes_entrainement = sorted(
        {
            int(entree["step"])
            for entree in log_history
            if "loss" in entree and "step" in entree
        }
    )

    pertes_validation_par_etape: dict[int, float] = {}
    for etape_eval, valeur in pertes_validation_par_etape_eval.items():
        if not etapes_entrainement:
            break
        etape_proche = min(etapes_entrainement, key=lambda e: abs(e - etape_eval))
        pertes_validation_par_etape[etape_proche] = valeur

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


IGNORE_INDEX = -100  # torch.nn.CrossEntropyLoss(ignore_index=-100) par defaut


def _tokeniser_exemple_masque(
    exemple: ExempleFormate, tokenizer: Any, assistant_only_loss: bool
) -> dict[str, list[int]]:
    """
    Tokenise `exemple.texte` TOUR PAR TOUR (pas en un seul appel) pour
    construire `labels` explicitement, plutot que de se fier a
    `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`
    (exige que le chat template definisse les balises jinja
    `{% generation %}`, jamais verifie sur le template Qwen3 reel, et
    de toute facon fragile face au bug de stripping `<think>` des tours
    assistant non-finaux, cf. docstring du module et `LimiteTour`).

    Si `assistant_only_loss` est faux, ou si `exemple.tours` est vide
    (bornes inconnues : `ExempleFormate` construit avant cette
    fonctionnalite), retombe sur une tokenisation en un seul appel avec
    `labels = input_ids` (perte pleine sequence, comportement
    historique).

    Pure fonction (aucun etat), testable avec un faux tokenizer sans
    GPU ni reseau (cf. tests/infrastructure/test_trl_sft_entraineur_masquage.py).
    """
    if not assistant_only_loss or not exemple.tours:
        ids = tokenizer(exemple.texte, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": list(ids),
        }

    input_ids: list[int] = []
    labels: list[int] = []
    for tour in exemple.tours:
        segment = exemple.texte[tour.debut : tour.fin]
        ids_segment = tokenizer(segment, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids_segment)
        if tour.role == "assistant":
            labels.extend(ids_segment)
        else:
            labels.extend([IGNORE_INDEX] * len(ids_segment))

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


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

    identifiant_modele_base: str
    configuration_quantification: ConfigurationQuantification
    repertoire_sortie: str = "outputs/sft-lora"
    attn_implementation: str = "sdpa"  # cf. docstring du module
    utiliser_liger_kernel: bool = False  # cf. docstring du module
    _modele: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

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
        dataset_train: Iterable[ExempleFormate],
        dataset_validation: Iterable[ExempleFormate],
        config_lora: ConfigurationLora,
        hyperparametres: HyperparametresEntrainement,
    ) -> ResultatEntrainementSFT:
        """Construit `peft.LoraConfig` + `trl.SFTConfig` + `trl.SFTTrainer`
        a partir des dataclasses pures du domaine, lance
        `trainer.train()`, sauvegarde le checkpoint LoRA, et reconstruit
        la courbe de metriques (`_courbe_depuis_log_history`).

        Si `hyperparametres.assistant_only_loss` est vrai, tokenise
        chaque `ExempleFormate` tour par tour via
        `_tokeniser_exemple_masque` et construit le dataset HF avec des
        colonnes `input_ids`/`attention_mask`/`labels` DEJA tokenisees
        (perte masquee aux tours assistant) ; `SFTTrainer` saute alors
        sa propre pipeline de tokenisation/masquage (comportement
        standard pour un dataset deja tokenize). Sinon, comportement
        historique : colonne `"text"` brute, perte pleine sequence,
        tokenisation geree par `SFTTrainer` lui-meme."""
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from trl import SFTConfig, SFTTrainer

        if hyperparametres.assistant_only_loss:
            dataset_train_hf = Dataset.from_list(
                [
                    _tokeniser_exemple_masque(exemple, self._tokenizer, True)
                    for exemple in dataset_train
                ]
            )
            dataset_validation_hf = Dataset.from_list(
                [
                    _tokeniser_exemple_masque(exemple, self._tokenizer, True)
                    for exemple in dataset_validation
                ]
            )
        else:
            dataset_train_hf = Dataset.from_list(
                [{"text": exemple.texte} for exemple in dataset_train]
            )
            dataset_validation_hf = Dataset.from_list(
                [{"text": exemple.texte} for exemple in dataset_validation]
            )

        lora_config = LoraConfig(
            r=config_lora.rang,
            lora_alpha=config_lora.alpha,
            lora_dropout=config_lora.dropout,
            target_modules=list(config_lora.modules_cibles),
            task_type=TaskType.CAUSAL_LM,
        )

        chemin_checkpoint = str(
            Path(self.repertoire_sortie) / f"run-{_horodatage_nom_run()}"
        )

        sft_config = SFTConfig(
            output_dir=chemin_checkpoint,
            learning_rate=hyperparametres.taux_apprentissage,
            num_train_epochs=hyperparametres.nombre_epoques,
            per_device_train_batch_size=hyperparametres.taille_lot,
            packing=hyperparametres.packing,
            loss_type=hyperparametres.type_perte,
            # assistant_only_loss de trl (texte conversationnel +
            # apply_chat_template interne) n'est jamais utilise ici :
            # le masquage, quand demande, est deja fait dans `labels`
            # par _tokeniser_exemple_masque (cf. docstring entrainer()).
            assistant_only_loss=False,
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
        return ResultatEntrainementSFT(
            chemin_checkpoint=chemin_checkpoint, courbe_metriques=courbe
        )
