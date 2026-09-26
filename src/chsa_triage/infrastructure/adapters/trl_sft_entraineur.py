"""
Adaptateur secondaire (Environnement B, GPU) : entrainement SFT-LoRA
reel via `trl.SFTTrainer` + `peft.LoraConfig`. Implemente le port
`EntraineurSupervise`. Seul module qui importe `trl`/`peft`/
`bitsandbytes`, imports fonction-scoped (`__init__`/`entrainer`) pour
que ce fichier reste importable en Environnement A sans ces paquets.

Statut reel : le premier run SFT-LoRA complet a tourne sur GPU cloud
L4 et a atteint le verdict de convergence SAINE (poids publies sur
`mombasstic/chsa-triage-sft-lora`, cf. README) — ce module a donc bien
ete valide de bout en bout, pas seulement par lecture de code.

Deux bugs reels trouves et corriges avant/pendant ce run :
1. `assistant_only_loss=True` (valeur du YAML) est INCOMPATIBLE avec la
   forme actuelle des donnees : `ExempleFormate.texte` est un texte
   ChatML deja rendu (colonne `"text"`), pas la forme "messages"
   structuree qu'exige `is_conversational()` pour ce flag — `SFTTrainer`
   leve un `ValueError` des la preparation du dataset. Par defaut ici,
   `assistant_only_loss=False` (backlog ouvert pour lever cette
   limite : cf. `m14-ocr-assistant-only-loss-estructurado`).
2. `type_perte=chunked_nll` (valeur du YAML) a fait echouer un job GPU
   facture reel : la cause n'etait pas "trl ne supporte pas
   chunked_nll" mais une incompatibilite de dependances — l'extra
   `remote` liste `unsloth` sans borne de version, qui force toute
   resolution fraiche (HF Jobs ne fige pas `uv.lock`) a plafonner `trl`
   a une version pre-1.0 sans `chunked_nll`. Corrige en repassant
   `type_perte` a `nll` (supporte par toutes les versions observees) ;
   `unsloth` reste dans l'extra sans etre cable au code, a trancher
   plus tard si le pic memoire l'exige.

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
    d'evaluation "eval_loss"/"step". Confirme sur le run SFT reel (cf.
    docstring du module)."""
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

    identifiant_modele_base: str
    configuration_quantification: ConfigurationQuantification
    repertoire_sortie: str = "outputs/sft-lora"
    assistant_only_loss: bool = False  # cf. docstring du module
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
        la courbe de metriques (`_courbe_depuis_log_history`)."""
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from trl import SFTConfig, SFTTrainer

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
        return ResultatEntrainementSFT(
            chemin_checkpoint=chemin_checkpoint, courbe_metriques=courbe
        )
