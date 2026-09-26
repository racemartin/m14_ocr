"""
Adaptateur secondaire (Environnement B, GPU) : entrainement DPO reel
via `trl.DPOTrainer`, continuant le checkpoint SFT-LoRA deja entraine.
Implemente le port `EntraineurPreference`. Seul module qui importe
`trl.DPOTrainer`/`trl.DPOConfig`, imports fonction-scoped pour rester
importable en Environnement A sans ces paquets.

Statut reel : plusieurs runs DPO ont tourne sur GPU cloud, jusqu'au
verdict de convergence SAINE (metriques reelles dans le README) — ce
module a ete valide de bout en bout.

Decision d'architecture cle sur `pi_ref` (verifiee par lecture du code
source de `trl.DPOTrainer.__init__`, pas devinee) : si `model` est deja
un `peft.PeftModel` et qu'aucun `ref_model` n'est fourni, `trl` ajoute
lui-meme un second adaptateur fige ("ref"), copie de l'adaptateur
"default" au chargement — aucun second modele complet n'est charge.
Passer `peft_config` a `DPOTrainer` creerait a la place un adaptateur
VIERGE qui EFFACERAIT le checkpoint SFT. Ce module suit donc le chemin
le moins couteux : charge le checkpoint SFT-LoRA une seule fois
(`is_trainable=True`, sinon l'adaptateur resterait gele et rien ne
s'entrainerait), ne passe jamais `peft_config`, laisse `ref_model=None`.
Consequence : `config_lora` recu par `entrainer()` (impose par le
contrat du port) n'est jamais transmis a `DPOTrainer` — il sert
seulement cote appelant a documenter l'architecture LoRA reelle dans
les metadonnees persistees.

Bug reel trouve et corrige : un job GPU a boucle son entrainement
jusqu'au bout puis plante en `torch.OutOfMemoryError` a l'evaluation.
Cause confirmee : `DPOConfig` fixait `per_device_train_batch_size` mais
jamais `per_device_eval_batch_size` (defaut reel `transformers` : 8, le
double du batch d'entrainement), et `concatenated_forward` (chosen+
rejected dans la meme passe) double deja la charge par element.
Corrige en alignant `per_device_eval_batch_size` sur le batch
d'entrainement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import (
    ExempleFormatePreference,
)
from chsa_triage.domain.ports.entraineur_preference import (
    ResultatEntrainementDPO,
)
from chsa_triage.infrastructure.adapters.trl_sft_entraineur import (
    _courbe_depuis_log_history,
    _horodatage_nom_run,
)

CHECKPOINT_POLITIQUE_DEPART_DEFAUT = "mombasstic/chsa-triage-sft-lora"

# Cles reelles de trl.DPOTrainer (rewards/*), sans equivalent SFT.
CLES_METRIQUES_RECOMPENSE = (
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
)


def _construire_dpo_config(
    chemin_checkpoint: str, hyperparametres: HyperparametresEntrainementDpo
) -> Any:
    """Construit le `trl.DPOConfig` reel a partir des hyperparametres du
    domaine. Extrait de `entrainer()` pour rester testable sans GPU.
    `per_device_eval_batch_size` est aligne sur `per_device_train_batch_size`
    (cf. docstring du module pour l'OOM reel que cela corrige)."""
    from trl import DPOConfig

    return DPOConfig(
        output_dir=chemin_checkpoint,
        beta=hyperparametres.beta,
        learning_rate=hyperparametres.taux_apprentissage,
        num_train_epochs=hyperparametres.nombre_epoques,
        per_device_train_batch_size=hyperparametres.taille_lot,
        per_device_eval_batch_size=hyperparametres.taille_lot,
        loss_type=[hyperparametres.type_perte],
        precompute_ref_log_probs=hyperparametres.precompute_ref_log_probs,
        eval_strategy="epoch",
        report_to="none",
    )


def _metriques_recompense_depuis_log_history(
    log_history: list[dict],
) -> dict[str, float]:
    """
    Valeurs agregees FINALES (derniere occurrence dans `log_history`,
    PAS une courbe par etape) des 4 metriques de recompense propres a
    `trl.DPOTrainer` : c'est le mode d'agregation attendu par
    `ResultatEntrainementDPO.metriques_recompense`, cf.
    `domain/ports/entraineur_preference.py`.
    """
    valeurs: dict[str, float] = {}
    for entree in log_history:
        for cle in CLES_METRIQUES_RECOMPENSE:
            if cle in entree:
                valeurs[cle] = entree[cle]
    return valeurs


@dataclass(slots=True)
class TrlDpoEntraineurAdapter:
    """Adaptateur GPU implementant `EntraineurPreference` via
    `trl.DPOTrainer`, continuant l'entrainement d'un checkpoint SFT-LoRA
    deja produit. Charge le modele de base quantifie plus l'adaptateur
    LoRA-SFT une seule fois dans `__init__`. `chemin_checkpoint_politique_depart`
    existe aussi comme parametre de `entrainer()` (impose par le port) :
    `entrainer()` verifie que les deux concordent plutot que d'ignorer
    l'un des deux."""

    identifiant_modele_base: str
    configuration_quantification: ConfigurationQuantification
    chemin_checkpoint_politique_depart: str = CHECKPOINT_POLITIQUE_DEPART_DEFAUT
    repertoire_sortie: str = "outputs/dpo-lora"
    attn_implementation: str = (
        "sdpa"  # cf. trl_sft_entraineur.py, non re-mesure ici
    )
    _modele: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
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

        modele_base = AutoModelForCausalLM.from_pretrained(
            self.identifiant_modele_base,
            quantization_config=bnb_config,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True,
        )
        modele_base = prepare_model_for_kbit_training(modele_base)
        # is_trainable=True : sans ce flag (defaut reel False), l'adaptateur
        # LoRA-SFT charge resterait gele et DPO n'entrainerait rien.
        self._modele = PeftModel.from_pretrained(
            modele_base,
            self.chemin_checkpoint_politique_depart,
            is_trainable=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.identifiant_modele_base, trust_remote_code=True
        )

    def entrainer(
        self,
        dataset_train: Iterable[ExempleFormatePreference],
        dataset_validation: Iterable[ExempleFormatePreference],
        config_lora: ConfigurationLora,
        hyperparametres: HyperparametresEntrainementDpo,
        chemin_checkpoint_politique_depart: str,
    ) -> ResultatEntrainementDPO:
        """Construit `trl.DPOConfig` + `trl.DPOTrainer`, lance
        `trainer.train()`, sauvegarde le checkpoint LoRA, et reconstruit
        la courbe + les metriques de recompense. `config_lora` est
        accepte (contrat du port) mais jamais transmis a `DPOTrainer`,
        et `ref_model` reste `None` (`trl` derive `pi_ref` lui-meme,
        cf. docstring du module). `chemin_checkpoint_politique_depart`
        doit concorder avec celui deja charge en `__init__` (cet
        adaptateur ne peut pas en recharger un autre a la volee) ;
        leve `ValueError` sinon.
        """
        if (
            chemin_checkpoint_politique_depart
            != self.chemin_checkpoint_politique_depart
        ):
            raise ValueError(
                f"chemin_checkpoint_politique_depart={chemin_checkpoint_politique_depart!r} passe a entrainer() "
                f"ne concorde pas avec {self.chemin_checkpoint_politique_depart!r} deja charge dans le constructeur "
                "de TrlDpoEntraineurAdapter : cet adaptateur charge son modele de depart UNE SEULE FOIS a la "
                "construction, il ne peut pas en recharger un autre a la volee."
            )

        from datasets import Dataset
        from trl import DPOTrainer

        dataset_train_hf = Dataset.from_list(
            [
                {
                    "prompt": e.texte_prompt,
                    "chosen": e.texte_chosen,
                    "rejected": e.texte_rejected,
                }
                for e in dataset_train
            ]
        )
        dataset_validation_hf = Dataset.from_list(
            [
                {
                    "prompt": e.texte_prompt,
                    "chosen": e.texte_chosen,
                    "rejected": e.texte_rejected,
                }
                for e in dataset_validation
            ]
        )

        chemin_checkpoint = str(
            Path(self.repertoire_sortie) / f"run-{_horodatage_nom_run()}"
        )

        dpo_config = _construire_dpo_config(chemin_checkpoint, hyperparametres)

        trainer = DPOTrainer(
            model=self._modele,
            ref_model=None,
            args=dpo_config,
            train_dataset=dataset_train_hf,
            eval_dataset=dataset_validation_hf,
            processing_class=self._tokenizer,
        )

        trainer.train()
        trainer.save_model(chemin_checkpoint)

        courbe = _courbe_depuis_log_history(trainer.state.log_history)
        metriques_recompense = _metriques_recompense_depuis_log_history(
            trainer.state.log_history
        )
        return ResultatEntrainementDPO(
            chemin_checkpoint=chemin_checkpoint,
            courbe_metriques=courbe,
            metriques_recompense=metriques_recompense,
        )
