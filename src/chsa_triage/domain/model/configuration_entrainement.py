"""
Configuration d'un run d'entrainement SFT-LoRA : trois dataclasses
figees deserialisees depuis `recipes/sft_qwen3_lora.yaml` (cf.
docs/03_etape2_sft/01_installation_configuration.md paragraphe 6).

Aucune dependance externe : ni `peft` ni `transformers` ne sont
importes ici (seule `infrastructure.adapters.trl_sft_entraineur` les
importera). Les champs ci-dessous ont ete verifies (pas supposes)
contre les signatures reelles de `peft.LoraConfig` (0.20.0) et
`transformers.BitsAndBytesConfig` (4.57.6) au moment de l'ecriture :
tous les autres parametres de ces deux classes ont une valeur par
defaut raisonnable pour un QLoRA NF4 standard, donc aucun champ
supplementaire n'est necessaire ici. `HyperparametresEntrainement` ne
couvre pas `assistant_only_loss` (present dans le YAML sous
`entrainement:`) : ce flag pilote `trl.SFTConfig`, pas
`transformers.TrainingArguments`, et sa verification reelle est
differee a l'ecriture de `infrastructure.adapters.trl_sft_entraineur`
(cf. guide d'implementation, etape 9).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfigurationQuantification:
    """
    Parametres de chargement quantifie du modele de base, transposables
    en `transformers.BitsAndBytesConfig` (`bits` pilote
    `load_in_4bit`/`load_in_8bit`, les trois autres champs se
    retrouvent tels quels sous `bnb_4bit_*`).
    """

    bits                  : int    # 4 ou 8
    type_quantification    : str   # ex. "nf4" -> bnb_4bit_quant_type
    double_quantification   : bool  # -> bnb_4bit_use_double_quant
    dtype_calcul              : str   # ex. "bfloat16" -> bnb_4bit_compute_dtype


@dataclass(frozen=True, slots=True)
class ConfigurationLora:
    """
    Parametres LoRA, transposables en `peft.LoraConfig` (`rang` -> `r`,
    `modules_cibles` -> `target_modules`, `alpha` -> `lora_alpha`,
    `dropout` -> `lora_dropout`).
    """

    rang           : int
    alpha           : int
    dropout          : float
    modules_cibles    : tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HyperparametresEntrainement:
    """Hyperparametres d'entrainement passes a EntraineurSupervise.entrainer()."""

    taux_apprentissage    : float
    nombre_epoques          : int
    taille_lot                : int
    packing                    : bool
    type_perte                  : str


@dataclass(frozen=True, slots=True)
class HyperparametresEntrainementDpo:
    """
    Hyperparametres d'entrainement DPO passes a
    EntraineurPreference.entrainer() (domain.ports.entraineur_preference).
    Champs repris tels quels de la section `entrainement:` de l'esquisse
    YAML `recipes/dpo_qwen3_lora.yaml`
    (docs/04_etape3_dpo/00_introduction_concepts.md §5), verifies
    (pas supposes) contre les champs reels de `trl.DPOConfig` (trl==1.13.0)
    propres au DPO, sans equivalent SFT.
    """

    beta                        : float
    taux_apprentissage           : float
    nombre_epoques                 : int
    taille_lot                       : int
    type_perte                        : str
    precompute_ref_log_probs           : bool
