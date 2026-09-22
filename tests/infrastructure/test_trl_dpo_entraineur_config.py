"""
Test unitaire, SANS GPU, de `_construire_dpo_config`
(`infrastructure/adapters/trl_dpo_entraineur.py`), extrait de
`TrlDpoEntraineurAdapter.entrainer()` specifiquement pour rester
testable ici : construire un `trl.DPOConfig` (`transformers.TrainingArguments`)
ne charge aucun modele et ne touche jamais le GPU, contrairement au
reste de l'adaptateur (cf. `pytest.importorskip("trl")` ci-dessous,
PAS de garde `torch.cuda.is_available()` comme
`test_trl_dpo_entraineur.py`).

Couvre la regression reelle du job GPU L4 100 exemples/--skip-reformulation
(22/09/2026) : `per_device_eval_batch_size` gardait le defaut reel de
`transformers.TrainingArguments` (8, le double du batch d'entrainement
configure), cause racine confirmee d'un `torch.OutOfMemoryError` pendant
l'evaluation post-epoque (cf. AVERTISSEMENT "MISE A JOUR REELLE
(22/09/2026)" en tete de module).
"""

from __future__ import annotations

import pytest

pytest.importorskip("trl")

from chsa_triage.domain.model.configuration_entrainement import HyperparametresEntrainementDpo
from chsa_triage.infrastructure.adapters.trl_dpo_entraineur import _construire_dpo_config


def test_construire_dpo_config_aligne_le_batch_eval_sur_le_batch_train(monkeypatch):
    # `trl.DPOConfig` met `bf16=True` par defaut (si `fp16` n'est pas
    # positionne) ; sa validation `TrainingArguments.__post_init__`
    # refuse ce defaut sans GPU bf16-capable. Ce test ne verifie que la
    # PROPAGATION des kwargs (pas un entrainement reel), donc ce garde-fou
    # GPU est court-circuite ici uniquement, sans toucher au code de
    # production (qui, lui, tourne toujours sur un vrai GPU bf16).
    monkeypatch.setattr(
        "transformers.training_args.is_torch_bf16_gpu_available", lambda: True
    )

    hyperparametres = HyperparametresEntrainementDpo(
        beta=0.1,
        taux_apprentissage=5e-6,
        nombre_epoques=1,
        taille_lot=4,
        type_perte="sigmoid",
        precompute_ref_log_probs=False,
    )

    dpo_config = _construire_dpo_config("outputs/dpo-lora/run-test", hyperparametres)

    assert dpo_config.per_device_train_batch_size == 4
    assert dpo_config.per_device_eval_batch_size == 4
    assert dpo_config.per_device_eval_batch_size == dpo_config.per_device_train_batch_size
