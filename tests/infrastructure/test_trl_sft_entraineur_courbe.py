"""
Tests de `_courbe_depuis_log_history` (reconstruction de la courbe de
metriques depuis `transformers.Trainer.state.log_history`), fonction
pure testable sans GPU/torch (cf. `test_trl_sft_entraineur_masquage.py`
pour le meme raisonnement d'import).

Le deuxieme scenario reproduit le `log_history` reel du job
`6abec439404719ba3761a42c` (run `sft-lora-essai-1`, 339 pas sur 3
epoques, bornes d'evaluation 113/226/339, aucune multiple de
`logging_steps=10`) : avant correction, les 3 `eval_loss` reellement
calcules par ce job (confirmes via `hf jobs logs`) n'etaient jamais
rattaches a aucun point de la courbe publiee.
"""

from __future__ import annotations

from chsa_triage.infrastructure.adapters.trl_sft_entraineur import (
    _courbe_depuis_log_history,
)


def test_courbe_rattache_eval_loss_au_pas_exact_quand_il_coincide():
    log_history = [
        {"loss": 2.0, "grad_norm": 0.5, "step": 10},
        {"eval_loss": 1.8, "step": 10},
        {"loss": 1.5, "grad_norm": 0.4, "step": 20},
    ]

    courbe = _courbe_depuis_log_history(log_history)

    assert len(courbe) == 2
    assert courbe[0].etape == 10
    assert courbe[0].perte_validation == 1.8
    assert courbe[1].perte_validation is None


def test_courbe_rattache_eval_loss_au_pas_entrainement_le_plus_proche_sans_egalite_exacte():
    """Reproduit le run reel sft-lora-essai-1 : eval_strategy="epoch"
    declenche l'evaluation aux bornes d'epoque (113/226/339), jamais un
    multiple de logging_steps=10 (10..330 publies). Avant correction,
    aucun des 3 eval_loss n'etait jamais rattache a un point."""
    log_history = [
        {"loss": 2.0, "grad_norm": 0.6, "step": 10},
        {"loss": 1.9, "grad_norm": 0.5, "step": 110},
        {"eval_loss": 1.7024, "step": 113},
        {"loss": 1.8, "grad_norm": 0.45, "step": 120},
        {"loss": 1.7, "grad_norm": 0.43, "step": 220},
        {"eval_loss": 1.6537, "step": 226},
        {"loss": 1.6, "grad_norm": 0.42, "step": 230},
        {"loss": 1.55, "grad_norm": 0.41, "step": 330},
        {"eval_loss": 1.6403, "step": 339},
    ]

    courbe = _courbe_depuis_log_history(log_history)

    par_etape = {point.etape: point.perte_validation for point in courbe}
    assert par_etape[110] == 1.7024  # |110-113|=3 < |120-113|=7
    assert par_etape[230] == 1.6537  # |230-226|=4 < |220-226|=6
    assert par_etape[330] == 1.6403  # seul pas d'entrainement proche de 339
    assert par_etape[10] is None
    assert par_etape[120] is None
    assert par_etape[220] is None


def test_courbe_sans_aucun_pas_entrainement_ne_leve_pas_d_exception():
    """Garde-fou : un log_history ne contenant que des evaluations (cas
    degenere, jamais observe reellement) ne doit pas planter."""
    log_history = [{"eval_loss": 1.0, "step": 5}]

    courbe = _courbe_depuis_log_history(log_history)

    assert courbe == ()
