"""
Tests de `evaluer_convergence` sur des courbes `MetriquesEntrainement`
synthetiques construites a la main (aucun port, aucune dependance
GPU/entrainement reelle).
"""

from __future__ import annotations

import math

import pytest

from chsa_triage.application.verdict_convergence import evaluer_convergence
from chsa_triage.domain.model.checkpoint_entraine import VerdictConvergence
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


def _point(etape: int, perte_train: float, perte_validation: float | None, norme_gradient: float = 1.0) -> MetriquesEntrainement:
    return MetriquesEntrainement(
        etape=etape, perte_train=perte_train, perte_validation=perte_validation, norme_gradient=norme_gradient
    )


def test_evaluer_convergence_leve_sur_courbe_vide():
    with pytest.raises(ValueError):
        evaluer_convergence(())


def test_evaluer_convergence_saine_quand_train_et_validation_baissent():
    courbe = (
        _point(1, perte_train=2.0, perte_validation=2.1),
        _point(2, perte_train=1.5, perte_validation=1.6),
        _point(3, perte_train=1.1, perte_validation=1.2),
        _point(4, perte_train=0.9, perte_validation=0.95),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.SAINE


def test_evaluer_convergence_surapprentissage_quand_validation_remonte():
    courbe = (
        _point(1, perte_train=2.0, perte_validation=2.0),
        _point(2, perte_train=1.5, perte_validation=1.4),
        _point(3, perte_train=1.0, perte_validation=1.3),
        _point(4, perte_train=0.7, perte_validation=1.8),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.SURAPPRENTISSAGE


def test_evaluer_convergence_sous_apprentissage_quand_train_stagne():
    courbe = (
        _point(1, perte_train=2.00, perte_validation=2.05),
        _point(2, perte_train=1.99, perte_validation=2.04),
        _point(3, perte_train=1.98, perte_validation=2.03),
        _point(4, perte_train=1.97, perte_validation=2.02),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.SOUS_APPRENTISSAGE


def test_evaluer_convergence_instable_sur_perte_nan():
    courbe = (
        _point(1, perte_train=2.0, perte_validation=2.0),
        _point(2, perte_train=math.nan, perte_validation=2.0),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.INSTABLE


def test_evaluer_convergence_instable_sur_gradient_infini():
    courbe = (
        _point(1, perte_train=2.0, perte_validation=2.0, norme_gradient=1.0),
        _point(2, perte_train=2.1, perte_validation=2.1, norme_gradient=math.inf),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.INSTABLE


def test_evaluer_convergence_instable_sur_divergence_gradient_sans_nan():
    courbe = (
        _point(1, perte_train=2.0, perte_validation=2.0, norme_gradient=1.0),
        _point(2, perte_train=2.5, perte_validation=2.4, norme_gradient=3.0),
        _point(3, perte_train=3.2, perte_validation=3.1, norme_gradient=15.0),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.INSTABLE


def test_evaluer_convergence_saine_sans_aucun_point_de_validation():
    """Aucune mesure de validation loggee (pas de val_loss a ce pas) : seule la perte train compte."""
    courbe = (
        _point(1, perte_train=2.0, perte_validation=None),
        _point(2, perte_train=1.0, perte_validation=None),
    )

    assert evaluer_convergence(courbe) == VerdictConvergence.SAINE
