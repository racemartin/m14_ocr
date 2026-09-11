"""
Tests de `candidat_suivant` avec une grille synthetique (pas de
dependance a `recipes/sft_qwen3_lora.yaml`).
"""

from __future__ import annotations

from chsa_triage.application.grille_hyperparametres import candidat_suivant
from chsa_triage.domain.model.configuration_entrainement import HyperparametresEntrainement


def _hyperparametres(taux_apprentissage: float) -> HyperparametresEntrainement:
    return HyperparametresEntrainement(
        taux_apprentissage=taux_apprentissage,
        nombre_epoques=3,
        taille_lot=4,
        packing=True,
        type_perte="chunked_nll",
    )


def test_candidat_suivant_retourne_le_premier_de_la_grille_quand_historique_vide():
    grille = [_hyperparametres(1e-4), _hyperparametres(2e-4), _hyperparametres(5e-4)]

    assert candidat_suivant(grille, historique=[]) == grille[0]


def test_candidat_suivant_saute_les_candidats_deja_essayes():
    grille = [_hyperparametres(1e-4), _hyperparametres(2e-4), _hyperparametres(5e-4)]
    historique = [grille[0]]

    assert candidat_suivant(grille, historique) == grille[1]


def test_candidat_suivant_ignore_lordre_de_lhistorique():
    grille = [_hyperparametres(1e-4), _hyperparametres(2e-4), _hyperparametres(5e-4)]
    historique = [grille[1], grille[0]]

    assert candidat_suivant(grille, historique) == grille[2]


def test_candidat_suivant_retourne_none_quand_la_grille_est_epuisee():
    grille = [_hyperparametres(1e-4), _hyperparametres(2e-4)]
    historique = [grille[0], grille[1]]

    assert candidat_suivant(grille, historique) is None


def test_candidat_suivant_sur_grille_vide_retourne_none():
    assert candidat_suivant(grille=[], historique=[]) is None


def test_candidat_suivant_historique_avec_doublons_et_candidats_hors_grille_ne_derange_pas():
    """L'historique peut contenir des essais anterieurs a une autre grille : seule l'appartenance compte."""
    grille = [_hyperparametres(1e-4), _hyperparametres(2e-4)]
    historique = [_hyperparametres(9e-4), grille[0], grille[0]]

    assert candidat_suivant(grille, historique) == grille[1]
