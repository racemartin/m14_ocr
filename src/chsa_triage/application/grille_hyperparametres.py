"""
Selection sequentielle du prochain jeu d'hyperparametres a essayer,
pour `AjusterBoucleHyperparametresSftUseCase`. Fonction pure, aucun
port. `grille` est une sequence deja entierement etalee de candidats
(le produit cartesien des axes YAML est calcule en amont par
l'appelant, `training/E2_04_sft_train.py`) ; retourne le premier
candidat absent de `historique`, ou `None` si la grille est epuisee.

Limite connue : l'axe `rang` pilote `ConfigurationLora`, pas
`HyperparametresEntrainement` ; faire varier `rang` en boucle
suppose donc de faire varier aussi `config_lora` en parallele, ce que
`AjusterBoucleHyperparametresSftUseCase` ne fait pas actuellement
(config_lora fixe).
"""

from __future__ import annotations

from collections.abc import Sequence

from chsa_triage.domain.model.configuration_entrainement import (
    HyperparametresEntrainement,
)


def candidat_suivant(
    grille: Sequence[HyperparametresEntrainement],
    historique: Sequence[HyperparametresEntrainement],
) -> HyperparametresEntrainement | None:
    """
    Retourne le premier element de `grille` absent de `historique`
    (comparaison par egalite de valeur, `HyperparametresEntrainement`
    etant `frozen=True`), ou `None` si `historique` couvre deja tous
    les candidats de `grille` (grille epuisee).
    """
    deja_essayes = set(historique)
    for candidat in grille:
        if candidat not in deja_essayes:
            return candidat
    return None
