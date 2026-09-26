"""
Port generique de l'entrainement par preference (DPO). Separe de
`EntraineurSupervise` (pas une generalisation) : la perte DPO compare
pi_theta/pi_ref sur `chosen` ET `rejected`, trois quantites distinctes
par exemple, jamais une seule sequence concatenee comme le SFT.
L'adaptateur concret charge les modeles (politique + reference) en
interne ; ce port ne voit que des donnees pures en entree/sortie.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import (
    ExempleFormatePreference,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


@dataclass(frozen=True, slots=True)
class ResultatEntrainementDPO:
    """
    Resultat d'un run DPO : chemin du checkpoint et courbe de metriques.
    Reutilise `MetriquesEntrainement` telle quelle (pas de variante DPO :
    `evaluer_convergence` diagnostique une courbe DPO comme une courbe
    SFT). `metriques_recompense` est un champ ajoute apres coup, absent
    de l'esquisse initiale du guide d'implementation : sans lui, les 4
    metriques propres a `trl.DPOTrainer` (`rewards/chosen`,
    `rewards/rejected`, `rewards/accuracies`, `rewards/margins`)
    n'auraient aucun canal pour atteindre `EntrainerDpoUseCase`. Valeurs
    finales agregees, pas une courbe par etape ; clefs optionnelles.
    """

    chemin_checkpoint: str
    courbe_metriques: tuple[MetriquesEntrainement, ...]
    metriques_recompense: dict[str, float] = field(default_factory=dict)


class EntraineurPreference(Protocol):
    """Port generique : entrainer une politique par preference DPO, a partir d'un checkpoint SFT-LoRA de depart/reference."""

    def entrainer(
        self,
        dataset_train: Iterable[ExempleFormatePreference],
        dataset_validation: Iterable[ExempleFormatePreference],
        config_lora: ConfigurationLora,
        hyperparametres: HyperparametresEntrainementDpo,
        chemin_checkpoint_politique_depart: str,
    ) -> ResultatEntrainementDPO:
        """Entraine le modele de base + LoRA-SFT (charge dans le constructeur de l'adaptateur) par preference DPO."""
        ...
