"""
Port generique de l'entrainement supervise (SFT-LoRA) : aucun objet
`torch`/`transformers`/`peft` ne traverse cette frontiere, meme
principe que `domain.ports.moteur_inference`. L'adaptateur concret
(`TrlSftEntraineurAdapter`, infrastructure, Etape B GPU) possede et
charge le modele quantifie en interne ; ce port n'expose que des
donnees pures en entree (`ExempleFormate`, `ConfigurationLora`,
`HyperparametresEntrainement`) et en sortie (`ResultatEntrainementSFT`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.model.exemple_formate import ExempleFormate


@dataclass(frozen=True, slots=True)
class MetriquesEntrainement:
    """Point de mesure unique de la courbe d'entrainement, a une etape (pas) donnee."""

    etape             : int
    perte_train        : float
    perte_validation    : float | None
    norme_gradient       : float


@dataclass(frozen=True, slots=True)
class ResultatEntrainementSFT:
    """Resultat complet d'un run d'entrainement : chemin du checkpoint et courbe de metriques."""

    chemin_checkpoint : str
    courbe_metriques   : tuple[MetriquesEntrainement, ...]


class EntraineurSupervise(Protocol):
    """Port generique : entrainer un adaptateur LoRA et retourner sa courbe de metriques."""

    def entrainer(
        self,
        dataset_train      : Iterable[ExempleFormate],
        dataset_validation  : Iterable[ExempleFormate],
        config_lora          : ConfigurationLora,
        hyperparametres       : HyperparametresEntrainement,
    ) -> ResultatEntrainementSFT:
        """Entraine le modele de base (charge dans le constructeur de l'adaptateur) avec LoRA."""
        ...
