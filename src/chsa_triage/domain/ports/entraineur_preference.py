"""
Port generique de l'entrainement par preference (DPO) : aucun objet
`torch`/`trl`/`peft` ne traverse cette frontiere, meme principe que
`domain.ports.entraineur_supervise`. Port SEPARE de `EntraineurSupervise`
(pas une generalisation) : la perte DPO compare pi_theta(y_w|x) a
pi_theta(y_l|x) (et pareil pour pi_ref), donc trois quantites distinctes
par exemple (prompt/chosen/rejected), jamais une seule sequence
concatenee comme le porte `ExempleFormate.texte` (SFT). Decision
tranchee et justifiee en docs/04_etape3_dpo/00_introduction_concepts.md
§4.

L'adaptateur concret (`TrlDpoEntraineurAdapter`, infrastructure,
Environnement B GPU) possede et charge les modeles (politique + reference)
en interne ; ce port n'expose que des donnees pures en entree
(`ExempleFormatePreference`, `ConfigurationLora`,
`HyperparametresEntrainementDpo`, le chemin du checkpoint SFT-LoRA de
depart) et en sortie (`ResultatEntrainementDPO`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


@dataclass(frozen=True, slots=True)
class ResultatEntrainementDPO:
    """
    Resultat complet d'un run d'entrainement DPO : chemin du checkpoint
    et courbe de metriques. Reutilise `MetriquesEntrainement` telle
    quelle pour `courbe_metriques` (aucune variante DPO de cette
    dataclass, decision actee en
    docs/04_etape3_dpo/00_introduction_concepts.md §4.4 :
    `application.verdict_convergence.evaluer_convergence` diagnostique
    une courbe DPO exactement comme une courbe SFT).

    ECART documente par rapport a l'esquisse du guide d'implementation
    (docs/04_etape3_dpo/03_guide_implementation_pas_a_pas.md etape 3,
    qui ne listait que `chemin_checkpoint`/`courbe_metriques`) :
    `metriques_recompense` est un champ AJOUTE ici, necessaire pour que
    l'exigence explicite de l'etape 9 du meme guide ("EntrainerDpoUseCase
    ... les 4 metriques de recompense DPO sont bien relayees", avec un
    cas de test dedie) soit reellement implementable. Sans lui, les 4
    metriques propres a `trl.DPOTrainer` (`rewards/chosen`,
    `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, cf. §4.4
    du document d'introduction) n'auraient litteralement aucun canal
    pour atteindre `EntrainerDpoUseCase.entrainer()`. Valeurs agregees
    finales (pas une courbe par etape, contrairement a
    `courbe_metriques`) : clefs optionnelles, l'adaptateur concret
    (`TrlDpoEntraineurAdapter`, etape 12, hors perimetre ici) peut n'en
    remplir qu'une partie.
    """

    chemin_checkpoint    : str
    courbe_metriques      : tuple[MetriquesEntrainement, ...]
    metriques_recompense    : dict[str, float] = field(default_factory=dict)


class EntraineurPreference(Protocol):
    """Port generique : entrainer une politique par preference DPO, a partir d'un checkpoint SFT-LoRA de depart/reference."""

    def entrainer(
        self,
        dataset_train                        : Iterable[ExempleFormatePreference],
        dataset_validation                    : Iterable[ExempleFormatePreference],
        config_lora                             : ConfigurationLora,
        hyperparametres                           : HyperparametresEntrainementDpo,
        chemin_checkpoint_politique_depart          : str,
    ) -> ResultatEntrainementDPO:
        """Entraine le modele de base + LoRA-SFT (charge dans le constructeur de l'adaptateur) par preference DPO."""
        ...
