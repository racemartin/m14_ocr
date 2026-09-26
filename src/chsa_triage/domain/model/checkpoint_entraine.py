"""
Entite CheckpointEntraine : metadonnees d'un checkpoint SFT-LoRA ou
DPO-LoRA, persistees separement des poids eux-memes (ecrits sur disque
par peft/trl). Reutilisee telle quelle pour le DPO (pas de classe
`CheckpointEntraineDpo` : `hyperparametres` accepte l'un ou l'autre type
et le stocke comme donnee opaque). Aucune dependance externe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


class VerdictConvergence(str, Enum):
    """Diagnostic pose sur une courbe d'entrainement par application.verdict_convergence."""

    SAINE = "saine"
    SURAPPRENTISSAGE = (
        "surapprentissage"  # perte train baisse, perte val remonte
    )
    SOUS_APPRENTISSAGE = "sous_apprentissage"  # les deux stagnent
    INSTABLE = "instable"  # norme de gradient diverge/NaN


@dataclass(frozen=True, slots=True)
class CheckpointEntraine:
    """Metadonnees d'un checkpoint SFT-LoRA, destinees a etre serialisees en JSONL."""

    identifiant: str  # ex. hash(recette + horodatage)
    chemin: str
    modele_base: str
    configuration_lora: ConfigurationLora
    hyperparametres: (
        HyperparametresEntrainement | HyperparametresEntrainementDpo
    )
    metriques_finales: MetriquesEntrainement
    verdict_convergence: VerdictConvergence
    horodatage: str
