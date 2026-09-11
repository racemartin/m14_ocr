"""
Entite CheckpointEntraine : enregistrement de metadonnees decrivant un
checkpoint SFT-LoRA produit par EntrainerSftUseCase, persiste separement
des poids eux-memes (ecrits sur disque par peft/trl directement, cf.
`ResultatEntrainementSFT.chemin_checkpoint`). Meme role que
`data/processed/rapport_anonymisation_rgpd.json` en Etape 1 :
indicateurs structures generes automatiquement plutot que recalcules
a la main, lus par l'Etape 3 (DPO) pour retrouver le meilleur run.

Aucune dependance externe (pas de mlflow, pas de peft ici) : seuls des
types du domaine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


class VerdictConvergence(str, Enum):
    """Diagnostic pose sur une courbe d'entrainement par application.verdict_convergence."""

    SAINE               = "saine"
    SURAPPRENTISSAGE      = "surapprentissage"   # perte train baisse, perte val remonte
    SOUS_APPRENTISSAGE     = "sous_apprentissage"  # les deux stagnent
    INSTABLE                = "instable"            # norme de gradient diverge/NaN


@dataclass(frozen=True, slots=True)
class CheckpointEntraine:
    """Metadonnees d'un checkpoint SFT-LoRA, destinees a etre serialisees en JSONL."""

    identifiant           : str    # ex. hash(recette + horodatage)
    chemin                  : str
    modele_base              : str
    configuration_lora        : ConfigurationLora
    hyperparametres            : HyperparametresEntrainement
    metriques_finales           : MetriquesEntrainement
    verdict_convergence          : VerdictConvergence
    horodatage                    : str
