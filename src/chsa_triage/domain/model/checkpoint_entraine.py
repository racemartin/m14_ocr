"""
Entite CheckpointEntraine : enregistrement de metadonnees decrivant un
checkpoint SFT-LoRA OU DPO-LoRA produit par EntrainerSftUseCase/
EntrainerDpoUseCase, persiste separement des poids eux-memes (ecrits sur
disque par peft/trl directement, cf. `ResultatEntrainementSFT`/
`ResultatEntrainementDPO.chemin_checkpoint`). Meme role que
`data/processed/rapport_anonymisation_rgpd.json` en Etape 1 :
indicateurs structures generes automatiquement plutot que recalcules
a la main.

Reutilisee telle quelle pour le DPO (aucune classe parallele
`CheckpointEntraineDpo`, decision actee en
docs/04_etape3_dpo/02_etapes_cas_usage.md §7) : `hyperparametres` accepte
soit un `HyperparametresEntrainement` (SFT) soit un
`HyperparametresEntrainementDpo` (DPO), stocke et reserialise en JSONL
comme donnee opaque, jamais lu champ par champ par cette classe ni par
`SauvegarderCheckpointSftUseCase`.

Aucune dependance externe (pas de mlflow, pas de peft ici) : seuls des
types du domaine.
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
    hyperparametres            : HyperparametresEntrainement | HyperparametresEntrainementDpo
    metriques_finales           : MetriquesEntrainement
    verdict_convergence          : VerdictConvergence
    horodatage                    : str
