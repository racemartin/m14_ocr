"""
Cas d'usage : persister un enregistrement de METADONNEES decrivant un
checkpoint SFT-LoRA deja produit.

Ne manipule jamais les poids de l'adaptateur LoRA eux-memes : ils sont
ecrits sur disque par `peft`/`trl` directement, via le
`chemin_checkpoint` deja retourne par `EntraineurSupervise.entrainer()`
(cf. `ResultatEntrainementSFT`). Ce cas d'usage se contente de
persister un `CheckpointEntraine` via une TROISIEME instance de
`RepositoryLectureEcriture` (apres `ExemplePivot` et `ExempleFormate`) :
meme port generique, troisieme type d'entite, cf.
docs/03_etape2_sft/02_etapes_cas_usage.md §6.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from chsa_triage.domain.model.checkpoint_entraine import CheckpointEntraine, VerdictConvergence
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.ports.dataset_repository import RepositoryLectureEcriture
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SauvegarderCheckpointSftUseCase:
    """Orchestre la persistance des metadonnees d'un checkpoint SFT-LoRA."""

    repository_checkpoints : RepositoryLectureEcriture
    horloge                   : Callable[[], str] = _horodatage_utc_iso

    def executer(
        self,
        identifiant          : str,
        chemin                 : str,
        modele_base             : str,
        configuration_lora       : ConfigurationLora,
        hyperparametres            : HyperparametresEntrainement | HyperparametresEntrainementDpo,
        metriques_finales           : MetriquesEntrainement,
        verdict_convergence          : VerdictConvergence,
    ) -> CheckpointEntraine:
        """Construit et persiste un `CheckpointEntraine` ; le retourne pour usage immediat par l'appelant."""
        checkpoint = CheckpointEntraine(
            identifiant=identifiant,
            chemin=chemin,
            modele_base=modele_base,
            configuration_lora=configuration_lora,
            hyperparametres=hyperparametres,
            metriques_finales=metriques_finales,
            verdict_convergence=verdict_convergence,
            horodatage=self.horloge(),
        )
        self.repository_checkpoints.sauvegarder(checkpoint)
        return checkpoint
