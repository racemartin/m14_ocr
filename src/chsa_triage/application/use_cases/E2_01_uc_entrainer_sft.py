"""
Cas d'usage : orchestrer un run d'entrainement SFT-LoRA unique.

Delegue l'entrainement lui-meme a `EntraineurSupervise` (port,
implemente hors GPU par un faux adaptateur en test, et par
`TrlSftEntraineurAdapter` en Environnement B, cf.
docs/03_etape2_sft/02_etapes_cas_usage.md §3) ; ce cas d'usage ne fait
que demarrer/terminer le suivi d'experimentation et relayer chaque
point de la courbe de metriques vers `SuiviExperimentation`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.ports.entraineur_supervise import EntraineurSupervise, ResultatEntrainementSFT
from chsa_triage.domain.ports.suivi_experimentation import SuiviExperimentation

NOM_RUN_PAR_DEFAUT = "sft-lora"


@dataclass(slots=True)
class EntrainerSftUseCase:
    """Orchestre un run d'entrainement SFT-LoRA complet, du demarrage a la cloture du suivi."""

    entraineur : EntraineurSupervise
    suivi        : SuiviExperimentation

    def entrainer(
        self,
        dataset_train      : Iterable[ExempleFormate],
        dataset_validation  : Iterable[ExempleFormate],
        config_lora          : ConfigurationLora,
        hyperparametres       : HyperparametresEntrainement,
        nom_run                : str = NOM_RUN_PAR_DEFAUT,
    ) -> ResultatEntrainementSFT:
        """
        Ouvre un run de suivi, delegue l'entrainement a
        `self.entraineur.entrainer(...)`, relit chaque metrique de la
        courbe resultante vers `self.suivi.logger_metrique(...)`, puis
        cloture le run. Retourne le `ResultatEntrainementSFT` complet
        (chemin du checkpoint + courbe), inchange.
        """
        parametres_run = {**asdict(config_lora), **asdict(hyperparametres)}
        self.suivi.demarrer_run(nom_run, parametres_run)

        resultat = self.entraineur.entrainer(dataset_train, dataset_validation, config_lora, hyperparametres)

        for point in resultat.courbe_metriques:
            self.suivi.logger_metrique("perte_train", point.perte_train, point.etape)
            if point.perte_validation is not None:
                self.suivi.logger_metrique("perte_validation", point.perte_validation, point.etape)
            self.suivi.logger_metrique("norme_gradient", point.norme_gradient, point.etape)

        self.suivi.terminer_run()
        return resultat
