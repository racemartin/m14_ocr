"""
Cas d'usage : orchestrer un run d'entrainement DPO unique, a partir
d'un checkpoint SFT-LoRA de depart. Meme structure d'orchestration que
`EntrainerSftUseCase` (Etape 2), adaptee au port `EntraineurPreference`
(cf. docs/04_etape3_dpo/02_etapes_cas_usage.md §4).

Difference a journaliser en plus du SFT : `trl.DPOTrainer` journalise
des metriques propres au DPO sans equivalent SFT (`rewards/chosen`,
`rewards/rejected`, `rewards/accuracies`, `rewards/margins`, noms reels
de l'API `trl`, cf. docs/04_etape3_dpo/00_introduction_concepts.md
§4.4). Relayees directement vers `SuiviExperimentation.logger_metrique(...)`,
en plus de `perte_train`/`perte_validation`/`norme_gradient`, sans
passer par `evaluer_convergence()` : une extension additive de la
boucle de suivi, jamais une modification du diagnostic de convergence
existant (reutilise tel quel, cf. `application/verdict_convergence.py`).
Ces 4 valeurs voyagent via `ResultatEntrainementDPO.metriques_recompense`,
champ AJOUTE au port (ecart documente dans
`domain/ports/entraineur_preference.py`, absent de l'esquisse du guide
d'implementation Etape 3).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.domain.ports.entraineur_preference import EntraineurPreference, ResultatEntrainementDPO
from chsa_triage.domain.ports.suivi_experimentation import SuiviExperimentation

NOM_RUN_PAR_DEFAUT = "dpo-lora"

# Metriques propres a trl.DPOTrainer, sans equivalent SFT (verifiees
# reellement, trl==1.13.0, cf. docs/04_etape3_dpo/00_introduction_concepts.md
# §4.4) : relayees telles quelles vers SuiviExperimentation, jamais
# consommees par evaluer_convergence().
CLES_METRIQUES_RECOMPENSE_DPO = (
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
)


@dataclass(slots=True)
class EntrainerDpoUseCase:
    """Orchestre un run d'entrainement DPO complet, du demarrage a la cloture du suivi."""

    entraineur : EntraineurPreference
    suivi        : SuiviExperimentation

    def entrainer(
        self,
        dataset_train                        : Iterable[ExempleFormatePreference],
        dataset_validation                    : Iterable[ExempleFormatePreference],
        config_lora                             : ConfigurationLora,
        hyperparametres                           : HyperparametresEntrainementDpo,
        chemin_checkpoint_politique_depart          : str,
        nom_run                                       : str = NOM_RUN_PAR_DEFAUT,
    ) -> ResultatEntrainementDPO:
        """
        Ouvre un run de suivi, delegue l'entrainement a
        `self.entraineur.entrainer(...)`, relaie chaque point de la
        courbe resultante (`perte_train`/`perte_validation`/
        `norme_gradient`) ainsi que les metriques de recompense DPO
        presentes dans `resultat.metriques_recompense`
        (`CLES_METRIQUES_RECOMPENSE_DPO`, valeurs agregees finales, pas
        par etape) vers `self.suivi.logger_metrique(...)`, puis cloture
        le run. Retourne le `ResultatEntrainementDPO` complet (chemin du
        checkpoint + courbe + metriques de recompense), inchange.
        """
        parametres_run = {
            **asdict(config_lora),
            **asdict(hyperparametres),
            "chemin_checkpoint_politique_depart": chemin_checkpoint_politique_depart,
        }
        self.suivi.demarrer_run(nom_run, parametres_run)

        resultat = self.entraineur.entrainer(
            dataset_train,
            dataset_validation,
            config_lora,
            hyperparametres,
            chemin_checkpoint_politique_depart,
        )

        for point in resultat.courbe_metriques:
            self.suivi.logger_metrique("perte_train", point.perte_train, point.etape)
            if point.perte_validation is not None:
                self.suivi.logger_metrique("perte_validation", point.perte_validation, point.etape)
            self.suivi.logger_metrique("norme_gradient", point.norme_gradient, point.etape)

        etape_finale = resultat.courbe_metriques[-1].etape if resultat.courbe_metriques else 0
        for cle in CLES_METRIQUES_RECOMPENSE_DPO:
            if cle in resultat.metriques_recompense:
                self.suivi.logger_metrique(cle, resultat.metriques_recompense[cle], etape_finale)

        self.suivi.terminer_run()
        return resultat
