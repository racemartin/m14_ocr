"""
Tests de `EntrainerDpoUseCase`, avec de faux adaptateurs en memoire
(pas de `trl`/`peft`, pas de mlflow/tensorboard reel), meme patron que
`tests/application/test_E2_01_uc_entrainer_sft.py`.
"""

from __future__ import annotations

from chsa_triage.application.use_cases.E3_02_uc_entrainer_dpo import EntrainerDpoUseCase
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.domain.ports.entraineur_preference import ResultatEntrainementDPO
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement

CHEMIN_CHECKPOINT_DEPART = "mombasstic/chsa-triage-sft-lora"


class FauxEntraineurPreference:
    """Faux adaptateur EntraineurPreference : retourne un resultat scripte, sans GPU."""

    def __init__(
        self,
        courbe: tuple[MetriquesEntrainement, ...],
        chemin_checkpoint: str = "checkpoints/dpo-faux/",
        metriques_recompense: dict[str, float] | None = None,
    ) -> None:
        self.courbe = courbe
        self.chemin_checkpoint = chemin_checkpoint
        self.metriques_recompense = metriques_recompense or {}
        self.appels: list[dict] = []

    def entrainer(
        self, dataset_train, dataset_validation, config_lora, hyperparametres, chemin_checkpoint_politique_depart
    ) -> ResultatEntrainementDPO:
        self.appels.append(
            {
                "dataset_train": list(dataset_train),
                "dataset_validation": list(dataset_validation),
                "config_lora": config_lora,
                "hyperparametres": hyperparametres,
                "chemin_checkpoint_politique_depart": chemin_checkpoint_politique_depart,
            }
        )
        return ResultatEntrainementDPO(
            chemin_checkpoint=self.chemin_checkpoint,
            courbe_metriques=self.courbe,
            metriques_recompense=self.metriques_recompense,
        )


class FauxSuiviExperimentation:
    """Faux adaptateur SuiviExperimentation : journalise les appels en memoire."""

    def __init__(self) -> None:
        self.runs_demarres: list[tuple[str, dict]] = []
        self.metriques_loggees: list[tuple[str, float, int]] = []
        self.runs_termines = 0

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        self.runs_demarres.append((nom, parametres))

    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None:
        self.metriques_loggees.append((nom, valeur, etape))

    def terminer_run(self) -> None:
        self.runs_termines += 1


def _config_lora() -> ConfigurationLora:
    return ConfigurationLora(rang=16, alpha=32, dropout=0.05, modules_cibles=("q_proj", "k_proj"))


def _hyperparametres() -> HyperparametresEntrainementDpo:
    return HyperparametresEntrainementDpo(
        beta=0.1, taux_apprentissage=5e-6, nombre_epoques=1, taille_lot=4,
        type_perte="sigmoid", precompute_ref_log_probs=False,
    )


def test_entrainer_dpo_delegue_a_lentraineur_et_retourne_son_resultat():
    courbe = (MetriquesEntrainement(etape=1, perte_train=0.6, perte_validation=0.65, norme_gradient=0.5),)
    entraineur = FauxEntraineurPreference(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerDpoUseCase(entraineur=entraineur, suivi=suivi)

    dataset_train = [ExempleFormatePreference(identifiant="a", texte_prompt="p", texte_chosen="c", texte_rejected="r")]
    dataset_validation = [ExempleFormatePreference(identifiant="b", texte_prompt="p", texte_chosen="c", texte_rejected="r")]
    config_lora = _config_lora()
    hyperparametres = _hyperparametres()

    resultat = cas_usage.entrainer(
        dataset_train, dataset_validation, config_lora, hyperparametres, CHEMIN_CHECKPOINT_DEPART
    )

    assert resultat.chemin_checkpoint == "checkpoints/dpo-faux/"
    assert resultat.courbe_metriques == courbe
    assert len(entraineur.appels) == 1
    assert entraineur.appels[0]["config_lora"] == config_lora
    assert entraineur.appels[0]["hyperparametres"] == hyperparametres
    assert entraineur.appels[0]["chemin_checkpoint_politique_depart"] == CHEMIN_CHECKPOINT_DEPART


def test_entrainer_dpo_demarre_et_termine_un_seul_run_de_suivi():
    courbe = (MetriquesEntrainement(etape=1, perte_train=0.6, perte_validation=0.65, norme_gradient=0.5),)
    entraineur = FauxEntraineurPreference(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerDpoUseCase(entraineur=entraineur, suivi=suivi)

    cas_usage.entrainer([], [], _config_lora(), _hyperparametres(), CHEMIN_CHECKPOINT_DEPART, nom_run="mon-run-dpo")

    assert len(suivi.runs_demarres) == 1
    assert suivi.runs_demarres[0][0] == "mon-run-dpo"
    assert suivi.runs_termines == 1


def test_entrainer_dpo_relaie_chaque_point_de_la_courbe_vers_le_suivi():
    courbe = (
        MetriquesEntrainement(etape=1, perte_train=1.0, perte_validation=1.1, norme_gradient=1.0),
        MetriquesEntrainement(etape=2, perte_train=0.8, perte_validation=None, norme_gradient=0.9),
    )
    entraineur = FauxEntraineurPreference(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerDpoUseCase(entraineur=entraineur, suivi=suivi)

    cas_usage.entrainer([], [], _config_lora(), _hyperparametres(), CHEMIN_CHECKPOINT_DEPART)

    assert ("perte_train", 1.0, 1) in suivi.metriques_loggees
    assert ("perte_validation", 1.1, 1) in suivi.metriques_loggees
    assert ("norme_gradient", 1.0, 1) in suivi.metriques_loggees
    assert ("perte_train", 0.8, 2) in suivi.metriques_loggees
    assert ("norme_gradient", 0.9, 2) in suivi.metriques_loggees
    assert not any(nom == "perte_validation" and etape == 2 for nom, _, etape in suivi.metriques_loggees)


def test_entrainer_dpo_relaie_les_4_metriques_de_recompense():
    """
    Difference structurelle par rapport au SFT (docs/04_etape3_dpo/
    02_etapes_cas_usage.md §4) : les 4 metriques `trl.DPOTrainer` sans
    equivalent SFT sont relayees en plus de la courbe generique.
    """
    courbe = (MetriquesEntrainement(etape=42, perte_train=0.5, perte_validation=0.55, norme_gradient=0.4),)
    metriques_recompense = {
        "rewards/chosen": 0.8,
        "rewards/rejected": -0.3,
        "rewards/accuracies": 0.75,
        "rewards/margins": 1.1,
    }
    entraineur = FauxEntraineurPreference(courbe, metriques_recompense=metriques_recompense)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerDpoUseCase(entraineur=entraineur, suivi=suivi)

    cas_usage.entrainer([], [], _config_lora(), _hyperparametres(), CHEMIN_CHECKPOINT_DEPART)

    for cle, valeur in metriques_recompense.items():
        assert (cle, valeur, 42) in suivi.metriques_loggees


def test_entrainer_dpo_sans_metriques_de_recompense_ne_loggue_rien_de_plus():
    courbe = (MetriquesEntrainement(etape=1, perte_train=0.6, perte_validation=0.65, norme_gradient=0.5),)
    entraineur = FauxEntraineurPreference(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerDpoUseCase(entraineur=entraineur, suivi=suivi)

    cas_usage.entrainer([], [], _config_lora(), _hyperparametres(), CHEMIN_CHECKPOINT_DEPART)

    noms_loggees = {nom for nom, _, _ in suivi.metriques_loggees}
    assert noms_loggees == {"perte_train", "perte_validation", "norme_gradient"}
