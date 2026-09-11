"""
Tests de `EntrainerSftUseCase`, avec de faux adaptateurs en memoire
(pas de `trl`/`peft`, pas de mlflow/tensorboard reel).
"""

from __future__ import annotations

from chsa_triage.application.use_cases import EntrainerSftUseCase
from chsa_triage.domain.model import ExempleFormate
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement, ResultatEntrainementSFT


class FauxEntraineurSupervise:
    """Faux adaptateur EntraineurSupervise : retourne une courbe scriptee, sans GPU."""

    def __init__(self, courbe: tuple[MetriquesEntrainement, ...], chemin_checkpoint: str = "checkpoints/faux/") -> None:
        self.courbe = courbe
        self.chemin_checkpoint = chemin_checkpoint
        self.appels: list[dict] = []

    def entrainer(self, dataset_train, dataset_validation, config_lora, hyperparametres) -> ResultatEntrainementSFT:
        self.appels.append(
            {
                "dataset_train": list(dataset_train),
                "dataset_validation": list(dataset_validation),
                "config_lora": config_lora,
                "hyperparametres": hyperparametres,
            }
        )
        return ResultatEntrainementSFT(chemin_checkpoint=self.chemin_checkpoint, courbe_metriques=self.courbe)


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


def _hyperparametres() -> HyperparametresEntrainement:
    return HyperparametresEntrainement(
        taux_apprentissage=2e-4, nombre_epoques=3, taille_lot=4, packing=True, type_perte="chunked_nll"
    )


def test_entrainer_sft_delegue_a_lentraineur_et_retourne_son_resultat():
    courbe = (MetriquesEntrainement(etape=1, perte_train=1.0, perte_validation=1.1, norme_gradient=0.5),)
    entraineur = FauxEntraineurSupervise(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerSftUseCase(entraineur=entraineur, suivi=suivi)

    dataset_train = [ExempleFormate(identifiant="a", texte="texte a")]
    dataset_validation = [ExempleFormate(identifiant="b", texte="texte b")]
    config_lora = _config_lora()
    hyperparametres = _hyperparametres()

    resultat = cas_usage.entrainer(dataset_train, dataset_validation, config_lora, hyperparametres)

    assert resultat.chemin_checkpoint == "checkpoints/faux/"
    assert resultat.courbe_metriques == courbe
    assert len(entraineur.appels) == 1
    assert entraineur.appels[0]["config_lora"] == config_lora
    assert entraineur.appels[0]["hyperparametres"] == hyperparametres


def test_entrainer_sft_demarre_et_termine_un_seul_run_de_suivi():
    courbe = (MetriquesEntrainement(etape=1, perte_train=1.0, perte_validation=1.1, norme_gradient=0.5),)
    entraineur = FauxEntraineurSupervise(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerSftUseCase(entraineur=entraineur, suivi=suivi)

    cas_usage.entrainer([], [], _config_lora(), _hyperparametres(), nom_run="mon-run")

    assert len(suivi.runs_demarres) == 1
    assert suivi.runs_demarres[0][0] == "mon-run"
    assert suivi.runs_termines == 1


def test_entrainer_sft_relaie_chaque_point_de_la_courbe_vers_le_suivi():
    courbe = (
        MetriquesEntrainement(etape=1, perte_train=2.0, perte_validation=2.1, norme_gradient=1.0),
        MetriquesEntrainement(etape=2, perte_train=1.5, perte_validation=None, norme_gradient=0.9),
    )
    entraineur = FauxEntraineurSupervise(courbe)
    suivi = FauxSuiviExperimentation()
    cas_usage = EntrainerSftUseCase(entraineur=entraineur, suivi=suivi)

    cas_usage.entrainer([], [], _config_lora(), _hyperparametres())

    # Etape 1 : perte_train, perte_validation, norme_gradient. Etape 2 : perte_validation absente (None non loggee).
    assert ("perte_train", 2.0, 1) in suivi.metriques_loggees
    assert ("perte_validation", 2.1, 1) in suivi.metriques_loggees
    assert ("norme_gradient", 1.0, 1) in suivi.metriques_loggees
    assert ("perte_train", 1.5, 2) in suivi.metriques_loggees
    assert ("norme_gradient", 0.9, 2) in suivi.metriques_loggees
    assert not any(nom == "perte_validation" and etape == 2 for nom, _, etape in suivi.metriques_loggees)
