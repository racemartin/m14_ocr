"""
Tests de `AjusterBoucleHyperparametresSftUseCase`, avec de faux
adaptateurs en memoire dont le faux `EntraineurSupervise` retourne des
courbes `MetriquesEntrainement` SCRIPTEES (une par appel, dans l'ordre)
pour exercer chaque branche de la boucle : converge au premier essai,
converge apres 1-2 reessais, epuise la grille sans converger.
"""

from __future__ import annotations

from chsa_triage.application.use_cases import (
    AjusterBoucleHyperparametresSftUseCase,
    EntrainerSftUseCase,
)
from chsa_triage.domain.model.checkpoint_entraine import VerdictConvergence
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement, ResultatEntrainementSFT


class FauxEntraineurSuperviseScripte:
    """
    Faux adaptateur EntraineurSupervise dont la courbe retournee est
    prise dans une liste SCRIPTEE, une par appel successif (dans
    l'ordre d'appel). Leve si la boucle appelle plus de fois que de
    courbes scriptees : signe d'une boucle non bornee.
    """

    def __init__(self, courbes: list[tuple[MetriquesEntrainement, ...]]) -> None:
        self.courbes = courbes
        self.appels: list[HyperparametresEntrainement] = []

    def entrainer(self, dataset_train, dataset_validation, config_lora, hyperparametres) -> ResultatEntrainementSFT:
        indice = len(self.appels)
        self.appels.append(hyperparametres)
        return ResultatEntrainementSFT(
            chemin_checkpoint=f"checkpoints/essai-{indice}/", courbe_metriques=self.courbes[indice]
        )


class FauxSuiviExperimentation:
    """Faux adaptateur SuiviExperimentation : ne fait qu'accepter les appels, rien a verifier ici."""

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        pass

    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None:
        pass

    def terminer_run(self) -> None:
        pass


def _config_lora() -> ConfigurationLora:
    return ConfigurationLora(rang=16, alpha=32, dropout=0.05, modules_cibles=("q_proj", "k_proj"))


def _hyperparametres(taux_apprentissage: float) -> HyperparametresEntrainement:
    return HyperparametresEntrainement(
        taux_apprentissage=taux_apprentissage, nombre_epoques=3, taille_lot=4, packing=True, type_perte="chunked_nll"
    )


def _courbe_saine() -> tuple[MetriquesEntrainement, ...]:
    return (
        MetriquesEntrainement(etape=1, perte_train=2.0, perte_validation=2.1, norme_gradient=1.0),
        MetriquesEntrainement(etape=2, perte_train=1.1, perte_validation=1.2, norme_gradient=0.9),
        MetriquesEntrainement(etape=3, perte_train=0.9, perte_validation=0.95, norme_gradient=0.8),
    )


def _courbe_surapprentissage() -> tuple[MetriquesEntrainement, ...]:
    return (
        MetriquesEntrainement(etape=1, perte_train=2.0, perte_validation=2.0, norme_gradient=1.0),
        MetriquesEntrainement(etape=2, perte_train=1.5, perte_validation=1.4, norme_gradient=0.9),
        MetriquesEntrainement(etape=3, perte_train=1.0, perte_validation=1.3, norme_gradient=0.9),
        MetriquesEntrainement(etape=4, perte_train=0.7, perte_validation=1.8, norme_gradient=0.9),
    )


def _courbe_sous_apprentissage() -> tuple[MetriquesEntrainement, ...]:
    return (
        MetriquesEntrainement(etape=1, perte_train=2.00, perte_validation=2.05, norme_gradient=1.0),
        MetriquesEntrainement(etape=2, perte_train=1.98, perte_validation=2.03, norme_gradient=1.0),
    )


def test_boucle_converge_des_le_premier_essai():
    hp_initiaux = _hyperparametres(1e-4)
    grille = [_hyperparametres(2e-4), _hyperparametres(5e-4)]  # jamais consommee
    entraineur = FauxEntraineurSuperviseScripte(courbes=[_courbe_saine()])
    cas_usage_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=FauxSuiviExperimentation())
    boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_usage_entrainement, grille=grille)

    resultat = boucle.executer([], [], _config_lora(), hp_initiaux)

    assert len(resultat.essais) == 1
    assert resultat.meilleur_essai.verdict == VerdictConvergence.SAINE
    assert resultat.meilleur_essai.hyperparametres == hp_initiaux
    assert entraineur.appels == [hp_initiaux]


def test_boucle_converge_apres_un_reessai():
    hp_initiaux = _hyperparametres(1e-4)
    hp_suivant = _hyperparametres(2e-4)
    grille = [hp_initiaux, hp_suivant]
    entraineur = FauxEntraineurSuperviseScripte(courbes=[_courbe_surapprentissage(), _courbe_saine()])
    cas_usage_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=FauxSuiviExperimentation())
    boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_usage_entrainement, grille=grille)

    resultat = boucle.executer([], [], _config_lora(), hp_initiaux)

    assert len(resultat.essais) == 2
    assert [e.verdict for e in resultat.essais] == [VerdictConvergence.SURAPPRENTISSAGE, VerdictConvergence.SAINE]
    assert resultat.meilleur_essai.verdict == VerdictConvergence.SAINE
    assert resultat.meilleur_essai.hyperparametres == hp_suivant
    assert entraineur.appels == [hp_initiaux, hp_suivant]


def test_boucle_converge_apres_deux_reessais():
    hp_initiaux = _hyperparametres(1e-4)
    hp_2 = _hyperparametres(2e-4)
    hp_3 = _hyperparametres(5e-4)
    grille = [hp_initiaux, hp_2, hp_3]
    entraineur = FauxEntraineurSuperviseScripte(
        courbes=[_courbe_surapprentissage(), _courbe_sous_apprentissage(), _courbe_saine()]
    )
    cas_usage_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=FauxSuiviExperimentation())
    boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_usage_entrainement, grille=grille)

    resultat = boucle.executer([], [], _config_lora(), hp_initiaux)

    assert len(resultat.essais) == 3
    assert resultat.meilleur_essai.verdict == VerdictConvergence.SAINE
    assert resultat.meilleur_essai.hyperparametres == hp_3
    assert entraineur.appels == [hp_initiaux, hp_2, hp_3]


def test_boucle_epuise_la_grille_sans_converger_retourne_le_meilleur_essai():
    hp_initiaux = _hyperparametres(1e-4)
    hp_2 = _hyperparametres(2e-4)
    grille = [hp_initiaux, hp_2]
    # Aucune des deux courbes n'est SAINE : la 1ere a une perte finale de validation plus basse (1.8 < 2.03).
    entraineur = FauxEntraineurSuperviseScripte(
        courbes=[_courbe_surapprentissage(), _courbe_sous_apprentissage()]
    )
    cas_usage_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=FauxSuiviExperimentation())
    boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_usage_entrainement, grille=grille)

    resultat = boucle.executer([], [], _config_lora(), hp_initiaux)

    assert len(resultat.essais) == 2
    assert entraineur.appels == [hp_initiaux, hp_2]  # grille epuisee : pas de 3e appel
    assert all(e.verdict != VerdictConvergence.SAINE for e in resultat.essais)
    # _courbe_surapprentissage se termine a perte_validation=1.8, _courbe_sous_apprentissage a 2.03 : le 1er gagne.
    assert resultat.meilleur_essai.hyperparametres == hp_initiaux
    assert resultat.meilleur_essai.verdict == VerdictConvergence.SURAPPRENTISSAGE


def test_boucle_materialise_les_iterables_epuisables_pour_chaque_essai():
    """dataset_train/validation passes en iterateur (pas juste liste) doivent survivre a plusieurs essais."""
    hp_initiaux = _hyperparametres(1e-4)
    hp_2 = _hyperparametres(2e-4)
    grille = [hp_initiaux, hp_2]
    entraineur = FauxEntraineurSuperviseScripte(courbes=[_courbe_surapprentissage(), _courbe_saine()])
    cas_usage_entrainement = EntrainerSftUseCase(entraineur=entraineur, suivi=FauxSuiviExperimentation())
    boucle = AjusterBoucleHyperparametresSftUseCase(cas_usage_entrainement=cas_usage_entrainement, grille=grille)

    resultat = boucle.executer(iter(["train-1", "train-2"]), iter(["val-1"]), _config_lora(), hp_initiaux)

    assert len(resultat.essais) == 2
