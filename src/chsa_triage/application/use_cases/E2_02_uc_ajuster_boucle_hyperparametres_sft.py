"""
Cas d'usage : boucle bornee d'ajustement d'hyperparametres SFT-LoRA.

Appelle `EntrainerSftUseCase`, passe la courbe resultante a
`application.verdict_convergence.evaluer_convergence`, et si le
verdict n'est pas `SAINE`, tire le prochain jeu d'hyperparametres
depuis `application.grille_hyperparametres.candidat_suivant(grille,
historique)` et relance. S'arrete sur `SAINE`, ou des que la grille est
epuisee (`candidat_suivant` retourne `None`) : dans ce dernier cas,
retourne le MEILLEUR essai observe (jamais une erreur silencieuse), cf.
docs/03_etape2_sft/02_etapes_cas_usage.md §5.

"Meilleur" (quand aucun essai n'est `SAINE`) : celui dont la perte du
DERNIER point de la courbe est la plus basse (perte de validation si
disponible a ce point, sinon perte d'entrainement) ; un essai `SAINE`
est toujours prefere a un essai non `SAINE`, quelle que soit sa perte
finale, puisque `SAINE` signifie deja une perte de validation qui ne
remonte pas.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from chsa_triage.application.grille_hyperparametres import candidat_suivant
from chsa_triage.application.use_cases.E2_01_uc_entrainer_sft import (
    NOM_RUN_PAR_DEFAUT,
    EntrainerSftUseCase,
)
from chsa_triage.application.verdict_convergence import evaluer_convergence
from chsa_triage.domain.model.checkpoint_entraine import VerdictConvergence
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.domain.ports.entraineur_supervise import ResultatEntrainementSFT


@dataclass(frozen=True, slots=True)
class EssaiHyperparametres:
    """Un essai unique de la boucle : jeu d'hyperparametres, resultat et verdict associes."""

    hyperparametres : HyperparametresEntrainement
    resultat          : ResultatEntrainementSFT
    verdict            : VerdictConvergence


@dataclass(frozen=True, slots=True)
class ResultatBoucleAjustement:
    """Issue complete de la boucle : meilleur essai retenu, et trace de tous les essais effectues."""

    meilleur_essai : EssaiHyperparametres
    essais           : tuple[EssaiHyperparametres, ...]


@dataclass(slots=True)
class AjusterBoucleHyperparametresSftUseCase:
    """Orchestre la boucle bornee d'ajustement d'hyperparametres SFT-LoRA."""

    cas_usage_entrainement : EntrainerSftUseCase
    grille                    : Sequence[HyperparametresEntrainement]

    def executer(
        self,
        dataset_train         : Iterable[ExempleFormate],
        dataset_validation     : Iterable[ExempleFormate],
        config_lora              : ConfigurationLora,
        hyperparametres_initiaux  : HyperparametresEntrainement,
        nom_run                    : str = NOM_RUN_PAR_DEFAUT,
    ) -> ResultatBoucleAjustement:
        """
        Boucle : entraine avec le jeu d'hyperparametres courant
        (premier essai : `hyperparametres_initiaux`), evalue la
        convergence, s'arrete sur `SAINE`. Sinon, tire le prochain
        candidat de `self.grille` non deja essaye ; s'arrete des que
        la grille est epuisee. `dataset_train`/`dataset_validation`
        sont materialises une fois (listes) pour pouvoir etre relus a
        chaque essai, un `Iterable` epuisable ne survivant pas a un
        second passage.
        """
        dataset_train = list(dataset_train)
        dataset_validation = list(dataset_validation)

        essais: list[EssaiHyperparametres] = []
        historique: list[HyperparametresEntrainement] = []
        candidat: HyperparametresEntrainement | None = hyperparametres_initiaux

        while candidat is not None:
            historique.append(candidat)
            nom_run_essai = f"{nom_run}-essai-{len(essais) + 1}"
            resultat = self.cas_usage_entrainement.entrainer(
                dataset_train, dataset_validation, config_lora, candidat, nom_run_essai
            )
            verdict = evaluer_convergence(resultat.courbe_metriques)
            essais.append(EssaiHyperparametres(hyperparametres=candidat, resultat=resultat, verdict=verdict))

            if verdict == VerdictConvergence.SAINE:
                break

            candidat = candidat_suivant(self.grille, historique)

        meilleur = min(essais, key=_cle_classement)
        return ResultatBoucleAjustement(meilleur_essai=meilleur, essais=tuple(essais))


def _cle_classement(essai: EssaiHyperparametres) -> tuple[int, float]:
    """SAINE (0) toujours avant non-SAINE (1) ; a egalite, perte finale la plus basse gagne."""
    rang_verdict = 0 if essai.verdict == VerdictConvergence.SAINE else 1
    return (rang_verdict, _perte_finale(essai.resultat))


def _perte_finale(resultat: ResultatEntrainementSFT) -> float:
    dernier_point = resultat.courbe_metriques[-1]
    if dernier_point.perte_validation is not None:
        return dernier_point.perte_validation
    return dernier_point.perte_train
