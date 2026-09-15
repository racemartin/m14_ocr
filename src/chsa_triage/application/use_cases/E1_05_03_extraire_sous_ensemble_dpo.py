"""
Cas d'usage : extraire le sous-ensemble DPO deja reparti en splits
(§7 du README, `DecouperSplitsUseCase`) qui doit etre publie (ex. sur
Hugging Face), en retirant les exemples portant une PII residuelle
confirmee ou encore en attente de decision humaine (cf.
`E1_04_01_reviser_pii_residuelle.ReviserPiiResiduelleUseCase.identifiants_a_exclure_publication`).

Meme patron exact que `E1_05_02_extraire_sous_ensemble_sft.py`
(cahier des charges §7, Livrable 1 : "SFT ~5000 paires + DPO"), en
filtrant `type_exemple == TypeExemple.DPO` au lieu de SFT : un
FILTRE/une SOUSTRACTION, jamais un nouveau muestreo, recoupe a
`taille_cible` par le meme echantillonnage stratifie
(`echantillon_stratifie`) en cas de surplus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chsa_triage.application.echantillonnage import echantillon_stratifie
from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.model.enums import TypeExemple
from chsa_triage.domain.ports import RepositoryLectureEcriture


@dataclass(slots=True)
class ExtraireSousEnsembleDpoUseCase:
    """Filtre les exemples DPO deja repartis en split, moins les identifiants exclus, recoupe a `taille_cible`."""

    repository             : RepositoryLectureEcriture
    identifiants_a_exclure : frozenset[str] = frozenset()
    taille_cible           : int = 5000

    nombre_avec_split       : int = field(default=0, init=False)
    nombre_exclus           : int = field(default=0, init=False)
    nombre_disponible_final : int = field(default=0, init=False)
    nombre_tronque          : int = field(default=0, init=False)

    def executer(self) -> list[ExemplePivot]:
        """
        Retourne les exemples `type_exemple == TypeExemple.DPO` avec
        `split is not None` dont l'identifiant n'est PAS dans
        `identifiants_a_exclure`, recoupes a `taille_cible` par
        echantillonnage stratifie (type_exemple, source) si le
        resultat filtre en contient plus. N'assigne, ne modifie ni ne
        persiste jamais rien (pure lecture) : le resultat est a ecrire
        par l'appelant (cf.
        `interfaces/cli/E1_05_03_extraire_sous_ensemble_dpo.py`).
        """
        avec_split = [
            e for e in self.repository.lister()
            if e.split is not None and e.type_exemple == TypeExemple.DPO
        ]
        self.nombre_avec_split = len(avec_split)

        resultat = [e for e in avec_split if e.identifiant not in self.identifiants_a_exclure]
        self.nombre_exclus = self.nombre_avec_split - len(resultat)
        self.nombre_disponible_final = len(resultat)

        if len(resultat) > self.taille_cible:
            resultat = echantillon_stratifie(resultat, self.taille_cible)
            self.nombre_tronque = self.nombre_disponible_final - len(resultat)
        else:
            self.nombre_tronque = 0

        return resultat

    @property
    def manque(self) -> int:
        """Nombre d'exemples manquants pour atteindre `taille_cible` (0 si deja atteint ou depasse)."""
        return max(0, self.taille_cible - self.nombre_disponible_final)
