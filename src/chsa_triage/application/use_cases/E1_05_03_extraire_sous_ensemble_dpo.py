"""
Cas d'usage : extraire le sous-ensemble DPO deja reparti en splits qui
doit etre publie, en retirant les exemples avec une PII residuelle
confirmee ou en attente de decision humaine. Meme patron exact que
`E1_05_02_extraire_sous_ensemble_sft.py`, filtrant `TypeExemple.DPO`
au lieu de SFT.
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

    repository: RepositoryLectureEcriture
    identifiants_a_exclure: frozenset[str] = frozenset()
    taille_cible: int = 5000

    nombre_avec_split: int = field(default=0, init=False)
    nombre_exclus: int = field(default=0, init=False)
    nombre_disponible_final: int = field(default=0, init=False)
    nombre_tronque: int = field(default=0, init=False)

    def executer(self) -> list[ExemplePivot]:
        """Exemples DPO deja repartis, moins `identifiants_a_exclure`,
        recoupes a `taille_cible` si le resultat en contient plus.
        Pure lecture : rien n'est persiste, a la charge de l'appelant."""
        avec_split = [
            e
            for e in self.repository.lister()
            if e.split is not None and e.type_exemple == TypeExemple.DPO
        ]
        self.nombre_avec_split = len(avec_split)

        resultat = [
            e
            for e in avec_split
            if e.identifiant not in self.identifiants_a_exclure
        ]
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
