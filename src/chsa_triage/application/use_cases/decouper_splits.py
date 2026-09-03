"""
Cas d'usage : repartir les ExemplePivot anonymises en jeux
train / val / test clinique, en respectant le point de vigilance de
la mission : le jeu de test ne doit jamais etre re-utilise en
entrainement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from chsa_triage.domain.model import TypeSplit
from chsa_triage.domain.ports import RepositoryLectureEcriture


@dataclass(slots=True)
class DecouperSplitsUseCase:
    """Orchestre le decoupage train/val/test clinique du dataset pivot."""

    repository        : RepositoryLectureEcriture
    graine_aleatoire    : int = 42
    proportion_val       : float = 0.10
    proportion_test       : float = 0.10

    def executer(self) -> dict[str, int]:
        """
        Assigne un split a chaque exemple anonymise et persiste le
        resultat. Retourne le decompte d'exemples par split.
        """
        exemples = list(self.repository.lister(filtre={"anonymise": True}))

        rng = random.Random(self.graine_aleatoire)
        rng.shuffle(exemples)

        n_total = len(exemples)
        n_test  = int(n_total * self.proportion_test)
        n_val   = int(n_total * self.proportion_val)

        decompte = {TypeSplit.TRAIN.value: 0, TypeSplit.VALIDATION.value: 0, TypeSplit.TEST_CLINIQUE.value: 0}

        for index, exemple in enumerate(exemples):
            if index < n_test:
                split = TypeSplit.TEST_CLINIQUE
            elif index < n_test + n_val:
                split = TypeSplit.VALIDATION
            else:
                split = TypeSplit.TRAIN

            exemple_avec_split = replace(exemple, split=split)
            self.repository.sauvegarder(exemple_avec_split)
            decompte[split.value] += 1

        return decompte
