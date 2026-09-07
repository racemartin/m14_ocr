"""
Cas d'usage : repartir les ExemplePivot anonymises en jeux
train / val / test clinique, en respectant le point de vigilance de
la mission : le jeu de test ne doit jamais etre re-utilise en
entrainement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from chsa_triage.domain.model import ExemplePivot, TypeSplit
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
        resultat en une seule operation. Retourne le decompte
        d'exemples par split.

        Decoupage stratifie par (type_exemple, source) (08/09/2026) :
        chaque strate est melangee et coupee selon les memes
        proportions test/val/train independamment des autres, plutot
        qu'un shuffle global -- ce qui garantit que train/val/test
        contiennent chacun une part de toutes les sources et des deux
        types d'exemple (SFT/DPO), meme quand certaines sources sont
        beaucoup plus petites que d'autres (ex. FrenchMedMCQA, 595
        exemples, face a UltraMedical-Preference, 109353).
        """
        exemples = list(self.repository.lister(filtre={"anonymise": True}))

        rng = random.Random(self.graine_aleatoire)

        groupes: dict[tuple[str, str], list[ExemplePivot]] = {}
        for exemple in exemples:
            cle = (exemple.type_exemple.value, exemple.source)
            groupes.setdefault(cle, []).append(exemple)

        decompte = {TypeSplit.TRAIN.value: 0, TypeSplit.VALIDATION.value: 0, TypeSplit.TEST_CLINIQUE.value: 0}
        exemples_avec_split: list[ExemplePivot] = []

        for cle in sorted(groupes):
            groupe = list(groupes[cle])
            rng.shuffle(groupe)

            n_total = len(groupe)
            n_test  = int(n_total * self.proportion_test)
            n_val   = int(n_total * self.proportion_val)

            for index, exemple in enumerate(groupe):
                if index < n_test:
                    split = TypeSplit.TEST_CLINIQUE
                elif index < n_test + n_val:
                    split = TypeSplit.VALIDATION
                else:
                    split = TypeSplit.TRAIN

                exemples_avec_split.append(replace(exemple, split=split))
                decompte[split.value] += 1

        # NOTE (08/09/2026, meme bug que celui corrige dans
        # AnonymiserDatasetUseCase) : `sauvegarder()` par iteration
        # relit/reecrit tout le fichier JSONL a chaque exemple -- O(n^2)
        # infaisable a l'echelle reelle. `sauvegarder_plusieurs` fait
        # la meme fusion par identifiant en une seule lecture/ecriture.
        self.repository.sauvegarder_plusieurs(exemples_avec_split)

        return decompte
