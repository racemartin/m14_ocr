"""
Cas d'usage : repartir les ExemplePivot anonymises en jeux
train / val / test clinique, en respectant le point de vigilance de
la mission : le jeu de test ne doit jamais etre re-utilise en
entrainement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from chsa_triage.application.echantillonnage import echantillon_stratifie
from chsa_triage.domain.model import ExemplePivot, TypeSplit
from chsa_triage.domain.ports import RepositoryLectureEcriture


@dataclass(slots=True)
class DecouperSplitsUseCase:
    """Orchestre le decoupage train/val/test clinique du dataset pivot."""

    repository            : RepositoryLectureEcriture
    graine_aleatoire        : int = 42
    proportion_val           : float = 0.10
    proportion_test           : float = 0.10
    n                        : int | None = None
    nombre_deja_assignes      : int = field(default=0, init=False)
    nombre_nouveaux            : int = field(default=0, init=False)

    def executer(self) -> dict[str, int]:
        """
        Assigne un split aux exemples anonymises qui n'en ont pas
        encore, et persiste UNIQUEMENT ces nouveaux exemples en une
        seule operation. Retourne le decompte TOTAL par split (exemples
        deja assignes lors d'executions anterieures + nouveaux de
        cette execution) -- c'est-a-dire la repartition complete
        actuelle du dataset, pas seulement ce qui vient d'etre ajoute
        (cf. `nombre_deja_assignes`/`nombre_nouveaux` pour distinguer
        les deux apres l'appel).

        Decoupage stratifie par (type_exemple, source) (08/09/2026) :
        chaque strate est melangee et coupee selon les memes
        proportions test/val/train independamment des autres, plutot
        qu'un shuffle global -- ce qui garantit que train/val/test
        contiennent chacun une part de toutes les sources et des deux
        types d'exemple (SFT/DPO), meme quand certaines sources sont
        beaucoup plus petites que d'autres (ex. FrenchMedMCQA, 595
        exemples, face a UltraMedical-Preference, 109353).

        Croissance stable, jamais de reordonnancement (10/09/2026,
        decision du capitaine -- remplace un comportement precedent
        qui recalculait TOUT le decoupage depuis zero a chaque
        execution). Un exemple qui a deja recu un `split` lors d'une
        execution anterieure n'est JAMAIS reassigne, quel que soit le
        `n` demande ensuite. Sans cette garantie, agrandir le jeu
        d'entrainement (relancer avec un `--n` plus grand, ou sans
        `--n` pour tout repartir) pouvait faire passer un exemple deja
        vu comme `train` vers `test` (ou l'inverse) a chaque nouvelle
        execution -- une fuite silencieuse de donnees d'entrainement
        dans l'evaluation, ce que le cahier des charges interdit
        explicitement ("le jeu de test ne doit jamais etre reutilise
        en entrainement").

        Algorithme : les exemples anonymises sont d'abord separes en
        deux groupes -- ceux qui ont deja un `split` (executions
        anterieures, jamais touches ici) et les candidats sans split.
        Seuls les candidats sans split peuvent devenir des "nouveaux"
        a repartir dans CETTE execution :
          - avec `n=N` : si `N` exemples sont deja assignes ou plus,
            il n'y a rien de nouveau a faire -- REDUIRE un decoupage
            deja fait n'est PAS supporte (le jeu ne peut que grandir,
            jamais retrecir : voir `nombre_nouveaux == 0` en sortie).
            Sinon, `N - nombre_deja_assignes` nouveaux exemples sont
            preleves parmi les candidats sans split par
            `echantillon_stratifie` (meme algorithme -- methode du
            plus grand reste -- que
            `AnonymiserDatasetUseCase._echantillon_stratifie`).
          - sans `n` (mode "tout") : TOUS les candidats sans split
            deviennent les "nouveaux" a repartir. Changement de
            comportement reel par rapport a l'ancien mode "tout" :
            avant, cela RECALCULAIT le decoupage de TOUT le dataset
            anonymise depuis zero (pouvait deplacer des exemples deja
            assignes) ; desormais cela COMPLETE ce qui manque, sans
            toucher aux exemples deja assignes.
        Les "nouveaux" de cette execution sont regroupes par
        (type_exemple, source), chaque groupe est melange et coupe
        selon `proportion_test`/`proportion_val`, exactement comme
        avant -- seule la POPULATION consideree a change (candidats
        sans split de cette execution), pas l'algorithme de
        repartition au sein d'un groupe.
        """
        exemples      = list(self.repository.lister(filtre={"anonymise": True}))
        deja_assignes = [e for e in exemples if e.split is not None]
        candidats     = [e for e in exemples if e.split is None]

        self.nombre_deja_assignes = len(deja_assignes)

        if self.n is None:
            nouveaux_candidats = candidats
        elif self.n <= self.nombre_deja_assignes:
            nouveaux_candidats = []
        else:
            n_a_prelever = self.n - self.nombre_deja_assignes
            if n_a_prelever < len(candidats):
                nouveaux_candidats = echantillon_stratifie(candidats, n_a_prelever, self.graine_aleatoire)
            else:
                nouveaux_candidats = candidats

        self.nombre_nouveaux = len(nouveaux_candidats)

        rng = random.Random(self.graine_aleatoire)

        groupes: dict[tuple[str, str], list[ExemplePivot]] = {}
        for exemple in nouveaux_candidats:
            cle = (exemple.type_exemple.value, exemple.source)
            groupes.setdefault(cle, []).append(exemple)

        decompte = {TypeSplit.TRAIN.value: 0, TypeSplit.VALIDATION.value: 0, TypeSplit.TEST_CLINIQUE.value: 0}
        for exemple in deja_assignes:
            decompte[exemple.split.value] += 1

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
        # N'est appele que s'il y a effectivement du nouveau : rien a
        # persister quand `nombre_nouveaux == 0` (cf. cas "reduire non
        # supporte" ci-dessus) evite une lecture/ecriture complete du
        # fichier pour rien.
        if exemples_avec_split:
            self.repository.sauvegarder_plusieurs(exemples_avec_split)

        return decompte
