"""
Cas d'usage : repartir les ExemplePivot anonymises en jeux
train / val / test clinique, en respectant le point de vigilance de
la mission : le jeu de test ne doit jamais etre re-utilise en
entrainement.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from chsa_triage.application.echantillonnage import echantillon_stratifie
from chsa_triage.domain.model import ExemplePivot, TypeSplit
from chsa_triage.domain.ports import RepositoryLectureEcriture


def _aucun_identifiant_en_attente() -> set[str]:
    """Valeur par defaut de `obtenir_identifiants_pii_en_attente` : aucune exclusion."""
    return set()


# Nom historique : couvre en realite deux motifs, cf.
# ReviserPiiResiduelleUseCase.identifiants_a_exclure_publication_set.


@dataclass(slots=True)
class DecouperSplitsUseCase:
    """Orchestre le decoupage train/val/test clinique du dataset pivot."""

    repository: RepositoryLectureEcriture
    graine_aleatoire: int = 42
    proportion_val: float = 0.10
    proportion_test: float = 0.10
    n: int | None = None
    obtenir_identifiants_pii_en_attente: Callable[[], set[str]] = (
        _aucun_identifiant_en_attente
    )
    nombre_deja_assignes: int = field(default=0, init=False)
    nombre_nouveaux: int = field(default=0, init=False)
    nombre_exclus_pii_en_attente: int = field(default=0, init=False)

    def executer(self) -> dict[str, int]:
        """
        Assigne un split aux exemples anonymises qui n'en ont pas
        encore, et persiste uniquement ces nouveaux exemples. Retourne
        le decompte TOTAL par split (deja assignes + nouveaux) ; cf.
        `nombre_deja_assignes`/`nombre_nouveaux` pour distinguer les
        deux apres l'appel.

        Garantie de croissance stable : un exemple deja `split` lors
        d'une execution anterieure n'est JAMAIS reassigne, quel que
        soit `n` ensuite. Sans cette garantie, agrandir le jeu
        (`--n` plus grand) pouvait faire passer un exemple de `train` a
        `test` d'une execution a l'autre — une fuite d'entrainement
        dans l'evaluation, interdite par le cahier des charges. Le
        decoupage est stratifie par (type_exemple, source), pour que
        train/val/test contiennent chacun une part de toutes les
        sources meme tres inegales en taille (ex. FrenchMedMCQA 595
        exemples vs UltraMedical-Preference 109353).

        Exclusion PII : `obtenir_identifiants_pii_en_attente` (defaut :
        aucune exclusion) retire des candidats, avant le decoupage, les
        exemples avec une PII residuelle CONFIRMEE ou encore SANS
        decision humaine — jamais les `deja_assignes`. Ils recevront un
        split lors d'une execution future, une fois la decision prise.
        """
        exemples = list(self.repository.lister(filtre={"anonymise": True}))
        deja_assignes = [e for e in exemples if e.split is not None]
        candidats = [e for e in exemples if e.split is None]

        # Exclusion PII par precaution : seuls les `candidats` (sans
        # split) sont concernes, jamais `deja_assignes` (croissance
        # stable, cf. docstring de executer()).
        identifiants_en_attente = self.obtenir_identifiants_pii_en_attente()
        if identifiants_en_attente:
            avant = len(candidats)
            candidats = [
                e
                for e in candidats
                if e.identifiant not in identifiants_en_attente
            ]
            self.nombre_exclus_pii_en_attente = avant - len(candidats)

        self.nombre_deja_assignes = len(deja_assignes)

        if self.n is None:
            nouveaux_candidats = candidats
        elif self.n <= self.nombre_deja_assignes:
            nouveaux_candidats = []
        else:
            n_a_prelever = self.n - self.nombre_deja_assignes
            if n_a_prelever < len(candidats):
                nouveaux_candidats = echantillon_stratifie(
                    candidats, n_a_prelever, self.graine_aleatoire
                )
            else:
                nouveaux_candidats = candidats

        self.nombre_nouveaux = len(nouveaux_candidats)

        rng = random.Random(self.graine_aleatoire)

        groupes: dict[tuple[str, str], list[ExemplePivot]] = {}
        for exemple in nouveaux_candidats:
            cle = (exemple.type_exemple.value, exemple.source)
            groupes.setdefault(cle, []).append(exemple)

        decompte = {
            TypeSplit.TRAIN.value: 0,
            TypeSplit.VALIDATION.value: 0,
            TypeSplit.TEST_CLINIQUE.value: 0,
        }
        for exemple in deja_assignes:
            decompte[exemple.split.value] += 1

        exemples_avec_split: list[ExemplePivot] = []

        for cle in sorted(groupes):
            groupe = list(groupes[cle])
            rng.shuffle(groupe)

            n_total = len(groupe)
            n_test = int(n_total * self.proportion_test)
            n_val = int(n_total * self.proportion_val)

            for index, exemple in enumerate(groupe):
                if index < n_test:
                    split = TypeSplit.TEST_CLINIQUE
                elif index < n_test + n_val:
                    split = TypeSplit.VALIDATION
                else:
                    split = TypeSplit.TRAIN

                exemples_avec_split.append(replace(exemple, split=split))
                decompte[split.value] += 1

        # Meme piege O(n^2) que AnonymiserDatasetUseCase : sauvegarder_plusieurs
        # en une seule ecriture, et seulement s'il y a du nouveau.
        if exemples_avec_split:
            self.repository.sauvegarder_plusieurs(exemples_avec_split)

        return decompte
