"""
Cas d'usage : calculer la repartition REELLE des splits par strate
(type_exemple, source), pour verifier apres coup que l'echantillonnage
stratifie (`DecouperSplitsUseCase`) est bien reste representatif dans
chaque split -- pas seulement au global (deja affiche par
`decouper_splits.py`), mais strate par strate.
"""

from __future__ import annotations

from dataclasses import dataclass

from chsa_triage.domain.ports import RepositoryLectureEcriture


@dataclass(slots=True)
class VerifierRepartitionSplitsUseCase:
    """Orchestre la lecture du dataset pivot deja reparti pour en calculer la repartition par strate."""

    repository: RepositoryLectureEcriture

    def executer(self) -> dict[tuple[str, str], dict[str, int]]:
        """
        Retourne, pour chaque strate `(type_exemple, source)`, le
        decompte d'exemples par split parmi les exemples deja
        anonymises ET repartis (`anonymise=True` et `split` non nul).
        """
        repartition: dict[tuple[str, str], dict[str, int]] = {}

        for exemple in self.repository.lister(filtre={"anonymise": True}):
            if exemple.split is None:
                continue
            cle = (exemple.type_exemple.value, exemple.source)
            compteur_strate = repartition.setdefault(cle, {})
            compteur_strate[exemple.split.value] = compteur_strate.get(exemple.split.value, 0) + 1

        return repartition
