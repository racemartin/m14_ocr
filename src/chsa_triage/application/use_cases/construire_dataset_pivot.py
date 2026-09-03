"""
Cas d'usage : convertir les enregistrements bruts d'une source vers
le schema pivot (ExemplePivot), puis les persister.

Le mapping "enregistrement brut -> ExemplePivot" est specifique a
chaque corpus source (MediQAl, FrenchMedMCQA, MedQuAD,
UltraMedical-Preference). Plutot que de mettre cette connaissance
dans le port (ce qui le rendrait specifique au domaine medical), on
l'injecte comme une fonction de mapping (Callable) fournie par
l'appelant (interfaces/cli), suivant le meme principe d'inversion de
dependance que pour les adaptateurs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.ports import LecteurCorpus, RepositoryLectureEcriture

FonctionMapping = Callable[[dict], "ExemplePivot | None"]


@dataclass(slots=True)
class ConstruireDatasetPivotUseCase:
    """Orchestre la conversion d'un corpus brut vers le schema pivot."""

    lecteur    : LecteurCorpus
    repository  : RepositoryLectureEcriture

    def executer(self, mapper: FonctionMapping) -> int:
        """
        Applique `mapper` a chaque enregistrement brut du corpus.
        Les enregistrements pour lesquels `mapper` retourne None sont
        ignores (ex. donnee incomplete ou hors perimetre).

        Retourne le nombre d'exemples pivot effectivement persistes.
        """
        exemples_valides: list[ExemplePivot] = []

        for enregistrement_brut in self.lecteur.lire_enregistrements():
            exemple = mapper(enregistrement_brut)
            if exemple is not None:
                exemples_valides.append(exemple)

        self.repository.sauvegarder_plusieurs(exemples_valides)
        return len(exemples_valides)
