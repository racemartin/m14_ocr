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
from dataclasses import dataclass, field

from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.ports import LecteurCorpus, RepositoryLectureEcriture

FonctionMapping = Callable[[dict], "ExemplePivot | None"]


@dataclass(slots=True)
class ConstruireDatasetPivotUseCase:
    """Orchestre la conversion d'un corpus brut vers le schema pivot.

    Dedoublonnage : `nouvel_identifiant` etant deterministe, deux
    enregistrements qui produisent le meme identifiant sont de vrais
    doublons, pas une collision. Seul le premier est conserve ; les
    suivants sont ecartes et exposes via `self.doublons` (jamais
    silencieusement perdus).
    """

    lecteur: LecteurCorpus
    repository: RepositoryLectureEcriture
    doublons: list[ExemplePivot] = field(default_factory=list, init=False)

    def executer(self, mapper: FonctionMapping) -> int:
        """
        Applique `mapper` a chaque enregistrement brut du corpus.
        Les enregistrements pour lesquels `mapper` retourne None sont
        ignores (ex. donnee incomplete ou hors perimetre).

        Retourne le nombre d'exemples pivot EFFECTIVEMENT persistes
        (doublons exclus ; cf. `self.doublons` pour ce qui a ete
        ecarte).
        """
        exemples_par_id: dict[str, ExemplePivot] = {}
        self.doublons = []

        for enregistrement_brut in self.lecteur.lire_enregistrements():
            exemple = mapper(enregistrement_brut)
            if exemple is None:
                continue
            if exemple.identifiant in exemples_par_id:
                self.doublons.append(exemple)
                continue
            exemples_par_id[exemple.identifiant] = exemple

        self.repository.sauvegarder_plusieurs(exemples_par_id.values())
        return len(exemples_par_id)
