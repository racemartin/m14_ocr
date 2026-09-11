"""
Cas d'usage : rendre en ChatML tous les `ExemplePivot` d'un split
donne, prets a etre tokenizes pour l'entrainement SFT.

Reutilise directement `RepositoryLectureEcriture` (domain/ports,
deja utilise partout en Etape 1) pour LIRE des `ExemplePivot`, et une
SECONDE instance du meme port generique, parametree cette fois sur
`ExempleFormate`, pour ECRIRE le resultat : preuve concrete que "un
port generique peut servir a n'importe quel type d'entite"
(docs/01_environnement/01_architecture_hexagonale.md), cf.
docs/03_etape2_sft/02_etapes_cas_usage.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass

from chsa_triage.domain.model.enums import TypeSplit
from chsa_triage.domain.ports.dataset_repository import RepositoryLectureEcriture
from chsa_triage.domain.ports.formateur_conversation import FormateurConversation


@dataclass(slots=True)
class FormaterDatasetChatMLUseCase:
    """Orchestre le rendu ChatML d'un split du dataset pivot."""

    repository_pivot   : RepositoryLectureEcriture
    repository_formate  : RepositoryLectureEcriture
    formateur            : FormateurConversation

    def executer(self, split: TypeSplit) -> int:
        """
        Lit tous les `ExemplePivot` du `split` demande, les rend en
        ChatML via `self.formateur`, et persiste le resultat en une
        seule operation (`sauvegarder_plusieurs`, jamais un
        `sauvegarder()` par item, cf. AGENTS.md sur le cout O(n^2) de
        `JsonlDatasetRepository.sauvegarder`). Retourne le nombre
        d'exemples formates.
        """
        candidats = list(self.repository_pivot.lister(filtre={"split": split}))
        exemples_formates = [self.formateur.formater(exemple) for exemple in candidats]
        self.repository_formate.sauvegarder_plusieurs(exemples_formates)
        return len(exemples_formates)
