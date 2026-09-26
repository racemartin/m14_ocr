"""
Cas d'usage : rendre en triplet texte (prompt/chosen/rejected) tous les
`ExemplePivot` DPO d'un split donne. Meme patron general que
`E2_00_uc_formater_dataset_chatml.py`, mais FUSIONNE deux sources en
memoire avant de formater (`ExemplePivot` d'origine + `ChosenReformule`
correspondant s'il existe, pour remplacer `chosen`), sans jamais muter
aucune des deux sources sur disque.

Formatage decouple de la reformulation : un exemple SANS `ChosenReformule`
(hors du sous-ensemble reformule, ou reformulation en echec) n'est PAS
exclu — le `chosen` ORIGINAL est utilise tel quel. Le pipeline DPO peut
donc tourner sur des paires originales sans dependre de la reformulation.

Filtre sur `TypeExemple.DPO` (jamais SFT, meme bug de fuite deja corrige
en Etape 1, cf. AGENTS.md "SFT/DPO type leak").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from chsa_triage.domain.model.enums import TypeExemple, TypeSplit
from chsa_triage.domain.ports.dataset_repository import (
    RepositoryLectureEcriture,
)
from chsa_triage.domain.ports.formateur_preference import FormateurPreference


@dataclass(slots=True)
class FormaterDatasetChatMLPreferenceUseCase:
    """Orchestre le rendu en triplet texte (prompt/chosen/rejected) d'un split du dataset pivot DPO."""

    repository_pivot: RepositoryLectureEcriture
    repository_reformule: RepositoryLectureEcriture | None
    repository_formate: RepositoryLectureEcriture
    formateur: FormateurPreference

    def executer(self, split: TypeSplit) -> int:
        """Lit les `ExemplePivot` DPO du `split` demande, les fusionne
        avec leur `ChosenReformule` s'il existe (sinon `chosen` original,
        cf. docstring du module), rend le triplet et persiste en une
        seule ecriture. Retourne le nombre d'exemples formates."""
        candidats = list(
            self.repository_pivot.lister(
                filtre={"split": split, "type_exemple": TypeExemple.DPO}
            )
        )

        exemples_formates = []
        for exemple in candidats:
            if self.repository_reformule is not None:
                chosen_reformule = self.repository_reformule.trouver_par_id(
                    exemple.identifiant
                )
            else:
                chosen_reformule = None
            if chosen_reformule is not None:
                exemple_fusionne = replace(
                    exemple, chosen=chosen_reformule.chosen_reformule
                )
            else:
                # Fallback au chosen original si aucune reformulation
                # n'existe pour cet identifiant (formatage decouple de
                # la reformulation, cf. docstring du module).
                exemple_fusionne = exemple
            exemples_formates.append(
                self.formateur.formater_preference(exemple_fusionne)
            )

        self.repository_formate.sauvegarder_plusieurs(exemples_formates)
        return len(exemples_formates)
