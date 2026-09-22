"""
Cas d'usage : rendre en triplet texte (prompt/chosen/rejected) tous les
`ExemplePivot` DPO d'un split donne, prets a etre tokenises pour
l'entrainement DPO. Meme patron general que
`E2_00_uc_formater_dataset_chatml.py` (Etape 2), avec une difference
structurelle documentee en docs/04_etape3_dpo/02_etapes_cas_usage.md §3 :
ce cas d'usage FUSIONNE deux sources en memoire avant de formater
(`ExemplePivot` d'origine + `ChosenReformule` correspondant s'il existe,
pour remplacer `chosen` par `chosen_reformule`), jamais de mutation
d'aucune des deux sources sur disque.

DECISION DESACOPLADA (19/09/2026) : un exemple SANS `ChosenReformule`
correspondant (hors du sous-ensemble reformule, ou reformulation en
echec) n'est PLUS exclu. Au lieu de cela, le `chosen` ORIGINAL est utilise
tel quel (fallback au texte medical direct). Ceci desacouple le
formatage DPO du processus de reformulation : le pipeline DPO peut
maintenant s'executer sur des paires chose/rejected originales sans
requerir la reformulation vers
+ JSON. La reformulation reste OPTIONNELLE : si `ChosenReformule` existe
pour un identifiant, il remplace `chosen` ; sinon, `chosen` original est
utilise.

Filtre sur `type_exemple == TypeExemple.DPO` (jamais SFT, meme bug de
fuite deja documente et corrige en Etape 1, cf. AGENTS.md "SFT/DPO type
leak").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from chsa_triage.domain.model.enums import TypeExemple, TypeSplit
from chsa_triage.domain.ports.dataset_repository import RepositoryLectureEcriture
from chsa_triage.domain.ports.formateur_preference import FormateurPreference


@dataclass(slots=True)
class FormaterDatasetChatMLPreferenceUseCase:
    """Orchestre le rendu en triplet texte (prompt/chosen/rejected) d'un split du dataset pivot DPO."""

    repository_pivot      : RepositoryLectureEcriture
    repository_reformule    : RepositoryLectureEcriture | None
    repository_formate       : RepositoryLectureEcriture
    formateur                  : FormateurPreference

    def executer(self, split: TypeSplit) -> int:
        """
        Lit tous les `ExemplePivot` de type `TypeExemple.DPO` du `split`
        demande, les fusionne avec leur `ChosenReformule` correspondant
        (`repository_reformule.trouver_par_id(identifiant)`) via
        `dataclasses.replace(exemple, chosen=chosen_reformule.chosen_reformule)`,
        PERSISTS les exemples SANS `ChosenReformule` en utilisant le
        `chosen` original tel quel (fallback, cf. docstring du module :
        la reformulation est desacoupee du formatage DPO). Rend le
        triplet fusionne via `self.formateur.formater_preference(...)`, et persiste
        le resultat en une seule operation (`sauvegarder_plusieurs`,
        jamais un `sauvegarder()` par item, cf. AGENTS.md sur le cout
        O(n^2)). Retourne le nombre d'exemples formates.
        """
        candidats = list(
            self.repository_pivot.lister(filtre={"split": split, "type_exemple": TypeExemple.DPO})
        )

        exemples_formates = []
        for exemple in candidats:
            if self.repository_reformule is not None:
                chosen_reformule = self.repository_reformule.trouver_par_id(exemple.identifiant)
            else:
                chosen_reformule = None
            if chosen_reformule is not None:
                exemple_fusionne = replace(exemple, chosen=chosen_reformule.chosen_reformule)
            else:
                # DECISION DESACOPLADA : fallback au chosen original si
                # aucune reformulation n'existe pour cet identifiant.
                # La reformulation est OPTIONNELLE, le pipeline DPO peut
                # tourner sur les paires originales sans  + JSON.
                exemple_fusionne = exemple
            exemples_formates.append(self.formateur.formater_preference(exemple_fusionne))

        self.repository_formate.sauvegarder_plusieurs(exemples_formates)
        return len(exemples_formates)
