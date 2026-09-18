"""
Cas d'usage : rendre en triplet texte (prompt/chosen/rejected) tous les
`ExemplePivot` DPO d'un split donne, prets a etre tokenizes pour
l'entrainement DPO. Meme patron general que
`E2_00_uc_formater_dataset_chatml.py` (Etape 2), avec une difference
structurelle documentee en docs/04_etape3_dpo/02_etapes_cas_usage.md §3 :
ce cas d'usage FUSIONNE deux sources en memoire avant de formater
(`ExemplePivot` d'origine + `ChosenReformule` correspondant s'il existe,
pour remplacer `chosen` par `chosen_reformule`), jamais de mutation
d'aucune des deux sources sur disque.

Un exemple SANS `ChosenReformule` correspondant (hors du sous-ensemble
reformule, ou reformulation en echec pour cet identifiant precis) est
EXCLU : il n'entre pas dans le jeu d'entrainement DPO tant que sa
reformulation n'existe pas, meme logique d'exclusion que
`E1_05_00_decouper_splits.py` excluant deja les identifiants avec PII
residuelle en attente (cf. AGENTS.md).

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
    repository_reformule    : RepositoryLectureEcriture
    repository_formate       : RepositoryLectureEcriture
    formateur                  : FormateurPreference

    def executer(self, split: TypeSplit) -> int:
        """
        Lit tous les `ExemplePivot` de type `TypeExemple.DPO` du `split`
        demande, les fusionne avec leur `ChosenReformule` correspondant
        (`repository_reformule.trouver_par_id(identifiant)`) via
        `dataclasses.replace(exemple, chosen=chosen_reformule.chosen_reformule)`,
        exclut ceux sans `ChosenReformule` correspondant, rend le triplet
        fusionne via `self.formateur.formater(...)`, et persiste le
        resultat en une seule operation (`sauvegarder_plusieurs`, jamais
        un `sauvegarder()` par item, cf. AGENTS.md sur le cout O(n^2)).
        Retourne le nombre d'exemples formates.
        """
        candidats = list(
            self.repository_pivot.lister(filtre={"split": split, "type_exemple": TypeExemple.DPO})
        )

        exemples_formates = []
        for exemple in candidats:
            chosen_reformule = self.repository_reformule.trouver_par_id(exemple.identifiant)
            if chosen_reformule is None:
                continue
            exemple_fusionne = replace(exemple, chosen=chosen_reformule.chosen_reformule)
            exemples_formates.append(self.formateur.formater(exemple_fusionne))

        self.repository_formate.sauvegarder_plusieurs(exemples_formates)
        return len(exemples_formates)
