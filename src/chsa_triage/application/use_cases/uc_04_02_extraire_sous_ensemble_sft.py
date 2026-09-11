"""
Cas d'usage : extraire le sous-ensemble deja reparti en splits
(§7 du README, `DecouperSplitsUseCase`) qui doit etre publie (ex. sur
Hugging Face), en retirant les exemples portant une PII residuelle
confirmee ou encore en attente de decision humaine (cf.
`uc_03_03_reviser_pii_residuelle.ReviserPiiResiduelleUseCase.identifiants_a_exclure_publication`).

Design (11/09/2026) : un FILTRE/une SOUSTRACTION,
pas un nouveau muestreo. Si le resultat, apres exclusion, est plus
petit que `taille_cible`, ce cas d'usage ne tente jamais de completer
automatiquement (ce role reste a `DecouperSplitsUseCase --n`, appele
separement avant de relancer l'extraction) ; il se contente de
rapporter honnetement combien d'exemples restent et combien manquent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.ports import RepositoryLectureEcriture


@dataclass(slots=True)
class ExtraireSousEnsembleSftUseCase:
    """Filtre les exemples deja repartis en split, moins les identifiants exclus, jusqu'a `taille_cible`."""

    repository             : RepositoryLectureEcriture
    identifiants_a_exclure : frozenset[str] = frozenset()
    taille_cible           : int = 5000

    nombre_avec_split       : int = field(default=0, init=False)
    nombre_exclus           : int = field(default=0, init=False)
    nombre_disponible_final : int = field(default=0, init=False)

    def executer(self) -> list[ExemplePivot]:
        """
        Retourne les exemples `split is not None` dont l'identifiant
        n'est PAS dans `identifiants_a_exclure`. N'assigne, ne modifie
        ni ne persiste jamais rien (pure lecture) : le resultat est a
        ecrire par l'appelant (cf. `interfaces/cli/extraire_sous_ensemble_sft.py`).
        """
        avec_split = [e for e in self.repository.lister() if e.split is not None]
        self.nombre_avec_split = len(avec_split)

        resultat = [e for e in avec_split if e.identifiant not in self.identifiants_a_exclure]
        self.nombre_exclus = self.nombre_avec_split - len(resultat)
        self.nombre_disponible_final = len(resultat)

        return resultat

    @property
    def manque(self) -> int:
        """Nombre d'exemples manquants pour atteindre `taille_cible` (0 si deja atteint ou depasse)."""
        return max(0, self.taille_cible - self.nombre_disponible_final)
