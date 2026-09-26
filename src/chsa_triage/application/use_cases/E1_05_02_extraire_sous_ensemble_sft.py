"""
Cas d'usage : extraire le sous-ensemble deja reparti en splits qui doit
etre publie (ex. Hugging Face), en retirant les exemples avec une PII
residuelle confirmee ou en attente de decision humaine. Filtre/soustrait
seulement (jamais de nouveau tirage) : si le resultat est trop petit,
rapporte honnetement le manque plutot que de completer automatiquement
(`DecouperSplitsUseCase --n` fait ce role, en amont) ; s'il est trop
grand, recoupe a `taille_cible` par echantillonnage stratifie.

Bug reel corrige : le filtre initial ne portait que sur `split is not
None`, sans `type_exemple` — un exemple DPO (UltraMedical-Preference)
pouvait donc fuiter dans le sous-ensemble "SFT" publie, puisque
`DecouperSplitsUseCase` reparti les deux types dans les memes splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chsa_triage.application.echantillonnage import echantillon_stratifie
from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.model.enums import TypeExemple
from chsa_triage.domain.ports import RepositoryLectureEcriture

ORDRE_SPLITS = ("train", "val", "test")


@dataclass(slots=True)
class ExtraireSousEnsembleSftUseCase:
    """Filtre les exemples SFT deja repartis en split, moins les identifiants exclus, recoupe a `taille_cible`."""

    repository: RepositoryLectureEcriture
    identifiants_a_exclure: frozenset[str] = frozenset()
    taille_cible: int = 5000

    nombre_avec_split: int = field(default=0, init=False)
    nombre_exclus: int = field(default=0, init=False)
    nombre_disponible_final: int = field(default=0, init=False)
    nombre_tronque: int = field(default=0, init=False)

    def executer(self) -> list[ExemplePivot]:
        """Exemples SFT deja repartis, moins `identifiants_a_exclure`,
        recoupes a `taille_cible` si le resultat en contient plus.
        Pure lecture : rien n'est persiste, a la charge de l'appelant."""
        avec_split = [
            e
            for e in self.repository.lister()
            if e.split is not None and e.type_exemple == TypeExemple.SFT
        ]
        self.nombre_avec_split = len(avec_split)

        resultat = [
            e
            for e in avec_split
            if e.identifiant not in self.identifiants_a_exclure
        ]
        self.nombre_exclus = self.nombre_avec_split - len(resultat)
        self.nombre_disponible_final = len(resultat)

        if len(resultat) > self.taille_cible:
            resultat = echantillon_stratifie(resultat, self.taille_cible)
            self.nombre_tronque = self.nombre_disponible_final - len(resultat)
        else:
            self.nombre_tronque = 0

        return resultat

    @property
    def manque(self) -> int:
        """Nombre d'exemples manquants pour atteindre `taille_cible` (0 si deja atteint ou depasse)."""
        return max(0, self.taille_cible - self.nombre_disponible_final)


def calculer_repartition_par_strate(
    exemples: list[ExemplePivot],
) -> dict[tuple[str, str], dict[str, int]]:
    """
    Calcule, pour chaque strate `(type_exemple, source)` des exemples
    donnes (deja en memoire, ex. le resultat recoupe de `executer()`),
    le decompte par split. Meme forme de retour que
    `VerifierRepartitionSplitsUseCase.executer()`, mais sans relire le
    repository : utilisable directement sur un resultat deja calcule.
    """
    repartition: dict[tuple[str, str], dict[str, int]] = {}
    for exemple in exemples:
        if exemple.split is None:
            continue
        cle = (exemple.type_exemple.value, exemple.source)
        compteur_strate = repartition.setdefault(cle, {})
        compteur_strate[exemple.split.value] = (
            compteur_strate.get(exemple.split.value, 0) + 1
        )
    return repartition


def formater_tableau_repartition(
    repartition: dict[tuple[str, str], dict[str, int]],
) -> str:
    """
    Formate `repartition` (cf. `calculer_repartition_par_strate`) en un
    tableau texte `Strate | Total | train (%) | val (%) | test (%)`,
    meme presentation que `interfaces/cli/E1_05_01_verifier_repartition_splits.py`.
    """
    lignes = [
        f"{'Strate':<45} {'Total':>7}  "
        + "  ".join(f"{s:>14}" for s in ORDRE_SPLITS)
    ]
    for cle, compteur_strate in sorted(repartition.items()):
        total_strate = sum(compteur_strate.values())
        colonnes = []
        for split in ORDRE_SPLITS:
            n = compteur_strate.get(split, 0)
            pourcentage = (n / total_strate * 100) if total_strate else 0.0
            colonnes.append(f"{n:>6} ({pourcentage:4.1f}%)")
        nom_strate = f"{cle[0]}/{cle[1]}"
        lignes.append(
            f"{nom_strate:<45} {total_strate:>7}  " + "  ".join(colonnes)
        )
    return "\n".join(lignes)
