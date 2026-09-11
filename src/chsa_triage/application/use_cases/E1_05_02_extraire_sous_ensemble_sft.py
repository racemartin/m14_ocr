"""
Cas d'usage : extraire le sous-ensemble deja reparti en splits
(§7 du README, `DecouperSplitsUseCase`) qui doit etre publie (ex. sur
Hugging Face), en retirant les exemples portant une PII residuelle
confirmee ou encore en attente de decision humaine (cf.
`E1_04_01_reviser_pii_residuelle.ReviserPiiResiduelleUseCase.identifiants_a_exclure_publication`).

Design (11/09/2026) : un FILTRE/une SOUSTRACTION,
pas un nouveau muestreo. Si le resultat, apres exclusion, est plus
petit que `taille_cible`, ce cas d'usage ne tente jamais de completer
automatiquement (ce role reste a `DecouperSplitsUseCase --n`, appele
separement avant de relancer l'extraction) ; il se contente de
rapporter honnetement combien d'exemples restent et combien manquent.

Correction (11/09/2026) : quand le resultat, apres exclusion, contient
PLUS d'exemples que `taille_cible`, le cas d'usage recoupe maintenant
a exactement `taille_cible` par echantillonnage stratifie
(`echantillon_stratifie`, meme algorithme du plus grand reste que
`AnonymiserDatasetUseCase`/`DecouperSplitsUseCase`), pour que la
taille publiee corresponde toujours a ce qui a ete demande au lieu de
publier tout le surplus disponible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chsa_triage.application.echantillonnage import echantillon_stratifie
from chsa_triage.domain.model import ExemplePivot
from chsa_triage.domain.ports import RepositoryLectureEcriture

ORDRE_SPLITS = ("train", "val", "test")


@dataclass(slots=True)
class ExtraireSousEnsembleSftUseCase:
    """Filtre les exemples deja repartis en split, moins les identifiants exclus, recoupe a `taille_cible`."""

    repository             : RepositoryLectureEcriture
    identifiants_a_exclure : frozenset[str] = frozenset()
    taille_cible           : int = 5000

    nombre_avec_split       : int = field(default=0, init=False)
    nombre_exclus           : int = field(default=0, init=False)
    nombre_disponible_final : int = field(default=0, init=False)
    nombre_tronque          : int = field(default=0, init=False)

    def executer(self) -> list[ExemplePivot]:
        """
        Retourne les exemples `split is not None` dont l'identifiant
        n'est PAS dans `identifiants_a_exclure`, recoupes a
        `taille_cible` par echantillonnage stratifie (type_exemple,
        source) si le resultat filtre en contient plus. N'assigne, ne
        modifie ni ne persiste jamais rien (pure lecture) : le
        resultat est a ecrire par l'appelant (cf.
        `interfaces/cli/E1_05_02_extraire_sous_ensemble_sft.py`).
        """
        avec_split = [e for e in self.repository.lister() if e.split is not None]
        self.nombre_avec_split = len(avec_split)

        resultat = [e for e in avec_split if e.identifiant not in self.identifiants_a_exclure]
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


def calculer_repartition_par_strate(exemples: list[ExemplePivot]) -> dict[tuple[str, str], dict[str, int]]:
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
        compteur_strate[exemple.split.value] = compteur_strate.get(exemple.split.value, 0) + 1
    return repartition


def formater_tableau_repartition(repartition: dict[tuple[str, str], dict[str, int]]) -> str:
    """
    Formate `repartition` (cf. `calculer_repartition_par_strate`) en un
    tableau texte `Strate | Total | train (%) | val (%) | test (%)`,
    meme presentation que `interfaces/cli/E1_05_01_verifier_repartition_splits.py`.
    """
    lignes = [f"{'Strate':<45} {'Total':>7}  " + "  ".join(f"{s:>14}" for s in ORDRE_SPLITS)]
    for cle, compteur_strate in sorted(repartition.items()):
        total_strate = sum(compteur_strate.values())
        colonnes = []
        for split in ORDRE_SPLITS:
            n = compteur_strate.get(split, 0)
            pourcentage = (n / total_strate * 100) if total_strate else 0.0
            colonnes.append(f"{n:>6} ({pourcentage:4.1f}%)")
        nom_strate = f"{cle[0]}/{cle[1]}"
        lignes.append(f"{nom_strate:<45} {total_strate:>7}  " + "  ".join(colonnes))
    return "\n".join(lignes)
