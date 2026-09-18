"""
Logique pure (parsing JSONL, pivot format long -> large, construction
de la courbe de convergence, detection des colonnes de recompense DPO
presentes) du dashboard Streamlit de suivi d'entrainement EN VIVO
(`monitoring/app_suivi_entrainement.py`), reutilisee telle quelle pour
un run SFT comme pour un run DPO.

Aucune dependance a Streamlit ni au reseau ici : testable directement
(`tests/monitoring/test_app_suivi_entrainement.py`), meme separation
que `application/verdict_convergence.py` (logique pure) vis-a-vis des
CLI qui l'appellent.

Lit le format LONG ecrit par
`infrastructure.adapters.hf_dataset_suivi_experimentation.HfDatasetSuiviExperimentation`
(une ligne JSON par appel a `logger_metrique` : `{"etape", "nom",
"valeur", "horodatage"}`), jamais le type domaine `MetriquesEntrainement`
directement : cette couche recompose ce type uniquement pour reutiliser
`evaluer_convergence`, elle ne le recoit jamais tout fait.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from chsa_triage.application.verdict_convergence import evaluer_convergence
from chsa_triage.domain.model.checkpoint_entraine import VerdictConvergence
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement

NOM_FICHIER_METRIQUES = "metriques.jsonl"

# Nombre minimal d'etapes completes (perte_train ET norme_gradient
# presentes) avant de calculer un verdict : un point unique ne peut
# jamais montrer de tendance (baisse/hausse).
NOMBRE_MINIMAL_ETAPES_POUR_VERDICT = 2

# Noms reels des metriques de recompense DPO, tels que journalises par
# `TrlDpoEntraineurAdapter` via `SuiviExperimentation.logger_metrique()`
# (API `trl.DPOTrainer`, cf. docs/04_etape3_dpo/02_etapes_cas_usage.md
# §4). Absentes d'un run SFT, presentes uniquement sur un run DPO.
COLONNES_RECOMPENSE_DPO = ("rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins")


@dataclass(frozen=True, slots=True)
class LigneMetrique:
    """Une ligne du JSONL format LONG, telle qu'ecrite par logger_metrique()."""

    etape       : int
    nom           : str
    valeur          : float
    horodatage        : float


def analyser_jsonl_metriques(texte: str) -> list[LigneMetrique]:
    """Parse le contenu JSONL format LONG ; ignore les lignes vides."""
    lignes = []
    for ligne_brute in texte.splitlines():
        ligne_brute = ligne_brute.strip()
        if not ligne_brute:
            continue
        objet = json.loads(ligne_brute)
        lignes.append(
            LigneMetrique(
                etape=int(objet["etape"]),
                nom=str(objet["nom"]),
                valeur=float(objet["valeur"]),
                horodatage=float(objet.get("horodatage", 0.0)),
            )
        )
    return lignes


def pivoter_par_etape(lignes: list[LigneMetrique]) -> list[dict]:
    """
    Pivote le format LONG vers une table LARGE, une entree par etape,
    triee par etape croissante. Une metrique n'apparait comme cle que
    dans les etapes ou elle a effectivement ete logguee (ex. pas de cle
    `perte_validation` si cette metrique n'a jamais ete envoyee).
    """
    par_etape: dict[int, dict] = {}
    for ligne in lignes:
        par_etape.setdefault(ligne.etape, {"etape": ligne.etape})[ligne.nom] = ligne.valeur
    return [par_etape[etape] for etape in sorted(par_etape)]


def construire_courbe_convergence(tableau_large: list[dict]) -> tuple[MetriquesEntrainement, ...]:
    """
    Reconstruit la `courbe_metriques` attendue par `evaluer_convergence`,
    en ignorant silencieusement les etapes incompletes (sans
    perte_train ou sans norme_gradient, ex. la derniere etape en cours
    de journalisation).
    """
    points = [
        MetriquesEntrainement(
            etape=ligne["etape"],
            perte_train=ligne["perte_train"],
            perte_validation=ligne.get("perte_validation"),
            norme_gradient=ligne["norme_gradient"],
        )
        for ligne in tableau_large
        if "perte_train" in ligne and "norme_gradient" in ligne
    ]
    return tuple(points)


def evaluer_convergence_en_vivo(tableau_large: list[dict]) -> tuple[VerdictConvergence | None, str]:
    """
    Calcule le verdict de convergence sur les points disponibles
    jusqu'ici. Ne leve jamais d'exception : retourne (None, message)
    tant qu'il n'y a pas assez de donnees pour un diagnostic utile.
    """
    courbe = construire_courbe_convergence(tableau_large)
    if len(courbe) < NOMBRE_MINIMAL_ETAPES_POUR_VERDICT:
        return None, (
            f"pas encore assez de donnees pour un verdict "
            f"({len(courbe)}/{NOMBRE_MINIMAL_ETAPES_POUR_VERDICT} etape(s) complete(s))"
        )

    verdict = evaluer_convergence(courbe)
    message = f"{len(courbe)} etapes analysees"
    if not any(point.perte_validation is not None for point in courbe):
        message += ", perte_validation absente : surapprentissage non detectable"
    return verdict, message


def filtrer_colonnes_presentes(tableau_large: list[dict], colonnes: tuple[str, ...]) -> list[str]:
    """
    Retourne, parmi `colonnes`, celles qui apparaissent effectivement
    dans au moins une ligne de `tableau_large` (ex. les 4 metriques de
    recompense DPO sur un run SFT : aucune n'a jamais ete journalisee,
    la liste retournee est vide).
    """
    return [colonne for colonne in colonnes if any(colonne in ligne for ligne in tableau_large)]


def extraire_noms_runs(chemins_fichiers: list[str], nom_fichier: str = NOM_FICHIER_METRIQUES) -> list[str]:
    """
    Extrait les noms de run a partir des chemins d'un depot dataset
    (`HfApi.list_repo_files`) : un run est le sous-repertoire contenant
    `nom_fichier` (`<nom_run>/metriques.jsonl`), tries alphabetiquement.
    """
    suffixe = f"/{nom_fichier}"
    return sorted(chemin[: -len(suffixe)] for chemin in chemins_fichiers if chemin.endswith(suffixe))
