"""
Export PNG statique des courbes d'entrainement (perte train/validation,
et recompenses DPO quand presentes), a partir des memes metriques deja
publiees que lit le dashboard Streamlit EN VIVO
(`app_suivi_entrainement.py`). Reutilise `logica_suivi_entrainement.py`
(parsing/pivot, deja teste, zero dependance Streamlit) : aucune
nouvelle logique de parsing ici, seulement le rendu matplotlib.

Comble le gap identifie par l'audit `m14-ocr-nfr-latencia-robustez-trazabilidad`
(§3.2, finding n°4) : jusqu'ici, la seule facon d'obtenir une figure de
courbe d'apprentissage etait une capture d'ecran manuelle du dashboard
Streamlit, non reproductible. Ce script produit des `.png` directement
depuis les metriques deja publiees (aucun nouveau run necessaire).

Usage (reseau, contre un depot dataset HF deja publie) :
    uv run python monitoring/exporter_courbes_png.py \
        --repo-id mombasstic/chsa-triage-sft-metrics --sortie-dir figures/

Usage (hors-ligne, rejoue un run deja telecharge, meme fixture que
`data/demos/move_chsa-triage-sft-metrics-fake_jston_to_mlflow_db.py`) :
    uv run python monitoring/exporter_courbes_png.py \
        --fichier-local data/demos/chsa-triage-sft-metrics-fake.json \
        --nom-run demo --sortie-dir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Meme raisonnement d'import que app_suivi_entrainement.py : ce script
# peut etre lance directement (`uv run python monitoring/...`), pas
# seulement importe comme module d'un paquet installe.
_RACINE_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RACINE_PROJET))
sys.path.insert(0, str(_RACINE_PROJET / "src"))

import matplotlib

matplotlib.use("Agg")  # aucun affichage interactif : export fichier seul
import matplotlib.pyplot as plt

from monitoring.logica_suivi_entrainement import (
    COLONNES_RECOMPENSE_DPO,
    analyser_jsonl_metriques,
    filtrer_colonnes_presentes,
    pivoter_par_etape,
)

NOM_FICHIER_COURBE_PERTES = "courbe_pertes.png"
NOM_FICHIER_COURBE_RECOMPENSES_DPO = "courbe_recompenses_dpo.png"

_LIBELLE_PAR_COLONNE_PERTE = {
    "perte_train": "Perte train",
    "perte_validation": "Perte validation",
}
_LIBELLE_PAR_COLONNE_RECOMPENSE = {
    "rewards/chosen": "Recompense chosen",
    "rewards/rejected": "Recompense rejected",
}


def exporter_courbe_pertes_png(
    tableau_large: list[dict], chemin_sortie: Path, titre: str
) -> bool:
    """
    Ecrit un PNG de la courbe perte_train (+ perte_validation si
    presente) dans `chemin_sortie`. Retourne `False` sans rien ecrire
    si aucune des deux colonnes n'est presente (ex. tableau vide),
    meme garde que `_afficher_courbe_pertes` du dashboard Streamlit.
    """
    colonnes = filtrer_colonnes_presentes(
        tableau_large, ("perte_train", "perte_validation")
    )
    if not colonnes:
        return False

    etapes = [ligne["etape"] for ligne in tableau_large]
    figure, axes = plt.subplots(figsize=(10, 6))
    for colonne in colonnes:
        valeurs = [ligne.get(colonne) for ligne in tableau_large]
        axes.plot(
            etapes, valeurs, marker="o", label=_LIBELLE_PAR_COLONNE_PERTE[colonne]
        )
    axes.set_xlabel("Etape")
    axes.set_ylabel("Perte")
    axes.set_title(titre)
    axes.legend()
    axes.grid(True, alpha=0.3)

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def exporter_courbe_recompenses_dpo_png(
    tableau_large: list[dict], chemin_sortie: Path, titre: str
) -> bool:
    """
    Ecrit un PNG des courbes `rewards/chosen`/`rewards/rejected` si
    presentes (run DPO uniquement). Retourne `False` sans rien ecrire
    sur un run SFT (aucune des deux colonnes jamais journalisee).
    """
    colonnes = filtrer_colonnes_presentes(
        tableau_large, ("rewards/chosen", "rewards/rejected")
    )
    if not colonnes:
        return False

    etapes = [ligne["etape"] for ligne in tableau_large]
    figure, axes = plt.subplots(figsize=(10, 6))
    for colonne in colonnes:
        valeurs = [ligne.get(colonne) for ligne in tableau_large]
        axes.plot(
            etapes,
            valeurs,
            marker="o",
            label=_LIBELLE_PAR_COLONNE_RECOMPENSE[colonne],
        )
    axes.set_xlabel("Etape")
    axes.set_ylabel("Recompense")
    axes.set_title(titre)
    axes.legend()
    axes.grid(True, alpha=0.3)

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def exporter_cartes_recompenses_dpo_png(
    tableau_large: list[dict], chemin_sortie: Path, titre: str
) -> bool:
    """
    Ecrit un PNG en cartes (barres) de la derniere valeur de chacune
    des 4 metriques de recompense DPO presentes (chosen/rejected,
    accuracies, margins), meme contenu que les 4 `st.metric` du
    dashboard EN VIVO, mais statique. Retourne `False` sans rien
    ecrire si aucune des 4 colonnes n'est presente.
    """
    colonnes = filtrer_colonnes_presentes(tableau_large, COLONNES_RECOMPENSE_DPO)
    if not colonnes:
        return False

    derniere_ligne = tableau_large[-1]
    libelles = [colonne for colonne in colonnes]
    valeurs = [derniere_ligne.get(colonne, 0.0) for colonne in colonnes]

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.bar(libelles, valeurs)
    axes.set_ylabel("Valeur (derniere etape)")
    axes.set_title(titre)
    axes.tick_params(axis="x", rotation=20)

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def exporter_toutes_les_courbes(
    tableau_large: list[dict], repertoire_sortie: Path, nom_run: str
) -> list[Path]:
    """
    Produit tous les PNG applicables a `tableau_large` (courbe de
    pertes systematiquement tentee, recompenses DPO seulement si
    presentes) sous `repertoire_sortie/<nom_run>/`. Retourne les
    chemins effectivement ecrits (jamais un chemin pour un graphique
    qui n'avait aucune donnee a montrer).
    """
    repertoire_run = repertoire_sortie / nom_run
    chemins_ecrits: list[Path] = []

    chemin_pertes = repertoire_run / NOM_FICHIER_COURBE_PERTES
    if exporter_courbe_pertes_png(
        tableau_large, chemin_pertes, titre=f"Courbe de perte - {nom_run}"
    ):
        chemins_ecrits.append(chemin_pertes)

    chemin_recompenses = repertoire_run / NOM_FICHIER_COURBE_RECOMPENSES_DPO
    if exporter_courbe_recompenses_dpo_png(
        tableau_large,
        chemin_recompenses,
        titre=f"Recompenses DPO - {nom_run}",
    ):
        chemins_ecrits.append(chemin_recompenses)

    chemin_cartes = repertoire_run / "cartes_recompenses_dpo.png"
    if exporter_cartes_recompenses_dpo_png(
        tableau_large,
        chemin_cartes,
        titre=f"Recompenses DPO (derniere etape) - {nom_run}",
    ):
        chemins_ecrits.append(chemin_cartes)

    return chemins_ecrits


def _tableau_depuis_fichier_local(chemin: Path) -> list[dict]:
    texte = chemin.read_text(encoding="utf-8")
    return pivoter_par_etape(analyser_jsonl_metriques(texte))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--repo-id",
        help="Depot dataset HF de metriques (ex. mombasstic/chsa-triage-sft-metrics)",
    )
    source.add_argument(
        "--fichier-local",
        help="Fichier JSONL local deja telecharge (format LONG, meme schema que metriques.jsonl)",
    )
    parser.add_argument(
        "--nom-run",
        help="Run a exporter (defaut : le dernier du depot, obligatoire avec --fichier-local)",
    )
    parser.add_argument(
        "--sortie-dir",
        default="figures",
        help="Repertoire racine de sortie (defaut figures/)",
    )
    arguments = parser.parse_args()

    repertoire_sortie = Path(arguments.sortie_dir)

    if arguments.fichier_local:
        if not arguments.nom_run:
            parser.error("--nom-run est obligatoire avec --fichier-local")
        tableau_large = _tableau_depuis_fichier_local(Path(arguments.fichier_local))
        nom_run = arguments.nom_run
    else:
        from monitoring.hf_dataset_runs import (
            lister_runs,
            telecharger_texte_metriques,
        )

        runs = lister_runs(arguments.repo_id)
        if not runs:
            print(f"Aucun run trouve dans {arguments.repo_id}.")
            return
        nom_run = arguments.nom_run or runs[-1]
        texte = telecharger_texte_metriques(arguments.repo_id, nom_run)
        tableau_large = pivoter_par_etape(analyser_jsonl_metriques(texte))

    if not tableau_large:
        print(f"Aucune metrique a exporter pour le run {nom_run}.")
        return

    chemins = exporter_toutes_les_courbes(tableau_large, repertoire_sortie, nom_run)
    if not chemins:
        print(f"Rien exporte pour {nom_run} (aucune colonne reconnue).")
        return
    for chemin in chemins:
        print(f"ecrit : {chemin}")


if __name__ == "__main__":
    main()
