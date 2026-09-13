"""
Dashboard Streamlit : visualisation EN VIVO de la courbe d'apprentissage
(perte train/validation) d'un run SFT-LoRA reel (Environnement B, GPU
sur HF Jobs), en lisant le dataset HF alimente par
`HfDatasetSuiviExperimentation` (`training/E2_04_sft_train.py`,
`suivi.backend: hf_dataset`).

Deploiement cible : Hugging Face Spaces (SDK Streamlit), cf. README
"Suivi d'entrainement en vivo" pour les commandes exactes de creation
du Space et de publication de ce code.

Vit hors `training/` (visualiseur passif, pas un pas execute par le job
d'entrainement) et hors `interfaces/cli/` (pas une CLI de la sequence
de cas d'usage E1/E2) : meme critere que
`domain/`/`ports/`/`infrastructure/adapters/`, d'ou l'absence de
prefixe `E1_`/`E2_` sur ce fichier (cf. AGENTS.md).

Toute la logique pure (parsing, pivot, verdict) est dans
`logica_suivi_entrainement.py`, testee sans Streamlit ni reseau.

Smoke test manuel (sans reseau, contre un JSONL de fixture local) :
    uv run streamlit run monitoring/app_suivi_entrainement.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Sur le Space, ce depot est clone tel quel (pas de `pip install -e .`) :
# ajoute la racine du projet (pour `import monitoring...`, ce script
# etant lance directement par `streamlit run`) et `src/` (pour
# `chsa_triage.application.verdict_convergence`, domaine pur, zero
# dependance externe) au chemin d'import.
_RACINE_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RACINE_PROJET))
sys.path.insert(0, str(_RACINE_PROJET / "src"))

import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from monitoring.logica_suivi_entrainement import (
    NOM_FICHIER_METRIQUES,
    analyser_jsonl_metriques,
    evaluer_convergence_en_vivo,
    extraire_noms_runs,
    pivoter_par_etape,
)

REPO_ID_PAR_DEFAUT = "mombasstic/chsa-triage-sft-metrics"
INTERVALLE_AUTO_ACTUALISATION_SECONDES = 15

_COULEUR_PAR_VERDICT = {
    "saine": "success",
    "surapprentissage": "warning",
    "sous_apprentissage": "warning",
    "instable": "error",
}


def _lister_runs(repo_id: str) -> list[str]:
    chemins = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    return extraire_noms_runs(chemins)


def _telecharger_tableau_metriques(repo_id: str, nom_run: str) -> list[dict]:
    chemin_local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{nom_run}/{NOM_FICHIER_METRIQUES}",
    )
    texte = Path(chemin_local).read_text(encoding="utf-8")
    return pivoter_par_etape(analyser_jsonl_metriques(texte))


def _afficher_cartes_derniere_etape(derniere_ligne: dict) -> None:
    colonnes = st.columns(4)
    colonnes[0].metric("Etape actuelle", derniere_ligne["etape"])
    for colonne, cle, libelle in (
        (colonnes[1], "perte_train", "Perte train"),
        (colonnes[2], "perte_validation", "Perte validation"),
        (colonnes[3], "norme_gradient", "Norme gradient"),
    ):
        valeur = derniere_ligne.get(cle)
        colonne.metric(libelle, f"{valeur:.4f}" if valeur is not None else "-")


def _afficher_verdict_convergence(tableau_large: list[dict]) -> None:
    verdict, message = evaluer_convergence_en_vivo(tableau_large)
    if verdict is None:
        st.info(f"Verdict de convergence : {message}.")
        return
    afficheur = getattr(st, _COULEUR_PAR_VERDICT[verdict.value])
    afficheur(f"Verdict de convergence : **{verdict.value}** ({message})")


def _afficher_courbe_pertes(tableau_large: list[dict]) -> None:
    colonnes_courbe = [c for c in ("perte_train", "perte_validation") if any(c in ligne for ligne in tableau_large)]
    if not colonnes_courbe:
        st.info("Aucune courbe de perte disponible pour l'instant.")
        return
    donnees = {"etape": [ligne["etape"] for ligne in tableau_large]}
    donnees.update({colonne: [ligne.get(colonne) for ligne in tableau_large] for colonne in colonnes_courbe})
    st.line_chart(donnees, x="etape", y=colonnes_courbe)


def main() -> None:
    st.set_page_config(page_title="Suivi entrainement SFT-LoRA", page_icon="📈", layout="wide")
    st.title("Suivi d'entrainement SFT-LoRA (en vivo)")

    with st.sidebar:
        repo_id = st.text_input("Depot dataset HF", value=REPO_ID_PAR_DEFAUT)
        try:
            runs = _lister_runs(repo_id)
        except Exception as erreur:  # noqa: BLE001 - frontiere UI : ne jamais crasher le dashboard sur une erreur reseau/HF Hub
            st.error(f"Impossible de lister les runs de {repo_id} : {erreur}")
            return
        if not runs:
            st.info("Aucun run trouve pour l'instant dans ce depot.")
            return
        nom_run = st.selectbox("Run", options=runs, index=len(runs) - 1)
        auto_actualiser = st.checkbox("Auto-actualiser", value=True)
        actualiser_maintenant = st.button("Actualiser maintenant")

    try:
        tableau_large = _telecharger_tableau_metriques(repo_id, nom_run)
    except EntryNotFoundError:
        st.info(f"Le run {nom_run} n'a pas encore de metriques.")
        return
    except Exception as erreur:  # noqa: BLE001 - frontiere UI : ne jamais crasher le dashboard sur une erreur reseau/HF Hub
        st.error(f"Erreur de telechargement : {erreur}")
        return

    if not tableau_large:
        st.info("Aucune metrique loguee pour l'instant.")
        return

    _afficher_cartes_derniere_etape(tableau_large[-1])
    _afficher_verdict_convergence(tableau_large)
    _afficher_courbe_pertes(tableau_large)

    if actualiser_maintenant:
        st.rerun()
    if auto_actualiser:
        time.sleep(INTERVALLE_AUTO_ACTUALISATION_SECONDES)
        st.rerun()


if __name__ == "__main__":
    main()
