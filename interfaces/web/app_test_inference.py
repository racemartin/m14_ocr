"""
Frontend Streamlit de TEST (Etape 4) : entretien clinique multi-tours
contre l'API FastAPI REELLE (`interfaces/api/`, `POST /conversations`,
`POST /conversations/{id}/messages`, `POST /conversations/{id}/diagnostic`),
pas un faux moteur. Distinct de `monitoring/app_suivi_entrainement.py`
(courbes d'apprentissage SFT/DPO, aucun chat, but different).

Cible de deploiement (jamais realisee dans cette tache : effet externe
reel, action reservee a l'operateur humain, meme patron que le reste
de cette session) : DEUX HF Spaces separes.
- Space Docker/GPU (couteux, allume seulement pendant les tests) :
  `interfaces/api/` + `Dockerfile` a la racine du depot (existant,
  Etape 4).
- Space Streamlit/CPU (leger, allume en permanence) : CE fichier, cf.
  `interfaces/web/README_space.md` pour le frontmatter YAML attendu
  par HF Spaces (SDK Streamlit) et `interfaces/web/requirements.txt`
  pour ses dependances a publier a la racine de ce Space.

Le Space Streamlit ne sait JAMAIS a l'avance si le Space GPU compagnon
est allume : il sonde `/sante` en boucle (polling leger, meme patron
`time.sleep()` + `st.rerun()` que `monitoring/app_suivi_entrainement.py`)
et affiche un etat d'attente clair tant que la reponse est negative ou
que l'API elle-meme est injoignable ; jamais d'erreur brute affichee
(cf. `logica_test_inference.py::interroger_sante`).

Variables d'environnement (jamais de valeur en dur dans le code ; sur
le Space, definies comme "Repository secrets") :
- `CHSA_API_URL_BASE` (defaut `http://127.0.0.1:7860`, utile pour un
  test local contre l'API lancee via `uvicorn`/`docker run`) : URL du
  Space Docker/GPU, DIFFERENTE de celle de ce Space Streamlit.
- `CHSA_API_CLE` (obligatoire, aucun defaut permissif) : meme valeur
  que `CHSA_CLE_API_DEMO` cote API (`interfaces/api/main.py`).

Smoke test manuel (necessite l'API reellement lancee en local, cf.
README §4.2 "API FastAPI") :
    export CHSA_API_URL_BASE=http://127.0.0.1:7860
    export CHSA_API_CLE=change-moi
    uv sync --extra web
    uv run streamlit run interfaces/web/app_test_inference.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Sur le Space, ce depot est clone tel quel (pas de `pip install -e .`) :
# ajoute la racine du projet au chemin d'import pour `import interfaces...`,
# ce script etant lance directement par `streamlit run`. Meme patron que
# `monitoring/app_suivi_entrainement.py`.
_RACINE_PROJET = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_RACINE_PROJET))

import httpx
import streamlit as st

from interfaces.web.logica_test_inference import (
    demarrer_conversation,
    interroger_sante,
    obtenir_diagnostic,
    poursuivre_conversation,
)

URL_API_PAR_DEFAUT = "http://127.0.0.1:7860"
INTERVALLE_SONDAGE_SANTE_SECONDES = 5


def _construire_client(url_base: str, cle_api: str) -> httpx.Client:
    return httpx.Client(base_url=url_base, headers={"X-API-Key": cle_api}, timeout=30.0)


def main() -> None:
    st.set_page_config(page_title="CHSA Triage - Test entretien", page_icon="🩺", layout="centered")
    st.title("Entretien clinique de test (CHSA Triage)")

    url_base = os.environ.get("CHSA_API_URL_BASE", URL_API_PAR_DEFAUT)
    cle_api = os.environ.get("CHSA_API_CLE")
    if not cle_api:
        st.error(
            "CHSA_API_CLE doit etre definie (secret du Space Streamlit, "
            "meme valeur que CHSA_CLE_API_DEMO cote API)."
        )
        return

    client = _construire_client(url_base, cle_api)

    sante = interroger_sante(client)
    if not sante.get("disponible"):
        st.info(f"⏳ En attente du modele... ({sante.get('detail', 'statut inconnu')})")
        st.caption(f"API interrogee : {url_base}/sante")
        time.sleep(INTERVALLE_SONDAGE_SANTE_SECONDES)
        st.rerun()

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = demarrer_conversation(client)
        st.session_state.historique = []

    for tour in st.session_state.historique:
        with st.chat_message(tour["role"]):
            st.write(tour["contenu"])

    message = st.chat_input("Decrivez les symptomes du patient...")
    if message:
        st.session_state.historique.append({"role": "user", "contenu": message})
        message_assistant = poursuivre_conversation(client, st.session_state.conversation_id, message)
        st.session_state.historique.append({"role": "assistant", "contenu": message_assistant})
        st.rerun()

    if st.session_state.historique and st.button("Obtenir le diagnostic"):
        diagnostic = obtenir_diagnostic(client, st.session_state.conversation_id)
        if diagnostic.get("format_respecte"):
            st.success(
                f"Niveau ESI {diagnostic['niveau']} : {diagnostic['categorie']} "
                f"({diagnostic['ressources_estimees']})"
            )
            st.caption(diagnostic.get("raisonnement") or "")
        else:
            st.warning("Format de diagnostic non respecte par le modele ; reponse brute :")
            st.write(diagnostic.get("texte_brut", ""))


if __name__ == "__main__":
    main()
