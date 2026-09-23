---
title: CHSA Triage Test Chat
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Space Streamlit/CPU, leger et bon marche, laisse allume en permanence
(a la difference du Space Docker/GPU compagnon, `README_space_gpu.md`
a la racine du depot, couteux et allume seulement pendant les tests).
Interface de test de l'entretien clinique CHSA Triage : sonde `GET
/sante` de l'API en boucle et affiche automatiquement l'interface de
chat des que le Space GPU devient disponible, sans synchronisation
manuelle des deux demarrages.

Ne contient aucun code d'entrainement ni d'inference lui-meme : parle
uniquement HTTP a l'API FastAPI reelle (`interfaces/api/`).

Secrets du Space a definir avant publication :
- `CHSA_API_URL_BASE` : URL du Space Docker/GPU compagnon.
- `CHSA_API_CLE` : meme cle que `CHSA_CLE_API_DEMO` cote API.

HF Spaces ne propose plus "Streamlit" comme SDK de premier niveau dans
son flux de creation actuel (confirme par une erreur serveur 400 sur
`hf repo create --sdk streamlit`, la docstring de
`HfApi.create_repo` est obsolete sur ce point) : "Streamlit" y
apparait desormais comme un TEMPLATE a l'interieur du SDK "Docker",
d'ou `sdk: docker` ci-dessus plutot que le SDK natif streamlit prevu
initialement.

Publication (jamais realisee dans cette tache, effet externe reel,
action reservee a l'operateur humain) : uploader
`interfaces/web/Dockerfile` (renomme `Dockerfile` a la racine du
Space), `interfaces/web/app_test_inference.py`,
`interfaces/web/logica_test_inference.py`, ce fichier (renomme
`README.md` a la racine du Space) et `interfaces/web/requirements.txt`
(renomme `requirements.txt` a la racine du Space, non utilise par le
build Docker lui-meme qui passe par `uv sync --extra web`, mais
conserve pour coherence avec le reste du depot) vers le depot HF Space
cible, meme patron que `deploy/space_gpu_api_vllm/README_space.md`.
