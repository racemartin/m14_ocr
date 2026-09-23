---
title: CHSA Triage API GPU
emoji: 🚑
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

Space Docker/GPU, couteux, a n'allumer que pendant les tests (a la
difference du Space Streamlit/CPU compagnon, leger, toujours allume,
cf. `interfaces/web/README_space.md`). Sert l'API FastAPI
(`interfaces/api/`) ET le serveur vLLM (LoRA DPO, jamais fusionne avec
la base) dans le MEME conteneur, via `Dockerfile` +
`demarrer.sh` de ce dossier.

`GET /sante` (`interfaces/api/app.py`) reste HTTP 200 des que uvicorn
demarre, meme si vLLM met plusieurs minutes a charger le modele : c'est
ce que le Space Streamlit compagnon sonde en boucle pour detecter
automatiquement la disponibilite reelle du modele, sans synchronisation
manuelle des deux demarrages.

Secrets du Space a definir avant publication :
- `CHSA_CLE_API_DEMO` : cle attendue en en-tete `X-API-Key`.

Publication (jamais realisee dans cette tache, effet externe reel,
GPU payant, a autoriser explicitement, action reservee a l'operateur
humain) : uploader
`deploy/space_gpu_api_vllm/Dockerfile` (renomme `Dockerfile` a la
racine du Space), `deploy/space_gpu_api_vllm/demarrer.sh`, ce fichier
(renomme `README.md` a la racine du Space), ainsi que `src/`,
`interfaces/`, `pyproject.toml`, `uv.lock` (memes chemins que le
`Dockerfile` racine du depot, cf. sa propre note de conception).
