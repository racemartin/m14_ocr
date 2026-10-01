#!/usr/bin/env bash
# Entrypoint du Space Docker/GPU "API FastAPI + vLLM + Streamlit" (cf.
# README_space.md dans ce meme dossier). Un Space Docker n'expose qu'UN
# seul port public (7860, cf. `app_port` dans README_space.md) : trois
# process tournent dans le MEME conteneur, mais seul Streamlit est
# branche dessus. vLLM (port 8000) et l'API FastAPI (port 8001) restent
# internes, injoignables depuis l'exterieur -- Streamlit est le seul
# client de l'API, exactement comme l'API est le seul client de vLLM.
# Decision produit du 01/10/2026 : avant cette date, Streamlit tournait
# uniquement EN LOCAL sur la machine de l'operateur humain (cf. historique
# git de interfaces/web/README_space.md) ; desormais le Space sert
# directement l'interface de test, sans rien a lancer en local.
#
# `GET /sante` (interfaces/api/app.py) reste HTTP 200 des que uvicorn
# demarre, meme si vLLM met plusieurs minutes a charger le modele :
# c'est ce que Streamlit sonde en boucle pour detecter automatiquement
# la disponibilite reelle du modele, sans synchronisation manuelle des
# demarrages.
#
# Deploiement reel du 24/09/2026 (mombasstic/chsa-triage-api). Un
# segfault natif systematique (huit pistes ciblees exclues -- voir le
# Dockerfile de ce dossier et le git log pour le detail complet ;
# signale en amont : https://github.com/vllm-project/vllm/issues/58616)
# a motive un changement de strategie : l'image part desormais de
# `vllm/vllm-openai` officielle plutot que d'une installation manuelle,
# cf. note en tete du Dockerfile. Le diagnostic `nvidia-smi` ci-dessous
# est garde (cout nul) : confirme un driver/CUDA sain (580.178.04,
# CUDA 13.0) sur la piste precedente, utile pour toute investigation
# future si le segfault persiste sur cette nouvelle base.
set -euo pipefail

echo "===== Diagnostic GPU/driver (avant vLLM) ====="
nvidia-smi || echo "nvidia-smi indisponible ou a echoue"
python3 -c "import torch; print('torch.version.cuda =', torch.version.cuda); print('torch CUDA disponible =', torch.cuda.is_available())" || echo "verification torch echouee"
echo "==============================================="

vllm serve Qwen/Qwen3-1.7B-Base \
    --enable-lora \
    --lora-modules dpo=mombasstic/chsa-triage-dpo-lora \
    --max-lora-rank 16 \
    --enforce-eager \
    --port 8000 &

export CHSA_MOTEUR_INFERENCE=distant
export CHSA_URL_MOTEUR_INFERENCE="http://127.0.0.1:8000"

uvicorn interfaces.api.main:app --host 0.0.0.0 --port 8001 &

# Streamlit (client de l'API interne) reutilise la meme cle que l'API
# attend (secret unique CHSA_CLE_API_DEMO, cf. README_space.md) -- pas
# de deuxieme secret a gerer pour une liaison purement interne au
# conteneur.
export CHSA_API_URL_BASE="http://127.0.0.1:8001"
export CHSA_API_CLE="${CHSA_CLE_API_DEMO:-}"

exec streamlit run interfaces/web/app_test_inference.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
