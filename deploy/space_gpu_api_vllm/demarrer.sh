#!/usr/bin/env bash
# Entrypoint du Space Docker/GPU "API FastAPI + vLLM" (cf.
# README_space.md dans ce meme dossier). Lance vLLM en arriere-plan
# (LoRA DPO servi nativement, meme commande que le README racine du
# depot, section "4.1 Adaptateur vLLM"), PUIS l'API FastAPI au premier
# plan : le Space Streamlit/CPU compagnon (interfaces/web/) detecte la
# disponibilite reelle du modele via GET /sante, jamais en attendant ce
# script (qui ne bloque pas sur le chargement de vLLM).
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

exec uvicorn interfaces.api.main:app --host 0.0.0.0 --port 7860
