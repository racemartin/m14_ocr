#!/usr/bin/env bash
# Entrypoint du Space Docker/GPU "API FastAPI + vLLM" (cf.
# README_space.md dans ce meme dossier). Lance vLLM en arriere-plan
# (LoRA DPO servi nativement, meme commande que le README racine du
# depot, section "4.1 Adaptateur vLLM"), PUIS l'API FastAPI au premier
# plan : le Space Streamlit/CPU compagnon (interfaces/web/) detecte la
# disponibilite reelle du modele via GET /sante, jamais en attendant ce
# script (qui ne bloque pas sur le chargement de vLLM).
#
# Deploiement reel du 24/09/2026 (mombasstic/chsa-triage-api). Journal
# condense des causes reelles rencontrees et reglees (voir historique
# git de ce fichier et du Dockerfile pour le detail complet) :
# - Compilateur C absent de l'image de base "runtime" -> build-essential
#   + python3.11-dev ajoutes cote Dockerfile (Triton a aussi besoin
#   d'un compilateur, meme sous --enforce-eager).
# - UID 1000 requis par tout Space Docker HF au runtime -> utilisateur
#   cree explicitement cote Dockerfile (correction legitime, gardee).
#
# Segfault natif (crash silencieux, sans message Python exploitable)
# juste apres le chargement des poids, pendant le "profile_run" interne,
# TOUJOURS au meme endroit exact, sur DEUX versions de vLLM (0.20.2 et
# 0.22.0). Pistes EXCLUES par tests reels, chacune avec le meme crash
# identique malgre le changement : multiprocessing V1, permissions UID
# 1000, LoRA/Punica (teste desactive sur les DEUX versions), backend
# d'attention (FLASHINFER), --no-async-scheduling, delai de demarrage
# fixe, vllm==0.22.0 (torch==2.11.0, compatibilite Python 3.11
# verifiee -- vllm==0.28.0 lui-meme exclu differemment : torch
# incompatible avec Python 3.11, jamais atteint le segfault). Preuve
# decisive : VLLM_TRACE_FUNCTION=1 (trace chaque appel Python, tres
# lent) permet un demarrage COMPLET et sain -- le ralentissement massif
# fait disparaitre le crash, signature d'une CONDITION DE COURSE
# dependante du temps reel d'execution.
#
# NON RESOLU. Reproduction solide sur deux versions de vLLM et sept
# pistes ciblees exclues : signalement fait en amont du projet vLLM
# (https://github.com/vllm-project/vllm/issues/58616).
#
# HYPOTHESE TESTEE ET EXCLUE (24/09/2026) : le Dockerfile officiel de
# vLLM utilise une image de base CUDA differente de la notre, et torch
# (2.11.0) telecharge des paquets nvidia-*-cu13 (CUDA 13) alors que
# notre image de base est CUDA 12.4.1 -- desaccord driver/CUDA
# suspecte. EXCLU par `nvidia-smi` reel (garde ci-dessous a titre de
# diagnostic permanent, cout nul) : driver 580.178.04, CUDA 13.0
# supporte nativement, `torch.version.cuda=13.0`,
# `torch.cuda.is_available()=True`. Le GPU/driver est sain ; ce n'est
# pas la cause du segfault.
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
