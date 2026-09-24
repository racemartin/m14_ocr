#!/usr/bin/env bash
# Entrypoint du Space Docker/GPU "API FastAPI + vLLM" (cf.
# README_space.md dans ce meme dossier). Lance vLLM en arriere-plan
# (LoRA DPO servi nativement, meme commande que le README racine du
# depot, section "4.1 Adaptateur vLLM"), PUIS l'API FastAPI au premier
# plan : le Space Streamlit/CPU compagnon (interfaces/web/) detecte la
# disponibilite reelle du modele via GET /sante, jamais en attendant ce
# script (qui ne bloque pas sur le chargement de vLLM).
#
# Deploiement reel du 24/09/2026 (mombasstic/chsa-triage-api) : l'image
# de base nvidia/cuda:12.4.1-runtime-ubuntu22.04 n'a pas de compilateur
# C par defaut (variante "runtime", pas "devel"), donc vLLM echouait au
# demarrage ("Failed to find C compiler") -- pas seulement pour
# torch.compile du graphe du modele (que --enforce-eager desactive),
# mais aussi pour un kernel Triton de tri/echantillonnage separe
# (topk_topp_sampler) qui compile quel que soit le mode eager/compile.
# Root cause reglee cote Dockerfile (build-essential ajoute) ; garde
# --enforce-eager ici en plus, cout de demarrage/memoire plus faible,
# acceptable pour ce POC. --max-lora-rank 16 fixe explicitement (rang
# reel du LoRA DPO, cf. recipes/dpo_qwen3_lora.yaml) plutot que de
# compter sur le defaut.
#
# Suite du deploiement reel du 24/09/2026 : une fois le compilateur
# regle, vLLM segfaultait (crash natif, sans message Python exploitable)
# juste apres le chargement des poids, pendant le "profile_run" interne
# (avant le calcul de la taille du cache KV), TOUJOURS au meme endroit
# exact. Pistes EXCLUES par des tests reels, chacune avec le MEME
# segfault identique malgre le changement : VLLM_ENABLE_V1_MULTIPROCESSING=0
# (sans effet, ignoree par `vllm serve`) ; utilisateur UID 1000 cote
# Dockerfile (correction legitime, gardee, mais n'a pas supprime le
# segfault) ; LoRA/Punica (teste desactive, meme crash) ;
# --attention-backend FLASHINFER a la place de FlashAttention2 (meme
# crash, log confirme "Using AttentionBackendEnum.FLASHINFER backend").
#
# DIAGNOSTIC EN COURS (24/09/2026) : VLLM_TRACE_FUNCTION=1 (recommande
# par le guide officiel de resolution de problemes de vLLM) trace
# chaque appel de fonction Python dans les logs pour identifier la
# ligne exacte avant le crash natif, au lieu de la trace generique
# inutilisable vue jusqu'ici. Ralentit enormement l'execution (avertit
# vLLM, >100x) mais le crash survient en quelques secondes normalement,
# donc reste exploitable pour ce diagnostic ponctuel. A RETIRER apres
# ce test, jamais en usage normal.
set -euo pipefail

export VLLM_TRACE_FUNCTION=1

vllm serve Qwen/Qwen3-1.7B-Base \
    --enable-lora \
    --lora-modules dpo=mombasstic/chsa-triage-dpo-lora \
    --max-lora-rank 16 \
    --enforce-eager \
    --port 8000 &

export CHSA_MOTEUR_INFERENCE=distant
export CHSA_URL_MOTEUR_INFERENCE="http://127.0.0.1:8000"

exec uvicorn interfaces.api.main:app --host 0.0.0.0 --port 7860
