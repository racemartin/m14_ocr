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
# regle, vLLM segfaultait au demarrage juste apres le chargement des
# poids ("Using PunicaWrapperGPU"), au tout premier appel reel d'un
# noyau Triton (Punica/LoRA). Deux hypotheses testees en conditions
# reelles, AUCUNE n'a resolu le segfault (meme trace identique a
# chaque fois) : (1) VLLM_ENABLE_V1_MULTIPROCESSING=0, sans effet
# verifie dans le code source de vLLM 0.20.2 (variable ignoree par le
# serveur `vllm serve`) ; (2) creation de l'utilisateur UID 1000 cote
# Dockerfile (requis par tout Space Docker HF au runtime), qui reglait
# une cause plausible de resolution de $HOME/cache JIT Triton mais n'a
# PAS supprime le segfault en pratique. Diagnostic en cours : plusieurs
# incidents similaires reels documentes dans le suivi de vLLM pointent
# vers le chemin Punica/LoRA specifiquement pendant le "profile_run"
# (juste apres le chargement des poids, avant le calcul de la taille
# du cache KV) -- exactement ou ce crash se produit.
#
# DIAGNOSTIC TEMPORAIRE (24/09/2026) : LoRA desactive ci-dessous pour
# isoler si le segfault vient specifiquement du chemin Punica/LoRA ou
# d'une cause plus profonde. A REVERTIR (remettre --enable-lora
# --lora-modules dpo=mombasstic/chsa-triage-dpo-lora --max-lora-rank 16)
# des que le resultat de ce test est connu.
set -euo pipefail

vllm serve Qwen/Qwen3-1.7B-Base \
    --enforce-eager \
    --port 8000 &

export CHSA_MOTEUR_INFERENCE=distant
export CHSA_URL_MOTEUR_INFERENCE="http://127.0.0.1:8000"

exec uvicorn interfaces.api.main:app --host 0.0.0.0 --port 7860
