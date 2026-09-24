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
# poids ("Using PunicaWrapperGPU"), dans le processus EngineCore
# separe du processus APIServer (architecture V1 : deux processus
# distincts, meme sur une seule carte, sans parallelisme). Aucun
# message d'erreur exploitable dans la trace (frames Python generiques
# uniquement). VLLM_ENABLE_V1_MULTIPROCESSING=0 est la recommandation
# officielle du guide de resolution de problemes de vLLM pour ce cas
# precis : garde le moteur dans le MEME processus que le serveur API,
# ce qui evite la cause du crash. Sans parallelisme (une seule GPU
# L4 ici), le cout de performance de ce mode est negligeable pour ce
# POC (une requete a la fois).
set -euo pipefail

export VLLM_ENABLE_V1_MULTIPROCESSING=0

vllm serve Qwen/Qwen3-1.7B-Base \
    --enable-lora \
    --lora-modules dpo=mombasstic/chsa-triage-dpo-lora \
    --max-lora-rank 16 \
    --enforce-eager \
    --port 8000 &

export CHSA_MOTEUR_INFERENCE=distant
export CHSA_URL_MOTEUR_INFERENCE="http://127.0.0.1:8000"

exec uvicorn interfaces.api.main:app --host 0.0.0.0 --port 7860
