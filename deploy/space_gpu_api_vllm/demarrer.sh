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
# Segfault natif (crash silencieux, sans message Python) juste apres le
# chargement des poids, pendant le "profile_run" interne, TOUJOURS au
# meme endroit exact, quoi que change dans la configuration. Pistes
# EXCLUES par tests reels (meme crash identique malgre le changement) :
# multiprocessing V1, permissions UID 1000, LoRA/Punica, backend
# d'attention (FLASHINFER), --no-async-scheduling. Preuve decisive :
# VLLM_TRACE_FUNCTION=1 (trace chaque appel Python, tres lent) a permis
# un demarrage COMPLET et sain -- le ralentissement massif fait
# disparaitre le crash, signature typique d'une CONDITION DE COURSE
# dependante du temps reel d'execution, pas d'une erreur de
# configuration isolable par un seul flag. NON RESOLU : ni
# VLLM_TRACE_FUNCTION=1 (>100x plus lent, inutilisable en usage reel)
# ni aucun flag cible testes a ce jour ne sont une solution de
# production viable. Pistes non testees pour la suite : version de
# vLLM differente (bug potentiel de cette version 0.20.2 precise),
# autre "flavor" de GPU HF (isoler une eventuelle cause materielle a
# cette instance L4 precise), ou signalement en amont du projet vLLM
# avec cette reproduction.
set -euo pipefail

vllm serve Qwen/Qwen3-1.7B-Base \
    --enable-lora \
    --lora-modules dpo=mombasstic/chsa-triage-dpo-lora \
    --max-lora-rank 16 \
    --enforce-eager \
    --port 8000 &

export CHSA_MOTEUR_INFERENCE=distant
export CHSA_URL_MOTEUR_INFERENCE="http://127.0.0.1:8000"

exec uvicorn interfaces.api.main:app --host 0.0.0.0 --port 7860
