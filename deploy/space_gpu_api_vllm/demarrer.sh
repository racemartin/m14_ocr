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
# ni aucun flag cible, ni une pause fixe de 20s avant le lancement de
# vLLM (teste aussi, meme segfault exact malgre le delai -- exclut
# l'hypothese d'un GPU/driver pas encore stabilise a l'allocation
# fraiche du conteneur) ne sont une solution de production viable a ce
# jour.
#
# vllm==0.28.0 (Dockerfile) exclu : torch incompatible avec Python 3.11
# de ce conteneur. vllm==0.22.0 (compatibilite Python 3.11 verifiee via
# PyPI avant test, cf. Dockerfile) : MEME segfault, mais trace native
# bien plus detaillee cette fois -- crash dans le dispatcher d'operateurs
# JIT de torch (torch::jit::invokeOperatorFromPython ->
# PythonKernelHolder), juste apres un nouveau log absent en 0.20.2 :
# "Using default LoRA kernel configs". Piste LoRA jamais testee
# specifiquement sur 0.22.0 (seulement sur 0.20.2, ou elle avait ete
# exclue) -- le chemin de kernel LoRA differe visiblement entre les deux
# versions.
#
# DIAGNOSTIC EN COURS : LoRA desactive ci-dessous, specifiquement pour
# vllm==0.22.0. A REVERTIR (remettre --enable-lora --lora-modules
# dpo=mombasstic/chsa-triage-dpo-lora --max-lora-rank 16) des que le
# resultat de ce test est connu.
set -euo pipefail

vllm serve Qwen/Qwen3-1.7B-Base \
    --enforce-eager \
    --port 8000 &

export CHSA_MOTEUR_INFERENCE=distant
export CHSA_URL_MOTEUR_INFERENCE="http://127.0.0.1:8000"

exec uvicorn interfaces.api.main:app --host 0.0.0.0 --port 7860
