# Image du Space HF Docker/GPU "API FastAPI + vLLM" (Etape 4, PREP DE
# DEPLOIEMENT UNIQUEMENT : ce fichier n'a JAMAIS ete build ni lance
# dans cette tache, aucun GPU/Docker disponible dans ce bac a sable,
# cf. AGENTS.md ; a verifier manuellement au premier deploiement reel).
#
# DIFFERENT du `Dockerfile` a la racine du depot (celui-ci reste
# l'image API SEULE, decouplee de vLLM, cf. sa propre note de
# conception : vLLM comme service HTTP externe). Ce fichier-ci sert le
# DEUXIEME deploiement, decision produit actee (cf. tache
# "healthcheck-streamlit") : DEUX HF Spaces separes, dont celui-ci, le
# Space Docker/GPU (couteux, allume seulement pendant les tests), qui
# fait tourner l'API ET vLLM dans le MEME conteneur (un seul cout GPU,
# un seul on/off) ; le second Space, Streamlit/CPU, leger et toujours
# allume, est prepare separement dans `interfaces/web/README_space.md`.
#
# `demarrer.sh` (ce dossier) lance vLLM en arriere-plan puis l'API
# FastAPI au premier plan ; `GET /sante` (interfaces/api/app.py) sonde
# `vLLM /health` en interne et reste toujours HTTP 200, meme si vLLM
# met plusieurs minutes a charger le modele (cf. ce fichier pour le
# raisonnement complet).
#
# Deploiement reel du 24/09/2026 (mombasstic/chsa-triage-api). Causes
# reelles rencontrees et reglees (voir historique git pour le detail
# complet) : compilateur C absent de l'image "runtime" -> build-essential
# + python3.11-dev ajoutes ; UID 1000 requis par tout Space Docker HF
# au runtime -> utilisateur cree explicitement ci-dessous
# (https://huggingface.co/docs/hub/spaces-sdks-docker#permissions).
#
# Segfault natif persistant ensuite, TOUJOURS au meme endroit (juste
# apres le chargement des poids, pendant le profile_run interne). Six
# pistes ciblees exclues par tests reels (meme crash identique malgre
# chaque changement) : multiprocessing V1, LoRA/Punica, backend
# d'attention, async scheduling, delai de demarrage fixe -- voir
# demarrer.sh pour le detail. Seul VLLM_TRACE_FUNCTION=1 (ralentit
# massivement l'execution, inutilisable en production) evite le crash,
# signature de condition de course dependante du temps reel.
#
# PISTE TESTEE ET EXCLUE : forcer vllm==0.28.0 (au lieu du 0.20.2
# verrouille par uv.lock) UNIQUEMENT dans cette image, via `uv pip
# install` apres le `uv sync --frozen`. Resultat reel : un saut de 8
# versions mineures entraine un torch incompatible avec ce conteneur
# (AttributeError: module 'sys' has no attribute
# 'get_int_max_str_digits', import de vllm.env_override) -- probleme
# DIFFERENT du segfault, jamais atteint. Retire ; une eventuelle
# version intermediaire (saut plus petit, compatibilite verifiee)
# resterait a explorer si cette piste est reprise.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3-pip curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

RUN useradd -m -u 1000 user
ENV HOME=/home/user
USER user
WORKDIR $HOME/app

COPY --chown=user pyproject.toml uv.lock ./
COPY --chown=user src/ src/
COPY --chown=user interfaces/ interfaces/
COPY --chown=user training/ training/
COPY --chown=user monitoring/ monitoring/

# `remote` (torch CUDA + vllm) ET `api` (FastAPI/uvicorn) dans la MEME
# image ici, contrairement au `Dockerfile` racine qui n'installe que
# `api` : c'est exactement ce qui distingue les deux fichiers.
RUN uv sync --extra api --extra remote --no-dev --frozen

ENV PATH="$HOME/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

COPY --chown=user deploy/space_gpu_api_vllm/demarrer.sh ./demarrer.sh
RUN chmod +x ./demarrer.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:7860/sante').raise_for_status()" || exit 1

CMD ["./demarrer.sh"]
