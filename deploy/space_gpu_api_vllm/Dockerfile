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
# Deploiement reel du 24/09/2026 (mombasstic/chsa-triage-api) : vLLM
# echoue au demarrage sans compilateur C ("Failed to find C compiler"),
# meme avec --enforce-eager (cf. demarrer.sh) -- ce flag desactive
# seulement torch.compile du graphe du modele, mais vLLM compile aussi
# via Triton un kernel de tri/echantillonnage separe
# (topk_topp_sampler) qui a lui aussi besoin d'un compilateur, quel que
# soit le mode eager/compile. build-essential est donc necessaire ici,
# l'image de base "runtime" (par opposition a "devel") n'en a pas par
# defaut.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY interfaces/ interfaces/
COPY training/ training/
COPY monitoring/ monitoring/

# `remote` (torch CUDA + vllm) ET `api` (FastAPI/uvicorn) dans la MEME
# image ici, contrairement au `Dockerfile` racine qui n'installe que
# `api` : c'est exactement ce qui distingue les deux fichiers.
RUN uv sync --extra api --extra remote --no-dev --frozen

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

COPY deploy/space_gpu_api_vllm/demarrer.sh ./demarrer.sh
RUN chmod +x ./demarrer.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:7860/sante').raise_for_status()" || exit 1

CMD ["./demarrer.sh"]
