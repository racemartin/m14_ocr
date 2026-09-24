# Image du Space HF Docker/GPU "API FastAPI + vLLM".
#
# DIFFERENT du `Dockerfile` a la racine du depot (celui-ci reste
# l'image API SEULE, decouplee de vLLM, cf. sa propre note de
# conception : vLLM comme service HTTP externe). Ce fichier-ci sert le
# DEUXIEME deploiement, decision produit actee : un unique Space
# Docker/GPU (couteux, allume seulement pendant les tests), qui fait
# tourner l'API ET vLLM dans le MEME conteneur (un seul cout GPU, un
# seul on/off) ; le second Space, Streamlit/CPU, leger et toujours
# allume, est prepare separement dans `interfaces/web/README_space.md`.
#
# `demarrer.sh` (ce dossier) lance vLLM en arriere-plan puis l'API
# FastAPI au premier plan ; `GET /sante` (interfaces/api/app.py) sonde
# `vLLM /health` en interne et reste toujours HTTP 200, meme si vLLM
# met plusieurs minutes a charger le modele.
#
# Deploiement reel du 24/09/2026 (mombasstic/chsa-triage-api). Historique
# complet dans le git log de ce fichier et de `demarrer.sh` -- resume :
# un segfault natif systematique juste apres le chargement des poids a
# resiste a HUIT pistes ciblees differentes (multiprocessing,
# permissions, LoRA, backend d'attention, async scheduling, delai de
# demarrage, version de vLLM 0.20.2->0.22.0, desaccord driver/CUDA
# exclu par nvidia-smi reel) sur une image construite a la main
# (nvidia/cuda + `uv sync`/`uv pip install vllm` manuel). Signale en
# amont : https://github.com/vllm-project/vllm/issues/58616
#
# CHANGEMENT DE STRATEGIE ICI (24/09/2026) : au lieu de continuer a
# installer vLLM nous-memes dans une image CUDA generique, on part
# desormais de l'image officielle `vllm/vllm-openai`, construite et
# testee par l'equipe vLLM elle-meme (environnement Python
# autonome, coherent avec ce que vLLM attend reellement) -- inspire
# des exemples officiels https://docs.vllm.ai/en/latest/examples/.
# Avantage supplementaire : plus besoin de jongler avec la
# compatibilite Python 3.11/3.12 d'une version de vLLM forcee dans
# NOTRE propre environnement (cf. piste 0.28.0 exclue ci-dessus) --
# l'image officielle a son propre Python coherent, quelle que soit sa
# version de vLLM.

FROM vllm/vllm-openai:latest

# Dependances API legeres, installees pendant que l'image reste root
# (utilisateur par defaut de cette image officielle) -- AVANT de
# creer/basculer vers l'utilisateur UID 1000 ci-dessous, qui n'aurait
# pas les droits d'ecriture sur l'environnement Python deja en place.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pydantic

# UID 1000 requis par tout Space Docker HF au runtime, quel que soit
# l'utilisateur du build (root par defaut sur cette image officielle)
# -- meme raisonnement que precedemment
# (https://huggingface.co/docs/hub/spaces-sdks-docker#permissions).
# Contrairement a notre precedente image nvidia/cuda, celle-ci a DEJA
# un compte UID 1000 (heritee de sa base Ubuntu -- `useradd -u 1000`
# echoue avec "UID 1000 is not unique"). On renomme ce compte existant
# plutot que d'en creer un nouveau.
RUN existant=$(getent passwd 1000 | cut -d: -f1); \
    if [ -n "$existant" ]; then \
        usermod --login user --home /home/user --move-home "$existant"; \
    else \
        useradd -m -u 1000 user; \
    fi
ENV HOME=/home/user
USER user
WORKDIR $HOME/app

COPY --chown=user src/ src/
COPY --chown=user interfaces/ interfaces/
COPY --chown=user training/ training/
COPY --chown=user monitoring/ monitoring/
COPY --chown=user deploy/space_gpu_api_vllm/demarrer.sh ./demarrer.sh
RUN chmod +x ./demarrer.sh

ENV PYTHONPATH="$HOME/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:7860/sante').raise_for_status()" || exit 1

# L'image officielle definit ENTRYPOINT ["vllm", "serve"] -- sans
# le vider ici, notre CMD serait ajoute comme ARGUMENT de cet
# entrypoint (`vllm serve ./demarrer.sh`) au lieu de le remplacer.
# Build reel : erreur "config file at './demarrer.sh' is not a valid
# JSON file" (vLLM essayait de charger le script comme s'il etait le
# modele).
ENTRYPOINT []
CMD ["./demarrer.sh"]
