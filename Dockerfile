# Image API seule (FastAPI + adaptateur vLLM HTTP), PAS vLLM lui-meme.
#
# Decision documentee (Etape 4, cahier des charges §6/§8) : vLLM tourne
# comme un service SEPARE (`vllm serve ...`, cf. README §4), jamais
# dans cette meme image. vLLM + ses dependances GPU (torch CUDA, ~5+
# Go) demandent un pilote NVIDIA/CUDA dans le conteneur hote, hors de
# portee d'une image API generique destinee a HF Spaces (Docker SDK,
# souvent sans GPU pour le seul frontal) ; l'adaptateur
# `VllmEndpointInferenceAdapter` parle a ce service vLLM par HTTP
# (`CHSA_URL_MOTEUR_INFERENCE`), exactement le decouplage que permet
# l'architecture hexagonale du projet (cf. AGENTS.md/docs/01_environnement/01_architecture_hexagonale.md).
# C'est le choix le plus simple a faire fonctionner pour un POC : une
# seule image legere, reconstruite rapidement, qui ne depend d'aucune
# dependance GPU au moment du build.
#
# Note connue : `torch` est une dependance BASE inconditionnelle du
# paquet (`pyproject.toml::[project.dependencies]`, herite des Etapes
# 1-3, hors perimetre de cette tache) ; `uv sync --extra api` l'installe
# donc aussi (CPU, via l'index PyTorch CPU deja configure dans
# `pyproject.toml`), meme si l'API elle-meme ne l'importe jamais
# (aucun adaptateur transformers/trl n'est charge en mode `distant`).
# Image plus lourde qu'un strict minimum FastAPI, mais fonctionnelle et
# reproductible ; alleger cette dependance de base est hors perimetre
# de cette tache.

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependances d'abord (cache Docker), code ensuite.
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY interfaces/ interfaces/
COPY training/ training/
COPY monitoring/ monitoring/

RUN uv sync --extra api --no-dev --frozen

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:7860/sante').raise_for_status()" || exit 1

CMD ["uvicorn", "interfaces.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
