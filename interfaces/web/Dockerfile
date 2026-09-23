# Image du Space HF Docker/CPU "frontend Streamlit de test" (Etape 4).
#
# HF Spaces ne propose plus "Streamlit" comme SDK de premier niveau
# dans son flux de creation actuel : "Streamlit" y apparait desormais
# comme un TEMPLATE a l'interieur du SDK "Docker" (confirmee par une
# erreur serveur 400 sur `hf repo create --sdk streamlit`, cf.
# AGENTS.md). Ce `Dockerfile` remplace donc le SDK natif `sdk:
# streamlit` initialement prevu dans `README_space.md` de ce dossier.
#
# Meme patron exact que le `Dockerfile` a la racine du depot (image API
# seule) : `python:3.11-slim`, `uv sync --extra <extra> --no-dev
# --frozen`, port 7860. DIFFERENT du `Dockerfile` racine (extra `api`,
# `uvicorn`) et de `deploy/space_gpu_api_vllm/Dockerfile` (extras
# `api`+`remote`, CUDA, vLLM) : celui-ci installe l'extra `web`
# (streamlit+httpx) et lance `streamlit run` sur
# `interfaces/web/app_test_inference.py`, qui ne parle qu'HTTP a l'API
# FastAPI reelle (aucun import de `src/chsa_triage`, cf. sa propre
# note de conception). Construit avec la racine du depot comme contexte
# de build (memes chemins COPY que les deux autres Dockerfile du
# projet), pas `interfaces/web/` malgre son emplacement dans ce
# dossier.

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependances d'abord (cache Docker), code ensuite.
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY interfaces/ interfaces/
COPY training/ training/
COPY monitoring/ monitoring/

RUN uv sync --extra web --no-dev --frozen

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["streamlit", "run", "interfaces/web/app_test_inference.py", "--server.port=7860", "--server.address=0.0.0.0"]
