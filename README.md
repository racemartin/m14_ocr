# CHSA Triage — Agent IA de Triage Médical (POC)

POC d'agent IA de triage médical pour le Centre Hospitalier
Saint-Aurélien, développé sous architecture hexagonale (ports &
adaptateurs). La documentation vit dans `docs/`, structurée en
sous-dossiers numérotés selon les étapes du projet :

| Dossier | Contenu |
|---|---|
| `docs/00_cadrage/` | Objectifs séquencés + décisions justifiées, cahier des charges |
| `docs/01_environnement/` | Installation (`uv` local, HF payant distant) + architecture hexagonale |
| `docs/02_etape1_donnees/` | Documentation spécifique à la préparation des données (à venir) |
| `docs/03_etape2_sft/` | Documentation SFT + LoRA (à venir) |
| `docs/04_etape3_dpo/` | Documentation alignement DPO (à venir) |
| `docs/05_etape4_deploiement/` | Documentation déploiement/évaluation (à venir) |
| `docs/diagrams/` | Diagrammes UML (activité, séquence, paquets, déploiement) par étape — `.puml`+`.png`+`.svg`+`.pdf`, voir `docs/diagrams/README.md` |

Chaque document se termine par un renvoi vers le suivant, pour lire
la documentation dans l'ordre du projet en partant de
`docs/00_cadrage/00_objectifs_du_projet.md`.

## Démarrage rapide (Étape 1 — données)

```bash
# Installation
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra local --extra dev
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_sm

# installer PyTorch compatible avec CPU/CUDA
uv add torch --index-url https://download.pytorch.org/whl/cpu

# Vérification de l'environnement
uv run python scripts/check_env_local.py

# Tests
uv run pytest tests/ -v

# Pipeline Étape 1 (les 4 corpus, fusionnes dans le meme dataset pivot)

# 1. Telechargement (Hugging Face Hub -> data/raw/)
# NB : la configuration "oeq" de MediQAl n'a qu'un split "test" (pas de "train") -- --split explicite requis
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration oeq --split test --sortie data/raw/mediqal_oeq.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqu --sortie data/raw/mediqal_mcqu.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqm --sortie data/raw/mediqal_mcqm.jsonl

uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub nthngdy/frenchmedmcqa      --sortie data/raw/frenchmedmcqa.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub keivalya/MedQuad-MedicalQnADataset  --sortie data/raw/medquad.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub TsinghuaC3I/UltraMedical-Preference --sortie data/raw/ultramedical_preference.jsonl

# 2. Profilage individuel (un rapport ydata-profiling par corpus)
uv run python interfaces/cli/profiler_corpus.py --source data/raw/mediqal_oeq.jsonl   --nom MediQAl-oeq
uv run python interfaces/cli/profiler_corpus.py --source data/raw/mediqal_mcqu.jsonl  --nom MediQAl-mcqu
uv run python interfaces/cli/profiler_corpus.py --source data/raw/mediqal_mcqm.jsonl  --nom MediQAl-mcqm

uv run python interfaces/cli/profiler_corpus.py --source data/raw/frenchmedmcqa.jsonl           --nom FrenchMedMCQA
uv run python interfaces/cli/profiler_corpus.py --source data/raw/medquad.jsonl                 --nom MedQuAD
uv run python interfaces/cli/profiler_corpus.py --source data/raw/ultramedical_preference.jsonl --nom UltraMedicalPreference

# 3. Construction du dataset pivot (meme --sortie : fusionne les corpus par identifiant)
# NB : seul MediQAl-oeq a un mapper aujourd'hui (schema question/answer).
# mediqal_mcqu.jsonl et mediqal_mcqm.jsonl (schema QCM) n'ont PAS encore
# de mapper -- cf. docs/02_etape1_donnees/00_couverture_exigences_officielles.md
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/mediqal_oeq.jsonl --corpus mediqal --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/frenchmedmcqa.jsonl --corpus frenchmedmcqa --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/medquad.jsonl --corpus medquad --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/ultramedical_preference.jsonl --corpus ultramedical_preference --sortie data/processed/dataset_pivot.jsonl

# 4. Anonymisation et decoupage en splits (une seule fois, sur le dataset pivot fusionne)
uv run python interfaces/cli/anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --strategie replace
uv run python interfaces/cli/decouper_splits.py --dataset data/processed/dataset_pivot.jsonl
```

## Structure (architecture hexagonale)

```
src/chsa_triage/
├── domain/            # entités + ports — zéro dépendance externe
├── application/       # cas d'usage — orchestrent les ports
└── infrastructure/    # adaptateurs concrets (JSONL, HF, Presidio, ydata-profiling, ...)
interfaces/            # adaptateurs primaires : cli/ (Étape 1), api/ et web/ (Étape 4)
training/              # scripts exécutés via HF Jobs (SFT, DPO) — Étapes 2-3
docker/                # Dockerfiles + docker-compose (frontend/backend) — Étape 4
```

Détail complet : `docs/01_environnement/01_architecture_hexagonale.md`.

## État d'avancement

- [x] Étape 0 — Cadrage, environnement, architecture
- [ ] Étape 1 — Préparation des données (squelette hexagonal prêt,
      mappers par corpus à affiner après profilage réel)
- [ ] Étape 2 — SFT + LoRA
- [ ] Étape 3 — DPO
- [ ] Étape 4 — Déploiement (FastAPI + Streamlit + vLLM + CI/CD)
