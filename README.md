<p align="center">
  <img src="docs/images/hospital-logo-design-vector-medical-cross/v987-18a.png" alt="CHSA" width="120">

  # CHSA Triage : Agent IA de Triage Médical (POC)


[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Qwen3--1.7B-FFD21E)](https://huggingface.co/docs/transformers)
[![TRL](https://img.shields.io/badge/TRL-SFT%20%2B%20DPO-FF6F00)](https://huggingface.co/docs/trl)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2)](https://huggingface.co/docs/peft)
[![vLLM](https://img.shields.io/badge/vLLM-inference-00B2A9)](https://docs.vllm.ai)
[![Presidio](https://img.shields.io/badge/Presidio-RGPD%20anonymisation-4B8BBE)](https://github.com/microsoft/presidio)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://www.docker.com)
[![HF Hub](https://img.shields.io/badge/🤗%20HF%20Hub-checkpoints-FFD21E)](https://huggingface.co)

 </p>

<table id="introduction" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Introduction</h1>
</td></tr></table>

POC d'agent IA de triage médical pour le Centre Hospitalier Saint-Aurélien,
développé sous architecture hexagonale. **Ce README est volontairement
court** : chaque section donne juste assez de contexte pour comprendre et
lancer la commande essentielle. Le détail complet (toutes les variantes de
commandes, tableaux de résultats run par run, hyperparamètres, dépannage
étendu) vit dans [`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md).

**Pipeline en 4 étapes :**

1. **Données** : 6 corpus sources fusionnés, dédupliqués, anonymisés (RGPD), répartis en splits.
2. **SFT + LoRA** : fine-tuning supervisé de `Qwen/Qwen3-1.7B-Base`.
3. **DPO** : alignement sur les préférences cliniques, continue le checkpoint SFT.
4. **Déploiement** : API FastAPI + vLLM, frontend Streamlit de test, Space HF.

**Documents de référence :**
- [PDF Support de présentation (M14)](docs/M14_Support_de_presentation_V3.pdf)
- [PDF Rapport technique MC4](docs/M14_Rapport_technique_CHSA_Triage_V3.pdf)
- [README_AVEC_DETAILS.md](README_AVEC_DETAILS.md) : toutes les commandes, tableaux de résultats et hyperparamètres
- Vue d'ensemble visuelle : [`docs/diagrams/00_vue_ensemble/vision_generale_etapes.png`](docs/diagrams/00_vue_ensemble/vision_generale_etapes.png)
- Version détaillée du schéma (scripts/adaptateurs/dépôts HF réels) : [`docs/diagrams/00_vue_ensemble/vision_generale_etapes_v3_detaille.png`](docs/diagrams/00_vue_ensemble/vision_generale_etapes_v3_detaille.png)

<table id="table-des-matières" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Table des matières</h1>
</td></tr></table>

- [Démarrage rapide](#démarrage-rapide)
- [1. Préparation de données](#1-préparation-de-données)
- [2. SFT + LoRA](#2-sft--lora)
- [3. DPO](#3-dpo)
- [4. Déploiement](#4-déploiement)
  - [4.1 Configuration GPU (vLLM, Space HF)](#41-configuration-gpu-vllm-space-hf)
  - [4.2 Configuration CPU gratuite (llama.cpp)](#42-configuration-cpu-gratuite-llamacpp)
  - [4.3 Outil de comparaison de précision (CPU, transformers+peft)](#43-outil-de-comparaison-de-précision-cpu-transformerspeft)
  - [4.4 Scénarios de test cliniques](#44-scénarios-de-test-cliniques)
- [Dépannage](#dépannage)

<table id="démarrage-rapide" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Démarrage rapide</h1>
</td></tr></table>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra local --extra dev
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_sm

uv tool install huggingface_hub[cli]
hf auth login
# Colle un token avec le rôle "write"
```

Vérifications d'environnement (à relancer avant chaque étape correspondante) :

```bash
uv run python scripts/check_env_local.py     # avant l'étape 1 (sans GPU)
uv run python scripts/check_env_gpu.py       # avant un run SFT/DPO
uv run python scripts/check_env_remote_hf.py # avant tout lancement HF Jobs/Spaces
```

```bash
uv run pytest tests/ -q
```

Détail complet (compte payant, HF Jobs, discipline de facturation, tableau
de tous les scripts par étape) : [`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#démarrage-rapide).

<table id="1-préparation-de-données" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h1 style="border-bottom:none; margin:0;">1. Préparation de données</h1>
</td></tr></table>

6 corpus sources sont téléchargés, fusionnés en un dataset pivot unique
(déduplication par identifiant déterministe), anonymisés (Presidio +
spaCy, jamais en modifiant le fichier original), contrôlés, puis
répartis en splits train/val/test stratifiés.

```bash
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/<corpus>.jsonl --corpus <nom> --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_04_00_anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --sortie data/processed/dataset_pivot_anonymise.jsonl --strategie replace --limite 5000
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl
```

**Résultat réel** : 147 204 enregistrements bruts fusionnés en **134 883
exemples pivot** (12 321 doublons exacts écartés), intégralement anonymisés
et répartis (37 802 SFT / 97 081 DPO).

Commandes complètes (téléchargement des 6 sources, profilage,
anonymisation par vagues, révision humaine des cas ambigus, extraction de
sous-ensembles publiables) : [`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#1-préparation-de-données).

<table id="2-sft--lora" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h1 style="border-bottom:none; margin:0;">2. SFT + LoRA</h1>
</td></tr></table>

Deux baselines zero-shot mesurées avant tout entraînement (CPU quantifié
et GPU pleine précision, pour isoler l'effet de la quantification), puis
l'entraînement SFT-LoRA réel (QLoRA 4-bit, rang 16) sur `Qwen/Qwen3-1.7B-Base`,
via HF Jobs (GPU L4).

```bash
hf jobs uv run \
    --flavor l4x1 --timeout 6h \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    -v hf://datasets/mombasstic/chsa-triage-sft-train-data:/mnt/train-data \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/training/E2_04_sft_train.py \
    --recette recipes/sft_qwen3_lora.yaml \
    --dataset /mnt/train-data/dataset_pivot_anonymise.jsonl \
    --suivi-hf-repo mombasstic/chsa-triage-sft-metrics \
    --checkpoint-hf-repo mombasstic/chsa-triage-sft-lora \
    --assistant-only-loss false
```

**Résultat le plus important du projet** (278 exemples `split=test`,
comparaison directe) :

| | F1 moyen (token) | Latence moyenne |
|---|---|---|
| Baseline CPU (Q4_K_M) | 0,037 | ~21,6 s |
| Baseline GPU (bf16) | 0,043 | ~7,3 s |
| **Post-SFT (bf16+LoRA)** | **0,112** | ~11,6 s |

Le F1 token **quasi triple** par rapport à la meilleure baseline. Premier
entraînement réel mené à terme avec succès (~20 min, 342 pas, 3 époques,
verdict **saine**).

Hyperparamètres détaillés (quantification, LoRA, grille de secours),
suivi d'entraînement (MLflow/Streamlit), évaluation post-SFT complète :
[`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#2-sft--lora).

<table id="3-dpo" style="width:100%;"><tr><td style="background-color:#f5cf47;">
<h1 style="border-bottom:none; margin:0;">3. DPO</h1>
</td></tr></table>

Continue le checkpoint SFT-LoRA sur les paires de préférence, via
`training/E3_03_dpo_train.py`.

```bash
hf jobs uv run \
    --flavor l4x1 --timeout 2h \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    -v hf://datasets/mombasstic/chsa-triage-dpo-train-data:/mnt/train-data \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/training/E3_03_dpo_train.py \
    --recette recipes/dpo_qwen3_lora.yaml \
    --dataset /mnt/train-data/dataset_chsa_triage_dpo_anonymise_5000.jsonl \
    --suivi-hf-repo mombasstic/chsa-triage-dpo-metrics \
    --checkpoint-hf-repo mombasstic/chsa-triage-dpo-lora \
    --skip-reformulation
```

**Résultat réel** (comparaison directe, mêmes 278 exemples) :

| | F1 moyen (token) | Latence moyenne |
|---|---|---|
| Baseline GPU (bf16) | 0,043 | ~7,3 s |
| Post-SFT (bf16+LoRA) | **0,112** | ~11,6 s |
| Post-DPO, premier essai (`beta=0,1`) | 0,049 | ~12,0 s |
| **Post-DPO, corrigé (`beta=0,3` + `repetition_penalty=1,2`)** | **0,110** | ~11,2 s |

Le premier run DPO (`beta=0,1`) a fait **régresser** le F1 : le modèle
dégénérait réellement (changements de langue aléatoires, répétitions),
signe d'un ancrage trop faible au modèle de référence. Diagnostiqué et
corrigé en relevant `beta=0,3` : le F1 revient au niveau du post-SFT, et
**ce gain tient à l'échelle réelle** (0,112 sur 100 exemples, 0,110 sur
le run complet à 5000, aucune régression). `beta=0,3` est désormais
l'hyperparamètre retenu.

Journal complet des runs (jobs, métriques `rewards/*`, diagnostic pas à
pas des 3 tentatives avant convergence) : [`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#3-dpo).

<table id="4-déploiement" style="width:100%;"><tr><td style="background-color:#f5b0e0;">
<h1 style="border-bottom:none; margin:0;">4. Déploiement</h1>
</td></tr></table>

Trois façons de tester l'API + le frontend Streamlit une fois le modèle
entraîné, selon ce qu'on a sous la main (budget GPU, ou rien du tout) :

| Configuration | Où tourne le modèle | Coût | Vitesse |
|---|---|---|---|
| [4.1 vLLM (Space HF)](#41-configuration-gpu-vllm-space-hf) | GPU distant (Space HF) | payant à l'heure | rapide |
| [4.2 llama.cpp (local)](#42-configuration-cpu-gratuite-llamacpp) | CPU local, quantifié Q4_K_M | gratuit | lent |
| [4.3 transformers+peft (local)](#43-outil-de-comparaison-de-précision-cpu-transformerspeft) | CPU local, pleine précision | gratuit | très lent, outil de comparaison ponctuelle uniquement |

Code testé (`uv run pytest tests/ -q`, **467 passed / 50 skipped**),
Space HF réellement créé et déployé (`mombasstic/chsa-triage-api`,
Docker/GPU), actuellement **en pause** pour maîtriser le coût. Garde-fou
de sécurité clinique NF4 (juge LLM) : décision produit encore ouverte,
non implémenté.

<table id="41-configuration-gpu-vllm-space-hf" style="width:100%;"><tr><td style="background-color:#f5b0e0;">
<h2 style="border-bottom:none; margin:0;">4.1 Configuration GPU (vLLM, Space HF)</h2>
</td></tr></table>

Depuis le 01/10/2026, le Space sert directement l'interface Streamlit
de test (plus besoin de la lancer en local) :

```bash
# Sortir le Space de pause (facture à la minute pendant qu'il tourne)
hf spaces restart mombasstic/chsa-triage-api
hf spaces wait mombasstic/chsa-triage-api
```

Puis ouvrir dans un navigateur : **https://mombasstic-chsa-triage-api.hf.space**
(l'URL affichée par `hf spaces restart`, ou la page du Space
[huggingface.co/spaces/mombasstic/chsa-triage-api](https://huggingface.co/spaces/mombasstic/chsa-triage-api)
qui affiche la même interface dans un cadre HF).

**Repasser le Space en pause après usage** (sinon la facturation continue) :

```bash
hf spaces pause mombasstic/chsa-triage-api
```

Alternative (frontend lancé en local contre l'API distante, utile par
exemple pour comparer deux configurations côte à côte) :
[`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#41-configuration-gpu-vllm-space-hf).

<table id="42-configuration-cpu-gratuite-llamacpp" style="width:100%;"><tr><td style="background-color:#f5b0e0;">
<h2 style="border-bottom:none; margin:0;">4.2 Configuration CPU gratuite (llama.cpp)</h2>
</td></tr></table>

Sert le modèle de base GGUF quantifié (Q4_K_M) avec l'adaptateur LoRA
chargé à part (`--lora`, jamais fusionné), sans aucun coût GPU.

```bash
# Terminal 1 : serveur du modèle
LD_LIBRARY_PATH=./llama-b10985 ./llama-b10985/llama-server \
    -m Qwen3-1.7B-Base.Q4_K_M.gguf --lora chsa-triage-dpo-lora.gguf \
    --port 8080 -c 1024 -t 2 --no-webui --host 0.0.0.0 --parallel 1

# Terminal 2 : API
export CHSA_CLE_API_DEMO=change-moi
export CHSA_MOTEUR_INFERENCE=local
uv run uvicorn interfaces.api.main:app --port 7860

# Terminal 3 : frontend
export CHSA_API_URL_BASE=http://127.0.0.1:7860
export CHSA_API_CLE=change-moi
uv run streamlit run interfaces/web/app_test_inference.py
```

Étapes de récupération/conversion des poids en GGUF (téléchargement du
modèle de base, conversion du LoRA) : [`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#47-test-local-gratuit-avec-le-checkpoint-dpo).

<table id="43-outil-de-comparaison-de-précision-cpu-transformerspeft" style="width:100%;"><tr><td style="background-color:#f5b0e0;">
<h2 style="border-bottom:none; margin:0;">4.3 Outil de comparaison de précision (CPU, transformers+peft)</h2>
</td></tr></table>

**Ceci n'est pas une troisième option de service**, juste un outil pour
comparer manuellement une réponse en pleine précision (float32) à celle
du chemin quantifié (§4.2) sur la même entrée. Volontairement lent (un
seul échange peut prendre plus d'une heure sur un CPU modeste) : jamais
pour du chat en direct, jamais plusieurs requêtes à la fois (sérialisé).
Aucun serveur de modèle séparé à lancer, tout est dans le même processus.

```bash
export CHSA_CLE_API_DEMO=change-moi
export CHSA_MOTEUR_INFERENCE=comparaison_precision_cpu
uv run uvicorn interfaces.api.main:app --port 8000
```

```bash
export CHSA_CLE="change-moi"
export CHSA_BASE="http://127.0.0.1:8000"
# puis les mêmes appels curl conversation/message/diagnostic que §4.4
```

<table id="44-scénarios-de-test-cliniques" style="width:100%;"><tr><td style="background-color:#f5b0e0;">
<h2 style="border-bottom:none; margin:0;">4.4 Scénarios de test cliniques</h2>
</td></tr></table>

Trois entretiens réels, réutilisables tels quels contre n'importe quelle
configuration ci-dessus (ajuster `$CHSA_BASE`/`$CHSA_CLE`) :

- **Scénario 1, urgent (ESI 2 probable)** : douleur thoracique, 62 ans, irradiation bras gauche, antécédent d'infarctus.
- **Scénario 2, modéré (ESI 3-4 probable)** : fièvre 38,5°C, toux sèche, 34 ans, sans antécédent.
- **Scénario 3, léger (ESI 5 probable)** : entorse cheville, 28 ans, douleur modérée.

```bash
CONVERSATION_ID=$(curl -sS -X POST "$CHSA_BASE/conversations" \
  -H "X-API-Key: $CHSA_CLE" | python3 -c "import sys,json; print(json.load(sys.stdin)['conversation_id'])")

curl -sS -X POST "$CHSA_BASE/conversations/$CONVERSATION_ID/messages" \
  -H "X-API-Key: $CHSA_CLE" -H "Content-Type: application/json" \
  -d '{"message": "<premier tour du scénario choisi>"}'
# répéter pour chaque tour du scénario

curl -sS -X POST "$CHSA_BASE/conversations/$CONVERSATION_ID/diagnostic" \
  -H "X-API-Key: $CHSA_CLE"
```

Texte complet des 3 scénarios, tour par tour : [`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#49-scenarios-de-test-cliniques).

<table id="dépannage" style="width:100%;"><tr><td style="background-color:#d9d9d9;">
<h1 style="border-bottom:none; margin:0;">Dépannage</h1>
</td></tr></table>

Problèmes réels déjà rencontrés et résolus, détail complet dans
[`README_AVEC_DETAILS.md`](README_AVEC_DETAILS.md#dépannage) :

- Segfault vLLM au démarrage (Space GPU) → construire depuis l'image officielle `vllm/vllm-openai`.
- Réponses interminables → `n_predict` jamais traduit vers `max_tokens` (vLLM).
- Dégénérescence en déploiement → `temperature`/`repetition_penalty` jamais explicites.
- Instabilité DPO (OOM éval, dégénérescence) → `per_device_eval_batch_size` non fixé, puis `beta` trop faible.
- Chargement du modèle en double / sortie corrompue (outil §4.3) → `generer()` non sérialisé entre requêtes concurrentes.
- `bf16` catastrophiquement lent sur CPU sans accélération matérielle (outil §4.3) → `float32` direct.

<table id="auteur" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Auteur</h1>
</td></tr></table>

**Rafael Cerezo Martín**

- Email : [rafael.cerezo.martin@icloud.com](mailto:rafael.cerezo.martin@icloud.com)
- GitHub : [@racemartin](https://github.com/racemartin)

<table id="licence" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Licence</h1>
</td></tr></table>

MIT License, voir [LICENSE](LICENSE) pour les détails.
