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
développé sous architecture hexagonale. Ce document est une version
**condensée** : chaque section suit le patron intro -> commande(s) réelle(s)
-> résultat obtenu, sans le détail d'implémentation.

Vue d'ensemble en un coup d'œil (entrée/sortie de chaque étape) :
[`docs/diagrams/00_vue_ensemble/vision_generale_etapes.png`](docs/diagrams/00_vue_ensemble/vision_generale_etapes.png).
Version détaillée (scripts/adaptateurs/dépôts HF réels, DPO marqué conceptuel) :
[`docs/diagrams/00_vue_ensemble/vision_generale_etapes_v3_detaille.png`](docs/diagrams/00_vue_ensemble/vision_generale_etapes_v3_detaille.png).


<table id="table-des-matières" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Table des matières</h1>
</td></tr></table>

- [Tableau récapitulatif des scripts](#tableau-récapitulatif-des-scripts)
- [1. Préparation de données](#1-préparation-de-données)
  - [1.1 Télécharger Corpus](#11-télécharger-corpus)
  - [1.2 Profiler Corpus](#12-profiler-corpus)
  - [1.3 Dataset Pivot](#13-dataset-pivot)
  - [1.4 Anonymisation](#14-anonymisation)
  - [1.5 Découpage en Splits](#15-découpage-en-splits)
- [2. SFT + LoRA](#2-sft--lora)
  - [2.1 Architecture](#21-architecture)
  - [2.2 Baseline Evaluation](#22-baseline-evaluation)
  - [2.3 SFT Train](#23-sft-train)
  - [2.4 SFT-LoRA Train](#24-sft-lora-train)
  - [2.5 Evaluation Post-SFT](#25-evaluation-post-sft)
  - [2.6 Suivi d'entraînement](#26-suivi-entrainement)
- [3. DPO](#3-dpo)
- [Vérifications d'environnement](#verifications-environnement)

<table id="tableau-récapitulatif-des-scripts" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Tableau récapitulatif des scripts</h1>
</td></tr></table>

Vue d'ensemble de tous les scripts exécutables du dépôt, classés par étape.


<table id="etape-1-preparation-des-donnees" style="width:100%; margin-left: 2cm;"><tr><td style="background-color:#b0f58c;">
<h3 style="border-bottom:none; margin:0;">Étape 1 — Préparation des données</h3>


| Script | Rôle |
|---|---|
| `interfaces/cli/E1_01_telecharger_corpus.py` | Télécharge un corpus brut depuis Hugging Face Hub et l'exporte en JSONL local (`data/raw/`). |
| `interfaces/cli/E1_02_profiler_corpus.py` | Génère un rapport de profilage ydata-profiling pour un corpus téléchargé. |
| `interfaces/cli/E1_03_00_construire_dataset_pivot.py` | Fusionne les corpus sources en un dataset pivot unique, dédupliqué par identifiant déterministe. |
| `interfaces/cli/E1_04_00_anonymiser_dataset.py` | Anonymise (Presidio/spaCy) le dataset pivot par vagues incrémentales, en écrivant dans un fichier séparé du pivot original. |
| `scripts/anonymiser_par_lots.sh` | Rappelle `E1_04_00_anonymiser_dataset.py` en boucle par vagues successives jusqu'à couverture complète du pivot. |
| `interfaces/cli/E1_04_02_controler_qualite_anonymisation.py` | Compare le pivot original et le fichier anonymisé sur un échantillon stratifié pour détecter de la PII résiduelle. |
| `interfaces/cli/E1_04_01_reviser_pii_residuelle.py` | Révision humaine persistée des candidats PII résiduelle et export de la liste d'exclusion pour la publication. |
| `interfaces/cli/E1_05_00_decouper_splits.py` | Répartit (stratifié train/val/test) les exemples du pivot anonymisé, de façon cumulative/incrémentale d'une exécution à l'autre. |
| `interfaces/cli/E1_05_01_verifier_repartition_splits.py` | Affiche la répartition des splits déjà assignés, par strate (type_exemple, source). |
| `interfaces/cli/E1_05_02_extraire_sous_ensemble_sft.py` / `E1_05_03_extraire_sous_ensemble_dpo.py` | Extraient les sous-ensembles SFT/DPO publiables (échantillons filtrés, hors périmètre de ce document condensé). |

</td>

</tr>
</table>
<table id="etape-1bis-baseline-zero-shot" style="width:100%; margin-left: 2cm;"><tr><td style="background-color:#d4f5b0;">
<h3 style="border-bottom:none; margin:0;">Étape 1bis — Baseline zero-shot</h3>


| Script | Rôle |
|---|---|
| `interfaces/cli/E1_06_00_evaluer_baseline.py` | Évalue la baseline zero-shot en local (CPU), via un `llama-server` sur un GGUF quantifié Q4_K_M. |
| `interfaces/cli/E1_06_01_evaluer_baseline_gpu.py` | Même évaluation baseline zero-shot mais en pleine précision (bf16, transformers) sur un job HF Jobs GPU. |

</td></tr></table>

<table id="etape-2-sft-lora" style="width:100%; margin-left: 2cm;"><tr><td style="background-color:#a6e3ff;">
<h3 style="border-bottom:none; margin:0;">Étape 2 — SFT + LoRA</h3>


| Script | Rôle |
|---|---|
| `interfaces/cli/E2_00_formater_dataset_chatml.py` | Point d'entrée autonome pour le rendu ChatML d'un split (mode didactique inclus). |
| `training/E2_04_sft_train.py` | Point d'entrée d'entraînement SFT-LoRA réel, exécuté via HF Jobs (GPU requis). |
| `interfaces/cli/E2_05_evaluer_post_sft.py` | Évaluation post-SFT : mêmes métriques/mêmes exemples que les baselines, mais via le modèle base+LoRA réellement entraîné. |
| `monitoring/app_suivi_entrainement.py`, `monitoring/importer_mlflow_local.py`, `monitoring/reconstruire_courbe_sft_depuis_log.py` | Suivi en direct (Streamlit/HF Space) et historisation locale (MLflow) de la courbe d'entraînement. |
| `monitoring/generer_presentation_etape2.py` | Régénère la présentation PowerPoint de synthèse à partir des chiffres mesurés (baselines, entraînement, évaluation post-SFT). |

</td></tr></table>

<table id="etape-3-dpo" style="width:100%; margin-left: 2cm;"><tr><td style="background-color:#f5cf47;">
<h3 style="border-bottom:none; margin:0;">Étape 3 — DPO</h3>


| Script | Rôle |
|---|---|
| `training/E3_03_dpo_train.py` | Point d'entrée d'entraînement DPO réel (continue le checkpoint SFT-LoRA), exécuté via HF Jobs (GPU requis) ; jamais lancé sur GPU à ce jour. |
| `interfaces/cli/E3_04_evaluer_post_dpo.py` | Évaluation post-DPO : mêmes métriques/même sous-ensemble que les baselines et le post-SFT, mais via le modèle base+LoRA DPO. |

</td></tr></table>

<table id="infrastructure" style="width:100%;  margin-left: 2cm;"><tr><td style="background-color:#d9d9d9;">
<h3 style="border-bottom:none; margin:0;">Infrastructure</h3>


| Script | Rôle |
|---|---|
| `scripts/check_env_local.py`, `scripts/check_env_gpu.py`, `scripts/check_env_remote_hf.py` | Vérifient que l'environnement (local, GPU, HF) est prêt avant chaque étape. |

</td></tr></table>

Détail de chaque commande dans les sections ci-dessous.



<table id="1-préparation-de-données" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h1 style="border-bottom:none; margin:0;">1. Préparation de données</h1>
</td></tr></table>

Les 6 fichiers sources sont fusionnés dans un dataset pivot unique, puis
anonymisés, contrôlés et répartis en splits train/val/test. Prérequis
d'installation (`uv sync`, modèles spaCy) : voir
`docs/01_environnement/00_guide_installation_environnement.md`.

<table id="11-télécharger-corpus" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h2 style="border-bottom:none; margin:0;">1.1 Télécharger Corpus</h2>
</td></tr></table>

Télécharge chacun des 6 corpus sources depuis Hugging Face Hub et les
exporte en JSONL local (`data/raw/`).

```bash
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration oeq --split test --sortie data/raw/mediqal_oeq.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqu --sortie data/raw/mediqal_mcqu.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqm --sortie data/raw/mediqal_mcqm.jsonl

uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub nthngdy/frenchmedmcqa      --sortie data/raw/frenchmedmcqa.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub keivalya/MedQuad-MedicalQnADataset  --sortie data/raw/medquad.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub TsinghuaC3I/UltraMedical-Preference --sortie data/raw/ultramedical_preference.jsonl
```

Les 6 fichiers sources sont récupérés dans `data/raw/`, prêts pour le
profilage puis la fusion en dataset pivot.

<table id="12-profiler-corpus" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h2 style="border-bottom:none; margin:0;">1.2 Profiler Corpus</h2>
</td></tr></table>

Génère un rapport ydata-profiling par corpus téléchargé, pour explorer sa
structure et sa qualité avant de le fusionner dans le pivot.

```bash
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_oeq.jsonl   --nom MediQAl-oeq
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_mcqu.jsonl  --nom MediQAl-mcqu
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_mcqm.jsonl  --nom MediQAl-mcqm

uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/frenchmedmcqa.jsonl           --nom FrenchMedMCQA
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/medquad.jsonl                 --nom MedQuAD
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/ultramedical_preference.jsonl --nom UltraMedicalPreference
```

Un rapport HTML par corpus a été généré, utilisé pour décider du mapping
de chaque source vers le schéma pivot (§1.3).

<table id="13-dataset-pivot" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h2 style="border-bottom:none; margin:0;">1.3 Dataset Pivot</h2>
</td></tr></table>

Fusionne les 6 corpus dans un dataset pivot unique, dédupliqué par un
identifiant déterministe (hash d'une clé naturelle propre à chaque source).

```bash
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/mediqal_oeq.jsonl --corpus mediqal_oeq --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/mediqal_mcqu.jsonl --corpus mediqal_mcqu --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/mediqal_mcqm.jsonl --corpus mediqal_mcqm --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/frenchmedmcqa.jsonl --corpus frenchmedmcqa --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/medquad.jsonl --corpus medquad --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/ultramedical_preference.jsonl --corpus ultramedical_preference --sortie data/processed/dataset_pivot.jsonl --taille-bloc 5000
```

Résultat réel : 147 204 enregistrements bruts fusionnés en **134 883
exemples pivot**, 12 321 doublons exacts détectés et écartés (archivés,
jamais perdus, dans `data/processed/doublons_supprimes.jsonl`). Cet
identifiant déterministe a mis au jour de vrais doublons entre sources qui
étaient invisibles avec des identifiants aléatoires. Détail par source :
`docs/02_etape1_donnees/00_couverture_exigences_officielles.md`.

<table id="14-anonymisation" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h2 style="border-bottom:none; margin:0;">1.4 Anonymisation</h2>
</td></tr></table>

Anonymise le pivot (Presidio + spaCy) par vagues incrémentales, en écrivant
dans un fichier **séparé** : le pivot original n'est jamais modifié.

```bash
uv run python interfaces/cli/E1_04_00_anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --sortie data/processed/dataset_pivot_anonymise.jsonl --strategie replace --limite 5000

# Pour enchaîner les vagues jusqu'à couverture complète du pivot :
scripts/anonymiser_par_lots.sh data/processed/dataset_pivot.jsonl data/processed/dataset_pivot_anonymise.jsonl replace 5000
```

Compare ensuite le pivot original et le fichier anonymisé sur un
échantillon stratifié pour détecter de la PII résiduelle :

```bash
uv run python interfaces/cli/E1_04_02_controler_qualite_anonymisation.py \
--dataset data/processed/dataset_pivot.jsonl \
--anonymise data/processed/dataset_pivot_anonymise.jsonl \
--taille-echantillon 200
```

Les candidats laissés en attente par ce contrôle passent par une
révision humaine persistée (une décision par candidat, jamais reperdue
d'une exécution à l'autre) :

```bash
uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py verify \
--dataset data/processed/dataset_pivot.jsonl \
--anonymise data/processed/dataset_pivot_anonymise.jsonl
```

L'anonymisation complète des 134 883 exemples a été menée à son terme par
vagues successives. Chaque exécution génère/fusionne un rapport RGPD cumulé
(entités détectées par type et par source), et un contrôle qualité par
comparaison de fichiers (`E1_04_02_controler_qualite_anonymisation.py`)
plus une révision humaine persistée des cas ambigus
(`E1_04_01_reviser_pii_residuelle.py`) ferment la boucle de validation
manuelle exigée par le cahier des charges. Détail complet (recognizer NIR,
normalisation des âges, méthodologie de révision) :
`docs/02_etape1_donnees/01_rapport_rgpd.md`.

<table id="15-découpage-en-splits" style="width:100%;"><tr><td style="background-color:#b0f58c;">
<h2 style="border-bottom:none; margin:0;">1.5 Découpage en Splits</h2>
</td></tr></table>

Répartit les exemples anonymisés en train/val/test, stratifié par
(type_exemple, source), de façon cumulative : un exemple déjà réparti n'est
jamais réassigné à une exécution ultérieure (aucune fuite train/test).

```bash
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl

# Vérifier la répartition obtenue, par strate :
uv run python interfaces/cli/E1_05_01_verifier_repartition_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl
```

Sur le pivot intégralement anonymisé, les 134 883 exemples ont été répartis
en splits (37 802 exemples SFT / 97 081 exemples DPO), vérifiés
représentatifs par strate. Les exemples portant un candidat de PII
résiduelle non résolu restent volontairement exclus du découpage tant
qu'aucune décision humaine n'est persistée (§1.4).

Pour publier un sous-ensemble filtré (échantillon stratifié), exporter
d'abord les identifiants à exclure puis extraire, ici pour le
sous-ensemble SFT :

```bash
uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py exporter \
--dataset data/processed/dataset_pivot.jsonl \
--anonymise data/processed/dataset_pivot_anonymise.jsonl

uv run python interfaces/cli/E1_05_02_extraire_sous_ensemble_sft.py \
--dataset data/processed/dataset_pivot_anonymise.jsonl \
--exclusions data/processed/identifiants_a_exclure_publication.jsonl \
--taille 5000
```

`E1_05_03_extraire_sous_ensemble_dpo.py` suit exactement le même
patron (mêmes `--dataset`/`--exclusions`/`--taille`) pour le
sous-ensemble DPO, en filtrant `type_exemple == DPO` au lieu de SFT.

<table id="2-sft--lora" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h1 style="border-bottom:none; margin:0;">2. SFT + LoRA</h1>
</td></tr></table>

Architecture hexagonale, deux baselines zero-shot mesurées avant tout
entraînement, l'entraînement SFT-LoRA réel, et son évaluation.

<table id="21-architecture" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h2 style="border-bottom:none; margin:0;">2.1 Architecture</h2>
</td></tr></table>

Domaine/ports/adaptateurs : `MoteurInference` (3 adaptateurs réels,
`LlamaCppInferenceAdapter`/`TransformersInferenceAdapter`/
`TransformersLoraInferenceAdapter`), `EntraineurSupervise`
(`TrlSftEntraineurAdapter`), `FormateurConversation`
(`ChatMLFormateurAdapter`), `SuiviExperimentation` (MLflow/TensorBoard/HF
dataset). Le CLI didactique ci-dessous exerce le formateur ChatML réel sur
un split, sans logique de rendu réimplémentée.

```bash
uv run python interfaces/cli/E2_00_formater_dataset_chatml.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --split train \
    --exemples 2
```

Affiche, pour 2 exemples réels, le tour prompt/completion brut puis le
texte ChatML final tel que produit par le tokenizer `Qwen/Qwen3-1.7B-Base`,
confirmant que le rendu utilisé à l'entraînement (§2.3) est bien celui-ci.
Schéma complet des classes/paquets/déploiement : `docs/diagrams/`.

<table id="22-baseline-evaluation" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h2 style="border-bottom:none; margin:0;">2.2 Baseline Evaluation</h2>
</td></tr></table>

Deux mesures indépendantes de `Qwen/Qwen3-1.7B-Base` **sans entraînement**,
sur le même sous-ensemble de 278 exemples `split=test` : CPU quantifié
Q4_K_M (llama.cpp) et GPU pleine précision bf16 (transformers, HF Jobs),
pour ne jamais mélanger l'effet de la quantification avec l'effet réel de
l'entraînement.

```bash
# CPU (llama.cpp), après avoir démarré un llama-server local :
LD_LIBRARY_PATH=./llama-b10985 ./llama-b10985/llama-server \
    -m Qwen3-1.7B-Base.Q4_K_M.gguf --port 8080 -c 1024 -t 2 --no-webui --host 0.0.0.0 --parallel 1

uv run python interfaces/cli/E1_06_00_evaluer_baseline.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --url-serveur http://127.0.0.1:8080

# GPU (transformers, bf16, HF Jobs) : le job distant lit le sous-ensemble de
# 278 exemples depuis un dépôt dataset HF dédié, à créer et publier une
# seule fois au préalable :
hf repo create mombasstic/chsa-triage-baseline-test --repo-type dataset --private
hf upload mombasstic/chsa-triage-baseline-test data/splits/dataset_pivot_test_sft.jsonl dataset_pivot_test_sft.jsonl --repo-type dataset

hf jobs uv run \
    --flavor l4x1 \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/interfaces/cli/E1_06_01_evaluer_baseline_gpu.py \
    --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
    --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics
```

Résultats réels, mêmes 278 exemples pour les deux runs :

| | Exact match | F1 moyen (token) | Latence moyenne | Échecs d'inférence |
|---|---|---|---|---|
| CPU (Q4_K_M) | 0,000 | 0,037 | ~21,6 s | 36/278 |
| GPU (bf16) | 0,000 | 0,043 | ~7,3 s | 0/278 |

La baseline GPU est plus rapide, plus fiable (zéro échec) et légèrement
meilleure en F1. Ces deux points zéro servent de référence mesurable pour
juger l'effet du SFT (§2.5).

Le run GPU (`hf jobs uv run`) journalise sur le dépôt HF
`mombasstic/chsa-triage-baseline-metrics` (aussi utilisé par
l'évaluation post-SFT, §2.5) ; pour le parcourir dans un MLflow local
(voir §2.6) :

```bash
uv run python monitoring/importer_mlflow_local.py \
    --repo-id mombasstic/chsa-triage-baseline-metrics \
    --base-sqlite data/processed/mlflow.db
```

<table id="23-sft-train" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h2 style="border-bottom:none; margin:0;">2.3 SFT Train</h2>
</td></tr></table>

Entraînement SFT-LoRA réel (QLoRA 4-bit, rang 16) sur `Qwen/Qwen3-1.7B-Base`,
lancé sur HF Jobs (GPU L4) via `training/E2_04_sft_train.py`, seul script
d'entraînement du projet.

Le pivot anonymisé complet est trop volumineux pour être retéléversé à
chaque lancement : il est monté depuis un dépôt dataset HF dédié, à
créer et publier une seule fois au préalable :

```bash
hf repo create mombasstic/chsa-triage-sft-train-data --repo-type dataset --private
hf upload mombasstic/chsa-triage-sft-train-data data/processed/dataset_pivot_anonymise.jsonl --repo-type dataset
```

Le dépôt modèle qui recevra les poids LoRA du meilleur essai
(`--checkpoint-hf-repo` ci-dessous) est optionnel à créer à l'avance :
le code le crée lui-même (`exist_ok=True`) au premier téléversement s'il
n'existe pas déjà. Commande manuelle équivalente, pour le créer soi-même
au préalable (par exemple pour en fixer la visibilité avant tout run) :

```bash
hf repo create mombasstic/chsa-triage-sft-lora --repo-type model --private
```

```bash
hf jobs uv run \
    --flavor l4x1 \
    --timeout 6h \
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

Le premier entraînement réel a été mené à son terme avec succès sur GPU L4
(~20 min, 342 pas, 3 époques) : verdict de convergence **SAINE**, poids
LoRA publiés durablement sur `mombasstic/chsa-triage-sft-lora`. La courbe
de perte train/validation de ce run a été reconstruite a posteriori depuis
le log brut du job (backend de suivi mal configuré à l'origine, corrigé
depuis) et republiée sur le dépôt de métriques.

**Explication du contenu de la rectte:**
<div style="margin-left: 2cm;">

---

## 1. Quantification — `BitsAndBytesConfig`

*Intervient au chargement du modèle base.*

| Paramètre | Valeur | À quoi ça sert / ce que ça implique | Autres options |
|---|---|---|---|
| `bits` | `4` | Niveau de quantification des poids gelés → divise par 4 la VRAM du modèle base. | 8/16-bit : + précis, + VRAM (**non viable sur GPU commercial**) |
| `type_quantification` | `nf4` | Type 4-bit optimal pour des poids à distribution ~gaussienne. | `fp4` : moins adapté à cette distribution |
| `double_quantification` | `true` | Quantifie aussi les constantes d'échelle → économie VRAM supplémentaire. | `false` : pas ce gain (négligeable en vitesse) |
| `dtype_calcul` | `bfloat16` | Précision de calcul forward/backward, bon range dynamique (GPU Ampere+). | `float16` : risque d'overflow / `float32` : + VRAM |

---

## 2. Adaptateur LoRA — `LoraConfig`

*Intervient à l'injection sur le modèle déjà quantifié.*

| Paramètre | Valeur | À quoi ça sert / ce que ça implique | Autres options |
|---|---|---|---|
| `rang (r)` | `16` | Capacité de l'adaptateur ; équilibre VRAM ↔ expressivité. | `r=8` : + léger / `r=32-64` : + capacité, risque de surapprentissage |
| `alpha` | `32` (2r) | Échelle de ΔW ; règle standard α=2r, stabilise le LR si r change. | `α=r` : échelle plus conservatrice |
| `dropout` | `0.05` | Régularisation sur l'entrée de la matrice A ; anti-surapprentissage (2246 exemples train). | `0.0` : aucune régularisation / `0.1+` : régularisation renforcée |
| `modules_cibles` | `q,k,v,o_proj` | Attention seulement → adaptateur léger. | `"all-linear"` (+MLP) : + expressif, + VRAM/paramètres |

---

## 3. Entraînement — `SFTConfig` / `SFTTrainer`

*Intervient à chaque pas d'optimisation.*

| Paramètre | Valeur | À quoi ça sert / ce que ça implique | Autres options |
|---|---|---|---|
| `taux_apprentissage` | `2e-4` | Valeur typique pour LoRA — a convergé « saine » dès le 1er essai. | `1e-4` : + prudent / `5e-4` : + rapide, risque d'instabilité |
| `nombre_epoques` | `3` | Passes complètes sur le dataset ; compromis apprentissage/mémorisation. | `1` : sous-apprentissage probable / `5+` : risque de surapprentissage |
| `taille_lot` | `4` | Exemples par pas et par GPU, limité par la VRAM disponible. | Valeur + haute si VRAM dispo : + stable, + lent par pas |
| `type_perte` | `nll` | Cross-entropy standard — imposée par une contrainte de dépendance. | `chunked_nll` : réduit le pic VRAM sur `lm_head` — écarté (trl figé en 0.24.0, sans support) |
| `assistant_only_loss` | `false` | Perte calculée sur toute la séquence (prompt + réponse), limitation technique connue (cf. `AGENTS.md`), pas un choix délibéré. | `true` (souhaité à terme) : masquerait (`-100`) les tokens system/user, mais **crash garanti** aujourd'hui, car `ExempleFormate` porte du ChatML déjà rendu en texte, pas des messages structurés par tour, seule forme acceptée par `trl.data_utils.is_conversational` |
| `packing` | `true` | Concatène les exemples courts → GPU utilisé à ~100%. | `false` : padding classique, jusqu'à 40-60% de FLOPs gaspillés |

### ↳ `packing=true` pilote en réalité 3 réglages de `SFTConfig`

> ⚠️ Absents du fichier YAML — tournent actuellement en valeur par défaut de `trl`.

| Paramètre (implicite) | Valeur actuelle | À quoi ça sert / ce que ça implique |
|---|---|---|
| `max_seq_length` | défaut trl | Longueur max par bloc empaqueté. Conditionne directement la VRAM (attention O(N²) ou FlashAttention-2 selon N). |
| `dataset_text_field` | auto (ChatML) | Colonne texte à empaqueter, ignorée car un formatting_func/chat template gère déjà le rendu (notre cas). |
| `dataset_kwargs` | défaut trl | Ex. `append_concat_token` (ajoute l'EOS entre exemples empaquetés) — **à vérifier explicitement** : un défaut erroné ici = risque de contamination inter-exemples. |

### ↳ `--attn-implementation`/`--liger-kernel` : flags CLI, absents du YAML

> ⚠️ Jamais mesurés empiriquement sur un GPU réel (aucun GPU disponible au moment de l'écriture).

| Paramètre | Valeur par défaut | À quoi ça sert / ce que ça implique |
|---|---|---|
| `attn_implementation` | `sdpa` | Intégré à PyTorch, aucune installation/compilation CUDA à part → le moins de risque d'échec sur un environnement cloud pas encore vérifié. Autres optimisations envisagées pour l'entraînement, jamais mesurées faute de GPU : <ul><li><code>sdpa</code> (par défaut, retenu ici)</li><li><code>flash_attention_2</code> : probablement plus rapide, mais nécessite le paquet <code>flash-attn</code> (compilation longue, échoue souvent sans le bon toolchain CUDA)</li><li><code>utiliser_liger_kernel=true</code> : champ réel de <code>trl.SFTConfig</code>, flag indépendant (pas une valeur de <code>attn_implementation</code>)</li><li><code>Unsloth</code> : non branché du tout dans le code, remplacerait tout le chemin de chargement du modèle, décision d'architecture plutôt qu'un simple flag</li></ul> |

---

## 4-5. Grille de secours & Suivi

*4 : si non-convergence · 5 : tout au long du run.*

| Paramètre | Valeur | À quoi ça sert / ce que ça implique | Autres options |
|---|---|---|---|
| `grille` (4) | 3×3 | LR `[1e-4, 2e-4, 5e-4]` × rang `[8, 16, 32]` — Optuna écarté pour ce POC. | Non utilisée : convergence « saine » dès le 1er essai |
| `suivi.backend` (5) | `hf_dataset` | Persiste hors du conteneur HF Jobs (disque non persistant). | `mlflow` : valide en local seulement — a fait perdre une courbe complète |
| `suivi.nom_experience` | `chsa-triage-sft` | Identifiant regroupant les runs dans le dataset de suivi. | — |

</div>



<table id="24-sft-lora-train" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h2 style="border-bottom:none; margin:0;">2.4 SFT-LoRA Train</h2>
</td></tr></table>

Dans ce projet, SFT et LoRA ne sont pas deux étapes distinctes : le seul
script d'entraînement (`training/E2_04_sft_train.py`, §2.3) applique déjà
LoRA nativement (QLoRA 4-bit) à chaque run. Il n'existe pas de variante
« SFT plein » séparée à documenter ici ; la commande et les résultats réels
sont ceux de la §2.3 ci-dessus.


<table id="25-evaluation-post-sft" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h2 style="border-bottom:none; margin:0;">2.5 Evaluation Post-SFT</h2>
</td></tr></table>

Évalue le modèle réellement entraîné (base + poids LoRA) sur le même
sous-ensemble de 278 exemples et les mêmes métriques que les baselines
(§2.2), pour une comparaison directe.

```bash
hf jobs uv run \
    --flavor l4x1 \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/interfaces/cli/E2_05_evaluer_post_sft.py \
    --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
    --depot-lora mombasstic/chsa-triage-sft-lora \
    --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics

# Visualize
watch -n 5 hf jobs list
watch -n 5 hf jobs stats   6aaa59765527934177ee9636
           hf jobs inspect 6aaab9a95527934177eeaac8 --format json | python3 -m json.tool
           hf jobs logs -f 6aaa59765527934177ee9636

```

**Résultat le plus important du projet à ce jour :**

| | Exact match | F1 moyen (token) | Latence moyenne |
|---|---|---|---|
| Baseline CPU (Q4_K_M) | 0,000 | 0,037 | ~21,6 s |
| Baseline GPU (bf16) | 0,000 | 0,043 | ~7,3 s |
| **Post-SFT (bf16+LoRA)** | 0,000 | **0,112** | ~11,6 s |

Le F1 token **quasi triple** par rapport à la meilleure baseline (0,043 ->
0,112), première preuve chiffrée que le SFT a un effet mesurable,
cohérente avec le verdict de convergence SAINE (§2.3). L'exact match reste
à 0,000 sur les trois runs : attendu, la métrique exige une correspondance
caractère-à-caractère avec des réponses de référence en langage libre.

Une fois ces résultats obtenus, régénérer la présentation PowerPoint de
synthèse (baselines, entraînement, évaluation post-SFT) à partir des
mêmes chiffres :

```bash
uv run --with python-pptx python monitoring/generer_presentation_etape2.py
```

<table id="26-suivi-entrainement" style="width:100%;"><tr><td style="background-color:#a6e3ff;">
<h2 style="border-bottom:none; margin:0;">2.6 Suivi d'entraînement</h2>
</td></tr></table>

Pendant un run réel, `training/E2_04_sft_train.py --suivi-hf-repo <repo>`
publie la courbe de perte en direct sur un dataset HF Hub, lue par un
dashboard Streamlit déployé sur HF Space :

```bash
uv run streamlit run monitoring/app_suivi_entrainement.py
```

Pour parcourir l'historique complet de tous les runs dans l'interface
MLflow habituelle, sans monter de serveur MLflow distant, importer
localement les runs du même dépôt HF (idempotent, `--forcer` pour
réimporter ; même fichier SQLite que l'import de §2.2 pour tout
retrouver au même endroit) :

```bash
uv run python monitoring/importer_mlflow_local.py \
    --repo-id mombasstic/chsa-triage-sft-metrics \
    --base-sqlite data/processed/mlflow.db
```

Puis ouvrir l'interface MLflow sur ce même fichier (baseline et SFT
confondus) :

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/processed/mlflow.db --host 0.0.0.0 --port 5000
```

Si la courbe d'un run déjà terminé n'a jamais atteint de backend
durable (cf. §2.3), elle peut être reconstruite a posteriori depuis le
log brut du job :

```bash
uv run python monitoring/reconstruire_courbe_sft_depuis_log.py \
--log /chemin/vers/le/log.log \
--nom-run sft-lora-16092026-reconstruit \
--publier
```

C'est exactement ce qui a permis de récupérer la courbe du premier run
réel (job `6aaab9a95527934177eeaac8`, §2.3), republiée avec succès dans
le même dépôt de métriques.

<table id="3-dpo" style="width:100%;"><tr><td style="background-color:#f5cf47;">
<h1 style="border-bottom:none; margin:0;">3. DPO</h1>
</td></tr></table>

Code complet implémenté et testé : les 13 étapes du guide
(`docs/04_etape3_dpo/03_guide_implementation_pas_a_pas.md`), du domaine
(port `EntraineurPreference`, cas d'usage `ReformulerPreferenceDpoUseCase`
/ `FormaterDatasetChatMLPreferenceUseCase` / `EntrainerDpoUseCase`)
jusqu'à l'adaptateur `TrlDpoEntraineurAdapter` et au point d'entrée
`training/E3_03_dpo_train.py` (recette `recipes/dpo_qwen3_lora.yaml`),
qui continue le checkpoint SFT-LoRA déjà entraîné
(`mombasstic/chsa-triage-sft-lora`, §2.3). Suite complète : 428 tests
passent, 4 ignorés (GPU absent).

Sous-ensemble réduit pour une vérification de lancement (même patron
que l'extraction SFT, §1.5) :

```bash
uv run python interfaces/cli/E1_05_03_extraire_sous_ensemble_dpo.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --exclusions data/processed/identifiants_a_exclure_publication.jsonl \
    --taille 100
```

Publié sur un dépôt dataset HF dédié (même patron que §2.3) :

```bash
hf repo create mombasstic/chsa-triage-dpo-train-data --repo-type dataset --private
hf upload mombasstic/chsa-triage-dpo-train-data data/processed/dataset_chsa_triage_dpo_anonymise_100.jsonl --repo-type dataset
```

Commande de lancement réelle, **jamais encore exécutée sur GPU**
(décision de lancement en attente) :

```bash
hf jobs uv run \
    --flavor l4x1 \
    --timeout 2h \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    -v hf://datasets/mombasstic/chsa-triage-dpo-train-data:/mnt/train-data \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/training/E3_03_dpo_train.py \
    --recette recipes/dpo_qwen3_lora.yaml \
    --dataset /mnt/train-data/dataset_chsa_triage_dpo_anonymise_100.jsonl \
    --suivi-hf-repo mombasstic/chsa-triage-dpo-metrics \
    --checkpoint-hf-repo mombasstic/chsa-triage-dpo-lora
```

Tout a été vérifié sans GPU (même méthode que pour le SFT avant son
premier run réel : installation temporaire de `trl`/`peft` pour
confirmer les signatures) ; deux bugs réels trouvés et corrigés au
passage : le chat template natif de Qwen3 retirait le bloc `<think>`
d'un `chosen` reformulé, et la désérialisation d'un checkpoint DPO
levait une `TypeError`. Schéma conceptuel de la double fonction d'une
seule passe DPO (préférence clinique + format de sortie JSON
contractuel) :
[`docs/diagrams/04_etape3_dpo/activite/dpo_double_fonction_entrainement.png`](docs/diagrams/04_etape3_dpo/activite/dpo_double_fonction_entrainement.png).

Une fois ce run réel effectué, évaluer le modèle aligné (mêmes
métriques/même sous-ensemble que les baselines et le post-SFT, §2.2/
§2.5, quatrième réemploi sans modification d'`EvaluerBaselineZeroShotUseCase`,
même patron exact que le post-SFT) :

```bash
hf jobs uv run \
    --flavor l4x1 \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/interfaces/cli/E3_04_evaluer_post_dpo.py \
    --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
    --depot-lora mombasstic/chsa-triage-dpo-lora \
    --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics
```


<table id="verifications-environnement" style="width:100%;"><tr><td style="background-color:#d9d9d9;">
<h1 style="border-bottom:none; margin:0;">Vérifications d'environnement</h1>
</td></tr></table>

Trois scripts, à lancer avant de démarrer l'étape correspondante :

```bash
uv run python scripts/check_env_local.py
```

```bash
uv run python scripts/check_env_gpu.py
```

```bash
uv run python scripts/check_env_remote_hf.py
```

`check_env_local.py` vérifie l'environnement local (Environnement A,
sans GPU, avant l'étape 1) ; `check_env_gpu.py` vérifie l'environnement
GPU (Environnement B, avant un run `SFTTrainer` coûteux, chat template,
tokens ChatML, chargement 4-bit) ; `check_env_remote_hf.py` vérifie
l'accès Hugging Face (Jobs + Spaces) avant tout lancement distant.


<table id="introduction" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Auteur</h1>
</td></tr></table>

**Rafael Cerezo Martín**

- Email : [rafael.cerezo.martin@icloud.com](mailto:rafael.cerezo.martin@icloud.com)
- GitHub : [@racemartin](https://github.com/racemartin)


<table id="introduction" style="width:100%;"><tr><td style="background-color:#c9f1edff;">
<h1 style="border-bottom:none; margin:0;">Licence</h1>
</td></tr></table>

MIT License, voir [LICENSE](LICENSE) pour les détails.