# CHSA Triage : Agent IA de Triage Médical (POC)

<table id="introduction" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Introduction</h1>
</td></tr></table>

POC d'agent IA de triage médical pour le Centre Hospitalier Saint-Aurélien,
développé sous architecture hexagonale. Ce document est une version
**condensée** : chaque section suit le patron intro -> commande(s) réelle(s)
-> résultat obtenu, sans le détail d'implémentation. Pour tout le detail
technique (hallucinations écartées, hallazgos, limites honnêtes, historique
complet des corrections), voir
[`README_IMPLEMENTACION_V1.md`](README_IMPLEMENTACION_V1.md).

<table id="tableau-récapitulatif-des-scripts" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Tableau récapitulatif des scripts</h1>
</td></tr></table>

Vue d'ensemble de tous les scripts exécutables du dépôt, classés par étape.

| Étape | Script | Rôle |
|---|---|---|
| Étape 1 (Préparation des données) | `interfaces/cli/E1_01_telecharger_corpus.py` | Télécharge un corpus brut depuis Hugging Face Hub et l'exporte en JSONL local (`data/raw/`). |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_02_profiler_corpus.py` | Génère un rapport de profilage ydata-profiling pour un corpus téléchargé. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_03_00_construire_dataset_pivot.py` | Fusionne les corpus sources en un dataset pivot unique, dédupliqué par identifiant déterministe. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_04_00_anonymiser_dataset.py` | Anonymise (Presidio/spaCy) le dataset pivot par vagues incrémentales, en écrivant dans un fichier séparé du pivot original. |
| Étape 1 (Préparation des données) | `scripts/anonymiser_par_lots.sh` | Rappelle `E1_04_00_anonymiser_dataset.py` en boucle par vagues successives jusqu'à couverture complète du pivot. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_04_02_controler_qualite_anonymisation.py` | Compare le pivot original et le fichier anonymisé sur un échantillon stratifié pour détecter de la PII résiduelle. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_04_01_reviser_pii_residuelle.py` | Révision humaine persistée des candidats PII résiduelle et export de la liste d'exclusion pour la publication. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_00_decouper_splits.py` | Répartit (stratifié train/val/test) les exemples du pivot anonymisé, de façon cumulative/incrémentale d'une exécution à l'autre. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_01_verifier_repartition_splits.py` | Affiche la répartition des splits déjà assignés, par strate (type_exemple, source). |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_02_extraire_sous_ensemble_sft.py` / `E1_05_03_extraire_sous_ensemble_dpo.py` | Extraient les sous-ensembles SFT/DPO publiables (échantillons filtrés, hors périmètre de ce document condensé). |
| Étape 1bis (Baseline zero-shot) | `interfaces/cli/E1_06_00_evaluer_baseline.py` | Évalue la baseline zero-shot en local (CPU), via un `llama-server` sur un GGUF quantifié Q4_K_M. |
| Étape 1bis (Baseline zero-shot) | `interfaces/cli/E1_06_01_evaluer_baseline_gpu.py` | Même évaluation baseline zero-shot mais en pleine précision (bf16, transformers) sur un job HF Jobs GPU. |
| Étape 2 (SFT + LoRA) | `interfaces/cli/E2_00_formater_dataset_chatml.py` | Point d'entrée autonome pour le rendu ChatML d'un split (mode didactique inclus). |
| Étape 2 (SFT + LoRA) | `training/E2_04_sft_train.py` | Point d'entrée d'entraînement SFT-LoRA réel, exécuté via HF Jobs (GPU requis). |
| Étape 2 (SFT + LoRA) | `interfaces/cli/E2_05_evaluer_post_sft.py` | Évaluation post-SFT : mêmes métriques/mêmes exemples que les baselines, mais via le modèle base+LoRA réellement entraîné. |
| Étape 2 (SFT + LoRA) | `monitoring/app_suivi_entrainement.py`, `monitoring/importer_mlflow_local.py`, `monitoring/reconstruire_courbe_sft_depuis_log.py` | Suivi en direct (Streamlit/HF Space) et historisation locale (MLflow) de la courbe d'entraînement. |
| Étape 3 (DPO) | *(aucun script à ce jour)* | Étape non implémentée dans le code. |
| Infrastructure | `scripts/check_env_local.py`, `scripts/check_env_gpu.py`, `scripts/check_env_remote_hf.py` | Vérifient que l'environnement (local, GPU, HF) est prêt avant chaque étape. |

Détail de chaque commande dans les sections ci-dessous, et détail exhaustif de tous les scripts (y compris ceux hors périmètre de ce document) dans `README_IMPLEMENTACION_V1.md`.

<table id="table-des-matières" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
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
- [3. DPO](#3-dpo)
- [Structure (architecture hexagonale)](#structure-architecture-hexagonale)

<table id="1-préparation-de-données" style="width:100%;"><tr><td style="background-color:#38a169;">
<h1 style="border-bottom:none; margin:0;">1. Préparation de données</h1>
</td></tr></table>

Les 6 fichiers sources sont fusionnés dans un dataset pivot unique, puis
anonymisés, contrôlés et répartis en splits train/val/test. Prérequis
d'installation (`uv sync`, modèles spaCy) : voir
`docs/01_environnement/00_guide_installation_environnement.md`.

<table id="11-télécharger-corpus" style="width:100%;"><tr><td style="background-color:#38a169;">
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

<table id="12-profiler-corpus" style="width:100%;"><tr><td style="background-color:#38a169;">
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

<table id="13-dataset-pivot" style="width:100%;"><tr><td style="background-color:#38a169;">
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

<table id="14-anonymisation" style="width:100%;"><tr><td style="background-color:#38a169;">
<h2 style="border-bottom:none; margin:0;">1.4 Anonymisation</h2>
</td></tr></table>

Anonymise le pivot (Presidio + spaCy) par vagues incrémentales, en écrivant
dans un fichier **séparé** : le pivot original n'est jamais modifié.

```bash
uv run python interfaces/cli/E1_04_00_anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --sortie data/processed/dataset_pivot_anonymise.jsonl --strategie replace --limite 5000

# Pour enchaîner les vagues jusqu'à couverture complète du pivot :
scripts/anonymiser_par_lots.sh data/processed/dataset_pivot.jsonl data/processed/dataset_pivot_anonymise.jsonl replace 5000
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

<table id="15-découpage-en-splits" style="width:100%;"><tr><td style="background-color:#38a169;">
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

<table id="2-sft--lora" style="width:100%;"><tr><td style="background-color:#02c39a;">
<h1 style="border-bottom:none; margin:0;">2. SFT + LoRA</h1>
</td></tr></table>

Architecture hexagonale, deux baselines zero-shot mesurées avant tout
entraînement, l'entraînement SFT-LoRA réel, et son évaluation.

<table id="21-architecture" style="width:100%;"><tr><td style="background-color:#02c39a;">
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

<table id="22-baseline-evaluation" style="width:100%;"><tr><td style="background-color:#02c39a;">
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

# GPU (transformers, bf16, HF Jobs) :
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

<table id="23-sft-train" style="width:100%;"><tr><td style="background-color:#02c39a;">
<h2 style="border-bottom:none; margin:0;">2.3 SFT Train</h2>
</td></tr></table>

Entraînement SFT-LoRA réel (QLoRA 4-bit, rang 16) sur `Qwen/Qwen3-1.7B-Base`,
lancé sur HF Jobs (GPU L4) via `training/E2_04_sft_train.py`, seul script
d'entraînement du projet.

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

<table id="24-sft-lora-train" style="width:100%;"><tr><td style="background-color:#02c39a;">
<h2 style="border-bottom:none; margin:0;">2.4 SFT-LoRA Train</h2>
</td></tr></table>

Dans ce projet, SFT et LoRA ne sont pas deux étapes distinctes : le seul
script d'entraînement (`training/E2_04_sft_train.py`, §2.3) applique déjà
LoRA nativement (QLoRA 4-bit) à chaque run. Il n'existe pas de variante
« SFT plein » séparée à documenter ici ; la commande et les résultats réels
sont ceux de la §2.3 ci-dessus.

<table id="25-evaluation-post-sft" style="width:100%;"><tr><td style="background-color:#02c39a;">
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

<table id="3-dpo" style="width:100%;"><tr><td style="background-color:#7aeae7;">
<h1 style="border-bottom:none; margin:0;">3. DPO</h1>
</td></tr></table>

**Non implémenté à ce jour.** Aucune commande ni étape n'existe encore dans
le code pour cette phase ; voir `docs/04_etape3_dpo/` (à venir).

<table id="structure-architecture-hexagonale" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Structure (architecture hexagonale)</h1>
</td></tr></table>

```
src/chsa_triage/
├── domain/            # entités + ports, zéro dépendance externe
├── application/       # cas d'usage : orchestrent les ports
└── infrastructure/    # adaptateurs concrets (JSONL, HF, Presidio, ydata-profiling, ...)
interfaces/            # adaptateurs primaires : cli/ (Étape 1-2), api/ et web/ (Étape 4)
training/              # scripts exécutés via HF Jobs (SFT, DPO) : Étapes 2-3
docker/                # Dockerfiles + docker-compose (frontend/backend) : Étape 4
```

Détail complet : `docs/01_environnement/01_architecture_hexagonale.md`.
