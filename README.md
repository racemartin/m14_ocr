# CHSA Triage : Agent IA de Triage Médical (POC)

<table id="introduction" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Introduction</h1>
</td></tr></table>

POC d'agent IA de triage médical pour le Centre Hospitalier
Saint-Aurélien, développé sous architecture hexagonale (ports &
adaptateurs). La documentation vit dans `docs/`, structurée en
sous-dossiers numérotés selon les étapes du projet :

| Dossier | Contenu |
|---|---|
| `docs/00_cadrage/` | Objectifs séquencés + décisions justifiées, cahier des charges |
| `docs/01_environnement/` | Installation (`uv` local, HF payant distant) + architecture hexagonale |
| `docs/02_etape1_donnees/` | Documentation spécifique à la préparation des données (à venir) |
| `docs/03_etape2_sft/` | Planification de l'entraînement SFT + LoRA (concepts, installation Environnement B, cas d'usage proposés, guide d'implémentation), écrite avant tout code d'entraînement |
| `docs/04_etape3_dpo/` | Documentation alignement DPO (à venir) |
| `docs/05_etape4_deploiement/` | Documentation déploiement/évaluation (à venir) |
| `docs/diagrams/` | Diagrammes UML (activité, séquence, paquets, déploiement) par étape : `.puml`+`.png`+`.svg`+`.pdf`, voir `docs/diagrams/README.md` |

Chaque document se termine par un renvoi vers le suivant, pour lire
la documentation dans l'ordre du projet en partant de
`docs/00_cadrage/00_objectifs_du_projet.md`.

<table id="tableau-récapitulatif-des-scripts" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Tableau récapitulatif des scripts</h1>
</td></tr></table>

Vue d'ensemble de tous les scripts exécutables du dépôt (`interfaces/cli/`,
`scripts/`, `training/`, `monitoring/`, plus quelques scripts isolés
trouvés ailleurs), classés par étape. Le détail de chaque commande
(options, exemples d'usage) reste dans les sections correspondantes
ci-dessous ; voir aussi la [Table des matières](#table-des-matières) juste
après pour naviguer par section plutôt que par script.

| Étape | Script | Rôle |
|---|---|---|
| Étape 1 (Préparation des données) | `interfaces/cli/E1_01_telecharger_corpus.py` | Télécharge un corpus brut depuis Hugging Face Hub et l'exporte en JSONL local (`data/raw/`). |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_02_profiler_corpus.py` | Génère un rapport de profilage ydata-profiling pour un corpus téléchargé. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_03_00_construire_dataset_pivot.py` | Fusionne les corpus sources en un dataset pivot unique, dédupliqué par identifiant déterministe. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_03_01_mappers_corpus.py` | Fonctions de mapping « enregistrement brut -> ExemplePivot », une par corpus source (module support importé par `E1_03_00_...`, pas un point d'entrée CLI à part entière). |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_04_00_anonymiser_dataset.py` | Anonymise (Presidio/spaCy) le dataset pivot par vagues incrémentales, en écrivant dans un fichier séparé du pivot original. |
| Étape 1 (Préparation des données) | `scripts/anonymiser_par_lots.sh` | Rappelle `E1_04_00_anonymiser_dataset.py` en boucle par vagues successives jusqu'à couverture complète du pivot (script shell, protection anti-boucle-infinie). |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_04_02_controler_qualite_anonymisation.py` | Compare le pivot original et le fichier anonymisé sur un échantillon stratifié pour détecter de la PII résiduelle. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_04_01_reviser_pii_residuelle.py` | Révision humaine persistée des candidats PII résiduelle (accepter/rejeter) et export de la liste d'exclusion pour la publication. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_00_decouper_splits.py` | Répartit (stratifié train/val/test) les exemples du pivot anonymisé, de façon cumulative/incrémentale d'une exécution à l'autre. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_01_verifier_repartition_splits.py` | Affiche la répartition des splits déjà assignés, par strate (type_exemple, source). |
| Étape 1 (Préparation des données) | `scripts/decouper_splits.py` | Matérialise le champ `split` déjà assigné en 3 fichiers JSONL séparés (train/val/test_baseline). N'est référencé nulle part dans le README documenté ; semble un utilitaire redondant/antérieur à `E1_05_00`/`E1_05_01`, à vérifier avant de le supprimer. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_02_extraire_sous_ensemble_sft.py` | Extrait, filtre (exclusions PII) et tronque à une taille cible le sous-ensemble SFT destiné à la publication Hugging Face. |
| Étape 1 (Préparation des données) | `interfaces/cli/E1_05_03_extraire_sous_ensemble_dpo.py` | Même algorithme que `E1_05_02_...` mais pour le sous-ensemble DPO. |
| Étape 1bis (Baseline zero-shot) | `interfaces/cli/E1_06_00_evaluer_baseline.py` | Évalue la baseline zero-shot en local (CPU), via un `llama-server` déjà lancé sur un GGUF quantifié Q4_K_M. |
| Étape 1bis (Baseline zero-shot) | `interfaces/cli/E1_06_01_evaluer_baseline_gpu.py` | Même évaluation baseline zero-shot mais en pleine précision (bf16, transformers) sur un job HF Jobs GPU, pour isoler l'effet de la quantification. |
| Étape 2 (SFT + LoRA) | `scripts/check_env_gpu.py` | Vérifie que l'environnement GPU (Environnement B) est prêt avant un run SFTTrainer/DPOTrainer coûteux (chat template, tokens ChatML, `assistant_only_loss`, chargement 4-bit). |
| Étape 2 (SFT + LoRA) | `interfaces/cli/E2_00_formater_dataset_chatml.py` | Point d'entrée autonome pour `FormaterDatasetChatMLUseCase` (rendu ChatML d'un split), avec un mode didactique optionnel (`--exemples N`, cape à 2) qui logue par `LogTool` l'`ExemplePivot` brut puis son rendu ChatML final. |
| Étape 2 (SFT + LoRA) | `training/E2_04_sft_train.py` | Point d'entrée d'entraînement SFT-LoRA réel (orchestre les 4 cas d'usage `E2_00`-`E2_03`), exécuté via HF Jobs (GPU requis). |
| Étape 2 (SFT + LoRA) | `monitoring/app_suivi_entrainement.py` | Dashboard Streamlit (déployé sur HF Spaces) de visualisation en direct de la courbe d'apprentissage d'un run SFT-LoRA. |
| Étape 2 (SFT + LoRA) | `monitoring/hf_dataset_runs.py` | Frontière réseau partagée (`HfApi.list_repo_files`/`hf_hub_download`) vers le dataset HF de métriques, réutilisée par le dashboard et l'importateur (module support, pas un script autonome). |
| Étape 2 (SFT + LoRA) | `monitoring/logica_suivi_entrainement.py` | Logique pure de parsing/pivot/convergence du dashboard, testable sans Streamlit ni réseau (module support, pas un script autonome). |
| Étape 2 (SFT + LoRA) | `monitoring/importer_mlflow_local.py` | Importe les runs du dataset HF de métriques dans un MLflow local (SQLite) pour parcourir l'historique complet sans serveur MLflow distant. |
| Étape 2 (SFT + LoRA) | `monitoring/reconstruire_courbe_sft_depuis_log.py` | Reconstruit a posteriori la courbe de métriques d'un run SFT déjà terminé à partir de son log brut, quand aucun backend `SuiviExperimentation` durable n'avait été branché. |
| Étape 2 (SFT + LoRA) | `interfaces/cli/E2_05_evaluer_post_sft.py` | Évaluation post-SFT : mêmes métriques/mêmes 278 exemples que les baselines zero-shot (§2.2), mais via le modèle base+LoRA réellement entraîné (`TransformersLoraInferenceAdapter`), pour comparaison directe. |
| Étape 2 (SFT + LoRA) | `data/demos/move_chsa-triage-sft-metrics-fake_jston_to_mlflow_db.py` | Chemins et nom de fichier codés en dur, sans `argparse` ni bloc `__main__` : brouillon/démo ponctuel important des métriques JSON factices dans MLflow ; aucun usage documenté au-delà de son propre nom de fichier, en pratique remplacé par `monitoring/importer_mlflow_local.py`. |
| Étape 3 (DPO) | *(aucun script à ce jour)* | Étape non implémentée dans le code ; voir `docs/04_etape3_dpo/` (à venir). |
| Infrastructure / vérifications transversales | `scripts/check_env_local.py` | Vérifie que l'environnement local (Environnement A, sans GPU) est prêt pour la préparation des données. |
| Infrastructure / vérifications transversales | `scripts/check_env_remote_hf.py` | Vérifie l'accès à l'environnement distant Hugging Face (Jobs + Spaces) avant de lancer un entraînement coûteux. |

Fichiers `.py` trouvés dans le même balayage mais volontairement absents
du tableau ci-dessus : les `__init__.py` de `interfaces/cli/`,
`monitoring/` et `training/` (marqueurs de package vides, aucun rôle
exécutable).

<table id="table-des-matières" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Table des matières</h1>
</td></tr></table>

- [Tableau récapitulatif des scripts](#tableau-récapitulatif-des-scripts)
- [1. Préparation des données (Étape 1)](#1-préparation-des-données-étape-1)
  - [1.1 Démarrage rapide (installation)](#11-démarrage-rapide-installation)
  - [1.2 Téléchargement (Hugging Face Hub -> data/raw/)](#12-telechargement-hugging-face-hub---dataraw)
  - [1.3 Profilage individuel](#13-profilage-individuel-un-rapport-ydata-profiling-par-corpus)
  - [1.4 Construction du dataset pivot](#14-construction-du-dataset-pivot-meme---sortie--fusionne-les-corpus-par-identifiant)
  - [1.5 Anonymisation](#15-anonymisation-incrementalereprenable-cf---limite-ecrit-dans-un-fichier-separe)
    - [Contrôle qualité de l'anonymisation](#controle-qualite-de-lanonymisation-comparaison-de-fichiers)
    - [Révision humaine persistée des candidats de PII résiduelle (NF2)](#revision-humaine-persistee-des-candidats-de-pii-residuelle-nf2)
  - [1.6 Découpage en splits (train / val / test, stratifié)](#16-decoupage-en-splits-train--val--test-stratifie)
    - [Vérification de la répartition des splits par strate](#verification-de-la-repartition-des-splits-par-strate)
  - [1.7 Extraction du sous-ensemble SFT publiable](#17-extraction-du-sous-ensemble-sft-5000-exemples-pour-publication-hugging-face)
  - [1.8 Extraction du sous-ensemble DPO publiable](#18-extraction-du-sous-ensemble-dpo-pour-publication-hugging-face)
- [2. SFT + LoRA (Étape 2)](#2-sft--lora-étape-2)
  - [2.1 Architecture](#21-architecture)
    - [Mode didactique : inspecter le rendu ChatML d'un split](#mode-didactique--inspecter-le-rendu-chatml-dun-split)
  - [2.2 Évaluation baseline zero-shot (Étape 1bis)](#22-évaluation-baseline-zero-shot-étape-1bis)
    - [Baseline CPU, via llama.cpp](#évaluation-baseline-zero-shot-étape-1bis-avant-sftdpo)
    - [Baseline GPU, via transformers sur HF Jobs](#évaluation-baseline-zero-shot-gpu-étape-1bis-sur-hf-jobs)
  - [2.3 Entraînement SFT-LoRA](#23-entrainement-sft-lora)
    - [Historique complet dans un MLflow local (importateur)](#historique-complet-dans-un-mlflow-local-importateur)
  - [2.4 Évaluation post-SFT](#24-évaluation-post-sft)
- [3. DPO (Étape 3)](#3-dpo-étape-3)
- [Structure (architecture hexagonale)](#structure-architecture-hexagonale)
- [État d'avancement](#état-davancement)

<table id="1-préparation-des-données-étape-1" style="width:100%;"><tr><td style="background-color:#38a169;">
<h1 style="border-bottom:none; margin:0;">1. Préparation des données (Étape 1)</h1>
</td></tr></table>

Les 6 fichiers sources sont fusionnés dans le même dataset pivot,
puis anonymisés, contrôlés, répartis en splits et enfin extraits en
sous-ensembles publiables (SFT et DPO). Détail méthodologique complet
dans `docs/02_etape1_donnees/`.

<table id="11-démarrage-rapide-installation" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.1 Démarrage rapide (installation)</h2>
</td></tr></table>

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
```

<table style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">1.1 Telecharger Corpus</h1>
</td></tr></table>


<table id="12-telechargement-hugging-face-hub---dataraw" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.2 Telechargement (Hugging Face Hub -> data/raw/)</h2>
</td></tr></table>

```bash
# NB : la configuration "oeq" de MediQAl n'a qu'un split "test" (pas de "train") ; --split explicite requis
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration oeq --split test --sortie data/raw/mediqal_oeq.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqu --sortie data/raw/mediqal_mcqu.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqm --sortie data/raw/mediqal_mcqm.jsonl

uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub nthngdy/frenchmedmcqa      --sortie data/raw/frenchmedmcqa.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub keivalya/MedQuad-MedicalQnADataset  --sortie data/raw/medquad.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub TsinghuaC3I/UltraMedical-Preference --sortie data/raw/ultramedical_preference.jsonl
```

<table id="13-profilage-individuel-un-rapport-ydata-profiling-par-corpus" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.3 Profilage individuel (un rapport ydata-profiling par corpus)</h2>
</td></tr></table>

```bash
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_oeq.jsonl   --nom MediQAl-oeq
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_mcqu.jsonl  --nom MediQAl-mcqu
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_mcqm.jsonl  --nom MediQAl-mcqm

uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/frenchmedmcqa.jsonl           --nom FrenchMedMCQA
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/medquad.jsonl                 --nom MedQuAD
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/ultramedical_preference.jsonl --nom UltraMedicalPreference
```

<table id="14-construction-du-dataset-pivot-meme---sortie--fusionne-les-corpus-par-identifiant" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.4 Construction du dataset pivot (meme --sortie : fusionne les corpus par identifiant)</h2>
</td></tr></table>

```bash
# NB : MediQAl a 3 configurations, avec 2 schemas differents : le
# mapper (donc la valeur --corpus) depend du schema, pas seulement de
# la source Hub. "oeq" (question/answer) -> mapper_mediqal ;
# "mcqu"/"mcqm" (QCM, answer_a..answer_e + correct_answers) ->
# mapper_mediqal_qcm. Voir interfaces/cli/E1_03_01_mappers_corpus.py.
# NB : --taille-bloc requis sur ultramedical_preference.jsonl (966 Mo,
# 109353 enregistrements) ; sans lecture par blocs, OOM reel confirme
# sur 5.8 Go de RAM disponibles. Voir --taille-bloc dans
# interfaces/cli/E1_03_00_construire_dataset_pivot.py.
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/mediqal_oeq.jsonl --corpus mediqal_oeq --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/mediqal_mcqu.jsonl --corpus mediqal_mcqu --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/mediqal_mcqm.jsonl --corpus mediqal_mcqm --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/frenchmedmcqa.jsonl --corpus frenchmedmcqa --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/medquad.jsonl --corpus medquad --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/E1_03_00_construire_dataset_pivot.py --source data/raw/ultramedical_preference.jsonl --corpus ultramedical_preference --sortie data/processed/dataset_pivot.jsonl --taille-bloc 5000
```

Resultat reel (08/09/2026, pivot regenere avec identifiants **deterministes**,
remplace l'execution du 07/09/2026 dont les identifiants etaient aleatoires) :

| Source | Corpus | Enregistrements bruts | Exemples pivot ecrits | Doublons exacts ecartes |
| --- | --- | --- | --- | --- |
| `data/raw/mediqal_oeq.jsonl` | `mediqal_oeq` | 4 969 | 4 969 | 0 |
| `data/raw/mediqal_mcqu.jsonl` | `mediqal_mcqu` | 10 113 | 10 113 | 0 |
| `data/raw/mediqal_mcqm.jsonl` | `mediqal_mcqm` | 5 767 | 5 767 | 0 |
| `data/raw/frenchmedmcqa.jsonl` | `frenchmedmcqa` | 595 | 594 | 1 |
| `data/raw/medquad.jsonl` | `medquad` | 16 407 | 16 359 | 48 |
| `data/raw/ultramedical_preference.jsonl` | `ultramedical_preference` | 109 353 | 97 081 | 12 272 |
| **TOTAL** | N/A | **147 204** | **134 883** | **12 321** |

L'identifiant deterministe (`ExemplePivot.nouvel_identifiant`, hash
stable derive d'une cle naturelle propre a chaque source : champ
`id` brut pour MediQAl/FrenchMedMCQA, hash Question+Answer pour
MedQuAD, `prompt_id`+`label_type`+chosen+rejected pour
UltraMedical-Preference) a mis au jour de VRAIS doublons exacts dans
les donnees brutes, invisibles auparavant (chaque appel generait un
UUID aleatoire different, donc jamais de collision). `--corpus`
determine desormais un **espace de noms** plus fin que la simple
`source` du pivot : verifie sur les fichiers reels, `mediqal_oeq.jsonl`
et `mediqal_mcqu.jsonl` partagent 1 492 valeurs de champ `id`
identiques bien que decrivant des registres differents.
`ConstruireDatasetPivotUseCase` dedoublonne reellement : un seul exemplaire par identifiant atterrit dans le
pivot, les doublons ecartes sont archives (jamais perdus) dans
`data/processed/doublons_supprimes.jsonl`. Detail complet (methodologie,
cle naturelle par source, investigation legere sur la cause probable
des doublons UltraMedical-Preference) dans
`docs/02_etape1_donnees/00_couverture_exigences_officielles.md`.

<table id="15-anonymisation-incrementalereprenable-cf---limite-ecrit-dans-un-fichier-separe" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.5 Anonymisation (incrementale/reprenable, cf. --limite), ecrit dans un fichier SEPARE</h2>
</td></tr></table>

```bash
# IMPORTANT (08/09/2026, design source/sortie separes) : --dataset (le pivot original) n'est JAMAIS modifie ;
# le resultat est ecrit dans --sortie, un fichier separe (defaut
# data/processed/dataset_pivot_anonymise.jsonl). "Deja anonymise" se
# determine par la presence de l'identifiant dans --sortie, pas par un
# champ mute sur le pivot source. Anonymisation complete du dataset
# (134883 exemples) mesuree a ~19h (cout NLP Presidio/spaCy) ;
# --limite (defaut 5000, l'objectif chiffre de la mission) anonymise
# un echantillon stratifie par (type_exemple, source) parmi les
# exemples du pivot pas encore presents dans --sortie ; le reste
# attend un appel ulterieur avec un N plus grand ou "full".
uv run python interfaces/cli/E1_04_00_anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --sortie data/processed/dataset_pivot_anonymise.jsonl --strategie replace --limite 5000
```

```bash
# Pour enchainer les vagues successives sans relancer la commande a
# la main a chaque fois : scripts/anonymiser_par_lots.sh rappelle
# E1_04_00_anonymiser_dataset.py en boucle jusqu'a couverture complete du
# pivot, sans jamais retraiter les exemples deja presents dans
# --sortie, avec protection anti-boucle-infinie si une vague
# n'avance plus. Les 4 arguments positionnels sont optionnels
# (valeurs par defaut identiques a celles de la commande ci-dessus).
# Logs par iteration dans logs/anonymisation/. Detail complet dans
# l'entete du script lui-meme.
scripts/anonymiser_par_lots.sh data/processed/dataset_pivot.jsonl data/processed/dataset_pivot_anonymise.jsonl replace 5000
```

Chaque execution genere/fusionne automatiquement un **rapport RGPD
cumule** (JSON + Markdown, `data/processed/rapport_anonymisation_rgpd.{json,md}`)
avec, par source et au total : registres traites (cumule sur toutes
les executions) et proportion reelle sur le total du dataset pivot
(compte a chaque execution, jamais code en dur), taux d'enregistrements
avec >=1 entite detectee, entites par type, et la liste tracable des
executions ayant contribue (horodatage, strategie, limite, graine).
Voir `application/use_cases/E1_04_03_rapport_anonymisation.py`.

`PresidioAnonymiseur` (08/09/2026, ameliorations avancees, voir
`docs/02_etape1_donnees/01_rapport_rgpd.md` §7 pour la justification
complete) ajoute un recognizer NIR francais (numero de securite
sociale, valide par cle de controle modulo 97, pas juste un motif "15
chiffres") et normalise les mentions explicites d'age
("age de X ans"/"X-year-old"/"aged X") en tranche clinique
(pediatrique/adolescent/adulte/personne agee) AVANT que Presidio ne
les analyse, pour que `DATE_TIME` ne les elimine pas comme une date de
naissance ; l'operateur `DATE_TIME` distingue en plus une date
calendaire absolue (masquee) d'une duree relative ("il y a 3
semaines", "depuis 2 mois"), laissee intacte, signal clinique pas
identifiant.

<table id="controle-qualite-de-lanonymisation-comparaison-de-fichiers" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Controle qualite de l'anonymisation (comparaison de fichiers)</h3>
</td></tr></table>

```bash
# Compare le pivot ORIGINAL (jamais modifie) au fichier ANONYMISE,
# croises par identifiant, sur un echantillon stratifie ; peut se
# relancer a tout moment, y compris retroactivement sur une vague
# anonymisee il y a longtemps (le pivot original existe toujours).
uv run python interfaces/cli/E1_04_02_controler_qualite_anonymisation.py --dataset data/processed/dataset_pivot.jsonl --anonymise data/processed/dataset_pivot_anonymise.jsonl --taille-echantillon 200
```

Detecte les candidats de PII residuelle sur le texte anonymise (regex
sans modele : emails, telephones, URLs, dates, bigrammes capitalises)
et les tranche avec une seconde opinion spaCy (memes
modeles que `PresidioAnonymiseur`, `fr_core_news_md`/`en_core_web_sm`,
jamais de LLM) : confirme, ecarte comme faux positif du regex, ou
marque explicitement "pendant_revision_humaine" si ni le regex ni
spaCy ne tranchent. Detecte aussi les candidats de sur-masquage
(termes originaux masques que spaCy ne reconnait pas comme entite
nommee) par diff texte original/anonymise. Tire en plus un stratum
DEDIE et independant (`--taille-echantillon-sans-entite`, 40 par
defaut) parmi les exemples ou Presidio n'a RIEN detecte du tout
(texte_original == texte_anonymise), ce qui distingue explicitement "rien
detecte" de "quelque chose detecte" pour la relecture manuelle,
plutot que de presumer ces cas corrects par defaut. Ecrit son propre
rapport (`data/processed/rapport_controle_qualite_anonymisation.{json,md}`),
avec des exemples reels inspectables par source et les compteurs
d'entites par type repris du rapport RGPD cumule (§1.5 ci-dessus, pas
recalcules). Voir `application/use_cases/E1_04_02_controler_qualite_anonymisation.py`.

**Muestreo INCREMENTAL** (09/09/2026, meme
patron que `--limite` ci-dessus) : `--registre-echantillons` (defaut
`data/processed/controle_qualite_identifiants_echantillonnes.jsonl`)
exclut du tirage les identifiants deja echantillonnes lors d'une
execution precedente, sur les deux strates ; chaque execution ne
compare que des identifiants NOUVEAUX. C'est ce qui rend les
decisions humaines de la section suivante cumulables entre
executions, au lieu d'un echantillon jete a chaque fois.

<table id="revision-humaine-persistee-des-candidats-de-pii-residuelle-nf2" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Revision humaine persistee des candidats de PII residuelle (NF2)</h3>
</td></tr></table>

```bash
# Revue interactive : recalcule TOUS les candidats "pendant_revision_humaine"
# deja echantillonnes (les deux strates, toutes executions confondues),
# exclut ceux ayant deja une decision, et persiste chaque reponse
# IMMEDIATEMENT (fermer le terminal a mi-parcours ne perd rien).
uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py verify --dataset data/processed/dataset_pivot.jsonl --anonymise data/processed/dataset_pivot_anonymise.jsonl

# Corriger une decision deja prise (sans repasser par toute la liste) :
uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py modify --identifiant chsa-xxxxxxxx
```

Ferme l'ecart identifie sur l'exigence NF2 du cahier des charges
("anonymisation validee **manuellement**") : avant ce script, le
verdict `pendant_revision_humaine` du controle qualite (ci-dessus) etait un
cul-de-sac ; aucune decision de personne n'etait jamais persistee.
Chaque decision (`accepte` = confirme non-PII, `rejete` = PII reelle
confirmee) est identifiee par une cle stable
`(source_liste, identifiant, champ, type_motif, debut, fin)` et
persistee dans `data/processed/decisions_revision_humaine.jsonl`. Le
rapport de `E1_04_02_controler_qualite_anonymisation.py` (ci-dessus) relit ce fichier
pour annoter chaque candidat en attente de son statut de decision
(accepte/rejete/encore en attente). Voir
`application/use_cases/E1_04_01_reviser_pii_residuelle.py` et
`docs/02_etape1_donnees/01_rapport_rgpd.md` §7.5 pour la methodologie
complete.

<table id="16-decoupage-en-splits-train--val--test-stratifie" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.6 Decoupage en splits (train / val / test, stratifie)</h2>
</td></tr></table>

```bash
# E1_05_00_decouper_splits.py opere sur le fichier ANONYMISE (dataset_pivot_anonymise.jsonl),
# PAS sur le pivot original ; le relancer apres chaque nouvelle vague
# d'anonymisation.
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl

# --n : taille CIBLE cumulee (pas la taille de cette seule execution).
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl --n 5000
```

**Croissance stable, jamais de reordonnancement (10/09/2026,
CHANGEMENT DE COMPORTEMENT reel).** Un exemple qui a
deja un `split` (execution anterieure) n'est JAMAIS reassigne, quel
que soit le `--n` demande ensuite ; agrandir le dataset ne fait QUE
completer ce qui manque, il ne recalcule plus jamais le decoupage
entier. Avant ce changement, relancer avec un `--n` different (ou sans
`--n`) pouvait deplacer un exemple deja vu de `train` vers `test` (ou
l'inverse), une fuite silencieuse d'exemples d'entrainement dans le
jeu de test ; ce que le cahier des charges interdit explicitement
("le jeu de test ne doit jamais etre reutilise en entrainement").

Exemple concret :

```bash
# Premiere execution : 5000 exemples, aucun split existant ; les 5000
# sont repartis stratifie (type_exemple, source) selon les proportions
# habituelles.
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl --n 5000

# Deuxieme execution, plus tard : --n 10000 preleve N - (deja assignes)
# = 10000 - 5000 = 5000 NOUVEAUX exemples (echantillon stratifie parmi
# ceux qui n'ont pas encore de split) et leur assigne un split. Les
# 5000 PREMIERS exemples GARDENT exactement le split qui leur a ete
# assigne lors de la premiere execution ; aucun n'est deplace entre
# train/val/test.
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl --n 10000
```

Si `--n N` est demande mais `N` est <= au nombre d'exemples deja
assignes, il n'y a rien de nouveau a faire : **reduire un decoupage
deja fait n'est pas supporte** (le jeu ne peut que grandir), un
avertissement est trace via LogTool (pas une erreur). Si `--n` est
omis, TOUS les exemples anonymises qui n'ont pas encore de split en
recoivent un (mode "completer ce qui manque", avant ce changement,
le mode sans `--n` recalculait le decoupage de tout le dataset anonymise
depuis zero). Le decompte affiche en sortie est le TOTAL cumule
(deja assignes + nouveaux de cette execution), distinct du nombre de
nouveaux exemples repartis lors de CETTE execution (affiche
separement).

**Exclusion des candidats PII confirmes ou en attente de revision
humaine (10/09/2026, etendue le 11/09/2026 aux candidats confirmes).**
Par precaution, un exemple portant au moins un candidat de PII
residuelle CONFIRME (fuite non ambigue) ou SANS decision humaine
persistee (§1.5 ci-dessus, `E1_04_01_reviser_pii_residuelle.py`) est exclu du decoupage de
cette execution : il reste sans `split` jusqu'a ce qu'une decision
soit prise (ou, pour un candidat confirme, indefiniment tant que le
texte n'est pas corrige). `E1_05_00_decouper_splits.py` accepte donc desormais les memes
adaptateurs que `E1_04_01_reviser_pii_residuelle.py verify` pour recalculer cet
ensemble : `--original` (defaut `data/processed/dataset_pivot.jsonl`),
`--registre-echantillons` (defaut
`data/processed/controle_qualite_identifiants_echantillonnes.jsonl`),
`--decisions` (defaut `data/processed/decisions_revision_humaine.jsonl`)
et `--jeton-masque`. Le nombre d'exemples exclus pour cette raison
lors de cette execution est affiche en sortie.

<table id="verification-de-la-repartition-des-splits-par-strate" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Verification de la repartition des splits par strate</h3>
</td></tr></table>

```bash
# E1_05_00_decouper_splits.py n'affiche que le total global (train/val/test).
# E1_05_01_verifier_repartition_splits.py relit le fichier anonymise deja
# reparti et affiche, pour chaque strate (type_exemple, source), le
# decompte ET le pourcentage par split, pour verifier visuellement
# que l'echantillonnage stratifie reste representatif DANS CHAQUE
# split (ex. une petite source comme FrenchMedMCQA doit rester
# ~80/10/10 comme les grosses sources, pas disparaitre de train ou de
# test). Fonctionne sans changement avec la croissance stable de
# E1_05_00_decouper_splits.py ci-dessus : il relit simplement les splits deja
# presents dans le fichier, quelle que soit la sequence d'executions
# --n qui les a produits.
uv run python interfaces/cli/E1_05_01_verifier_repartition_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl
```

`E1_04_00_anonymiser_dataset.py` affiche une barre de progression `tqdm` pendant le traitement (peut durer plusieurs dizaines de minutes sur un gros dataset).

<table id="17-extraction-du-sous-ensemble-sft-5000-exemples-pour-publication-hugging-face" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.7 Extraction du sous-ensemble SFT (5000 exemples, pour publication Hugging Face)</h2>
</td></tr></table>

Deux etapes : exporter les identifiants a exclure (candidats de PII
residuelle confirmes ou en attente), puis soustraire ce fichier du
pivot anonymise deja reparti en splits.

**Correction (12/09/2026) :** `ExtraireSousEnsembleSftUseCase` ne
filtrait auparavant que sur `split != null`, sans filtrer
`type_exemple` ; `DecouperSplitsUseCase` (§1.6) reparti A DESSEIN les
deux types (SFT et DPO) dans le meme fichier anonymise, donc des
exemples DPO (ex. UltraMedical-Preference, `chosen`/`rejected`
renseignes, `completion` vide) se retrouvaient dans le sous-ensemble
cense n'etre que du SFT (constate : un fichier de 5000 lignes etait
72% DPO). Le filtre `type_exemple == TypeExemple.SFT` est maintenant
explicite, avant tout comptage/exclusion/recoupe (idem pour
`FormaterDatasetChatMLUseCase`, §2 Etape 2). Consequence directe : la
taille reellement ecrite depend du nombre d'exemples PUREMENT SFT deja
repartis en split, pas du pool total (SFT+DPO) comme avant ; `--taille`
reste un PLAFOND, jamais un nombre garanti (cf. `manque` ci-dessous) :
si le pivot anonymise n'a encore couvert qu'une fraction du corpus
complet (vagues incrementales de `E1_04_00_anonymiser_dataset.py`,
§1.5), le pool SFT disponible peut etre plus petit que `--taille` et le
fichier ecrit contiendra alors HONNETEMENT moins de lignes (renommer
le fichier de sortie pour que son nom reflete ce compte reel, cf.
`chemin_sortie_defaut`). Sur le pivot COMPLETEMENT anonymise et
reparti (134 883 exemples, 37 802 SFT/97 081 DPO, §1.5/§1.6), le pool SFT
disponible large permet d'atteindre les 5000 demandes sans y toucher.

```bash
# 1. Export non interactif (lecture seule) des identifiants a exclure :
#    au moins un candidat VERDICT_CONFIRME (fuite non ambigue, jamais
#    soumise a decision humaine), ou au moins un candidat
#    VERDICT_REVISION_HUMAINE sans decision DECISION_ACCEPTE persistee
#    (candidat encore ouvert, ou explicitement rejete). Meme replay
#    deterministe que `E1_04_01_reviser_pii_residuelle.py verify` (§1.5), etendu
#    aux candidats CONFIRME.
uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py exporter \
    --dataset data/processed/dataset_pivot.jsonl \
    --anonymise data/processed/dataset_pivot_anonymise.jsonl

# 2. Filtre `type_exemple == SFT` ET `split != null` MOINS les
#    identifiants exclus a l'etape 1, puis RECOUPE a exactement
#    --taille (echantillonnage stratifie type_exemple/source) si le
#    resultat filtre en contient plus. Pure soustraction + recoupage :
#    ne rajoute jamais d'exemples pour compenser un manque.
uv run python interfaces/cli/E1_05_02_extraire_sous_ensemble_sft.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --exclusions data/processed/identifiants_a_exclure_publication.jsonl \
    --taille 5000
```

Exemple reel (12/09/2026, apres correction du filtre `type_exemple` ET
apres avoir complete l'anonymisation/le decoupage sur les 134 883
exemples du pivot, §1.5/§1.6 ; controle qualite §1.5 elargi a 1350
identifiants echantillonnes cumules, dont 444 de sources SFT, avant
cette extraction) :

```
Exemples avec split (avant exclusion) : 37772
Exclus (PII confirmee ou en attente de revision humaine) : 10
Disponibles apres exclusion : 37762
Recoupes par echantillonnage stratifie : -32762 (surplus au-dela de 5000)
Ecrits dans data/processed/dataset_chsa_triage_sft_anonymise_5000.jsonl : 5000 exemple(s).

Repartition du sous-ensemble ecrit par strate (type_exemple, source) :
Strate                                          Total           train             val            test
sft/FrenchMedMCQA                                  79      61 (77.2%)      10 (12.7%)       8 (10.1%)
sft/MedQuAD                                      2161    1702 (78.8%)     216 (10.0%)     243 (11.2%)
sft/MediQAl                                      2760    2220 (80.4%)     257 ( 9.3%)     283 (10.3%)
Taille cible atteinte (5000 == 5000).
```

Note bien : aucune strate `dpo/UltraMedical-Preference` dans le
tableau ci-dessus, contrairement a avant la correction (confirme aussi
par `grep -c '"type_exemple": "dpo"' data/processed/dataset_chsa_triage_sft_anonymise_5000.jsonl`
retournant 0).

Si le resultat, apres exclusion, contient MOINS d'exemples que
`--taille` (cas rencontre plus tot dans cette meme investigation,
lorsque seule une fraction du pivot avait ete anonymisee),
`E1_05_02_extraire_sous_ensemble_sft.py` ne tente jamais de completer
automatiquement (ce n'est qu'un filtre/une soustraction, pas un
nouveau muestreo) : il affiche clairement combien d'exemples restent
et combien manquent, et suggere d'elargir l'anonymisation
(`E1_04_00_anonymiser_dataset.py --limite <N>`, §1.5) puis le decoupage
des splits (`E1_05_00_decouper_splits.py --n <N>` ou sans `--n` pour
tout completer, §1.6) avant de relancer l'extraction ; elargir seulement
`--n` sans avoir d'abord anonymise davantage n'ajoute pas de nouveaux
exemples SFT si le pool SFT anonymise lui-meme est deja epuise.

Le pivot anonymise complet (`dataset_pivot_anonymise.jsonl`) et le
fichier d'exclusions restent locaux sous `data/processed/` (deja
exclus de Git, voir le commentaire correspondant dans `.gitignore`) ;
le fichier filtre de 5000 exemples est reproductible a tout moment a
partir du pivot complet via les deux commandes ci-dessus.

**Publication sur Hugging Face Hub.** Depot cible : `mombasstic/dataset_chsa_triage_sft_anonymise_5000`
(le suffixe numerique doit toujours correspondre au compte REEL du
fichier publie, pas seulement a `--taille` demande : verifier que les
deux correspondent avant publication), prive par defaut, meme s'il ne
contient que les 5000 exemples filtres et non le pivot complet : il
s'agit toujours de texte medical anonymise, et la visibilite privee
minimise l'exposition publique tant que la couverture du controle
qualite (§1.5) reste partielle (voir la limite de couverture
ci-dessous). Le depot pourra etre rendu public plus tard depuis
l'interface web de Hugging Face si souhaite.

Nécessite une session Hugging Face ouverte au prealable avec un jeton
de role "write" (`hf auth login`, voir
`docs/01_environnement/00_guide_installation_environnement.md` §1.4).

```bash
# 1. Creer le depot (prive, type dataset) :
hf repo create mombasstic/dataset_chsa_triage_sft_anonymise_5000 --repo-type dataset --private

# 2. Publier le fichier de 5000 exemples :
hf upload mombasstic/dataset_chsa_triage_sft_anonymise_5000 data/processed/dataset_chsa_triage_sft_anonymise_5000.jsonl --repo-type dataset

# 3. Verifier la publication ET le nombre de lignes cote Hub (pas
#    seulement en local) : le plus fiable est de retelecharger le
#    fichier depuis le Hub puis de compter les lignes, sans dependre
#    du "dataset viewer" de HF qui peut prendre du temps a traiter un
#    fichier tout juste publie, surtout sur un depot prive.
hf download mombasstic/dataset_chsa_triage_sft_anonymise_5000 dataset_chsa_triage_sft_anonymise_5000.jsonl --repo-type dataset --local-dir /tmp/verificacion_hf
wc -l /tmp/verificacion_hf/dataset_chsa_triage_sft_anonymise_5000.jsonl   # doit correspondre aux lignes du fichier local
```

**Limite de couverture connue :** l'exclusion ci-dessus ne peut porter
que sur ce qui a deja ete AUDITE. Seul un sous-ensemble du pivot a ete
echantillonne par le controle qualite (§1.5, 1350 identifiants cumules
sur 134 883 a ce jour, ~1%) ; un exemple jamais echantillonne peut
donc encore contenir une PII residuelle non detectee, meme apres
l'etape 1. Augmenter la couverture de
`E1_04_02_controler_qualite_anonymisation.py` (§1.5) avant publication reduit ce
risque, mais ne l'elimine pas completement sans audit exhaustif.

<table id="18-extraction-du-sous-ensemble-dpo-pour-publication-hugging-face" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">1.8 Extraction du sous-ensemble DPO (pour publication Hugging Face)</h2>
</td></tr></table>

Meme patron exact que §1.7 (`ExtraireSousEnsembleSftUseCase`), en
filtrant `type_exemple == TypeExemple.DPO` au lieu de SFT : le cahier
des charges (`docs/00_cadrage/01_cahier_des_charges.md` §7, Livrable
1) exige "SFT ~5000 paires + DPO" dans le dataset publie, et
`E1_05_03_extraire_sous_ensemble_dpo.py` produit ce second sous-ensemble
a partir du meme pivot anonymise et du meme fichier d'exclusions PII
(reutilise tel quel, sans le regenerer). Meme fichier d'exclusions
(§1.7 etape 1) : un identifiant y figure independamment du type
d'exemple, l'export ne le regenere donc pas.

```bash
# Reutilise le meme fichier d'exclusions que §1.7 etape 1 (pas besoin de
# le regenerer si deja fait) :
uv run python interfaces/cli/E1_04_01_reviser_pii_residuelle.py exporter \
    --dataset data/processed/dataset_pivot.jsonl \
    --anonymise data/processed/dataset_pivot_anonymise.jsonl

# Filtre `type_exemple == DPO` ET `split != null` MOINS les
# identifiants exclus, puis RECOUPE a exactement --taille
# (echantillonnage stratifie type_exemple/source) si le resultat
# filtre en contient plus.
uv run python interfaces/cli/E1_05_03_extraire_sous_ensemble_dpo.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --exclusions data/processed/identifiants_a_exclure_publication.jsonl \
    --taille 5000
```

Sortie attendue (meme forme que §1.7, `Strate` prefixee `dpo/` au lieu
de `sft/` puisque `calculer_repartition_par_strate` groupe par
`type_exemple`) :

```
Exemples avec split (avant exclusion) : <N>
Exclus (PII confirmee ou en attente de revision humaine) : <n>
Disponibles apres exclusion : <N-n>
Ecrits dans data/processed/dataset_chsa_triage_dpo_anonymise_<taille>.jsonl : <taille> exemple(s).

Repartition du sous-ensemble ecrit par strate (type_exemple, source) :
Strate                                          Total           train             val            test
dpo/UltraMedical-Preference                    <...>           <...>            <...>            <...>
```

Sur le pivot completement anonymise et reparti (134 883 exemples,
37 802 SFT / 97 081 DPO au 12/09/2026, §1.5/§1.6), le pool DPO disponible
(uniquement `UltraMedical-Preference` a ce jour) est largement
suffisant pour atteindre les 5000 demandes sans y toucher ; comme en
§1.7, `--taille` reste un PLAFOND jamais garanti si le pivot anonymise
ne couvre encore qu'une fraction du corpus complet.

**Publication sur Hugging Face Hub.** Meme depot que §1.7 recommande une
publication SEPAREE (deux fichiers distincts dans le meme depot
prive), pour que le nom du fichier continue de refleter honnetement
son contenu et son compte reel :

```bash
# Le depot existe deja depuis §1.7 (creer une seule fois) :
hf repo create mombasstic/dataset_chsa_triage_sft_anonymise_5000 --repo-type dataset --private

# Publier le fichier DPO a cote du fichier SFT deja publie :
hf upload mombasstic/dataset_chsa_triage_sft_anonymise_5000 data/processed/dataset_chsa_triage_dpo_anonymise_5000.jsonl --repo-type dataset

# Publier/mettre a jour la dataset card (README.md du depot HF) :
hf upload mombasstic/dataset_chsa_triage_sft_anonymise_5000 docs/02_etape1_donnees/dataset_card_dpo_hf.md README.md --repo-type dataset

# Verifier le compte de lignes cote Hub (meme methode que §1.7, ne pas
# se fier au dataset viewer qui peut prendre du temps sur un depot prive) :
hf download mombasstic/dataset_chsa_triage_sft_anonymise_5000 dataset_chsa_triage_dpo_anonymise_5000.jsonl --repo-type dataset --local-dir /tmp/verificacion_hf_dpo
wc -l /tmp/verificacion_hf_dpo/dataset_chsa_triage_dpo_anonymise_5000.jsonl
```

Meme limite de couverture qu'en §1.7 : l'exclusion ne porte que sur ce
qui a deja ete audite par le controle qualite (§1.5).

<table id="2-sft--lora-étape-2" style="width:100%;"><tr><td style="background-color:#02c39a;">
<h1 style="border-bottom:none; margin:0;">2. SFT + LoRA (Étape 2)</h1>
</td></tr></table>

Architecture retenue, deux mesures de baseline zero-shot (CPU
quantifié et GPU pleine précision, pour ne jamais mélanger l'effet de
la quantification avec l'effet réel de l'entraînement), l'entraînement
SFT-LoRA réel avec suivi en direct, et l'évaluation post-entraînement
(pas encore implémentée).

<table id="21-architecture" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">2.1 Architecture</h2>
</td></tr></table>

L'architecture hexagonale (domain/application/infrastructure,
interfaces/, training/) est commune à tout le projet ; voir
[Structure (architecture hexagonale)](#structure-architecture-hexagonale)
plus bas pour le schéma des dossiers, et le détail complet dans
`docs/01_environnement/01_architecture_hexagonale.md`. La partie
spécifique à l'Étape 2 (SFT + LoRA), domaine/ports/application/
adaptateurs, est documentée dans `docs/03_etape2_sft/` (concepts,
installation Environnement B, cas d'usage `E2_NN_uc_*`, guide
d'implémentation pas à pas) ; voir aussi les notes correspondantes
dans `AGENTS.md`.

<table id="mode-didactique--inspecter-le-rendu-chatml-dun-split" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Mode didactique : inspecter le rendu ChatML d'un split</h3>
</td></tr></table>

`E2_00_uc_formater_dataset_chatml.py`/`FormaterDatasetChatMLUseCase`
n'a longtemps été invoqué que depuis l'intérieur de
`training/E2_04_sft_train.py` (étape 1 du pipeline d'entraînement,
GPU requis). `interfaces/cli/E2_00_formater_dataset_chatml.py` en
expose un point d'entrée autonome et bon marché, avec les mêmes
adaptateurs réels (`JsonlDatasetRepository` + `ChatMLFormateurAdapter`) :

```bash
uv run python interfaces/cli/E2_00_formater_dataset_chatml.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --split train \
    --exemples 2
```

`--exemples N` (capé à 2) logue par `LogTool`, en plus de l'écriture
normale vers `--dataset-formate`, l'`ExemplePivot` brut (tours
prompt/completion) puis le texte ChatML final rendu par
`ChatMLFormateurAdapter.formater()` pour les N premiers exemples du
split, sans aucune logique de rendu réimplémentée. Purement additif et
lecture seule/console : l'écriture vers `--dataset-formate` est
strictement identique avec ou sans ce flag, et `training/
E2_04_sft_train.py` n'est pas modifié (il continue d'invoquer le cas
d'usage directement).

<table id="22-évaluation-baseline-zero-shot-étape-1bis" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">2.2 Évaluation baseline zero-shot (Étape 1bis)</h2>
</td></tr></table>

Deux mesures indépendantes de la performance de `Qwen/Qwen3-1.7B-Base`
SANS entrainement, avant SFT/DPO, pour disposer d'un point de
comparaison mesurable (cahier des charges §9 : "l'accuracy...
depasse la baseline zero-shot de facon mesurable") : CPU quantifie
Q4_K_M via llama.cpp (Environnement A, local), et GPU pleine precision
bf16 via transformers (Environnement B, HF Jobs), pour ne jamais
melanger l'effet de la quantification avec l'effet reel de
l'entrainement.

<table id="évaluation-baseline-zero-shot-étape-1bis-avant-sftdpo" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Évaluation baseline zero-shot (Étape 1bis, avant SFT/DPO)</h3>
</td></tr></table>

Mesure la performance de `Qwen/Qwen3-1.7B-Base` SANS entrainement sur
le split `test` DEJA EXISTANT du pivot anonymise (champ `split`, §1.6),
pour disposer d'un point de comparaison mesurable avant SFT/DPO
(cahier des charges §9 : "l'accuracy... depasse la baseline zero-shot
de facon mesurable"). Environnement A (local, sans GPU) : l'inference
passe par un serveur `llama-server` (llama.cpp) DEJA LANCE en local,
servant un GGUF quantifie du modele.

**Prerequis : demarrer un `llama-server` local.** Aucune conversion
GGUF a faire soi-meme : une quantification publique exacte de
`Qwen/Qwen3-1.7B-Base` existe deja sur le Hub, `mradermacher/Qwen3-1.7B-Base-GGUF`.

```bash
# 1. Telecharger le GGUF (Q4_K_M, ~1.1 Go) :
uv run python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('mradermacher/Qwen3-1.7B-Base-GGUF', 'Qwen3-1.7B-Base.Q4_K_M.gguf', local_dir='.'))
"

# 2. Recuperer le binaire llama-server (build CPU officiel, Ubuntu x64) :
curl -sL -o llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b10985/llama-b10985-bin-ubuntu-x64.tar.gz
tar -xzf llama.tar.gz

# 3. Demarrer le serveur (contexte reduit : suffisant pour des invites
#    zero-shot courtes, adapte a une machine avec peu de RAM) :
LD_LIBRARY_PATH=./llama-b10985 ./llama-b10985/llama-server \
    -m Qwen3-1.7B-Base.Q4_K_M.gguf --port 8080 -c 1024 -t 2 --no-webui --host 0.0.0.0 --parallel 1

curl http://127.0.0.1:8080/health
{"status":"ok"}.

```

**`--parallel 1` est necessaire, pas cosmetique (confirme le 16/09/2026) :**
sans ce flag, `llama-server` choisit `--parallel`/`-np` automatiquement
(defaut `-1` = auto) et a reparti le `-c 1024` ci-dessus entre 4 slots
paralleles sur cette machine, soit ~256 tokens de contexte REELS par
requete (verifie avec `curl http://127.0.0.1:8080/props`, champ
`total_slots`), pas 1024. Une invite medicale reelle (question +
gabarit de chat rendu) depasse facilement 256 tokens, ce qui a produit
un `500 Internal Server Error` sur `/completion` apres plusieurs
minutes de generation. `EvaluerBaselineZeroShotUseCase.executer()`
(`src/chsa_triage/application/use_cases/E1_06_00_evaluer_baseline_zero_shot.py`)
genere ses requetes SEQUENTIELLEMENT, jamais en parallele : les slots
supplementaires n'apportent donc aucun benefice ici, ils ne font que
voler du contexte a l'unique requete reellement utilisee. Avec
`--parallel 1`, `/props` confirme `total_slots: 1` et les 1024 tokens
de contexte demandes sont bien tous disponibles pour cette requete.

Puis, dans un second terminal :

```bash
uv run python interfaces/cli/E1_06_00_evaluer_baseline.py \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --url-serveur http://127.0.0.1:8080
```

Le run est journalise dans MLflow (`SuiviExperimentation`, meme
mecanisme que `training/E2_04_sft_train.py`, pas un nouveau systeme de
tracking) sous le nom `baseline-zero-shot`, dans
`sqlite:///data/processed/mlflow.db` par defaut (`--suivi-uri` pour un
autre chemin). Execute LOCALEMENT (pas sur un job HF), le run atterrit
directement dans le MLflow local : PAS besoin de
`monitoring/importer_mlflow_local.py` pour le retrouver (ce script
rapatrie des runs distants publies sur un depot HF, ce qui n'est pas
le cas ici).

**Limite honnete mesuree (15/09/2026) :** sur une machine a RAM
limitee (~6 Go, cf. AGENTS.md), charger le GGUF de 1,1 Go dans
`llama-server` peut prendre plusieurs minutes (pression memoire), et
la generation CPU peut descendre a ~0,2-0,5 tokens/seconde selon la
charge de la machine. Une evaluation complete du split test (plusieurs
milliers d'exemples) est donc lente sur une machine sans GPU dedie ;
prevoir le temps necessaire ou reduire `--n-predict`.

**Metriques calculees** (`application/metriques_evaluation_baseline.py`,
fonctions pures, testables sans reseau/GPU) : exact match et F1 token
sur la reponse complete (s'appliquent tels quels au dataset REEL
actuel), et exactitude de classification du niveau ESI (extrait le
champ `niveau` d'un JSON `{niveau, categorie, ressources_estimees}`,
cf. cahier des charges F3). Point de vigilance honnete : les
`completion` reels du pivot actuel (MediQAl, FrenchMedMCQA, MedQuAD)
sont des reponses en langage naturel, pas ce format JSON
(`docs/03_etape2_sft/00_introduction_concepts.md`), donc cette
derniere metrique aura tres peu (voire aucune) paire comparable sur le
dataset d'aujourd'hui ; le CLI l'indique clairement plutot que
d'afficher un pourcentage trompeur calcule sur une poignee de
coincidences.

**Resultat reel (16/09/2026), run `baseline-zero-shot` dans le MLflow
local (`sqlite:///data/processed/mlflow.db`), meme sous-ensemble de 278
exemples `split=test`/`type_exemple=sft` que la baseline GPU ci-dessous,
apres les fixes `--parallel 1` et de resilience aux echecs isoles
(tous les deux documentes plus haut) :** exact match 0.000, F1 moyen
(token) 0.037 (valeur exacte 0.03662720909500452), exactitude de
classification du niveau ESI non calculable (0/242 paires comparables,
cf. le point de vigilance sur le format JSON ci-dessus), latence
moyenne 21571,9 ms (~21,6 s) par generation, et 36/278 exemples
ignores pour echec d'inference isole (`nombre_echecs_inference`) ; les
242 exemples restants sont ceux effectivement compares pour ces
metriques.

<table id="évaluation-baseline-zero-shot-gpu-étape-1bis-sur-hf-jobs" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Évaluation baseline zero-shot GPU (Étape 1bis, sur HF Jobs)</h3>
</td></tr></table>

Meme mesure que ci-dessus (`Qwen/Qwen3-1.7B-Base` SANS entrainement, meme
sous-ensemble de 278 exemples `split=test`/`type_exemple=sft`), mais
sur GPU reel via `transformers` en pleine precision bf16
(`TransformersInferenceAdapter`), PAS sur GGUF quantifie Q4_K_M
(ci-dessus, `LlamaCppInferenceAdapter`) : pour comparer plus tard au modele
SFT/DPO (probablement lui aussi evalue via `transformers`/GPU) SANS
melanger l'effet de la quantification avec l'effet reel de
l'entrainement. `EvaluerBaselineZeroShotUseCase` (application) est
REUTILISE SANS MODIFICATION entre les deux baselines ci-dessus : seul l'adaptateur
d'inference change.

Le dataset a evaluer vit sur un depot dataset HF PRIVE deja publie,
`mombasstic/chsa-triage-baseline-test` (fichier
`dataset_pivot_test_sft.jsonl`, 278 exemples, EXACTEMENT le meme
sous-ensemble que la baseline CPU ci-dessus, pour que les deux resultats soient comparables) ;
il vit aussi, versionne, dans
`data/splits/dataset_pivot_test_sft.jsonl` de ce depot.
`interfaces/cli/E1_06_01_evaluer_baseline_gpu.py` le telecharge lui-meme
via `huggingface_hub.hf_hub_download` (pas de clone du depot de donnees
sur le job distant, qui n'a de toute facon pas acces a `data/processed/`,
gitignore).

```bash
uv run python interfaces/cli/E1_06_01_evaluer_baseline_gpu.py \
    --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
    --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics
```

**Verification reelle effectuee (16/09/2026), et sa limite honnete :**
le telechargement du dataset depuis `mombasstic/chsa-triage-baseline-test`
(278 lignes, confirme), le rendu ChatML du VRAI tokenizer
`Qwen/Qwen3-1.7B-Base` et le refus explicite, fail-fast, de
`TransformersInferenceAdapter` en l'absence de GPU CUDA (`RuntimeError`,
jamais un repli silencieux vers le CPU) ont ete verifies pour de vrai en
executant la commande ci-dessus dans CET environnement de developpement
(credentials HF reelles disponibles ici, contrairement a §2.3 ci-dessous).
Comme prevu (aucun GPU
disponible ici), les 278 exemples echouent tous a l'inference avec le
meme `RuntimeError`, et le cas d'usage leve `ValueError` ("rien a
agreger") : ceci confirme le CABLAGE de bout en bout, PAS les vrais
chiffres de baseline GPU, qui restent a produire sur un job HF Jobs
GPU reel (jamais lance ici : couterait une session GPU payante pour un
resultat deja connu par construction, la commande ne fait qu'echouer
plus vite sans GPU).

**Commande `hf jobs uv run` (syntaxe verifiee via `hf jobs uv run --help`
dans cet environnement, la commande elle-meme JAMAIS EXECUTEE : lancer
un vrai job GPU est une action payante/irreversible, hors perimetre
d'une verification de syntaxe) :**

```bash
hf jobs uv run \
    --flavor l4x1 \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/interfaces/cli/E1_06_01_evaluer_baseline_gpu.py \
    --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
    --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics
```

**Erreur reelle en production, session GPU payante (16/09/2026), cause
identifiee et corrigee ici :** cette commande a ete reellement lancee sur HF
Jobs (GPU reel, payant) avec `chsa-triage[local]` (au lieu de `[remote]`
comme ci-dessus) et a echoue sur TOUS les exemples avec la meme erreur :
`Using a device_map, tp_plan, torch.device context manager or setting
torch.set_default_device(device) requires accelerate. You can install it
with pip install accelerate`. Cause reelle, verifiee dans `pyproject.toml` :
l'extra `local` (Environnement A, sans GPU, Etape 1, cf. plus haut) ne
declare ni `accelerate` ni de version de `torch` pour GPU ; l'extra `remote`
(Environnement B, GPU, Etapes 2 et 3, cf. plus haut) declare deja
`torch>=2.3`, `transformers>=4.44` et `accelerate>=0.33`, exactement ce dont
`TransformersInferenceAdapter` a besoin pour `device_map="cuda"`. Aucune
dependance ne manquait dans `pyproject.toml` : seul le groupe d'extras passe
a `--with` etait errone, corrige ci-dessus en `chsa-triage[remote]`.

**Limite honnete de cette commande, documentee plutot que masquee :**
`hf jobs uv run SCRIPT` execute un fichier UNIQUE (local ou URL), avec
ses dependances declarees en metadonnees PEP 723 (`# /// script`) OU via
`--with` ; il ne clone PAS le depot GitHub pour rendre `src/chsa_triage/`,
`interfaces/`, `src/tools/` disponibles au script telecharge par URL brute.
`E1_06_01_evaluer_baseline_gpu.py` importe `chsa_triage.*` et
`tools.rafael.log_tool` : sans le paquet installe, l'import echoue des la
premiere ligne. `--with "chsa-triage[remote] @ git+https://...@main"`
installe le paquet DEPUIS GitHub (le depot expose deja `[build-system]`
hatchling + `[tool.hatch.build.targets.wheel] packages = [...]` incluant
`interfaces`, `src/tools`, `training`, `monitoring`, cf. `pyproject.toml`)
avant d'executer le script telecharge : c'est l'alternative REELLEMENT
verifiee ici, PAS inventee -
`uv run --with "chsa-triage @ git+https://github.com/racemartin/m14_ocr.git@main" --no-project python -c "import chsa_triage"`
a ete execute pour de vrai dans cet environnement (reseau GitHub reel,
resolution `uv` reelle) et a reussi. Ceci ne verifie que la
RESOLUTION/INSTALLATION du paquet, pas l'execution complete du script sur
l'infrastructure HF Jobs elle-meme (jamais lancee, cf. ci-dessus).
`--secrets HF_TOKEN` transmet le token HF necessaire au telechargement du
depot dataset PRIVE `--dataset-hf-repo` depuis le job distant.

**Resultat reel (16/09/2026), job `baseline-zero-shot-gpu` relance avec
succes sur GPU reel (`--flavor l4x1`) :** apres les deux corrections
reelles issues du premier essai ci-dessus (extra `[local]` -> `[remote]`
de `pyproject.toml`, cf. le paragraphe dedie plus haut et AGENTS.md ; et
l'usage correct de `--secrets HF_TOKEN` pour transmettre le token, plutot
que de coller sa valeur en clair dans la commande), le job a ete relance
pour de vrai sur une GPU L4 et a complete avec succes, journalise dans le
meme MLflow local que la baseline CPU. Sur les memes 278 exemples
`split=test`/`type_exemple=sft` : 278 evalues et 278 comparables (ZERO
echec d'inference), exact match 0.000, F1 moyen (token) 0.043 (valeur
exacte 0.04338116533022368), exactitude de classification du niveau ESI
non calculable (0/278, meme raison que la baseline CPU ci-dessus),
latence moyenne 7295,4 ms (~7,3 s) par generation.

**Ce que cette baseline etablit, et ce qu'elle ne cherche pas a etablir :**
un F1/exact match bas ici est attendu, pas un echec : `Qwen/Qwen3-1.7B-Base`
n'a encore rien vu du format de triage cible, et l'objectif de cette etape
n'est pas d'obtenir de bons scores mais de fixer le point zero mesurable
contre lequel la progression reelle du SFT (§2.3) puis du DPO (§3) sera
jugee (cahier des charges §9). Le resultat tout aussi important de ce run
est independant des scores de qualite : le pipeline d'evaluation
lui-meme (chargement du modele, generation, scoring) est valide de bout
en bout sur les 278 exemples, sans aucun echec technique. La latence
mesuree ici (~7,3 s/generation) est une reference "brute", sur HF Jobs
sans moteur d'inference optimise ; vLLM, deja impose par le cahier des
charges (§6, "Moteur d'inference impose : vLLM (PagedAttention)") et
planifie pour l'Etape 4/Semaine 4 (§8, "Deploiement vLLM/Docker/CI-CD"),
est le levier prevu pour reduire cette latence plus tard, pas quelque
chose a optimiser des cette etape.

**Comparaison des deux baselines :** la baseline GPU/bf16 est nettement
plus rapide que la baseline CPU/Q4_K_M (~7,3 s contre ~21,6 s par
generation), ne subit aucun echec d'inference (0/278 contre 36/278 en
CPU), et obtient un F1 token legerement superieur (0.043 contre 0.037),
vraisemblablement du a la precision complete plutot qu'a la
quantification Q4_K_M.

**Prochaine etape explicite :** l'entrainement SFT-LoRA (§2.3 ci-dessous),
avec pour objectif direct que le modele commence a produire la structure
JSON de triage attendue et depasse ce plancher mesure ici en F1/exact
match, de facon mesurable (cahier des charges §9).

<table id="23-entrainement-sft-lora" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">2.3 Entrainement SFT-LoRA</h2>
</td></tr></table>

Pendant un run SFT-LoRA reel (Environnement B, GPU sur HF Jobs),
`training/E2_04_sft_train.py --suivi-hf-repo <repo>` (recette
`suivi.backend: hf_dataset`) publie la courbe de perte train/validation
en continu vers un dataset Hugging Face Hub, lu EN VIVO par un
dashboard Streamlit deploye a part sur un Space
(`monitoring/app_suivi_entrainement.py`). Les etapes 1 a 4 ci-dessous
(creation des depots metriques/Space, publication du dashboard) restent
**NON EXECUTEES/NON VERIFIEES en reseau reel** (documentees, syntaxe
verifiee uniquement). Le LANCEMENT lui-meme (etape 5, `hf jobs uv run`)
a en revanche ete verifie pour de vrai le 16/09/2026, cf.
"Verification reelle du lancement HF Jobs" plus bas : credentials HF
reelles disponibles dans cet environnement de verification, contrairement
a la redaction initiale de cette section.

```bash
# 1. Creer le depot dataset qui recevra les metriques (prive) :
hf repo create mombasstic/chsa-triage-sft-metrics --repo-type dataset --private

# 2. Creer le Space Streamlit qui les visualise (prive) :
hf repo create mombasstic/chsa-triage-sft-monitor --repo-type space --space_sdk streamlit --private

# 3. Publier le code du dashboard sur le Space : au minimum
#    monitoring/, src/chsa_triage/domain/ et
#    src/chsa_triage/application/verdict_convergence.py (logique de
#    verdict reutilisee telle quelle, zero dependance externe).
#    monitoring/requirements.txt (streamlit, huggingface_hub UNIQUEMENT :
#    PAS le pyproject.toml complet du projet, qui installerait
#    torch/trl/peft inutilement) doit atterrir a la RACINE du Space
#    (HF Spaces l'exige), de meme que le README.md du Space (frontmatter
#    YAML `sdk: streamlit`, `app_file: monitoring/app_suivi_entrainement.py`) :
#    monitoring/README_space.md est PRET A COPIER tel quel, aucune
#    redaction manuelle necessaire.
hf upload mombasstic/chsa-triage-sft-monitor monitoring/ monitoring/ --repo-type space
hf upload mombasstic/chsa-triage-sft-monitor src/chsa_triage/domain/ src/chsa_triage/domain/ --repo-type space
hf upload mombasstic/chsa-triage-sft-monitor src/chsa_triage/application/verdict_convergence.py src/chsa_triage/application/verdict_convergence.py --repo-type space
hf upload mombasstic/chsa-triage-sft-monitor monitoring/requirements.txt requirements.txt --repo-type space
hf upload mombasstic/chsa-triage-sft-monitor monitoring/README_space.md README.md --repo-type space

# 4. (Optionnel mais recommande avant tout run GPU reel) Verifier le
#    dashboard de bout en bout avec des metriques FACTICES :
#    data/demos/chsa-triage-sft-metrics-fake.json (31 etapes, perte
#    train/validation + norme gradient, verdict SAINE confirme contre
#    monitoring/logica_suivi_entrainement.py au moment de sa creation).
#    Le nom de destination (demo_datos_ficticios/metriques.jsonl) est ce
#    qui fait apparaitre "demo_datos_ficticios" comme run selectionnable
#    dans le dashboard ; le nom du fichier local n'a pas besoin de
#    correspondre. Supprimer ce run factice du depot avant le premier
#    run reel pour ne pas encombrer le selecteur.
hf upload mombasstic/chsa-triage-sft-metrics \
    data/demos/chsa-triage-sft-metrics-fake.json \
    demo_datos_ficticios/metriques.jsonl \
    --repo-type dataset

# **Test de connectivité (quelques centimes, confirme que la carte bancaire fonctionne avec HF Jobs)**
hf jobs uv run --flavor t4-small python -c "import torch; print(torch.cuda.get_device_name())"

# *Cela prend quelques secondes, ne coûte presque rien et valide l'ensemble du circuit de facturation avant de lancer un job plus important.*

# 5. Lancer l'entrainement en pointant vers le depot de metriques
#    cree a l'etape 1, pour que le Space ait des vraies donnees a lire.
#    Commande LOCALE (Environnement B avec GPU deja provisionne) :
uv run python training/E2_04_sft_train.py \
    --recette recipes/sft_qwen3_lora.yaml \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --suivi-hf-repo mombasstic/chsa-triage-sft-metrics \
    --checkpoint-hf-repo mombasstic/chsa-triage-sft-lora \
    --assistant-only-loss false
```

**Commande `hf jobs uv run` (VERIFIEE REELLEMENT de bout en bout le
16/09/2026, cf. le detail complet plus bas ; jamais lancee avec
`--flavor` GPU reel/payant, uniquement avec `--flavor cpu-basic`, ce
qui suffit a verifier tout le cablage jusqu'au guard
`assistant_only_loss`) :**

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

<table id="verification-reelle-du-lancement-hf-jobs-16092026" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Verification reelle du lancement HF Jobs (16/09/2026)</h3>
</td></tr></table>

Meme demarche que la baseline GPU (§2.2) : le lancement REEL du
training COMPLET (GPU payant, `--flavor` GPU) n'a jamais ete effectue
(cout/duree bien superieurs a l'evaluation baseline), mais tout ce qui
peut etre verifie SANS charger le modele l'a ete pour de vrai,
credentials HF disponibles ici (`hf auth whoami` -> `mombasstic`).

**1. `recipes/*.yaml` n'est PAS installe par `--with "chsa-triage[remote] @ git+..."`,
verifie localement (pas besoin de reseau HF) :**
`uv build --wheel` sur ce depot puis inspection du `.whl` produit
confirme que `training/E2_04_sft_train.py` y est bien inclus (`packages
= ["src/chsa_triage", "interfaces", "src/tools", "training", "monitoring"]`,
`pyproject.toml`), mais AUCUN fichier sous `recipes/` : ce dossier n'est
pas un paquet Python, il n'apparait dans aucune entree de `packages`.
Consequence : `--recette recipes/sft_qwen3_lora.yaml` echouerait sur un
job qui execute le script par URL brute SANS moyen de fournir ce YAML
autrement.

**2. Mecanisme reel trouve pour livrer `--recette`/`--dataset` a un job
distant : PAS un wrapper `hf_hub_download` custom (a la difference de
la baseline GPU), le CLI `hf jobs uv run` gere deja ca nativement.**
Lu directement dans le code source de
`huggingface_hub.HfApi._create_uv_command_env_and_secrets` (installe
localement, `hf` CLI) : tout argument du script qui EST un fichier
local existant est automatiquement televerse vers un bucket HF
propre au job puis l'argument est reecrit vers le chemin monte
(`/data/<nom_fichier>`) ; tout argument qui N'EST PAS un fichier local
existant ET dont le suffixe est `.py`/`.sh`/`.yaml`/`.yml`/`.toml` (et
qui n'est pas une URL `http(s)://`) fait echouer la commande cote
CLIENT, AVANT meme de soumettre le job, avec `FileNotFoundError` :
reproduit reellement en passant `--recette /data/sft_qwen3_lora.yaml`
(chemin cense n'exister que dans le conteneur distant). Consequence
pratique : `--recette recipes/sft_qwen3_lora.yaml` (chemin LOCAL reel,
sur la machine qui lance `hf jobs uv run`) fonctionne tel quel, sans
aucun code ni depot supplementaire : le CLI le televerse et reecrit
l'argument automatiquement. Le chemin de bucket reserve pour ce
mecanisme est `/data` (`huggingface_hub.constants.HF_JOBS_ARTIFACTS_MOUNT_PATH`) :
tout volume monte manuellement via `-v` DOIT utiliser un autre chemin
de montage (`/mnt/train-data` ci-dessus), sinon `hf jobs uv run` refuse
avec "Mount path '/data' is reserved for Jobs artifacts...".

**3. `--dataset` : le pivot anonymise complet est trop volumineux
(576 Mo) pour repasser par l'auto-televersement ci-dessus a chaque
lancement (retente/grille d'hyperparametres) ; mecanisme retenu a la
place : `-v hf://datasets/<depot>:/mnt/train-data`, verifie reellement
en conditions reelles (job ci-dessous). Ceci confirme aussi, en lisant
`FormaterDatasetChatMLUseCase.executer()` (filtre deja
`type_exemple == SFT` ET `split` demande a la lecture), que `--dataset`
doit pointer vers le pivot anonymise COMPLET reparti en splits
(`data/processed/dataset_pivot_anonymise.jsonl`, 37 802 exemples SFT
avec split une fois le pivot integralement anonymise/reparti, cf.
AGENTS.md), PAS vers le sous-ensemble de 5000 exemples deja extrait
pour publication (§1.7) : ce dernier est un ECHANTILLON stratifie
delibere de 5000 lignes, bien plus petit que le pool SFT reellement
disponible pour l'entrainement.

**Depot utilise, et pourquoi ce n'est PAS le depot deja publie
`mombasstic/dataset_chsa_triage_sft_anonymise_5000` (§1.7) :** verifie
reellement en le telechargeant (`hf download`) que ce depot existant
contient bien 5000 lignes avec les bons champs, MAIS avec une
repartition `type_exemple` de `{dpo: 3591, sft: 1409}` : c'est
l'export PRE-CORRECTIF de la fuite SFT/DPO documentee dans AGENTS.md
("un export SFT de 5000 lignes etait a 72% DPO, corrige le
12/09/2026"), jamais republie depuis. Le fichier LOCAL correspondant
(`data/processed/dataset_chsa_triage_sft_anonymise_5000.jsonl`) est
lui bien corrige (5000/5000 `sft`), confirmant que c'est bien le depot
Hub qui est reste perime, pas un faux-positif de lecture. Combine au
point precedent (mauvaise taille de toute facon pour l'entrainement),
un nouveau depot dedie a ete cree : `mombasstic/chsa-triage-sft-train-data`
(prive), avec pour l'instant seulement `recipes/sft_qwen3_lora.yaml`
et un ECHANTILLON de 200 lignes (`dataset_pivot_anonymise_echantillon.jsonl`,
extrait du fichier LOCAL corrige, pas du depot perime) : suffisant pour
verifier le mecanisme de montage/lecture de bout en bout sans le cout
d'un televersement de 576 Mo pour une verification. **Avant un vrai
lancement de production**, remplacer ce fichier par le pivot anonymise
COMPLET (`hf upload mombasstic/chsa-triage-sft-train-data
data/processed/dataset_pivot_anonymise.jsonl --repo-type dataset`) ;
filtrer au prealable sur `type_exemple == sft` reduirait le
televersement d'environ 72% (mais n'est pas necessaire : le cas
d'usage filtre deja a la lecture, cf. point 3 ci-dessus). Corriger le
depot `dataset_chsa_triage_sft_anonymise_5000` perime est un probleme
REEL mais SEPARE (il alimente la publication communautaire §1.7/§1.8,
pas l'entrainement), signale ici plutot que corrige, hors perimetre de
cette verification.

**4. Job reel lance pour verifier tout ce qui precede, `--flavor
cpu-basic` (aucun GPU charge, guard atteint avant tout cout reel) :**

```bash
hf jobs uv run \
    --flavor cpu-basic \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    -v hf://datasets/mombasstic/chsa-triage-sft-train-data:/mnt/train-data \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/training/E2_04_sft_train.py \
    --recette recipes/sft_qwen3_lora.yaml \
    --dataset /mnt/train-data/dataset_pivot_anonymise_echantillon.jsonl
```

Job reel `sft-lora-verif-hybrid`
(`https://huggingface.co/jobs/mombasstic/6aaa8a2d5527934177ee9fac`,
39 s au total, ~35 s d'execution, cout de l'ordre de `$0.0002`) :
les 256 paquets de `chsa-triage[remote]` (torch/trl/peft/bitsandbytes/
unsloth/vllm compris) se sont installes avec succes sur l'infrastructure
HF Jobs reelle, `recipes/sft_qwen3_lora.yaml` est apparu televerse a
`/data/sft_qwen3_lora.yaml` (auto-televersement, point 2), le dataset
monte est apparu a `/mnt/train-data/...` (volume, point 3), et le
script a atteint et declenche pour de vrai le guard
`assistant_only_loss` (log `ERROR` explicite, sortie non-zero
ATTENDUE : le job est marque "echoue" par design, c'est le
comportement voulu, pas un bug). Ceci confirme le CABLAGE complet
(paquet, recette, dataset, guard) sur l'infrastructure reelle, PAS
l'entrainement lui-meme (jamais lance avec un GPU).

**5. Choix du flavor GPU pour un vrai lancement, raisonnement (pas de
mesure GPU reelle disponible ici) :** contrairement a l'inference
baseline (§2.2), l'entrainement QLoRA 4-bit d'un modele de 1.7B a une
empreinte VRAM modeste (poids 4-bit ~1 Go + adaptateurs LoRA rang 16
sur 4 modules, negligeables + etats d'optimiseur AdamW sur les seuls
poids LoRA) : `l4x1` (24 Go) reste largement suffisant COTE VRAM,
meme marge que pour l'inference deja validee sur ce meme pipeline. La
vraie difference avec la baseline (278 exemples, une seule passe
forward chacun) est le TEMPS : plusieurs milliers de pas d'optimisation
sur jusqu'a ~30 000 exemples d'entrainement (avant `packing`) x 3
epoques. D'ou `--timeout 6h` ajoute explicitement ci-dessus (le defaut
de `hf jobs uv run` n'est pas documente dans `--help`, mieux vaut le
fixer soi-meme) plutot qu'un flavor plus gros : si un run reel montre
que `l4x1` est trop lent pour le budget de temps/cout accepte,
`a10g-large` (meme VRAM, plus de vCPU/RAM, cf. `hf jobs hardware`) est
l'escalade naturelle, a mesurer sur un run reel, pas devinee ici.

**6. Persistance des poids du meilleur checkpoint, gap trouve et
corrige (16/09/2026) :** avant ce correctif, `training/E2_04_sft_train.py`
n'ecrivait les poids LoRA qu'en LOCAL (`--repertoire-sortie-checkpoints`,
`trainer.save_model()`), jamais vers un depot HF ; `grep -rn
"push_to_hub|upload_folder" src/ training/` ne trouvait qu'UN usage
(les METRIQUES, `HfDatasetSuiviExperimentation`), jamais le modele
lui-meme. Sur un job HF Jobs distant, dont le disque ne survit pas au
job (meme fait deja documente pour la baseline GPU, §2.2), un
entrainement reel facture aurait donc perdu le modele entraine,
seules les metriques auraient survecu. Corrige en ajoutant
`--checkpoint-hf-repo` (nouvelle fonction `_publier_checkpoint_hf`,
`training/E2_04_sft_train.py`) : publie, UNE SEULE FOIS apres selection
du meilleur essai de la grille (jamais les essais intermediaires
rejetes), le dossier local du checkpoint vers un depot modele HF prive
via `huggingface_hub.upload_folder` (`HfApi.create_repo(...,
exist_ok=True)` d'abord, donc pas besoin de creer le depot a la main
au prealable, a la difference des depots dataset/Space ci-dessus).
VERIFIE SANS GPU : signatures reelles de `trl.SFTConfig`/
`transformers.Trainer` (`push_to_hub`/`hub_model_id`/`push_to_hub()`
existent nativement, mais les brancher directement dans
`TrlSftEntraineurAdapter.entrainer()` publierait CHAQUE essai de la
grille au lieu du seul meilleur, d'ou le choix de publier explicitement
dans `E2_04_sft_train.py` plutot que dans l'adaptateur) et
`HfApi.create_repo`/`upload_folder` (signatures reelles inspectees,
tests `tests/training/test_E2_04_sft_train.py` avec `HfApi`
remplace, aucun reseau reel). NON VERIFIE : la publication d'un
checkpoint REEL (poids produits par un vrai entrainement), faute de
GPU disponible ici.

**7. Premier vrai lancement GPU (L4, payant), echec a la construction
de `SFTTrainer`, cause reelle et correction (16/09/2026) :** un vrai
job GPU L4 facture a ete lance (pas seulement le job de verification
`cpu-basic` ci-dessus) : il est alle loin (telechargement reel du
dataset, formatage ChatML, tokenisation, `packing`, guard
`assistant_only_loss` deja franchi), puis a echoue a la construction de
`SFTTrainer` avec `ValueError: Invalid loss_type chunked_nll passed.
Supported values are 'nll' and 'dft'.` `recipes/sft_qwen3_lora.yaml::
entrainement.type_perte` valait `chunked_nll`, une optimisation
memoire reelle (calcule la perte par morceaux de la sequence pour
eviter de materialiser le tenseur de logits complet, cf. §2.1/00_
introduction_concepts.md) jamais confrontee au trl REELLEMENT installe
par ce job. **Cause reelle, pas un simple typo** : `chunked_nll`
EXISTE bien dans `trl` depuis la version 1.12 (verifie par lecture du
code source de plusieurs versions de `trl`, installees temporairement
dans un venv jetable), et est meme la valeur PAR DEFAUT de
`SFTConfig.loss_type`. Mais l'extra `remote` de `pyproject.toml` liste
`"unsloth"` SANS borne de version, alors qu'Unsloth n'est PAS cable
dans le code (cf. AVERTISSEMENT dans
`infrastructure/adapters/trl_sft_entraineur.py`) ; la derniere version
publiee d'`unsloth` (verifie reellement via l'API JSON PyPI,
`unsloth==2026.9.4`) exige sans condition `trl<=0.24.0`, un trl
pre-1.0 ecrit avant que `chunked_nll` existe (reproduit pour de vrai :
`uv pip install "trl>=0.9" "unsloth"` dans un venv jetable, sans
lockfile, resout bien `trl==0.24.0` + `unsloth==2026.9.4`, et l'erreur
`ValueError` obtenue est identique mot pour mot a celle du job reel).
Le `uv.lock` commite de ce depot masque le probleme en local (il a
fige un `unsloth==2024.8` tres ancien, sans contrainte sur `trl`,
laissant `trl` remonter a 1.13.0 lors d'un `uv sync --extra remote`),
mais `hf jobs uv run --with "chsa-triage[remote] @ git+..."` (point 4
ci-dessus) n'utilise PAS `uv.lock` : chaque lancement resout les
dependances a neuf contre l'etat REEL de PyPI, pas l'etat fige
localement. **Correction appliquee** : `recipes/sft_qwen3_lora.yaml::
entrainement.type_perte` passe a `nll` (la valeur standard, supportee
par toutes les versions de `trl` observees ici, de 0.24.0 a 1.13.0) ;
`training/E2_04_sft_train.py` verifie desormais `type_perte` contre un
ensemble de valeurs sures AVANT de charger le modele
(`_verifier_type_perte_valide`, meme patron que le guard
`assistant_only_loss`), pour qu'une future config invalide echoue tout
de suite plutot qu'apres avoir facture le telechargement/formatage/
tokenisation. **Marge memoire sans `chunked_nll`** : le raisonnement
VRAM deja documente au point 5 ci-dessus (LoRA rang 16 sur 4 modules
d'un modele 1.7B quantifie NF4, `l4x1` 24 Go, marge large) reste
valide : il ne reposait pas sur `chunked_nll`, deja ecrit pour
`loss_type="nll"` standard ; `chunked_nll` aurait ete un coussin de
securite supplementaire, pas une condition de faisabilite. Si le pic
memoire s'avere reellement trop juste sur un futur run, retirer
`unsloth` (non utilise) de l'extra `remote` permettrait a `trl>=1.12`
de se resoudre et de recuperer `chunked_nll` pour de vrai : option NON
appliquee ici (changement de dependances plus large que ce correctif).
Voir l'AVERTISSEMENT complet dans
`infrastructure/adapters/trl_sft_entraineur.py` et AGENTS.md.

**8. Premier entrainement complet REELLEMENT lance et reussi (GPU L4,
verdict "saine", poids publies), mais courbe de suivi perdue, cause
reelle et correction (16/09/2026) :** apres la correction du point 7,
un run GPU L4 facture complet a ete lance pour de vrai (~20 min, 342
pas) et a converge (verdict `SAINE`), avec les poids LoRA publies avec
succes sur `mombasstic/chsa-triage-sft-lora` (point 6 ci-dessus). Mais
les metriques de la courbe d'entrainement (`perte_train`,
`perte_validation`, `norme_gradient`) ne sont arrivees NULLE PART de
durable : perdues. **Cause reelle, verifiee dans le code** : la
commande de lancement passait bien `--suivi-hf-repo
mombasstic/chsa-triage-sft-metrics`, mais `recipes/
sft_qwen3_lora.yaml::suivi.backend` valait encore `mlflow` (la valeur
par defaut a ce moment-la) ; `training/E2_04_sft_train.py::
_construire_suivi()` ne lit `arguments.suivi_hf_repo` QUE quand
`backend == "hf_dataset"`, donc ce flag a ete ignore en silence et les
metriques ecrites dans un SQLite LOCAL (`data/processed/mlflow.db` par
defaut) A L'INTERIEUR du conteneur ephemere du job HF Jobs, qui ne
survit pas au job (meme fait deja documente pour les poids au point 6
et pour la baseline GPU au §2.2). Confirme en telechargeant
`mombasstic/chsa-triage-sft-metrics` : il ne contient que le fixture
`demo_datos_ficticios`, rien du run `sft-lora` reel. **Correction
appliquee, en deux temps deliberement redondants** : (1) `recipes/
sft_qwen3_lora.yaml::suivi.backend` vaut maintenant `hf_dataset` par
defaut (`mlflow` ne redevient pertinent que pour un futur run
GENUINEMENT local sur GPU propre, disque persistant ; les deux
commandes reellement documentees ci-dessus, locale et `hf jobs uv run`,
passent deja `--suivi-hf-repo` ensemble avec `--checkpoint-hf-repo`) ;
(2) nouvelle fonction `_verifier_suivi_hf_repo_coherent()` (meme patron
que les guards `assistant_only_loss`/`type_perte`, appelee AVANT tout
chargement de modele/GPU) refuse de demarrer si `--suivi-hf-repo` est
fourni alors que `suivi.backend != hf_dataset`, pour que ce flag ne
puisse plus JAMAIS etre ignore en silence, meme si la recette est
modifiee de nouveau a l'avenir. VERIFIE SANS GPU NI RESEAU : les deux
correctifs sont testes (`tests/domain/test_configuration_entrainement.py`
pour la valeur par defaut de la recette, `tests/training/
test_E2_04_sft_train.py::TestVerifierSuiviHfRepoCoherent` pour le
guard). NON VERIFIE ici : la reconstruction de la courbe deja perdue de
ce run precis, menee separement a partir du log brut du job (hors
perimetre de cette correction, cf. point 9 ci-dessous). Voir AGENTS.md
pour le meme avertissement, redige au niveau du code.

**9. Courbe de metriques de CE run precis (job `6aaab9a95527934177eeaac8`,
point 8 ci-dessus) reconstruite a posteriori (16/09/2026) :**
`monitoring/reconstruire_courbe_sft_depuis_log.py` reconstruit cette
courbe precise a partir du LOG BRUT sauvegarde du job (915 lignes,
dictionnaires `{'loss': ...}`/`{'eval_loss': ...}` imprimes par
`transformers`/`trl`) et la republie dans le meme depot dataset
(`mombasstic/chsa-triage-sft-metrics/sft-lora-16092026-reconstruit/`,
verifie reellement selectionnable par `monitoring/app_suivi_entrainement.py`
via `hf_dataset_runs.lister_runs`). Les `etape` viennent du numero de
pas tqdm reellement imprime (pas d'une simple interpolation par
`epoch`). Les `horodatage`, en revanche, sont RECONSTRUITS et non
mesures ligne a ligne : le job n'imprime pas d'horodatage a chaque pas,
seulement dans les barres tqdm (temps ecoule depuis le debut de la
boucle) et dans le nom du repertoire de checkpoint cree juste avant
`trainer.train()` (`run-YYYYmmddTHHMMSSZ`, horodatage reel utilise
comme ancrage). Coherence verifiee : l'ecart ainsi reconstruit au
dernier pas (~1182 s) concorde a moins d'une seconde avec le
`train_runtime` rapporte par `trl` lui-meme en fin de run (1182.9999 s).
Le fichier `parametres.json` associe marque explicitement
`reconstruit_depuis_log: true` et le `job_id` source, et embarque la
recette REELLE au commit utilise par le job (`git show <sha>:recipes/
sft_qwen3_lora.yaml`, jamais le fichier courant du worktree, qui a pu
changer depuis, notamment via le correctif du point 8). Constat annexe,
non corrige ici (dashboard existant, hors perimetre) : comme
`E2_01_uc_entrainer_sft.py` logue `perte_train` tous les 10 pas mais les
evaluations tombent aux bornes d'epoque (pas 114/228/342, jamais
multiples de 10 sauf 342 lui-meme absent des pas de log d'entrainement),
`monitoring/logica_suivi_entrainement.py::construire_courbe_convergence`
(qui exige `perte_train` ET `norme_gradient` sur la MEME ligne pivotee)
n'inclut aucun des 3 points `perte_validation` dans son verdict de
convergence : un run reel futur avec les memes
`logging_steps`/`eval_strategy` afficherait le meme comportement, ce
n'est pas un artefact de la reconstruction. Script teste sans reseau
(`tests/monitoring/test_reconstruire_courbe_sft_depuis_log.py`,
fragment de log fidele au format reel).

**Panne serveur connue sur `hf repo create`/`hf repos create --repo-type
dataset` :** confirme sur ce projet le 16/09/2026, la commande peut
echouer avec une vraie `500 Internal Server Error` renvoyee par le
serveur de Hugging Face lui-meme, sur les deux formes de la commande
(l'ancienne `hf repo create`, deja marquee "deprecated", et la nouvelle
`hf repos create`), toutes deux contre le meme endpoint
`https://huggingface.co/api/repos/create`. Rien a voir avec les
identifiants ou la syntaxe de la commande : c'est cote HF. Contournement :
reessayer la commande (l'erreur est generalement transitoire), ou, si
elle persiste, creer le depot manuellement depuis l'interface web de
Hugging Face puis continuer avec `hf upload` normalement (le reste du
flux ne depend pas de la creation du depot via le CLI).

Smoke test local (verifie, sans reseau, contre un JSONL de fixture) :
`uv run streamlit run monitoring/app_suivi_entrainement.py` (necessite
`uv sync --extra web --extra local`, groupes `streamlit`/`huggingface_hub`).
La logique pure (parsing JSONL, pivot, verdict de convergence) est
testee dans `tests/monitoring/test_app_suivi_entrainement.py`, sans
Streamlit ni reseau. `data/demos/chsa-triage-sft-metrics-fake.json`
(etape 4 ci-dessus) permet en plus de verifier le dashboard COMPLET
(graphique, cartes, verdict en direct) sans attendre un run GPU reel,
une fois publie sur le depot de metriques.

<table id="historique-complet-dans-un-mlflow-local-importateur" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h3 style="border-bottom:none; margin:0;">Historique complet dans un MLflow local (importateur)</h3>
</td></tr></table>

Le dashboard Streamlit ne montre que le run selectionne, en vivo,
depuis un Space distant : pour parcourir l'HISTORIQUE COMPLET de tous
les runs (SFT, et DPO plus tard, meme mecanisme) avec l'interface MLflow
habituelle (comparaison de runs, tri par metrique, etc.), sans monter
de serveur MLflow distant (ecarte : aurait exige Postgres + Docker +
authentification d'un Space prive, trop d'infrastructure pour ce POC),
`monitoring/importer_mlflow_local.py` telecharge les runs du meme
depot dataset HF et les reproduit dans un MLflow LOCAL (SQLite), via
l'adaptateur `MlflowSuiviExperimentation` deja utilise par
`training/E2_04_sft_train.py` (`suivi.backend: mlflow`). Idempotent :
relancer la commande n'importe que les runs pas encore presents dans
ce MLflow local (`--forcer` pour reimporter).

```bash
# Rafraichir le MLflow local (par defaut : ~/.chsa-triage/mlflow.db) :
uv run python monitoring/importer_mlflow_local.py

Depot dataset HF : mombasstic/chsa-triage-sft-metrics
MLflow local : sqlite:////home/rafael/.chsa-triage/mlflow.db
2026/09/15 16:31:27 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/09/15 16:31:27 INFO mlflow.store.db.utils: Updating database tables
metriques.jsonl: 5.76kB [00:00, 2.56MB/s]
importe : demo_datos_ficticios (69 metriques)
1 run(s) importe(s) : demo_datos_ficticios
Ouvrir l'interface : uv run mlflow ui --backend-store-uri sqlite:////home/rafael/.chsa-triage/mlflow.db --host 0.0.0.0

uv run mlflow ui \
  --backend-store-uri sqlite:////home/rafael/.chsa-triage/mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --cors-allowed-origins "*"

# Ouvrir l'interface MLflow sur ce meme fichier :
uv run mlflow ui --backend-store-uri sqlite:///$HOME/.chsa-triage/mlflow.db --host 0.0.0.0
```

La frontiere reseau HF Hub (`HfApi.list_repo_files`/`hf_hub_download`)
est partagee avec le dashboard dans `monitoring/hf_dataset_runs.py`,
pour ne pas la dupliquer entre les deux. Verifie de bout en bout SANS
reseau HF reel (le depot `mombasstic/chsa-triage-sft-metrics` necessite
une authentification HF non disponible ici, cf. plus haut) en
alimentant directement la logique d'import avec le contenu de
`data/demos/chsa-triage-sft-metrics-fake.json` : les 31 etapes
apparaissent bien dans le MLflow local (metriques + parametres
relisibles via `MlflowClient`), et une seconde execution n'importe
rien de plus (idempotence confirmee). Tests :
`tests/monitoring/test_importer_mlflow_local.py` (integration MLflow
reelle sur sqlite temporaire, source HF injectee) et
`tests/monitoring/test_hf_dataset_runs.py` (frontiere HF Hub,
`HfApi`/`hf_hub_download` remplaces).

<table id="24-évaluation-post-sft" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h2 style="border-bottom:none; margin:0;">2.4 Évaluation post-SFT</h2>
</td></tr></table>

Mesure la performance du modèle RÉELLEMENT entraîné par SFT-LoRA
(poids publiés sur `mombasstic/chsa-triage-sft-lora`, verdict `SAINE`,
premier entraînement réel de cette session, §2.3 ci-dessus), sur le
MÊME sous-ensemble de 278 exemples `split=test`/`type_exemple=sft`
que les deux baselines zero-shot (`mombasstic/chsa-triage-baseline-test`,
§2.2), avec les MÊMES métriques
(`application/metriques_evaluation_baseline.py`), pour une comparaison
numéro-contre-numéro directe.

**Décision de conception : réutilisation, jamais un renommage.**
`EvaluerBaselineZeroShotUseCase`
(`application/use_cases/E1_06_00_evaluer_baseline_zero_shot.py`) est
agnostique au modèle injecté : il ne connaît qu'un `MoteurInference`
(port), un `RepositoryLectureEcriture`, un `FormateurInviteZeroShot`
et un `SuiviExperimentation`, jamais une classe concrète. Confirmé en
le relisant : rien dans son code ne suppose que le modèle est
"zero-shot" au sens strict, seulement qu'on génère une réponse par
exemple du split test et qu'on la compare à la référence. Il est donc
réutilisé SANS AUCUNE MODIFICATION une troisième fois (après CPU/GGUF
puis GPU/bf16 zero-shot, §2.2), exactement le même patron que
`E1_06_01_evaluer_baseline_gpu.py` vs `E1_06_00_evaluer_baseline.py` :
seul l'adaptateur d'inférence change.
`interfaces/cli/E2_05_evaluer_post_sft.py` est ce troisième point
d'entrée. Renommer le cas d'usage ("Baseline"/"ZeroShot" ne colle plus
littéralement à un modèle entraîné) a été considéré et écarté : cela
aurait touché 3 scripts CLI déjà publiés (les deux baselines et
celui-ci) et leurs tests pour un gain purement cosmétique, sans changer
le comportement ; le docstring du nouveau CLI documente explicitement
ce réemploi plutôt que de le masquer. Aucun nouveau cas d'usage fin
n'a été ajouté non plus : un simple wrapper qui ne ferait que déléguer
à `EvaluerBaselineZeroShotUseCase` sans rien y ajouter aurait été une
abstraction sans valeur.

**Adaptateur nouveau : `TransformersLoraInferenceAdapter`**
(`infrastructure/adapters/transformers_lora_inference_adapter.py`),
copie quasi conforme de `TransformersInferenceAdapter` (§2.2) : même
contrat `parametres["invite_deja_rendue"]`, même mesure de
`latence_ms`, même refus explicite (`RuntimeError`) sans GPU CUDA
(jamais de repli silencieux vers le CPU). Seule différence réelle :
`_obtenir_modele_et_tokenizer()` charge le modèle de base
(`Qwen/Qwen3-1.7B-Base`, bf16) PUIS l'enveloppe avec
`peft.PeftModel.from_pretrained(modele_base, depot_lora)` pour
appliquer les poids LoRA entraînés par-dessus. Réutilise directement
`_parametres_generation_transformers` de
`transformers_inference_adapter.py` (import, jamais dupliquée) :
seule la construction du modèle diffère entre les deux adaptateurs, la
traduction des paramètres de génération est strictement identique.
`peft>=0.12` est déjà dans l'extra `remote` de `pyproject.toml`,
aucune nouvelle dépendance nécessaire.

```bash
uv run python interfaces/cli/E2_05_evaluer_post_sft.py \
    --dataset-hf-repo mombasstic/chsa-triage-baseline-test \
    --depot-lora mombasstic/chsa-triage-sft-lora \
    --suivi-hf-repo mombasstic/chsa-triage-baseline-metrics
```

**Commande `hf jobs uv run` (syntaxe vérifiée via `hf jobs uv run --help`
dans cet environnement ; jamais lancée avec un `--flavor` GPU réel,
coût/attente d'une session GPU payante hors périmètre de cette tâche
d'après le budget accordé) :**

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

`--flavor l4x1` : même raisonnement que la baseline GPU (§2.2), pas
une nouvelle mesure GPU réelle. C'est de l'INFÉRENCE (une passe forward
par exemple), pas de l'entraînement : coût/durée attendus du même ordre
de grandeur que la baseline GPU (~7,3 s/génération mesurés en §2.2 sur
ce même `l4x1`, donc ~34 min pour les 278 exemples), très inférieur au
coût de l'entraînement lui-même (~20 min, §2.3 point 8). Cette commande
suppose que cette branche a déjà été fusionnée sur `main` et poussée
sur GitHub : `hf jobs uv run` télécharge le script par URL brute
directement depuis `racemartin/m14_ocr@main`, pas depuis ce worktree
local (même limite déjà documentée pour les deux baselines, §2.2).

**Vérification réelle effectuée dans cet environnement (17/09/2026),
et sa limite honnête, même démarche que §2.2/§2.3 :** credentials HF
réelles disponibles ici (`hf auth whoami` → `mombasstic`). Vérifié pour
de vrai, sans dépenser de session GPU :
- Téléchargement réel du dataset `mombasstic/chsa-triage-baseline-test`
  (`dataset_pivot_test_sft.jsonl`, 278 lignes confirmées, identique au
  fichier déjà utilisé par les deux baselines).
- Le dépôt `mombasstic/chsa-triage-sft-lora` existe réellement et
  contient bien `adapter_config.json`/`adapter_model.safetensors` à sa
  racine (poids du MEILLEUR essai, jamais les checkpoints
  intermédiaires, cf. `_publier_checkpoint_hf`, §2.3 point 6) ;
  `adapter_config.json::base_model_name_or_path` vaut bien
  `Qwen/Qwen3-1.7B-Base`, confirmant que ce dépôt LoRA correspond
  réellement au modèle de base utilisé par `--modele-base` par défaut.
- `peft==0.21.0` installé temporairement (comme pour les vérifications
  de signature précédentes de cette session, sans GPU) : signature
  réelle de `peft.PeftModel.from_pretrained(model, model_id, ...,
  is_trainable=False, ...)` confirmée, correspond à l'usage fait par
  l'adaptateur (`is_trainable` reste à son défaut `False`, l'usage ici
  est de l'inférence, jamais un réentraînement).
- `interfaces/cli/E2_05_evaluer_post_sft.py` exécuté pour de vrai dans
  cet environnement (sans `--flavor`, donc en local, sans GPU) : le
  téléchargement du dataset et la tentative de chargement du modèle
  LoRA se déroulent réellement, puis `TransformersLoraInferenceAdapter`
  refuse explicitement (`RuntimeError`, jamais un repli silencieux)
  sur les 278/278 exemples faute de GPU CUDA, et le cas d'usage lève
  `ValueError` ("rien à agréger"), exactement le même comportement que
  la vérification de la baseline GPU en §2.2. Ceci confirme le CÂBLAGE
  de bout en bout (dataset → adaptateur base+LoRA → métriques → suivi),
  PAS les vrais chiffres post-SFT, qui restent à produire sur un job HF
  Jobs GPU réel (jamais lancé ici, même raison de coût que §2.2/§2.3).

**Résultat réel (17/09/2026), job GPU réel (`--flavor l4x1`) lancé avec
succès, run `evaluation-post-sft` journalisé dans le même dépôt
métriques que les deux baselines (`mombasstic/chsa-triage-baseline-metrics`) :**
sur les mêmes 278 exemples `split=test`/`type_exemple=sft` : 278
évalués et 278 comparables (ZÉRO échec d'inférence), exact match
0.000, F1 moyen (token) 0.112 (valeur exacte 0.11166994355349093),
exactitude de classification du niveau ESI non calculable (0/278,
même raison structurelle que les deux baselines, cf. §2.2), latence
moyenne 11553,4 ms (~11,6 s) par génération.

**Comparaison des trois runs** (mêmes 278 exemples, mêmes métriques,
tous les trois désormais journalisés dans le même dépôt
`mombasstic/chsa-triage-baseline-metrics`) :

| | Exact match | F1 moyen (token) | Latence moyenne |
|---|---|---|---|
| Baseline CPU (Q4_K_M) | 0,000 | 0,037 | ~21,6 s |
| Baseline GPU (bf16) | 0,000 | 0,043 | ~7,3 s |
| Post-SFT (bf16+LoRA) | 0,000 | **0,112** | ~11,6 s |

**Le F1 quasi triple par rapport au meilleur des deux baselines
(0,043 -> 0,112) :** première preuve chiffrée que le SFT a un effet
mesurable (cahier des charges §9), cohérente avec le verdict de
convergence `SAINE` déjà obtenu côté courbe d'entraînement (§2.3).
L'exact match reste à 0,000 sur les trois runs : à ne pas lire comme
un échec du SFT, c'est une métrique très stricte qui exige une
correspondance caractère-à-caractère avec une réponse de référence en
langage libre (MediQAl/FrenchMedMCQA/MedQuAD, cf. le point de
vigilance JSON en §2.2), quasi inatteignable même avec une réelle
amélioration du modèle ; le F1 token, moins strict, est la métrique
qui porte le signal ici.

**La latence augmente par rapport à la baseline GPU (~7,3 s ->
~11,6 s), et l'hypothèse retenue pour l'expliquer n'a pas pu être
vérifiée directement :** `EvaluerBaselineZeroShotUseCase` ne
journalise aucune longueur de génération par exemple (seuls
`exact_match`/`f1_moyen`/`exactitude_niveau_triage`/`latence_ms_moyenne`/
`nombre_echecs_inference` sont loggés, confirmé en relisant
`executer()`, et aucun texte de génération brut n'est conservé nulle
part), donc la longueur moyenne réelle des générations n'est pas
calculable a posteriori sans relancer une inférence (nouveau coût
GPU, hors périmètre ici). Élément indirect disponible : les deux runs
partagent exactement les mêmes paramètres de génération
(`temperature: 0.0`, `n_predict: 256`, confirmé dans `parametres.json`
des deux runs) et le même `--flavor l4x1`, donc le ratio de latence
(~1,58×) est cohérent avec un nombre de tokens générés significativement
plus élevé en moyenne côté post-SFT (le modèle affiné, désormais
exposé au format de triage cible, produit vraisemblablement des
réponses plus longues/plus structurées, se rapprochant davantage de la
limite `n_predict=256` plutôt que de s'arrêter tôt comme le modèle de
base non entraîné). Ceci reste une hypothèse plausible, pas un fait
vérifié : documentée comme telle plutôt que présentée comme confirmée.

**Ce que ce résultat établit :** le pipeline d'évaluation post-SFT est
validé de bout en bout sur GPU réel (chargement modèle de base + poids
LoRA, génération, scoring, journalisation), et le SFT-LoRA a un effet
mesurable et positif sur le F1 token par rapport aux deux baselines
zero-shot, conformément à l'objectif du cahier des charges §9.

Synthèse de ces résultats (baselines → entraînement → évaluation
post-SFT) sous forme de présentation PowerPoint :
[`docs/03_etape2_sft/04_presentation_synthese_sft_lora.pptx`](docs/03_etape2_sft/04_presentation_synthese_sft_lora.pptx)
(régénérable via `scripts/generer_presentation_etape2.py`).

<table id="3-dpo-étape-3" style="width:100%;"><tr><td style="background-color:#7aeae7;">
<h1 style="border-bottom:none; margin:0;">3. DPO (Étape 3)</h1>
</td></tr></table>

**Non implémenté à ce jour.** Aucune commande ni étape n'existe encore
dans le code pour cette phase ; voir `docs/04_etape3_dpo/` (à venir).

<table id="structure-architecture-hexagonale" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">Structure (architecture hexagonale)</h1>
</td></tr></table>

```
src/chsa_triage/
├── domain/            # entités + ports, zéro dépendance externe
├── application/       # cas d'usage : orchestrent les ports
└── infrastructure/    # adaptateurs concrets (JSONL, HF, Presidio, ydata-profiling, ...)
interfaces/            # adaptateurs primaires : cli/ (Étape 1), api/ et web/ (Étape 4)
training/              # scripts exécutés via HF Jobs (SFT, DPO) : Étapes 2-3
docker/                # Dockerfiles + docker-compose (frontend/backend) : Étape 4
```

Détail complet : `docs/01_environnement/01_architecture_hexagonale.md`.

<table id="état-davancement" style="width:100%;"><tr><td style="background-color:#c9f1ed;">
<h1 style="border-bottom:none; margin:0;">État d'avancement</h1>
</td></tr></table>

- [x] Étape 0 : Cadrage, environnement, architecture
- [ ] Étape 1 : Préparation des données : dataset pivot **régénéré**
      (08/09/2026) avec identifiants **déterministes** sur les 6
      fichiers réels : **134 883 exemples** (147 204 registres bruts,
      **12 321 doublons exacts dédoublonnés réellement**, archivés
      dans `data/processed/doublons_supprimes.jsonl`, jamais perdus) ;
      anonymisation écrit désormais dans un fichier **séparé**
      (`dataset_pivot_anonymise.jsonl`, le pivot original n'est plus
      jamais modifié), complète mesurée à ~19h (coût NLP
      Presidio/spaCy), rendue incrémentale/reprenable via `--limite`
      (échantillonnage stratifié par type_exemple+source) ; chaque
      exécution génère/fusionne automatiquement un **rapport RGPD
      cumulé** (JSON + Markdown) ; contrôle qualité **automatisé** par
      comparaison de fichiers (regex + seconde opinion spaCy,
      `E1_04_02_controler_qualite_anonymisation.py`) ; **première vague
      exécutée sur le pivot régénéré (5 000/134 883 exemples, 90,6 %
      avec ≥1 entité détectée, 64 667 entités) et découpée en splits
      (4 004/498/498, vérifiée représentative par strate)** ; 200
      exemples contrôlés automatiquement (0 PII résiduelle confirmée,
      35 candidats explicitement en attente de révision humaine) ;
      voir `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`.
      Muestreo du contrôle qualité rendu **incrémental** (09/09/2026,
      `--registre-echantillons`) et les 35 candidats en attente
      peuvent désormais être tranchés avec une décision humaine
      **persistée** (`E1_04_01_reviser_pii_residuelle.py`,
      `data/processed/decisions_revision_humaine.jsonl`) ; voir §1.5
      ci-dessus et `docs/02_etape1_donnees/01_rapport_rgpd.md` §7.5.
      **Vagues ultérieures** : à relancer avec `--limite` plus grand
      (ou `full`) avant le SFT/DPO ; réévaluer d'abord le risque de
      saturation/surapprentissage d'un entraînement sur un
      sous-échantillon trop petit face au dataset complet (à étudier
      à ce moment-là, pas tranché ici)
- [ ] Étape 2 : SFT + LoRA
- [ ] Étape 3 : DPO
- [ ] Étape 4 : Déploiement (FastAPI + Streamlit + vLLM + CI/CD)
