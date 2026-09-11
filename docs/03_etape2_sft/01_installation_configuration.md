\newpage

# Étape 2 : Installation et configuration

> Les trois phases sans GPU de l'Étape 2 sont écrites et testées (voir
> `02_etapes_cas_usage.md`) ; ce document en revanche décrit
> spécifiquement l'Environnement B (GPU), qui reste entièrement à
> provisionner : `training/sft_train.py` n'est pas encore écrit,
> `recipes/sft_qwen3_lora.yaml` n'existe pas encore. Les commandes et
> journaux ci-dessous restent des exemples illustratifs de ce qui est
> attendu une fois ces deux éléments en place : ils ne sont pas
> exécutés ni mesurés à ce stade.

## 1. Quel environnement pour cette étape

`docs/01_environnement/00_guide_installation_environnement.md`
distingue déjà **Environnement A** (local, WSL2, 5 Go RAM, sans GPU :
utilisé pour toute l'Étape 1) et **Environnement B** (distant, HF
Jobs / Spaces Dev Mode, GPU à la demande : réservé aux Étapes 2-3).
L'Étape 2 se déroule entièrement en **Environnement B** : le
chargement 4-bit d'un modèle de 1,7 milliard de paramètres et
l'entraînement LoRA nécessitent un GPU, absent en Environnement A.

`pyproject.toml` définit déjà le groupe de dépendances correspondant
(`[project.optional-dependencies] remote`), pas à recréer :

```toml
remote = [
    "torch>=2.3", "transformers>=4.44", "trl>=0.9", "peft>=0.12",
    "accelerate>=0.33", "bitsandbytes>=0.43", "unsloth", "liger-kernel",
    "datasets>=2.20", "huggingface_hub>=0.24", "vllm>=0.5",
    "mlflow>=2.14", "tensorboard>=2.17",
]
```

`vllm` est listé dans ce même groupe pour l'Étape 4 (inférence) : il
n'est pas nécessaire pour le SFT lui-même, mais partager le groupe
`remote` évite une resynchronisation d'environnement entre les Étapes
2, 3 et 4 qui tournent toutes sur la même machine GPU distante.

## 2. Provisionner une session GPU (Spaces Dev Mode)

Reprend `docs/01_environnement/00_guide_installation_environnement.md`
§3.3, avec le hardware adapté à un run SFT réel plutôt qu'à du
débogage léger :

```bash
# Depuis le poste local (Environnement A) : créer/ouvrir le Space,
# activer Dev Mode avec un GPU A10G (T4 suffit pour valider le
# chargement 4-bit et un pas d'entraînement, pas pour un run complet
# dans un temps raisonnable : cf. §3.4 discipline de facturation)
ssh chsa-dev   # alias défini dans ~/.ssh/config, cf. guide d'installation

# Dans la session Dev Mode (GPU) :
uv sync --extra remote --extra dev
```

## 3. Authentification et secrets

Deux identifiants Hugging Face distincts sont nécessaires, tous deux
via le même `HF_TOKEN` (rôle "write", déjà utilisé à l'Étape 1) :

| Usage | Ressource HF | Statut |
|---|---|---|
| Télécharger les poids | `Qwen/Qwen3-1.7B-Base` | Modèle public au moment de l'écriture ; vérifier la page du modèle avant le premier téléchargement (certains modèles Qwen exigent une acceptation de licence) |
| Charger le dataset d'entraînement | Dataset CHSA versionné sur HF Hub | **Pas encore poussé** : le roadmap Étape 1 prévoit "Verser le dataset versionné sur HF Hub (Livrable 1)" comme dernière étape, non encore réalisée. Voir §4 ci-dessous |
| Pousser le checkpoint SFT-LoRA | Repo HF `models` dédié (à créer) | Pas encore créé |

```bash
hf auth login   # dans la session Dev Mode, si pas déjà fait via .env
```

`.env` (Environnement B, jamais commité, même fichier que
l'Environnement A si la session Dev Mode le récupère par `git pull`) :

```bash
HF_TOKEN=hf_xxx
MLFLOW_TRACKING_URI=<uri_partagee_avec_environnement_A>
```

Le fait de réutiliser `MLFLOW_TRACKING_URI` (et non une instance MLflow
séparée par environnement) permet de comparer, dans un même tableau de
bord, la baseline (Étape 1bis, calculable en Environnement A puisque
l'inférence zéro-shot n'a pas besoin de 4-bit pour un test léger) et
les runs SFT (Étape 2, Environnement B).

## 4. Dépendance non résolue : le dataset d'entraînement

Le pipeline Étape 2 a besoin d'un dataset d'entraînement/validation.
Deux sources sont possibles à ce jour :

1. **HF Hub, dataset versionné (Livrable 1)** : prévu par le roadmap
   comme livrable de fin d'Étape 1, **pas encore réalisé**. C'est la
   source cible à terme (reproductible, versionnée, chargeable
   directement dans `training/sft_train.py` via
   `datasets.load_dataset`).
2. **Fichiers locaux déjà produits** :
   `data/processed/dataset_pivot_anonymise.jsonl` découpé en splits
   (`data/splits/{train,val,test}.jsonl` via
   `interfaces/cli/E1_05_00_decouper_splits.py`) : au moment de l'écriture de
   ce document, **5 000 exemples anonymisés sur 134 883** (4 004
   train / 498 val / 498 test), le reste du dataset pivot restant à
   anonymiser par vagues successives (`--limite`, cf.
   `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`).

Cette dépendance n'est pas bloquante pour la planification (ce
document), mais elle conditionne un choix d'implémentation réel :
`training/sft_train.py` doit-il lire depuis HF Hub ou depuis
`data/splits/*.jsonl` transférés manuellement dans l'Environnement B ?
Voir `03_guide_implementation_pas_a_pas.md` §"Chargement du dataset"
pour la proposition de conception (repository JSONL réutilisé en
premier, adaptateur HF Hub branché sans changer l'application le jour
où le Livrable 1 est poussé : c'est exactement le bénéfice attendu du
port générique `RepositoryLectureEcriture`, cf.
`docs/01_environnement/01_architecture_hexagonale.md` §4).

## 5. Vérifier l'environnement GPU

`scripts/check_env_gpu.py` **existe déjà** dans le dépôt (écrit en
prévision de cette étape, cf. roadmap "ETAPE 0 : Cadrage &
Environnement" → `check_env_local.py` / `check_env_gpu.py`). Il
vérifie, dans cet ordre : disponibilité CUDA, présence de `trl` avec
le flag `assistant_only_loss` sur `SFTConfig`, atomicité des tokens de
contrôle ChatML (`<|im_start|>`, `<|im_end|>`) pour le modèle cible, et
chargement effectif en 4-bit NF4 :

```bash
uv run python scripts/check_env_gpu.py --model Qwen/Qwen3-1.7B-Base
```

Journal attendu (exemple illustratif : jamais exécuté, aucun GPU
disponible en Environnement A) :

```
================================================================================
VERIFICATION DE L'ENVIRONNEMENT GPU (Environnement B - Cloud)
================================================================================
  CUDA disponible.......: True                 [OK]
  GPU detecte...........: NVIDIA A10G          [OK]
  VRAM totale...........: 24.0 Go               [OK]
  Version trl...........: 0.9.x                [OK]
  assistant_only_loss...: disponible            [OK]
  Chat template (Qwen/Qwen3-1.7B-Base)...: present [OK]
  Token <|im_start|>    : atomique              [OK]
  Token <|im_end|>      : atomique              [OK]
  Token EOS.............: <|im_end|>            [OK]
  Chargement 4-bit (Qwen/Qwen3-1.7B-Base)...: reussi [OK]
================================================================================
Environnement GPU pret pour lancer SFTTrainer / DPOTrainer.
```

**Décision (écart identifié dans `check_env_gpu.py` lui-même)** :
l'argument `--model` de ce script a pour défaut `Qwen/Qwen3-1.7B` (le
modèle **instruct**), alors que le cahier des charges (§6) impose
explicitement `Qwen3-1.7B-Base` comme point de départ du SFT : passer
`--model Qwen/Qwen3-1.7B-Base` explicitement (comme dans l'exemple
ci-dessus) est nécessaire tant que le défaut n'est pas corrigé. Ce
correctif est un changement de code d'une ligne, hors périmètre de ce
document purement documentaire ; à traiter au moment où
`training/sft_train.py` est effectivement écrit (`03_guide_implementation_pas_a_pas.md`).

**Décision (vérification manquante à ajouter, pas encore couverte par
`check_env_gpu.py`)** : les quatre vérifications existantes confirment
que `assistant_only_loss` est un flag *disponible* sur `SFTConfig`,
pas qu'il fonctionne *correctement* avec la chat template exacte du
modèle cible : ce mécanisme s'appuie sur la capacité du tokenizer à
distinguer, token par token, ce qui appartient à un tour `assistant`
(typiquement via un support de type "masque de génération" dans le
gabarit Jinja de la chat template). Proposition : ajouter une
cinquième vérification (`verifier_assistant_only_loss_reel`) qui
applique la chat template sur un exemple à deux tours et contrôle que
le masque produit correspond bien aux tokens du tour `assistant`
attendu, plutôt que de supposer que la présence du flag suffit. À
écrire en même temps que `training/sft_train.py` (voir
`03_guide_implementation_pas_a_pas.md`), pas avant : le test n'a de
sens qu'une fois un GPU/tokenizer réel disponible pour le vérifier.

## 6. Fichier de recette (`recipes/`)

`docs/01_environnement/00_guide_installation_environnement.md` §4
prévoit déjà un dossier `recipes/` pour les fichiers YAML de
configuration d'entraînement (pas encore créé). Schéma proposé pour
`recipes/sft_qwen3_lora.yaml` : regroupe tout ce qui doit être
reproductible d'un run à l'autre (NF3, cahier des charges) dans un
fichier versionné plutôt que dans des arguments CLI dispersés :

```yaml
modele_base: Qwen/Qwen3-1.7B-Base

quantification:
  bits: 4
  type_quantification: nf4
  double_quantification: true
  dtype_calcul: bfloat16

lora:
  rang: 16
  alpha: 32
  dropout: 0.05
  modules_cibles: [q_proj, k_proj, v_proj, o_proj]

entrainement:
  taux_apprentissage: 2.0e-4
  nombre_epoques: 3
  taille_lot: 4
  packing: true
  type_perte: chunked_nll
  assistant_only_loss: true

grille_hyperparametres:
  # Grid restreint (Optuna explicitement écarté pour ce POC, cf.
  # roadmap et 02_etapes_cas_usage.md) : utilisé seulement si la
  # première passe ne converge pas proprement.
  taux_apprentissage: [1.0e-4, 2.0e-4, 5.0e-4]
  rang: [8, 16, 32]

suivi:
  backend: mlflow   # ou tensorboard
  nom_experience: chsa-triage-sft
```

Les valeurs numériques ci-dessus (`rang=16`, `alpha=32`,
`taux_apprentissage=2e-4`) sont des points de départ usuels pour un
QLoRA sur un modèle de cette taille, **pas des valeurs validées sur ce
projet** : elles sont le point d'entrée de la grille d'ajustement
décrite en `02_etapes_cas_usage.md`, à ajuster une fois un premier run
réel observé.

## 7. Checklist de démarrage (Étape 2)

- [ ] `docs/01_environnement/00_guide_installation_environnement.md`
      suivi jusqu'au bout (Environnement A **et** B).
- [ ] Session Spaces Dev Mode GPU active, `uv sync --extra remote`
      exécuté dedans.
- [ ] `HF_TOKEN` valide dans `.env` de la session Dev Mode.
- [ ] `scripts/check_env_gpu.py --model Qwen/Qwen3-1.7B-Base` passe
      (les 4 vérifications existantes, cf. §5).
- [ ] Dataset d'entraînement disponible dans l'Environnement B : soit
      `data/splits/*.jsonl` transférés manuellement, soit le dataset
      HF Hub une fois le Livrable 1 poussé (§4).
- [ ] `recipes/sft_qwen3_lora.yaml` créé (§6).
- [ ] `MLFLOW_TRACKING_URI` pointant vers la même instance
      qu'en Environnement A.

## Document suivant

`02_etapes_cas_usage.md` : les cas d'usage proposés pour orchestrer
chaque étape du pipeline SFT, et les décisions de conception encore
ouvertes.
