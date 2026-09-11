\newpage

# Étape 2 : Guide d'implémentation pas à pas

> Les étapes 1 à 8 ci-dessous sont écrites et testées (221 tests en
> vert au moment de l'écriture de cette mise à jour) : ce document
> décrit désormais l'ordre réellement suivi, pas seulement un ordre
> proposé. Seules les étapes 9 (`TrlSftEntraineurAdapter`) et 10
> (`training/sft_train.py` + `recipes/sft_qwen3_lora.yaml`) restent
> **[CONCEPTION]** : elles nécessitent un GPU réel (Environnement B),
> absent au moment de l'écriture.

## Principe directeur

L'Étape 1 a été construite dans l'ordre `domain` → `ports` →
`application` → `infrastructure` → `interfaces/cli`, chaque couche
testée avant d'attaquer la suivante (`tests/domain/`,
`tests/application/` avec de faux adaptateurs en mémoire,
`tests/infrastructure/` en dernier, contre les vraies bibliothèques).
Ce même ordre a été suivi pour l'Étape 2, avec une contrainte
supplémentaire propre à cette étape : une partie du code peut être
écrite et testée en **Environnement A** (sans GPU), et une partie
nécessite impérativement l'**Environnement B**. Écrire et valider tout
ce qui peut l'être en Environnement A avant de consommer du temps GPU
facturé limite le coût réel de l'itération : c'est la même discipline
de facturation que `docs/01_environnement/00_guide_installation_environnement.md`
§3.4 demande déjà pour l'usage de Dev Mode.

## Ordre suivi

### 1. [FAIT] `domain/model/exemple_formate.py` : `ExempleFormate`

Dataclass pure (`identifiant`, `texte`), sans dépendance externe,
même règle que le reste de `domain/model`. **Validation** : test
unitaire trivial (construction, égalité) dans
`tests/domain/test_exemple_formate.py`, aucun adaptateur nécessaire.

### 2. [FAIT] `domain/model/configuration_entrainement.py` : `ConfigurationQuantification`, `ConfigurationLora`, `HyperparametresEntrainement`

Trois dataclasses figées (`frozen=True, slots=True`, comme
`ConstantesVitales`). Les champs proposés en
`01_installation_configuration.md` §6 (`rang`, `alpha`, `dropout`,
`modules_cibles` pour LoRA ; `bits`, `type_quantification`,
`double_quantification`, `dtype_calcul` pour la quantification) ont
été **vérifiés (pas supposés)** contre les signatures réelles de
`peft.LoraConfig` (0.20.0) et `transformers.BitsAndBytesConfig`
(4.57.6) : ils couvrent tout ce que ces deux classes exigent en
pratique, tous les autres paramètres ayant une valeur par défaut
raisonnable pour un QLoRA NF4 standard (cf. le docstring du module).
`HyperparametresEntrainement` ne couvre volontairement pas
`assistant_only_loss` (présent dans le YAML sous `entrainement:`) :
ce flag pilote `trl.SFTConfig`, pas `transformers.TrainingArguments`,
sa vérification réelle est différée à l'écriture de
`infrastructure.adapters.trl_sft_entraineur` (étape 9). **Validation** :
test unitaire (les valeurs de `recipes/sft_qwen3_lora.yaml` se
désérialisent correctement vers ces dataclasses).

### 3. [FAIT] `domain/model/checkpoint_entraine.py` : `CheckpointEntraine`, `VerdictConvergence`

Cf. schéma JSON en `02_etapes_cas_usage.md` §6. **Validation** :
test unitaire de (dé)sérialisation JSON, sur le modèle de
`tests/domain/test_exemple_pivot.py`.

### 4. [FAIT] `domain/ports/formateur_conversation.py`, `entraineur_supervise.py`, `suivi_experimentation.py`

Trois `Protocol`, sans implémentation. **Validation** : aucune
(un `Protocol` seul ne s'exécute pas) : leur signature est validée
indirectement à l'étape 6 ci-dessous, où de faux adaptateurs les
implémentent pour de vrai.

### 5. [FAIT] `application/verdict_convergence.py`, `application/grille_hyperparametres.py`

Fonctions pures, aucun port. **Validation** : tests unitaires sur des
courbes `MetriquesEntrainement` synthétiques construites à la main
(perte qui diverge, perte qui stagne, perte qui converge proprement) :
sur le modèle de `tests/application/test_detection_pii_residuelle.py`
et `tests/application/test_rapport_anonymisation.py`, aucune
dépendance à un GPU ni à une vraie bibliothèque d'entraînement.
Les quatre seuils numériques séparant `SAINE` de
`SURAPPRENTISSAGE`/`SOUS_APPRENTISSAGE`/`INSTABLE` sont documentés
dans le module comme des **paramètres à calibrer**, pas des valeurs
mesurées : un choix provisoire tant qu'aucune vraie courbe n'a été
observée (`TrlSftEntraineurAdapter` n'existe pas encore), à ajuster
une fois un premier run réel disponible plutôt qu'à présenter comme
calibré.

### 6. [FAIT] `application/use_cases/E2_00_uc_formater_dataset_chatml.py` à `E2_03_uc_sauvegarder_checkpoint_sft.py`

Écrits et testés avec de **faux adaptateurs en mémoire**, définis
directement dans les fichiers de test (`FauxFormateurConversation`,
`FauxEntraineurSupervise`, `FauxSuiviExperimentation`) : exactement le
patron déjà utilisé par
`tests/application/test_anonymiser_dataset.py` (`FauxRepository`,
`FauxAnonymiseur`), pas de nouveaux adaptateurs de production juste
pour les tests. Ceci permet de valider toute la logique
d'orchestration (dont la boucle d'ajustement d'hyperparamètres,
`AjusterBoucleHyperparametresSftUseCase`) **sans jamais toucher un
GPU** : le faux `EntraineurSupervise` retourne des courbes
`MetriquesEntrainement` scriptées (converge au 2ᵉ essai, n'atteint
jamais la convergence, etc.) pour exercer chaque branche de la boucle.
**Validation** : `tests/application/test_E2_01_uc_entrainer_sft.py` et
`tests/application/test_E2_02_uc_ajuster_boucle_hyperparametres_sft.py`
(nommage sur le modèle de `test_anonymiser_dataset.py`).

### 7. [FAIT] `infrastructure/adapters/chatml_formateur_adapter.py` : `ChatMLFormateurAdapter`

Premier adaptateur réel de cette étape, et le seul testable en
intégration dès l'**Environnement A** : `AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")`
et `apply_chat_template` ne nécessitent pas de GPU, seulement le
tokenizer (léger, quelques Mo) : `transformers` est déjà dans l'extra
`local` de `pyproject.toml`. **Validation** : test d'intégration réel
(pas mocké) dans `tests/infrastructure/test_chatml_formateur_adapter.py`,
qui vérifie concrètement, sur un `ExemplePivot` construit à la main,
que le texte rendu contient bien `<|im_start|>assistant` et que les
tokens de contrôle restent atomiques (même logique que
`scripts/check_env_gpu.py::verifier_chat_template`, réutilisable ici).
Auto-ignoré si `transformers` n'est pas installé ou si le token HF
n'est pas configuré, sur le modèle de
`tests/infrastructure/test_presidio_anonymiseur.py`.

### 8. [FAIT] `infrastructure/adapters/mlflow_suivi_experimentation.py`, `tensorboard_suivi_experimentation.py`

`mlflow`/`tensorboard` sont déjà dans l'extra `local` (pas seulement
`remote`) : ces deux adaptateurs sont également testables en
Environnement A, contre une instance MLflow locale ou un répertoire de
logs TensorBoard temporaire. **Validation** :
`tests/infrastructure/test_mlflow_suivi_experimentation.py`
(`mlflow.set_tracking_uri` vers un dossier temporaire, vérifier que
les métriques loggées sont bien relisibles ensuite).

### 9. [CONCEPTION] `infrastructure/adapters/trl_sft_entraineur.py` : `TrlSftEntraineurAdapter`

Seul module qui nécessite réellement l'Environnement B (GPU). À
n'écrire qu'une fois les étapes 1 à 8 validées : c'est la seule partie
qui consomme du temps GPU facturé pour être testée.
**Décisions à trancher à l'écriture** :
- Quelles optimisations (Unsloth / Liger Kernel / FlashAttention-2)
  activer par défaut, lesquelles rendre optionnelles : à mesurer
  empiriquement sur `Qwen3-1.7B-Base` (temps par epoch, VRAM de pic),
  pas à décider a priori.
- Comment câbler `assistant_only_loss` avec la chat template réelle du
  modèle (cf. vérification proposée en
  `01_installation_configuration.md` §5) avant de faire confiance aux
  courbes de perte produites.

**Validation** : `tests/infrastructure/test_trl_sft_entraineur.py`,
auto-ignoré si `torch.cuda.is_available()` est faux (même principe que
les tests Presidio/spaCy, adapté au GPU plutôt qu'à un modèle spaCy) :
un run minimal réel serait quelques dizaines de pas sur un
sous-échantillon de 50-100 exemples, suffisant pour confirmer que la
perte décroît et qu'aucune erreur CUDA ne survient, sans consommer un
run complet.

### 10. [CONCEPTION] `training/sft_train.py` + `recipes/sft_qwen3_lora.yaml`

Point d'entrée, sur le modèle argparse + `LogTool` + résumé console de
`interfaces/cli/E1_04_00_anonymiser_dataset.py` : charge la recette YAML,
construit les adaptateurs (dependency injection, comme dans tous les
scripts `interfaces/cli/` existants), enchaîne les quatre cas d'usage
dans l'ordre du diagramme d'activité
(`docs/diagrams/03_etape2_sft/activite/pipeline_sft_lora.puml`).
**Validation finale** : exécution réelle de bout en bout en
Environnement B sur un petit sous-échantillon, avec
`scripts/check_env_gpu.py` passé en amont comme porte d'entrée
(cf. `01_installation_configuration.md` §5) : pas de run complet tant
que ce test réduit n'a pas produit une courbe de perte qui décroît.

## Table récapitulative

| Ordre | Fichier | Couche | Statut | Testable sans GPU | Validation |
|---|---|---|---|---|---|
| 1 | `domain/model/exemple_formate.py` | domain | [FAIT] | oui | `tests/domain/test_exemple_formate.py` |
| 2 | `domain/model/configuration_entrainement.py` | domain | [FAIT] | oui | `tests/domain/test_configuration_entrainement.py` |
| 3 | `domain/model/checkpoint_entraine.py` | domain | [FAIT] | oui | `tests/domain/test_checkpoint_entraine.py` |
| 4 | `domain/ports/*.py` (3 fichiers) | domain | [FAIT] | oui (aucun test direct) | (aucune, validée indirectement à l'étape 6) |
| 5 | `application/verdict_convergence.py`, `grille_hyperparametres.py` | application | [FAIT] | oui | `tests/application/test_verdict_convergence.py`, `test_grille_hyperparametres.py` |
| 6 | `application/use_cases/E2_00..03_uc_*.py` | application | [FAIT] | oui (faux adaptateurs) | `tests/application/test_E2_00..03_uc_*.py` |
| 7 | `infrastructure/adapters/chatml_formateur_adapter.py` | infrastructure | [FAIT] | **oui** (tokenizer seul) | `tests/infrastructure/test_chatml_formateur_adapter.py`, intégration réelle, Environnement A |
| 8 | `infrastructure/adapters/mlflow_suivi_experimentation.py` / `tensorboard_...` | infrastructure | [FAIT] | **oui** (déjà dans l'extra `local`) | `tests/infrastructure/test_mlflow_suivi_experimentation.py`, `test_tensorboard_suivi_experimentation.py`, intégration réelle, Environnement A |
| 9 | `infrastructure/adapters/trl_sft_entraineur.py` | infrastructure | [CONCEPTION] | non | intégration réelle, Environnement B, GPU requis |
| 10 | `training/sft_train.py`, `recipes/sft_qwen3_lora.yaml` | training (point d'entrée) | [CONCEPTION] | non (dépend de 9) | exécution réelle réduite, Environnement B |

Huit des dix éléments (1 à 8) sont donc écrits et testés sans jamais
ouvrir de session GPU facturée : seuls les deux derniers restent à
écrire et exigent l'Environnement B, seule partie non couverte par les
221 tests actuels.

## Décisions encore ouvertes avant de continuer

Récapitulatif des points signalés dans les documents précédents,
toujours ouverts après les étapes 1 à 8 : à trancher avant (ou
pendant) l'écriture des étapes 9-10, pas des détails à découvrir en
cours de route :

1. **Format de sortie cible absent des données actuelles**
   (`<think>` + JSON strict, cahier des charges F3-F4) : trois options
   posées en `02_etapes_cas_usage.md` §2, aucune retenue ;
   `ChatMLFormateurAdapter` (étape 7) ne les tranche pas non plus, il
   rend tel quel le contenu `prompt`/`completion` existant.
2. **Source du dataset d'entraînement** : HF Hub (Livrable 1, pas
   encore poussé) ou `data/splits/*.jsonl` transférés manuellement :
   `01_installation_configuration.md` §4.
3. **Seuils numériques de convergence** (`evaluer_convergence`) :
   documentés comme provisoires dans le code tant qu'aucune vraie
   courbe n'a été observée, §5 ci-dessus.
4. **`--model` par défaut de `scripts/check_env_gpu.py`** pointe vers
   la variante instruct (`Qwen/Qwen3-1.7B`) plutôt que
   `Qwen3-1.7B-Base` : correctif d'une ligne, toujours pas fait, à
   faire au moment d'écrire `training/sft_train.py`
   (`01_installation_configuration.md` §5).
5. **Vérification `assistant_only_loss` réelle** (pas seulement la
   présence du flag) : vérification supplémentaire proposée pour
   `check_env_gpu.py`, toujours pas écrite, `01_installation_configuration.md` §5.

## Document précédent / document suivant

Ce guide clôt la documentation des trois phases sans GPU de l'Étape 2
(étapes 1 à 8, [FAIT]). Documents de cette étape, dans l'ordre de
lecture : `00_introduction_concepts.md` →
`01_installation_configuration.md` → `02_etapes_cas_usage.md` →
diagrammes (`docs/diagrams/03_etape2_sft/`) → ce document. La suite
directe est l'écriture des étapes 9-10 ci-dessus
(`TrlSftEntraineurAdapter` puis `training/sft_train.py`), qui
nécessite un GPU réel (Environnement B) et sort du périmètre de cette
mise à jour purement documentaire.

La suite (Étape 3 : DPO, `docs/04_etape3_dpo/`) n'est pas planifiée
par ces documents : elle dépend directement du checkpoint SFT-LoRA
produit ici (cf. `CheckpointEntraine`, §6 de
`02_etapes_cas_usage.md`) et des décisions encore ouvertes listées
ci-dessus.
