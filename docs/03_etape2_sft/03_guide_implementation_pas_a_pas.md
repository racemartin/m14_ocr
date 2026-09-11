\newpage

# Étape 2 : Guide d'implémentation pas à pas

> Document de planification : décrit l'ordre et la méthode
> d'implémentation proposés, pas du code. Aucun fichier listé
> ci-dessous n'existe dans le dépôt au moment de l'écriture de ce
> guide.

## Principe directeur

L'Étape 1 a été construite dans l'ordre `domain` → `ports` →
`application` → `infrastructure` → `interfaces/cli`, chaque couche
testée avant d'attaquer la suivante (`tests/domain/`,
`tests/application/` avec de faux adaptateurs en mémoire,
`tests/infrastructure/` en dernier, contre les vraies bibliothèques).
Ce même ordre est proposé pour l'Étape 2, avec une contrainte
supplémentaire propre à cette étape : une partie du code peut être
écrite et testée en **Environnement A** (sans GPU), et une partie
nécessite impérativement l'**Environnement B**. Écrire et valider tout
ce qui peut l'être en Environnement A avant de consommer du temps GPU
facturé limite le coût réel de l'itération : c'est la même discipline
de facturation que `docs/01_environnement/00_guide_installation_environnement.md`
§3.4 demande déjà pour l'usage de Dev Mode.

## Ordre proposé

### 1. `domain/model/exemple_formate.py` : `ExempleFormate`

Dataclass pure (`identifiant`, `texte`), sans dépendance externe,
même règle que le reste de `domain/model`. **Validation** : test
unitaire trivial (construction, égalité) dans
`tests/domain/test_exemple_formate.py`, aucun adaptateur nécessaire.

### 2. `domain/model/configuration_entrainement.py` : `ConfigurationQuantification`, `ConfigurationLora`, `HyperparametresEntrainement`

Trois dataclasses figées (`frozen=True, slots=True`, comme
`ConstantesVitales`). **Décision de conception à prendre ici** :
valider que les champs proposés en
`01_installation_configuration.md` §6 (`rang`, `alpha`, `dropout`,
`modules_cibles` pour LoRA ; `bits`, `type_quantification`,
`double_quantification`, `dtype_calcul` pour la quantification)
couvrent tout ce que `peft.LoraConfig`/`transformers.BitsAndBytesConfig`
exigent en pratique : à confirmer en lisant leur signature réelle au
moment de l'implémentation, pas supposé ici. **Validation** : test
unitaire (les valeurs de `recipes/sft_qwen3_lora.yaml` se
désérialisent correctement vers ces dataclasses).

### 3. `domain/model/checkpoint_entraine.py` : `CheckpointEntraine`, `VerdictConvergence`

Cf. schéma JSON proposé en `02_etapes_cas_usage.md` §6. **Validation** :
test unitaire de (dé)sérialisation JSON, sur le modèle de
`tests/domain/test_exemple_pivot.py`.

### 4. `domain/ports/formateur_conversation.py`, `entraineur_supervise.py`, `suivi_experimentation.py`

Trois `Protocol`, sans implémentation. **Validation** : aucune
(un `Protocol` seul ne s'exécute pas) : leur signature est validée
indirectement à l'étape 6 ci-dessous, quand un faux adaptateur doit
les implémenter pour de vrai.

### 5. `application/verdict_convergence.py`, `application/grille_hyperparametres.py`

Fonctions pures, aucun port. **Validation** : tests unitaires sur des
courbes `MetriquesEntrainement` synthétiques construites à la main
(perte qui diverge, perte qui stagne, perte qui converge proprement) :
sur le modèle de `tests/application/test_detection_pii_residuelle.py`
et `tests/application/test_rapport_anonymisation.py`, aucune
dépendance à un GPU ni à une vraie bibliothèque d'entraînement.
**Décision à prendre ici** : fixer le seuil numérique exact séparant
`SAINE` de `SURAPPRENTISSAGE` (ex. delta de perte validation sur les N
derniers pas de log) : un choix arbitraire tant qu'aucune vraie courbe
n'a été observée ; documenter ce seuil comme provisoire dans le code
plutôt que de le présenter comme calibré.

### 6. `application/use_cases/uc_05_00_formater_dataset_chatml.py` à `uc_05_03_sauvegarder_checkpoint_sft.py`

Écrits et testés avec de **faux adaptateurs en mémoire**, définis
directement dans les fichiers de test (`FauxFormateurConversation`,
`FauxEntraineurSupervise`, `FauxSuiviExperimentation`) : exactement le
patron déjà utilisé par
`tests/application/test_anonymiser_dataset.py` (`FauxRepository`,
`FauxAnonymiseur`), pas de nouveaux adaptateurs de production juste
pour les tests. Ceci permet de valider toute la logique
d'orchestration (dont la boucle d'ajustement d'hyperparamètres,
`AjusterBoucleHyperparametresSftUseCase`) **sans jamais toucher un
GPU** : le faux `EntraineurSupervise` peut retourner des courbes
`MetriquesEntrainement` scriptées (converge au 2ᵉ essai, n'atteint
jamais la convergence, etc.) pour exercer chaque branche de la boucle.
**Validation** : `tests/application/test_entrainer_sft.py` et
`tests/application/test_ajuster_boucle_hyperparametres_sft.py`
(nommage sur le modèle de `test_anonymiser_dataset.py`).

### 7. `infrastructure/adapters/chatml_formateur_adapter.py` : `ChatMLFormateurAdapter`

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

### 8. `infrastructure/adapters/mlflow_suivi_experimentation.py`, `tensorboard_suivi_experimentation.py`

`mlflow`/`tensorboard` sont déjà dans l'extra `local` (pas seulement
`remote`) : ces deux adaptateurs sont également testables en
Environnement A, contre une instance MLflow locale ou un répertoire de
logs TensorBoard temporaire. **Validation** :
`tests/infrastructure/test_mlflow_suivi_experimentation.py`
(`mlflow.set_tracking_uri` vers un dossier temporaire, vérifier que
les métriques loggées sont bien relisibles ensuite).

### 9. `infrastructure/adapters/trl_sft_entraineur.py` : `TrlSftEntraineurAdapter`

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

### 10. `training/sft_train.py` + `recipes/sft_qwen3_lora.yaml`

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

| Ordre | Fichier | Couche | Testable sans GPU | Validation |
|---|---|---|---|---|
| 1 | `domain/model/exemple_formate.py` | domain | oui | unitaire |
| 2 | `domain/model/configuration_entrainement.py` | domain | oui | unitaire |
| 3 | `domain/model/checkpoint_entraine.py` | domain | oui | unitaire |
| 4 | `domain/ports/*.py` (3 fichiers) | domain | oui (aucun test direct) | (aucune) |
| 5 | `application/verdict_convergence.py`, `grille_hyperparametres.py` | application | oui | unitaire, courbes synthétiques |
| 6 | `application/use_cases/uc_05_00..03_*.py` | application | oui (faux adaptateurs) | unitaire avec faux adaptateurs |
| 7 | `infrastructure/adapters/chatml_formateur_adapter.py` | infrastructure | **oui** (tokenizer seul) | intégration réelle, Environnement A |
| 8 | `infrastructure/adapters/mlflow_suivi_experimentation.py` / `tensorboard_...` | infrastructure | **oui** (déjà dans l'extra `local`) | intégration réelle, Environnement A |
| 9 | `infrastructure/adapters/trl_sft_entraineur.py` | infrastructure | non | intégration réelle, Environnement B, GPU requis |
| 10 | `training/sft_train.py`, `recipes/sft_qwen3_lora.yaml` | training (point d'entrée) | non (dépend de 9) | exécution réelle réduite, Environnement B |

Sept des dix éléments (1 à 8) sont donc écrivables et testables sans
jamais ouvrir de session GPU facturée : seuls les deux derniers
exigent l'Environnement B.

## Décisions encore ouvertes avant de commencer

Récapitulatif des points signalés dans les documents précédents, à
trancher avant (ou pendant) l'implémentation réelle, pas des détails à
découvrir en cours de route :

1. **Format de sortie cible absent des données actuelles**
   (`<think>` + JSON strict, cahier des charges F3-F4) : trois options
   posées en `02_etapes_cas_usage.md` §2, aucune retenue.
2. **Source du dataset d'entraînement** : HF Hub (Livrable 1, pas
   encore poussé) ou `data/splits/*.jsonl` transférés manuellement :
   `01_installation_configuration.md` §4.
3. **Seuil numérique de convergence** (`evaluer_convergence`) :
   provisoire tant qu'aucune vraie courbe n'a été observée, §5
   ci-dessus.
4. **`--model` par défaut de `scripts/check_env_gpu.py`** pointe vers
   la variante instruct (`Qwen/Qwen3-1.7B`) plutôt que
   `Qwen3-1.7B-Base` : correctif d'une ligne, à faire au moment
   d'écrire `training/sft_train.py` (`01_installation_configuration.md`
   §5).
5. **Vérification `assistant_only_loss` réelle** (pas seulement la
   présence du flag) : vérification supplémentaire proposée pour
   `check_env_gpu.py`, `01_installation_configuration.md` §5.

## Document précédent / document suivant

Ce guide clôt la planification de l'Étape 2. Documents de cette
étape, dans l'ordre de lecture : `00_introduction_concepts.md` →
`01_installation_configuration.md` → `02_etapes_cas_usage.md` →
diagrammes (`docs/diagrams/03_etape2_sft/`) → ce document.

La suite (Étape 3 : DPO, `docs/04_etape3_dpo/`) n'est pas planifiée
par ces documents : elle dépend directement du checkpoint SFT-LoRA
produit ici (cf. `CheckpointEntraine`, §6 de
`02_etapes_cas_usage.md`) et des décisions encore ouvertes listées
ci-dessus.
