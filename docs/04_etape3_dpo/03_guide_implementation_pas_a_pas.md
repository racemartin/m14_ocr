\newpage

# Étape 3 : Guide d'implémentation pas à pas

> **Statut réel (19/09/2026) : rien n'est écrit.** Vérifié par grep
> réel sur `src/chsa_triage/`, `training/` et `recipes/` avant
> d'écrire ce document (pas supposé) : aucun des fichiers listés
> ci-dessous n'existe (`domain/model/exemple_formate_preference.py`,
> `domain/model/preference_reformulee.py`,
> `domain/ports/entraineur_preference.py`,
> `domain/ports/formateur_preference.py`,
> `application/validation_reformulation_dpo.py`, aucun `E3_NN_uc_*`,
> aucun `trl_dpo_entraineur.py`, aucun `training/E3_0X_dpo_train.py`,
> aucun `recipes/dpo_qwen3_lora.yaml`) ; seule exception, une édition
> mineure d'un fichier Étape 2 déjà écrit
> (`domain/model/checkpoint_entraine.py`, cf. étape 5 ci-dessous, type-hint
> encore `HyperparametresEntrainement` seul, pas encore élargi). Les 13
> étapes ci-dessous sont donc toutes **[CONCEPTION]** : ce document ne
> tranche aucune nouvelle décision (elles sont déjà prises dans
> `00_introduction_concepts.md` et `02_etapes_cas_usage.md`, lus
> entiers avant d'écrire celui-ci), il se contente de les **séquencer**
> dans l'ordre d'implémentation réel, sur le modèle de
> `docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md`.

## Principe directeur

Même principe que l'Étape 2 (`docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md`,
"Principe directeur") : ordre `domain` → `ports` → `application` →
`infrastructure` → `interfaces/cli`/`training`, chaque couche testée
avant d'attaquer la suivante (`tests/domain/`, `tests/application/`
avec de faux adaptateurs en mémoire, `tests/infrastructure/` en
dernier, contre les vraies bibliothèques). La même contrainte
Environnement A / Environnement B s'applique, avec une différence
notable par rapport au SFT : l'Étape 3 a **deux** points qui exigent
un GPU réel plutôt qu'un seul.

- **Environnement A (sans GPU)** : domaine, ports, fonction pure de
  validation, cas d'usage avec faux adaptateurs, et une des deux
  méthodes de `ChatMLFormateurAdapter` (`formater_preference()`,
  tokenizer seul, comme `formater()`/`formater_invite_zero_shot()` en
  Étape 2).
- **Environnement B (GPU requis)** : `ReformulerPreferenceDpoUseCase`
  en exécution réelle (le `MoteurInference` sous-jacent,
  `TransformersLoraInferenceAdapter`, charge un modèle 1.7B en
  mémoire GPU, même contrainte que l'évaluation post-SFT) **et**
  `TrlDpoEntraineurAdapter`/`training/E3_0X_dpo_train.py` (l'entraînement
  DPO lui-même). Écrire et valider tout ce qui peut l'être en
  Environnement A avant de consommer du temps GPU facturé reste la
  même discipline de facturation que
  `docs/01_environnement/00_guide_installation_environnement.md` §3.4
  et que le principe directeur du document SFT.

## Ordre suivi

### 1. [CONCEPTION] `domain/model/exemple_formate_preference.py` : `ExempleFormatePreference`

Dataclass pure (`frozen=True, slots=True`), même famille que
`domain/model/exemple_formate.py` (Étape 2), mais avec un triplet de
champs plutôt qu'un texte unique (décision tranchée en
`00_introduction_concepts.md` §4.3, esquisse reprise telle quelle) :
`identifiant`, `texte_prompt`, `texte_chosen`, `texte_rejected`. Aucune
dépendance externe. **Validation** : test unitaire trivial
(construction, égalité) dans
`tests/domain/test_exemple_formate_preference.py`, même patron que
`tests/domain/test_exemple_formate.py`, aucun adaptateur nécessaire.

### 2. [CONCEPTION] `domain/model/preference_reformulee.py` : `ChosenReformule`

Dataclass pure (`frozen=True, slots=True`), esquisse reprise telle
quelle de `02_etapes_cas_usage.md` §1.5 : `identifiant`,
`chosen_reformule: tuple[Message, ...]`, `horodatage`. Importe `Message`
depuis `domain.model.exemple_pivot` (import par sous-module direct,
jamais `from chsa_triage.domain.model import Message`, cf. AGENTS.md
sur la règle anti-cycle déjà posée en Étape 2). **Validation** : test
unitaire de (dé)sérialisation JSON dans
`tests/domain/test_preference_reformulee.py`, sur le modèle de
`tests/domain/test_exemple_pivot.py`/`test_checkpoint_entraine.py`.

### 3. [CONCEPTION] `domain/ports/entraineur_preference.py` : `EntraineurPreference`, `ResultatEntrainementDPO`

`Protocol`, même patron que `domain/ports/entraineur_supervise.py`
(Étape 2). Esquisse déjà posée en `00_introduction_concepts.md` §4.2 :
`entrainer(dataset_train: Iterable[ExempleFormatePreference],
dataset_validation: Iterable[ExempleFormatePreference], config_lora:
ConfigurationLora, hyperparametres: HyperparametresEntrainementDpo,
chemin_checkpoint_politique_depart: str) -> ResultatEntrainementDPO`.
`ResultatEntrainementDPO` réutilise `MetriquesEntrainement` telle
quelle (`chemin_checkpoint`, `courbe_metriques:
tuple[MetriquesEntrainement, ...]`), décision déjà actée en
`00_introduction_concepts.md` §4.4 (aucune variante DPO de cette
dataclass). Importe les types `domain/model` par sous-module direct,
jamais via le package `domain.model` agrégé (même règle anti-cycle
qu'à l'étape 2 ci-dessus ; c'est exactement le type de violation qui a
provoqué le crash circulaire documenté en AGENTS.md lors du câblage de
`CheckpointEntraine`/`MetriquesEntrainement`).
`HyperparametresEntrainementDpo` (dataclass figée, mêmes champs que
`entrainement:` dans l'esquisse YAML de `00_introduction_concepts.md`
§5 : `beta`, `taux_apprentissage`, `nombre_epoques`, `taille_lot`,
`type_perte`, `precompute_ref_log_probs`) est définie dans
`domain/model/configuration_entrainement.py`, à côté de
`HyperparametresEntrainement` (SFT), pas dans ce fichier de port.
**Validation** : aucune (un `Protocol` seul ne s'exécute pas), comme
en Étape 2 : la signature est validée indirectement à l'étape 8
ci-dessous, où un faux adaptateur l'implémente pour de vrai.

### 4. [CONCEPTION] `domain/ports/formateur_preference.py` : `FormateurPreference`

`Protocol` à une seule méthode, esquisse reprise telle quelle de
`02_etapes_cas_usage.md` §3 : `formater(self, exemple: ExemplePivot) ->
ExempleFormatePreference`. Même remarque de non-exécutabilité que
l'étape 3 ci-dessus. **Validation** : aucune directement ; validée
indirectement à l'étape 9 (nouvelle méthode de
`ChatMLFormateurAdapter`) et à l'étape 7 (cas d'usage avec faux
adaptateur).

### 5. [CONCEPTION] Réserve de typage : `domain/model/checkpoint_entraine.py`

Édition mineure d'un fichier Étape 2 déjà **[FAIT]**, pas une nouvelle
entité : `CheckpointEntraine.hyperparametres` (ligne 43) et le
paramètre `hyperparametres` de
`SauvegarderCheckpointSftUseCase.executer()` passent de
`HyperparametresEntrainement` à `HyperparametresEntrainement |
HyperparametresEntrainementDpo`, et le docstring du module (ligne 1-8,
actuellement "checkpoint SFT-LoRA produit par EntrainerSftUseCase")
est généralisé au-delà de "SFT", même réserve honnête déjà notée en
`02_etapes_cas_usage.md` §7. Aucun renommage de
`SauvegarderCheckpointSftUseCase` (même décision déjà prise ailleurs
dans le projet de ne pas renommer une classe existante pour un gain
cosmétique seul, réutilisée ici par analogie, cf.
`02_etapes_cas_usage.md` §7). Dépend de l'étape 3 ci-dessus
(`HyperparametresEntrainementDpo` doit exister avant que cette union de
types soit valide). **Validation** : test de non-régression dans
`tests/domain/test_checkpoint_entraine.py` (un `CheckpointEntraine`
construit avec `HyperparametresEntrainementDpo` se sérialise/désérialise
correctement, en plus du cas SFT déjà couvert).

### 6. [CONCEPTION] `application/validation_reformulation_dpo.py` : `parser_reformulation_stricte`

Fonction pure, aucun port, même famille que
`application/verdict_convergence.py`/`application/detection_pii_residuelle.py`
(Étape 1/2) : esquisse déjà posée en `02_etapes_cas_usage.md` §1.4,
`parser_reformulation_stricte(texte: str) -> tuple[Message, ...] |
None`. Extrait le bloc `<think>...</think>`, parse le reste comme JSON
strict avec exactement les clés `niveau`/`categorie`/`ressources_estimees`
(cahier des charges F3-F4), retourne `None` (jamais une exception) dès
que le format n'est pas respecté. **Validation** : tests unitaires sur
des chaînes construites à la main (JSON valide, JSON invalide, clé
manquante, clé en trop, bloc `<think>` absent, bloc `<think>` vide),
sur le modèle de `tests/application/test_detection_pii_residuelle.py` :
aucune dépendance à un GPU ni à `MoteurInference`.

### 7. [CONCEPTION] `application/use_cases/E3_00_uc_reformuler_preference_dpo.py` : `ReformulerPreferenceDpoUseCase`

Consomme `MoteurInference` (réutilisé, aucun nouveau port, décision
actée en `02_etapes_cas_usage.md` §1.2) et deux instances du port
générique `RepositoryLectureEcriture` (lecture du pivot DPO anonymisé,
écriture des `ChosenReformule`). `executer(self, source_dpo:
Iterable[ExemplePivot]) -> int` filtre sur `type_exemple ==
TypeExemple.DPO`, exclut les identifiants déjà présents dans le fichier
de sortie (`identifiants_existants()`, patron incrémental/resumable
déjà utilisé par `AnonymiserDatasetUseCase --limite`, cf. AGENTS.md),
appelle `self.moteur.generer(messages)` puis
`parser_reformulation_stricte(reponse.texte)` (étape 6) pour chaque
candidat restant jusqu'à `taille_cible`, compte les échecs
(`nombre_echecs_reformulation`) sans jamais écrire un exemple partiel,
et persiste les `ChosenReformule` valides en un seul
`sauvegarder_plusieurs()` (jamais un `sauvegarder()` par item, cf.
AGENTS.md, coût O(n²)). Fichier de sortie proposé (§1.5 de
`02_etapes_cas_usage.md`) : `data/processed/dataset_dpo_chosen_reformule.jsonl`.
Testable entièrement avec un **faux** `MoteurInference` en mémoire
(scripté pour retourner tantôt un JSON valide, tantôt un format
dégénéré) : aucun GPU nécessaire pour valider la logique
d'orchestration/exclusion/comptage elle-même, seule l'exécution *réelle*
(étape 12 ci-dessous) exige un GPU. **Validation** :
`tests/application/test_E3_00_uc_reformuler_preference_dpo.py`, faux
`MoteurInference`/`RepositoryLectureEcriture` définis directement dans
le fichier de test, même patron que
`tests/application/test_E2_01_uc_entrainer_sft.py`.

### 8. [CONCEPTION] `application/use_cases/E3_01_uc_formater_dataset_chatml_preference.py` : `FormaterDatasetChatMLPreferenceUseCase`

Consomme `FormateurPreference` (étape 4) et deux instances du port
générique `RepositoryLectureEcriture`, même patron que
`E2_00_uc_formater_dataset_chatml.py`, avec une différence
structurelle documentée en `02_etapes_cas_usage.md` §3 : ce cas
d'usage **fusionne** deux sources en mémoire avant de formater
(`ExemplePivot` d'origine + `ChosenReformule` correspondant s'il
existe, via `dataclasses.replace(exemple, chosen=chosen_reformule.chosen_reformule)`),
jamais de mutation d'aucune des deux sources sur disque. Un exemple
sans `ChosenReformule` correspondant est exclu (hors sous-ensemble
reformulé, ou reformulation en échec pour cet identifiant), même
logique d'exclusion que `E1_05_00_decouper_splits.py` pour la PII
résiduelle en attente (cf. AGENTS.md). Filtre sur `type_exemple ==
TypeExemple.DPO` (jamais SFT, même bug de fuite déjà documenté et
corrigé en Étape 1, AGENTS.md "SFT/DPO type leak"). Testable
entièrement avec un faux `FormateurPreference` en mémoire, sans
tokenizer réel. **Validation** :
`tests/application/test_E3_01_uc_formater_dataset_chatml_preference.py`,
même patron que `tests/application/test_E2_00_uc_formater_dataset_chatml.py`,
avec un cas de test dédié à l'exclusion des exemples sans
`ChosenReformule`.

### 9. [CONCEPTION] `application/use_cases/E3_02_uc_entrainer_dpo.py` : `EntrainerDpoUseCase`

Même structure d'orchestration que `EntrainerSftUseCase` (Étape 2),
adaptée au port `EntraineurPreference` (étape 3) :
`entrainer(dataset_train, dataset_validation, config_lora,
hyperparametres, chemin_checkpoint_politique_depart,
nom_run=NOM_RUN_PAR_DEFAUT)` appelle
`SuiviExperimentation.demarrer_run(...)`, délègue à
`EntraineurPreference.entrainer(...)`, relaie chaque point de
`ResultatEntrainementDPO.courbe_metriques` (`MetriquesEntrainement`
réutilisée telle quelle) vers `SuiviExperimentation.logger_metrique(...)`,
puis `terminer_run()`. Différence à journaliser en plus (§4 de
`02_etapes_cas_usage.md`) : les quatre métriques `trl.DPOTrainer` sans
équivalent SFT (`rewards/chosen`, `rewards/rejected`,
`rewards/accuracies`, `rewards/margins`) sont relayées directement vers
`SuiviExperimentation.logger_metrique(...)`, en plus de
`perte_train`/`perte_validation`/`norme_gradient`, sans passer par
`evaluer_convergence()` (extension additive de la boucle de suivi,
décision déjà actée en `00_introduction_concepts.md` §4.4). Testable
entièrement avec un faux `EntraineurPreference` en mémoire (scripté
pour retourner des courbes `MetriquesEntrainement` synthétiques),
exactement comme `EntrainerSftUseCase` avant l'écriture de
`TrlSftEntraineurAdapter` en Étape 2. **Validation** :
`tests/application/test_E3_02_uc_entrainer_dpo.py`, faux
`EntraineurPreference`/`SuiviExperimentation` définis dans le fichier
de test, même patron que `tests/application/test_E2_01_uc_entrainer_sft.py`
(y compris un cas de test qui vérifie explicitement que les 4 métriques
de récompense sont bien relayées).

### 10. [CONCEPTION] `infrastructure/adapters/chatml_formateur_adapter.py` : nouvelle méthode `formater_preference()`

Édition d'un fichier Étape 2 déjà **[FAIT]**, pas un nouvel adaptateur :
une nouvelle méthode `formater_preference(self, exemple: ExemplePivot)
-> ExempleFormatePreference` ajoutée à `ChatMLFormateurAdapter`, à côté
de `formater()`/`formater_invite_zero_shot()` déjà écrites (lignes 40 et
57 du fichier actuel), réutilisant le même tokenizer déjà chargé
paresseusement (même précédent que la coexistence de ces deux méthodes
sur le même adaptateur, cf. AGENTS.md "note du 15/09/2026"). Détail
exact du gabarit de rendu (`texte_prompt` vraisemblablement identique au
rendu de `formater_invite_zero_shot()`, `add_generation_prompt=True` ;
`texte_chosen`/`texte_rejected` rendant chacun le tour assistant seul,
sans `system` ni `user`) reste, comme en `02_etapes_cas_usage.md` §3,
un détail différé à l'implémentation réelle : c'est l'un des deux
points explicitement laissés ouverts par `00_introduction_concepts.md`
("Point de vigilance", mapping `Message` → colonnes `trl`), à vérifier
contre un vrai appel `DPOTrainer.train()` plutôt qu'à figer ici.
**Testable en Environnement A** (tokenizer seul, pas de GPU), même
statut que `formater()`/`formater_invite_zero_shot()`. **Validation** :
test d'intégration réel (pas mocké) ajouté à
`tests/infrastructure/test_chatml_formateur_adapter.py`, qui vérifie
concrètement que `texte_chosen` contient bien le format `<think>`+JSON
attendu sur un exemple reformulé construit à la main, et que
`texte_prompt` ne contient jamais le tour assistant. Auto-ignoré si
`transformers` n'est pas installé ou si le token HF n'est pas
configuré, même garde que le reste du fichier.

### 11. [CONCEPTION] Réutilisations sans nouveau code, à documenter comme étape

Trois réutilisations déjà décidées, aucune ligne de code
supplémentaire au-delà de ce qui existe déjà, listées ici uniquement
pour que l'ordre d'implémentation reste complet :

- **`TransformersLoraInferenceAdapter`** (déjà écrit, Étape 1bis) comme
  `MoteurInference` injecté dans `ReformulerPreferenceDpoUseCase`
  (étape 7), pointé vers `mombasstic/chsa-triage-sft-lora` : quatrième
  réutilisation de ce port dans le projet, aucune modification de
  l'adaptateur (`02_etapes_cas_usage.md` §1.2).
- **`SauvegarderCheckpointSftUseCase`/`CheckpointEntraine`** (déjà
  écrits, Étape 2, après l'édition mineure de l'étape 5 ci-dessus) pour
  persister les métadonnées d'un checkpoint DPO dans le même
  `data/processed/checkpoints_sft.jsonl`, distingué des checkpoints SFT
  par `configuration_lora`/`hyperparametres`/`modele_base`, jamais un
  fichier séparé (`02_etapes_cas_usage.md` §7).
- **`SuiviExperimentation`/adaptateurs déjà écrits**
  (`MlflowSuiviExperimentation`, `TensorboardSuiviExperimentation`,
  `HfDatasetSuiviExperimentation`) injectés dans `EntrainerDpoUseCase`
  (étape 9), aucune connaissance du backend concret requise côté cas
  d'usage, même principe d'injection qu'en Étape 2
  (`02_etapes_cas_usage.md` §8).

**Validation** : aucune nouvelle, ces trois éléments restent couverts
par leurs tests Étape 1bis/Étape 2 déjà existants.

### 12. [CONCEPTION] `infrastructure/adapters/trl_dpo_entraineur.py` : `TrlDpoEntraineurAdapter`

**Nécessite un GPU réel (Environnement B)**, même statut que
`trl_sft_entraineur.py` avant son propre premier run (Étape 2, étape
9). Seul point du projet qui importe `trl.DPOTrainer`/`trl.DPOConfig`,
même règle "seule `infrastructure/adapters` importe des bibliothèques
externes" déjà respectée par `TrlSftEntraineurAdapter`. À la
différence de `TrlSftEntraineurAdapter`, le constructeur prend un
paramètre supplémentaire, `chemin_checkpoint_politique_depart` (le
dépôt LoRA-SFT, `mombasstic/chsa-triage-sft-lora` par défaut), chargé
une fois comme point de départ de `π_θ`. **Décisions à trancher à
l'écriture, pas ici** (`02_etapes_cas_usage.md` §2) :
- Recharger une seconde copie gelée du même base+LoRA-SFT comme
  `π_ref`, ou déléguer à `trl` (`ref_model=None` dans `trl.DPOConfig`,
  qui dérive `π_ref` en interne à partir de `π_θ` avant le premier
  pas) : laissé ouvert par `00_introduction_concepts.md` §0/§2.
- `precompute_ref_log_probs` (défaut réel vérifié `False`, cf.
  `00_introduction_concepts.md` §5) : à mesurer une fois un GPU réel
  disponible, même prudence que les optimisations Unsloth/Liger/
  FlashAttention-2 non validées en Étape 2.

**Validation** : `tests/infrastructure/test_trl_dpo_entraineur.py`,
auto-ignoré si `torch.cuda.is_available()` est faux, même principe que
`test_trl_sft_entraineur.py` : un run minimal réel serait quelques
dizaines de pas sur un sous-échantillon de 50-100 exemples déjà
reformulés (étape 7), suffisant pour confirmer que la perte décroît et
qu'aucune erreur CUDA ne survient, sans consommer un run complet.

### 13. [CONCEPTION] `training/E3_0X_dpo_train.py` + `recipes/dpo_qwen3_lora.yaml`

**Environnement B**, dépend entièrement de l'étape 12. Point d'entrée,
sur le modèle argparse + `LogTool` + résumé console de
`training/E2_04_sft_train.py` : charge la recette YAML (esquisse déjà
posée en `00_introduction_concepts.md` §5), construit les adaptateurs
par injection de dépendances, enchaîne dans l'ordre :
`ReformulerPreferenceDpoUseCase` (étape 7, si le sous-ensemble
reformulé cible n'est pas déjà complet, même patron incrémental que
l'anonymisation) → `FormaterDatasetChatMLPreferenceUseCase` (étape 8)
→ `EntrainerDpoUseCase` (étape 9) → sauvegarde du checkpoint (étape 5
et 11). Nom exact du fichier (`E3_0X`) à fixer au moment de l'écriture,
convention `AGENTS.md` déjà établie (numérotation `E3_NN` des cas
d'usage réservée aux vraies classes de cas d'usage ; ce point d'entrée
suit plutôt le précédent de nommage de `training/E2_04_sft_train.py`,
qui n'est pas non plus un `E2_NN_uc_*`). Reprend **telles quelles** les
deux gardes de démarrage déjà réelles dans
`training/E2_04_sft_train.py` (`_verifier_type_perte_valide`,
`_verifier_suivi_hf_repo_coherent`, lignes 238 et 264 du fichier
actuel), plutôt que de les redécouvrir : le second guard en particulier
(cohérence `--suivi-hf-repo`/`suivi.backend`) s'applique à l'identique
à un futur job DPO sur HF Jobs, même disque éphémère, même risque de
perte de courbe que celui documenté en AGENTS.md pour le premier run
SFT payant. `recipes/dpo_qwen3_lora.yaml` fixe déjà `suivi.backend:
hf_dataset` par défaut dans son esquisse (`00_introduction_concepts.md`
§5), cohérent avec cette découverte réelle. **Validation finale** :
exécution réelle de bout en bout en Environnement B sur un petit
sous-échantillon (les 50-100 exemples de l'étape 12), avec
`scripts/check_env_gpu.py` passé en amont comme porte d'entrée, pas de
run complet tant que ce test réduit n'a pas produit une courbe de perte
qui décroît, même discipline que la validation finale du document SFT.

## Table récapitulative

| Ordre | Fichier | Couche | Statut | Environnement | Validation |
|---|---|---|---|---|---|
| 1 | `domain/model/exemple_formate_preference.py` | domain | [CONCEPTION] | A | `tests/domain/test_exemple_formate_preference.py` |
| 2 | `domain/model/preference_reformulee.py` | domain | [CONCEPTION] | A | `tests/domain/test_preference_reformulee.py` |
| 3 | `domain/ports/entraineur_preference.py` (+ `HyperparametresEntrainementDpo`) | domain | [CONCEPTION] | A | (aucune directe, validée indirectement à l'étape 8) |
| 4 | `domain/ports/formateur_preference.py` | domain | [CONCEPTION] | A | (aucune directe, validée indirectement aux étapes 7/9) |
| 5 | `domain/model/checkpoint_entraine.py` (édition, union de types) | domain | [CONCEPTION] | A | `tests/domain/test_checkpoint_entraine.py` (régression) |
| 6 | `application/validation_reformulation_dpo.py` | application | [CONCEPTION] | A | tests unitaires dédiés, aucun adaptateur |
| 7 | `application/use_cases/E3_00_uc_reformuler_preference_dpo.py` | application | [CONCEPTION] | A (logique) / B (exécution réelle) | `tests/application/test_E3_00_uc_reformuler_preference_dpo.py` (faux adaptateurs) |
| 8 | `application/use_cases/E3_01_uc_formater_dataset_chatml_preference.py` | application | [CONCEPTION] | A | `tests/application/test_E3_01_uc_formater_dataset_chatml_preference.py` (faux adaptateurs) |
| 9 | `application/use_cases/E3_02_uc_entrainer_dpo.py` | application | [CONCEPTION] | A (logique) / B (exécution réelle) | `tests/application/test_E3_02_uc_entrainer_dpo.py` (faux adaptateurs) |
| 10 | `infrastructure/adapters/chatml_formateur_adapter.py` (nouvelle méthode) | infrastructure | [CONCEPTION] | **A** (tokenizer seul) | `tests/infrastructure/test_chatml_formateur_adapter.py` (cas ajouté) |
| 11 | Réutilisations sans code neuf (`TransformersLoraInferenceAdapter`, `SauvegarderCheckpointSftUseCase`, `SuiviExperimentation`) | infrastructure | [CONCEPTION] | A/B selon l'adaptateur | (déjà couvertes par les tests Étape 1bis/2 existants) |
| 12 | `infrastructure/adapters/trl_dpo_entraineur.py` | infrastructure | [CONCEPTION] | **B** | `tests/infrastructure/test_trl_dpo_entraineur.py`, GPU requis |
| 13 | `training/E3_0X_dpo_train.py`, `recipes/dpo_qwen3_lora.yaml` | training (point d'entrée) | [CONCEPTION] | **B** (dépend de 12) | exécution réelle réduite, Environnement B |

Sur les 13 étapes, 11 (1 à 11) sont entièrement conçevables et
testables en **Environnement A**, sans jamais ouvrir de session GPU
facturée (la logique d'orchestration des étapes 7 et 9 y compris,
via de faux adaptateurs, seule leur exécution *réelle* attend un GPU).
Seules les étapes 12 et 13 exigent l'**Environnement B** : une de plus
qu'en Étape 2 (qui n'en comptait qu'une, `TrlSftEntraineurAdapter`),
parce que l'Étape 3 introduit un deuxième point de contact GPU
(`ReformulerPreferenceDpoUseCase` en exécution réelle, via
`TransformersLoraInferenceAdapter`) avant même d'atteindre
l'entraînement DPO proprement dit.

## Décisions encore ouvertes avant de continuer

Récapitulatif des points signalés dans les deux documents précédents,
toujours ouverts après ce guide de séquencement : à trancher pendant
l'écriture des étapes concernées, pas des détails à découvrir en cours
de route sans repère.

1. **Quel prompt exact de reformulation** (`PROMPT_REFORMULATION_CHOSEN`,
   étape 7) : laissé ouvert par `00_introduction_concepts.md`,
   "Point de vigilance".
2. **Mapping exact `Message` → colonnes `prompt`/`chosen`/`rejected`**
   au niveau du rendu texte (étape 10, `formater_preference()`) :
   esquissé par analogie avec `formater_invite_zero_shot()`, jamais
   vérifié contre un vrai appel `DPOTrainer.train()`, second point
   laissé ouvert par `00_introduction_concepts.md`, "Point de
   vigilance".
3. **`π_ref` : seconde copie gelée rechargée, ou `ref_model=None`
   délégué à `trl`** (étape 12) : laissé ouvert en
   `00_introduction_concepts.md` §0/§2 et repris en
   `02_etapes_cas_usage.md` §2.
4. **`precompute_ref_log_probs`** (étape 12) : `False` par défaut dans
   l'esquisse de recette, à mesurer une fois un GPU réel disponible
   (`00_introduction_concepts.md` §5).
5. **Nom exact de `training/E3_0X_dpo_train.py`** (étape 13) : `0X` non
   fixé, à choisir en cohérence avec la numérotation `E3_NN` des cas
   d'usage au moment de l'écriture.
6. **Grille d'hyperparamètres DPO** : explicitement hors périmètre pour
   l'instant (`02_etapes_cas_usage.md` §6) : prématurée avant
   d'observer une vraie courbe DPO, comme les seuils de
   `evaluer_convergence()` l'étaient avant le premier run SFT réel.
   Si elle devient nécessaire, elle nécessiterait une classe parallèle
   à `AjusterBoucleHyperparametresSftUseCase` (typée sur
   `HyperparametresEntrainementDpo`/`EntrainerDpoUseCase`/
   `ExempleFormatePreference`), pas une généralisation de la classe
   SFT existante (contrainte déjà documentée, cf.
   `02_etapes_cas_usage.md` §6).

## Document précédent / document suivant

Documents de cette étape, dans l'ordre de lecture :
`00_introduction_concepts.md` → `02_etapes_cas_usage.md` → diagrammes
(`docs/diagrams/04_etape3_dpo/`, seul `dpo_double_fonction_entrainement.puml`
existe à ce jour) → ce document. La suite directe est l'écriture des
étapes 1 à 13 ci-dessus, dans l'ordre indiqué, avec la même discipline
qu'en Étape 2 : valider entièrement l'Environnement A (étapes 1 à 11)
avant d'ouvrir la moindre session GPU facturée pour les étapes 12-13,
qui sortent du périmètre de cette mise à jour purement documentaire.
