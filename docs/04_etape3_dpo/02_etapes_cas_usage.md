\newpage

# Étape 3 : Étapes du pipeline et cas d'usage (DPO)

> **Statut réel (18/09/2026) : rien n'est encore écrit pour l'Étape 3.**
> Contrairement à `docs/03_etape2_sft/02_etapes_cas_usage.md` (où la
> majorité des cas d'usage étaient déjà **[FAIT]** au moment de la
> rédaction), toutes les lignes de ce document sont **[CONCEPTION]** :
> ni `domain/ports/entraineur_preference.py`, ni
> `domain/model/exemple_formate_preference.py`, ni aucun fichier
> `E3_NN_uc_*` n'existent à ce jour (vérifié par grep réel sur
> `src/chsa_triage/`). Ce document traduit les décisions déjà tranchées
> dans `00_introduction_concepts.md` (lu entier avant d'écrire celui-ci,
> à ne pas relire ici : uniquement référencé) en classes/cas d'usage
> concrets, plus **une décision supplémentaire tranchée ici pour la
> première fois** (§1 : qui/comment reformule le `chosen`, laissée
> explicitement ouverte par `00_introduction_concepts.md`, "Point de
> vigilance"). Comme pour l'Étape 2 avant l'écriture du code, c'est un
> document de conception, pas un guide d'implémentation pas-à-pas.

## Vue d'ensemble

| Étape du roadmap | Cas d'usage / module | Port(s) | Adaptateur(s) | Statut |
|---|---|---|---|---|
| Reformuler `chosen` vers `<think>`+JSON sur un sous-ensemble | `E3_00_uc_reformuler_preference_dpo.py` → `ReformulerPreferenceDpoUseCase` | `MoteurInference` (réutilisé), `RepositoryLectureEcriture` (×2) | `TransformersLoraInferenceAdapter` (réutilisé), `JsonlDatasetRepository`/adaptateur JSONL dédié (réutilisé) | [CONCEPTION] |
| Charger `Qwen3-1.7B-Base` + LoRA-SFT comme `π_θ`/`π_ref` | *(pas de cas d'usage dédié : voir §2 Décision)* | `EntraineurPreference` | `TrlDpoEntraineurAdapter` (à écrire, charge les modèles dans son constructeur) | [CONCEPTION] |
| Rendre `prompt`/`chosen`/`rejected` en triplet texte | `E3_01_uc_formater_dataset_chatml_preference.py` → `FormaterDatasetChatMLPreferenceUseCase` | `FormateurPreference`, `RepositoryLectureEcriture` (×2) | `ChatMLFormateurAdapter` (réutilisé, nouvelle méthode), `JsonlDatasetRepository`/adaptateur JSONL dédié (réutilisé) | [CONCEPTION] |
| Lancer `DPOTrainer` | `E3_02_uc_entrainer_dpo.py` → `EntrainerDpoUseCase` | `EntraineurPreference`, `SuiviExperimentation` | `TrlDpoEntraineurAdapter` (à écrire), `HfDatasetSuiviExperimentation`/`MlflowSuiviExperimentation` (réutilisés) | [CONCEPTION] |
| Évaluer sur validation intermédiaire | fonction pure `application/verdict_convergence.py::evaluer_convergence` | (aucun, réutilisé tel quel) | (aucun) | [FAIT] côté Étape 2, réutilisé sans modification (§5) |
| Boucle d'ajustement d'hyperparamètres DPO | *(pas de cas d'usage pour l'instant : voir §6 Décision)* | (aucun) | (aucun) | [HORS PÉRIMÈTRE] pour l'instant |
| Sauvegarder le checkpoint DPO | *(pas de nouveau cas d'usage : voir §7 Décision)* | `RepositoryLectureEcriture` | `JsonlDatasetRepository`/adaptateur JSONL checkpoint (réutilisés) | [CONCEPTION], dépend d'un petit ajustement de typage hérité de l'Étape 2 |

Diagramme d'activité existant : `docs/diagrams/04_etape3_dpo/activite/dpo_double_fonction_entrainement.puml`
(écrit AVANT le code, marqué `[CONCEPTUEL/PRÉVU]`, illustre la double
fonction d'une seule passe DPO référencée en §3-§4 ci-dessous). Le code
étant désormais réellement écrit et testé (13 étapes, 428 tests, cf.
AGENTS.md), les diagrammes de séquence, de paquets et de déploiement de
l'Étape 3 existent aussi (20/09/2026, `docs/diagrams/04_etape3_dpo/{sequence,paquets,deploiement}/`),
même patron que l'Étape 2 : voir `docs/diagrams/README.md` pour l'index
complet et l'état de couverture à jour.

## 1. Décision (nouvelle, tranchée ici) : `E3_00_uc_reformuler_preference_dpo.py` → `ReformulerPreferenceDpoUseCase`

### 1.1. La question laissée ouverte, et la décision prise

`00_introduction_concepts.md` §3.2 décide **quoi** faire (reformuler
`chosen` vers `<think>`+JSON) mais laisse explicitement ouvert **qui**
reformule ("Point de vigilance", dernier paragraphe). Décision tranchée
ici, à consigner pour la première fois :
**le checkpoint SFT-LoRA déjà entraîné** (`mombasstic/chsa-triage-sft-lora`,
verdict `SAINE`, cf. §0 de `00_introduction_concepts.md`), en mode
inférence, via un prompt de reformulation dédié.

Trois options étaient sur la table, écartées explicitement :

- **Une API externe payante** (un modèle plus grand type GPT-4/Claude) :
  écartée pour coût et nouvelle dépendance externe, hors du budget NF5
  (cahier des charges) d'un POC de 4 semaines qui a déjà consommé son
  budget GPU sur le SFT et prévoit d'en consommer davantage sur le DPO
  lui-même (§8, "S3").
- **Des règles programmatiques déterministes** (ex. heuristiques
  regex/mots-clés pour dériver `niveau`/`categorie` du texte `chosen`) :
  écartées pour le même risque déjà identifié honnêtement dans
  `00_introduction_concepts.md` §3.2 ("inventer une classification
  clinique sans fondement réel dans le texte") : une règle
  déterministe n'a aucune compréhension du contenu clinique, elle ne
  ferait que déplacer ce risque d'un prompt de reformulation vers un
  ensemble de règles tout aussi arbitraire, sans le bénéfice de pouvoir
  s'appuyer sur un modèle déjà exposé au vocabulaire médical du corpus
  (le SFT).
- **Le modèle de base nu** (`Qwen3-1.7B-Base` sans LoRA), suggéré comme
  option secondaire par `00_introduction_concepts.md` §"Point de
  vigilance" ("ou `Qwen3-1.7B-Base` lui-même en few-shot") : écarté au
  profit du checkpoint SFT-LoRA parce que ce dernier a déjà, par
  construction, vu le vocabulaire et le format du domaine pendant le
  SFT (verdict `SAINE` déjà obtenu) ; le modèle de base nu n'aurait
  aucun avantage particulier ici et forcerait un prompt few-shot plus
  long pour compenser, pour un gain incertain.

**Pourquoi le checkpoint SFT-LoRA l'emporte** : c'est la ressource la
moins chère disponible (aucune nouvelle dépendance, aucun nouveau coût
d'API, le poids est déjà publié et réutilisable tel quel), et c'est
déjà, très concrètement, un modèle qui *connaît* le corpus médical de
ce projet (le SFT l'a entraîné dessus). Ce n'est pas prétendu être une
garantie de justesse clinique (la limite honnête de `00_introduction_concepts.md`
§3.2 reste entière : le prompt de reformulation doit projeter le
contenu de `chosen` vers le schéma cible, jamais l'inventer sans lien
avec le texte source), seulement le choix le plus cohérent avec le
budget et les ressources déjà en place.

### 1.2. Port : réutilisation de `MoteurInference`, pas de nouveau port

**Décision** : `ReformulerPreferenceDpoUseCase` consomme le port
`MoteurInference` déjà écrit (`domain/ports/moteur_inference.py`,
`generer(messages: list[dict], parametres: dict | None) -> ReponseModele`),
sans nouveau port dédié type `ReformulateurPreference`.

**Pourquoi, et pourquoi ce n'est pas la même situation que §4.2 de
`00_introduction_concepts.md`** (qui, lui, justifie un nouveau port,
`EntraineurPreference`, plutôt que de réutiliser `EntraineurSupervise`) :
la distinction posée dans ce document était structurelle, la forme de
donnée diffère (un texte concaténé vs. un triplet `prompt`/`chosen`/`rejected`
distinct). Ici, il n'y a **aucune** différence structurelle : la
reformulation est, au sens le plus strict, "envoyer des messages,
recevoir une réponse", exactement le contrat déjà exposé par
`MoteurInference.generer()`. C'est d'ailleurs déjà la **quatrième**
réutilisation de ce même port dans ce projet, avec un prompt/mode
différent à chaque fois, jamais une modification du port lui-même :
`LlamaCppInferenceAdapter` (baseline zero-shot CPU/GGUF),
`TransformersInferenceAdapter` (baseline zero-shot GPU/bf16),
`TransformersLoraInferenceAdapter` (évaluation post-SFT), et ici un
quatrième usage du **même** `TransformersLoraInferenceAdapter` déjà
écrit (cf. `infrastructure/adapters/transformers_lora_inference_adapter.py`,
lu directement avant d'écrire cette section), pointé vers le même
dépôt `mombasstic/chsa-triage-sft-lora`. Créer un port
`ReformulateurPreference` séparé n'ajouterait aucune capacité : ce
serait un simple alias d'un sous-ensemble de `MoteurInference.generer()`,
l'exact type d'abstraction prématurée que ce projet évite ailleurs
(cf. AGENTS.md, "pas d'abstraction au-delà de ce que la tâche exige").

Concrètement, `ReformulerPreferenceDpoUseCase` appelle :

```python
messages = [
    {"role": "system", "content": PROMPT_REFORMULATION_CHOSEN},
    {"role": "user", "content": texte_chosen_original},  # rendu du tuple Message d'origine
]
reponse = self.moteur.generer(messages)  # invite_deja_rendue absent : mode par defaut
```

Même mode par défaut (`invite_deja_rendue` absent/`False`) que
l'évaluation post-SFT : le tokenizer applique son propre chat template
via `apply_chat_template(..., add_generation_prompt=True)`, cohérent
avec le fait qu'on demande ici une **génération** (une reformulation),
pas la relecture d'un texte déjà entièrement rendu (`PROMPT_REFORMULATION_CHOSEN`,
le texte exact du prompt, est un détail d'implémentation différé,
comme le "quel prompt exact" resté ouvert par `00_introduction_concepts.md`).

### 1.3. Sur quel sous-ensemble, et pourquoi cette taille

**Décision** : reformuler un sous-ensemble de l'ordre de **quelques
milliers d'exemples**, pas les 97 081 exemples DPO disponibles.
Concrètement, ce document propose d'aligner ce sous-ensemble sur celui
**déjà** produit par `E1_05_03_extraire_sous_ensemble_dpo.py`
(`ExtraireSousEnsembleDpoUseCase`, déjà écrit, `taille_cible: int = 5000`
par défaut, cf. `src/chsa_triage/application/use_cases/E1_05_03_extraire_sous_ensemble_dpo.py`),
plutôt qu'un chiffre choisi indépendamment. Raisonnement :

- Cette taille (~5000) est **déjà** la décision de publication retenue
  pour le sous-ensemble DPO (même ordre de grandeur que le Livrable 1
  SFT, cahier des charges §7, "SFT ~5000 paires + DPO"), déjà
  implémentée et testée. Réutiliser exactement ce sous-ensemble pour la
  reformulation évite de faire deux choix de dimensionnement
  indépendants (un pour "quoi publier", un pour "quoi reformuler") qui
  pourraient diverger sans raison ; reformuler autre chose que ce qui
  sera effectivement entraîné/publié serait du travail perdu.
- Le calendrier de l'Étape 3 est d'une semaine (cahier des charges §8,
  "S3") : reformuler l'intégralité des 97 081 exemples via inférence
  GPU (même ordre de grandeur de coût que l'anonymisation Presidio en
  Étape 1, ~19h de calcul pour un traitement champ par champ complet,
  cf. AGENTS.md) serait un chantier d'échelle séparé, hors de ce budget.
- Après répartition train/val/test (§5 de `00_introduction_concepts.md`,
  proportions ~80/10/10 déjà observées ailleurs dans le projet), ~5000
  exemples donnent un ordre de grandeur d'environ 4000 exemples train +
  500 val, suffisant pour observer une première courbe DPO réelle sans
  consommer un budget GPU disproportionné par rapport au SFT (342 pas,
  ~20 minutes de calcul réel, cf. AGENTS.md).

**Honnêteté explicite à documenter, même patron que partout ailleurs
dans ce projet** : ce sous-ensemble reformulé est **partiel**, pas le
corpus DPO complet. Un futur passage pourrait élargir la reformulation
(relancer `ReformulerPreferenceDpoUseCase` sur un `taille_cible` plus
grand, §1.5) exactement comme `E1_05_00_decouper_splits.py --n` est
déjà conçu comme une cible cumulative extensible plutôt qu'un tirage
figé une fois pour toutes (cf. AGENTS.md, "Croissance stable"). Ce
document ne prétend pas que 5000 exemples couvrent la diversité clinique
du corpus source (`UltraMedical-Preference`, ~109k exemples avant
dédoublonnage) : c'est un point de départ suffisant pour un premier
entraînement DPO réel dans le calendrier du POC, pas une couverture
revendiquée.

### 1.4. Validation stricte, et le patron "fail loudly" déjà établi dans ce projet

**Décision** : chaque sortie de reformulation est parsée comme JSON
**strict** ; tout échec de parsing ou de schéma **écarte** l'exemple du
sous-ensemble reformulé (il n'est jamais écrit tel quel, ni "réparé"
silencieusement). C'est le même principe déjà appliqué dans plusieurs
gardes-fous réels de ce projet, deux formes différentes selon le
niveau où l'échec se produit :

- **Au niveau d'un `Iterable` de N exemples** (échec ponctuel, pas
  systémique) : `EvaluerBaselineZeroShotUseCase.executer()` (Étape
  1bis) enveloppe chaque appel `MoteurInference.generer()` dans un
  `except Exception` par exemple, compte les échecs
  (`nombre_echecs_inference`) et continue plutôt que d'interrompre tout
  le run (cf. AGENTS.md, "`EvaluerBaselineZeroShotUseCase.executer()`
  n'aborte plus tout le run sur un exemple dégénéré"). C'est le patron
  le plus proche de ce dont `ReformulerPreferenceDpoUseCase` a besoin :
  un échec de reformulation sur UN exemple (JSON malformé, bloc
  `<think>` absent, clé manquante) ne doit jamais faire perdre tout le
  lot déjà reformulé.
- **Au niveau du démarrage d'un run entier** (précondition globale,
  jamais un cas par cas) : `training/E2_04_sft_train.py::_verifier_type_perte_valide`/
  `_verifier_suivi_hf_repo_coherent` (lus directement avant d'écrire
  cette section) refusent de démarrer tout l'entraînement (`SystemExit`)
  si une configuration est incohérente, **avant** tout chargement de
  modèle/GPU coûteux. Ce n'est **pas** le patron applicable ici (la
  reformulation n'a pas de précondition de configuration globale à
  vérifier avant de commencer), mais il reste la référence citée par la
  tâche pour le principe général "jamais d'échec silencieux" : cité ici
  comme précédent de philosophie, pas réimplémenté.

Concrètement, une fonction pure dédiée (même famille que
`application/verdict_convergence.py`/`application/detection_pii_residuelle.py` :
aucun port, testable sans GPU) :

```python
# application/validation_reformulation_dpo.py (a ecrire)

def parser_reformulation_stricte(texte: str) -> tuple[Message, ...] | None:
    """
    Extrait le bloc <think>...</think> puis parse le reste comme JSON
    strict avec exactement les cles niveau/categorie/ressources_estimees
    (cahier des charges F3-F4). Retourne None (jamais une exception) des
    que le format n'est pas respecte : bloc <think> absent/vide, JSON
    invalide, cle manquante ou en trop. Fonction pure, aucun acces
    reseau/GPU : la separation "generer" (ReformulerPreferenceDpoUseCase,
    via MoteurInference) / "valider" (ici) suit le meme principe que
    detection_pii_residuelle.py, testable independamment de tout modele.
    """
```

`ReformulerPreferenceDpoUseCase.executer()` appelle
`self.moteur.generer(...)` puis `parser_reformulation_stricte(reponse.texte)`
pour chaque exemple du sous-ensemble ; sur `None`, l'exemple est compté
(`nombre_echecs_reformulation`, journalisé, même esprit que
`nombre_echecs_inference`) et **absent** du fichier de sortie (§1.5),
jamais écrit avec un contenu partiel ou invalide.

### 1.5. Entrées/sorties : jamais de mutation du pivot existant

Cohérent avec `00_introduction_concepts.md` §3.2 ("Conséquence sur le
pipeline de données") : la reformulation **ne modifie jamais**
`dataset_pivot.jsonl` ni `dataset_pivot_anonymise.jsonl`. Même principe
déjà en place pour la séparation pivot/anonymisé (AGENTS.md, "Pivot/anonymized-file
split") et pour le formatage ChatML SFT (`ExempleFormate` écrit dans un
fichier séparé, jamais dans le pivot) : le résultat de la reformulation
est un **nouveau type d'entité**, persisté dans un fichier dérivé, via
une **quatrième** instance du port générique `RepositoryLectureEcriture`
(après `ExemplePivot`, `ExempleFormate` et `CheckpointEntraine` en
Étape 2).

```python
# domain/model/preference_reformulee.py (a ecrire)

@dataclass(frozen=True, slots=True)
class ChosenReformule:
    """Chosen reformule vers <think>+JSON pour un ExemplePivot DPO donne."""

    identifiant       : str               # repris de ExemplePivot.identifiant, jamais regenere
    chosen_reformule   : tuple[Message, ...]  # nouveau tour assistant, remplace chosen a l'usage
    horodatage           : str
```

`executer(self, source_dpo: Iterable[ExemplePivot]) -> int` filtre sur
`type_exemple == TypeExemple.DPO`, exclut les identifiants déjà présents
dans le fichier de sortie (`identifiants_existants()`, même patron
incrémental/resumable que `AnonymiserDatasetUseCase --limite`, cf.
AGENTS.md) pour permettre d'élargir le sous-ensemble reformulé par
vagues successives sans jamais retraiter ce qui l'est déjà, appelle
`MoteurInference.generer()` + `parser_reformulation_stricte()` pour
chaque candidat restant (jusqu'à `taille_cible`, §1.3), et persiste les
`ChosenReformule` valides en un seul `sauvegarder_plusieurs()` (jamais
un `sauvegarder()` par item, cf. AGENTS.md sur le coût O(n²)). Fichier
proposé : `data/processed/dataset_dpo_chosen_reformule.jsonl`.

**Sur `rejected`** : aucun traitement, cohérent avec §3.2 de
`00_introduction_concepts.md` ("reformuler `chosen` uniquement... est
délibéré"). `ReformulerPreferenceDpoUseCase` ne lit ni n'écrit jamais
`rejected`.

## 2. Charger `Qwen3-1.7B-Base` + LoRA-SFT comme `π_θ`/`π_ref` : pas de cas d'usage dédié

**Décision de conception**, même raisonnement que
`docs/03_etape2_sft/02_etapes_cas_usage.md` §1 (SFT) : aucun cas
d'usage séparé pour charger les modèles. C'est une responsabilité
interne de `TrlDpoEntraineurAdapter` (à écrire,
`infrastructure/adapters/trl_dpo_entraineur.py`), même précédent que
`MoteurInference`/`EntraineurSupervise`/`PresidioAnonymiseur` déjà cité
dans ce document (§1) et dans le document SFT : le domaine ne représente
jamais un modèle chargé (poids `torch`, tokenizer, wrapper PEFT) comme
une donnée, seul l'adaptateur concret le possède.

**Ce qui diffère du SFT ici** : `TrlDpoEntraineurAdapter.__init__` prend
un paramètre supplémentaire que `TrlSftEntraineurAdapter` n'avait pas,
`chemin_checkpoint_politique_depart` (le dépôt LoRA-SFT, `mombasstic/chsa-triage-sft-lora`
par défaut, cf. `EntraineurPreference.entrainer()` en
`00_introduction_concepts.md` §4.2), chargé une fois comme point de
départ de `π_θ`. Pour `π_ref` : la même esquisse §4.2 laisse ouvert,
comme le fait déjà `00_introduction_concepts.md` §0, le choix entre
recharger une seconde copie gelée du même base+LoRA-SFT, ou déléguer à
`trl` (`ref_model=None` dans `trl.DPOConfig`, qui dérive `π_ref` en
interne à partir de `π_θ` avant le premier pas) : ce choix reste, comme
dans le document précédent, un détail d'implémentation différé à
l'écriture réelle de `TrlDpoEntraineurAdapter`, pas tranché ici.

## 3. `E3_01_uc_formater_dataset_chatml_preference.py` : `FormaterDatasetChatMLPreferenceUseCase`

Traduit `00_introduction_concepts.md` §4.3 (déjà tranché : un type
parallèle `ExempleFormatePreference`, pas une variante d'`ExempleFormate`)
sous la forme "cas d'usage", même patron exact que
`E2_00_uc_formater_dataset_chatml.py` (§2 du document SFT) : lit les
`ExemplePivot` du split demandé, filtrés sur
`type_exemple == TypeExemple.DPO` (jamais SFT, même bug de fuite déjà
documenté et corrigé en Étape 1, cf. AGENTS.md, "SFT/DPO type leak"),
appelle `FormateurPreference.formater(exemple)` pour chacun, persiste
en un seul `sauvegarder_plusieurs()` via une cinquième instance du port
générique (paramétrée sur `ExempleFormatePreference`).

**Différence structurelle avec `FormaterDatasetChatMLUseCase`, à
documenter explicitement** : contrairement au SFT, où `formater()`
reçoit directement l'`ExemplePivot` du pivot anonymisé, ce cas d'usage
doit d'abord **fusionner** deux sources en mémoire, jamais réécrire
aucune des deux sur disque (même principe de non-mutation que §1.5) :

1. le `ExemplePivot` d'origine (`prompt`/`rejected` inchangés) ;
2. le `ChosenReformule` correspondant (§1.5), s'il existe, pour
   remplacer `chosen` par `chosen_reformule` avant de rendre le triplet.

Un exemple **sans** `ChosenReformule` correspondant (hors du
sous-ensemble reformulé de §1.3, ou reformulation en échec pour cet
identifiant précis, §1.4) est **exclu** de ce cas d'usage : il n'entre
pas dans le jeu d'entraînement DPO tant que sa reformulation n'existe
pas, même logique d'exclusion que `E1_05_00_decouper_splits.py`
excluant déjà les identifiants avec PII résiduelle en attente (cf.
AGENTS.md) plutôt qu'un blocage bloquant tout le pipeline. Concrètement,
`executer()` construit l'exemple fusionné via `dataclasses.replace(exemple,
chosen=chosen_reformule.chosen_reformule)` avant d'appeler
`FormateurPreference.formater()`, qui garde ainsi la signature déjà
esquissée en §4.3 de `00_introduction_concepts.md`
(`formater(exemple: ExemplePivot) -> ExempleFormatePreference`), sans
avoir besoin d'un second paramètre : la fusion reste une responsabilité
du cas d'usage, pas du port de formatage.

Port (`domain/ports/formateur_preference.py`, à écrire, esquisse
reprise telle quelle de `00_introduction_concepts.md` §4.3) :

```python
class FormateurPreference(Protocol):
    def formater(self, exemple: ExemplePivot) -> ExempleFormatePreference: ...
```

Adaptateur : `ChatMLFormateurAdapter` (déjà écrit,
`infrastructure/adapters/chatml_formateur_adapter.py`), une nouvelle
méthode `formater_preference()` à ajouter, réutilisant le même
tokenizer déjà chargé paresseusement (même précédent que
`formater()`/`formater_invite_zero_shot()` coexistant sur le même
adaptateur, cf. AGENTS.md, "note du 15/09/2026"). `texte_prompt` suit
vraisemblablement le même rendu que `formater_invite_zero_shot()`
(`add_generation_prompt=True`, prompt seul) ; `texte_chosen`/`texte_rejected`
rendent chacun le tour assistant correspondant seul, sans `system` ni
`user`. Détail exact du gabarit différé à l'implémentation réelle,
cohérent avec le fait que ce mapping `Message` → colonnes `trl` reste
explicitement l'un des deux points laissés ouverts par
`00_introduction_concepts.md` ("Point de vigilance").

## 4. `E3_02_uc_entrainer_dpo.py` : `EntrainerDpoUseCase`

Même structure d'orchestration que `EntrainerSftUseCase`
(`docs/03_etape2_sft/02_etapes_cas_usage.md` §3), adaptée au port
`EntraineurPreference` déjà esquissé en `00_introduction_concepts.md`
§4.2 : `entrainer(dataset_train, dataset_validation, config_lora,
hyperparametres, chemin_checkpoint_politique_depart, nom_run=NOM_RUN_PAR_DEFAUT)`
appelle `SuiviExperimentation.demarrer_run(...)`, délègue à
`EntraineurPreference.entrainer(...)`, relaie chaque point de la courbe
retournée (`ResultatEntrainementDPO.courbe_metriques`, réutilisant
`MetriquesEntrainement` tel quel, §5 ci-dessous) vers
`SuiviExperimentation.logger_metrique(...)`, puis `terminer_run()`.

**Ce qui diffère du SFT, à journaliser en plus** : `trl.DPOTrainer`
journalise des métriques propres au DPO sans équivalent SFT
(`rewards/chosen`, `rewards/rejected`, `rewards/accuracies`,
`rewards/margins`, noms réels de l'API `trl`, cf.
`00_introduction_concepts.md` §4.4). `EntrainerDpoUseCase` les relaie
directement vers `SuiviExperimentation.logger_metrique(...)`, en plus
de `perte_train`/`perte_validation`/`norme_gradient` déjà couverts par
`MetriquesEntrainement`, sans passer par `evaluer_convergence()`
(diagnostic générique, §5) : une extension additive de la boucle de
suivi, décision déjà actée en §4.4 de `00_introduction_concepts.md`,
reprise ici sous forme d'implémentation concrète du cas d'usage.

`TrlDpoEntraineurAdapter` (à écrire,
`infrastructure/adapters/trl_dpo_entraineur.py`) sera le seul point du
projet qui importe `trl.DPOTrainer`/`trl.DPOConfig`, même règle "seule
`infrastructure/adapters` importe des bibliothèques externes" déjà
respectée par `TrlSftEntraineurAdapter`. En attendant, comme pour
`EntrainerSftUseCase` avant l'écriture de son adaptateur GPU,
`EntrainerDpoUseCase` resterait testable avec un faux
`EntraineurPreference` en mémoire : toute la logique d'orchestration
(démarrage/clôture du run, relais des métriques génériques et
spécifiques DPO) serait validée indépendamment du futur adaptateur GPU,
même stratégie de test que l'Étape 2 (cf. AGENTS.md sur
`test_E2_01_uc_entrainer_sft.py`).

## 5. Évaluation de convergence : réutilisée telle quelle, sans modification

Pas de section à écrire ici au sens propre : `00_introduction_concepts.md`
§4.4 tranche déjà que `application/verdict_convergence.py::evaluer_convergence`
est réutilisée **sans aucune modification** pour le DPO, les quatre
verdicts (`SAINE`/`SURAPPRENTISSAGE`/`SOUS_APPRENTISSAGE`/`INSTABLE`)
et leurs quatre seuils provisoires (§4 du document SFT) se diagnostiquant
sur une courbe `MetriquesEntrainement` générique, sans aucune
spécificité DPO. `EntrainerDpoUseCase` (§4 ci-dessus) l'appelle
exactement comme `EntrainerSftUseCase` le fait déjà. Rien de nouveau à
concevoir : mentionné ici uniquement pour la complétude de la
correspondance décision → cas d'usage demandée par ce document.

## 6. Décision : pas de boucle d'ajustement d'hyperparamètres DPO pour l'instant

**Décision** : contrairement au SFT (`AjusterBoucleHyperparametresSftUseCase`,
document SFT §5), aucun cas d'usage `AjusterBoucleHyperparametresDpoUseCase`
n'est proposé ici. Ce n'est pas un oubli : `00_introduction_concepts.md`
§5 ("Pourquoi pas de grille d'hyperparamètres DPO dans ce document")
explique déjà pourquoi trancher les axes concrets d'une grille DPO
(quels `beta`, quels `taux_apprentissage` essayer) est **prématuré**
avant d'avoir observé une vraie courbe DPO, exactement le même
raisonnement que celui qui a gardé les seuils de `evaluer_convergence()`
"provisoires" jusqu'au premier run SFT réel.

**Précision technique, propre à ce document** : même si la décision
d'attendre était prise, une éventuelle boucle DPO ne pourrait de toute
façon **pas** réutiliser `AjusterBoucleHyperparametresSftUseCase`
tel quel. Vérifié par lecture directe du code (pas supposé,
`src/chsa_triage/application/use_cases/E2_02_uc_ajuster_boucle_hyperparametres_sft.py`) :
cette classe est typée concrètement sur `HyperparametresEntrainement`,
`EntrainerSftUseCase` et `ExempleFormate` (pas de paramétrage générique
`Generic[T]`), donc une grille DPO nécessiterait une classe parallèle,
`AjusterBoucleHyperparametresDpoUseCase`, typée sur
`HyperparametresEntrainementDpo`/`EntrainerDpoUseCase`/`ExempleFormatePreference`,
réutilisant telle quelle la fonction pure
`application/grille_hyperparametres.py::candidat_suivant` (déjà
générique sur n'importe quel `Sequence[T]` comparable par égalité, cf.
document SFT §5). Ce point est noté ici pour que l'implémentation
future n'ait pas à re-découvrir cette contrainte, sans pour autant
écrire cette classe maintenant : prématuré tant que la grille elle-même
n'est pas décidée.

## 7. Décision : sauvegarde du checkpoint DPO, réutilisation avec une réserve honnête

**Décision** : pas de nouveau cas d'usage ni de nouvelle dataclass.
`CheckpointEntraine` et `SauvegarderCheckpointSftUseCase` (Étape 2,
déjà écrits) sont réutilisés tels quels pour persister les métadonnées
d'un checkpoint DPO : littéralement la **même** troisième instance du
port générique `RepositoryLectureEcriture` déjà utilisée pour
`CheckpointEntraine` en Étape 2 (après `ExemplePivot` et `ExempleFormate`),
pas une nouvelle instance séparée, contrairement à `ChosenReformule`/
`ExempleFormatePreference` (§1.5, §3 ci-dessus) qui, eux, introduisent
chacun un nouveau type d'entité et donc une nouvelle instance du port.

**Pourquoi la réutilisation l'emporte ici, contrairement à `EntraineurSupervise`/`EntraineurPreference`
(§4.2 de `00_introduction_concepts.md`)** : la distinction qui justifiait
un nouveau port ailleurs dans ce document (§1.2) était une différence
**structurelle** de forme de donnée (triplet vs texte unique). Ici, la
forme est **identique** : un enregistrement de métadonnées plat
(`identifiant`/`chemin`/`modele_base`/`configuration_lora`/`hyperparametres`/
`metriques_finales`/`verdict_convergence`/`horodatage`), aucune méthode
de `CheckpointEntraine` ni de `SauvegarderCheckpointSftUseCase` ne
dispatch sur le type concret d'`hyperparametres` (stocké et resérialisé
en JSONL comme donnée opaque, jamais lu champ par champ par le cas
d'usage lui-même). Écrire une classe parallèle
`CheckpointEntraineDpo`/`SauvegarderCheckpointDpoUseCase` dupliquerait
~30 lignes de code identique pour zéro comportement nouveau : l'exact
type de duplication que ce projet évite ailleurs.

**La réserve honnête, à corriger au moment de l'implémentation, pas
ici** : `CheckpointEntraine.hyperparametres` et le paramètre
`hyperparametres` de `SauvegarderCheckpointSftUseCase.executer()` sont
aujourd'hui type-hintés spécifiquement `HyperparametresEntrainement`
(SFT), et le docstring du module (`domain/model/checkpoint_entraine.py`)
dit explicitement "checkpoint SFT-LoRA". Un appel DPO y passerait en
réalité une instance de `HyperparametresEntrainementDpo` (§4.2 de
`00_introduction_concepts.md`) : Python ne l'empêche pas à l'exécution
(pas de vérification de type au runtime), mais le type-hint mentirait.
La correction propre, différée au guide d'implémentation plutôt que
tranchée ici (édition mineure de code Étape 2 déjà écrit, hors périmètre
d'un document de conception Étape 3) : élargir le type-hint en
`HyperparametresEntrainement | HyperparametresEntrainementDpo` et
généraliser le docstring/nom au-delà de "SFT" (`SauvegarderCheckpointSftUseCase`
garderait probablement son nom, cf. document SFT §"Évaluation post-SFT",
même décision déjà prise de ne pas renommer une classe existante pour un
gain cosmétique seul, appliquée ici par analogie plutôt que retranchée).

`data/processed/checkpoints_sft.jsonl` (proposé en Étape 2) accueillerait
donc, tel quel, aussi bien les checkpoints SFT que DPO, distingués par
leur `configuration_lora`/`hyperparametres`/`modele_base` respectifs :
pas de fichier séparé `checkpoints_dpo.jsonl` proposé ici, cohérent avec
la réutilisation du même port/fichier pour toutes les entités
`CheckpointEntraine` déjà produites par le pipeline.

## 8. Suivi d'expérimentation : port partagé, réutilisé sans modification

Même port, mêmes adaptateurs que `docs/03_etape2_sft/02_etapes_cas_usage.md`
§7 (`SuiviExperimentation`, `MlflowSuiviExperimentation`,
`TensorboardSuiviExperimentation`, `HfDatasetSuiviExperimentation`) :
rien de nouveau à écrire côté suivi pour l'Étape 3. `recipes/dpo_qwen3_lora.yaml`
(esquissé en `00_introduction_concepts.md` §5) fixe déjà
`suivi.backend: hf_dataset` par défaut, cohérent avec la découverte
réelle documentée en AGENTS.md (le premier run SFT-LoRA payant a perdu
sa courbe complète faute de ce réglage) : `EntrainerDpoUseCase` (§4
ci-dessus) n'a besoin d'aucune connaissance du backend concret, même
principe d'injection que pour le SFT. Le futur point d'entrée
d'entraînement DPO (`training/E3_0X_dpo_train.py`, non nommé
précisément ici, pas encore écrit) devrait reprendre telles quelles les
deux gardes de démarrage déjà réelles dans `training/E2_04_sft_train.py`
(`_verifier_type_perte_valide`/`_verifier_suivi_hf_repo_coherent`, cf.
§1.4 ci-dessus) plutôt que de les redécouvrir : le second guard en
particulier (cohérence `--suivi-hf-repo`/`suivi.backend`) s'applique à
l'identique à un futur job DPO sur HF Jobs, même disque éphémère, même
risque de perte de courbe.

Côté visualisation, le dashboard EN VIVO
(`monitoring/app_suivi_entrainement.py`) sait déjà afficher ces quatre
métriques de récompense (`rewards/chosen`/`rewards/rejected`/`rewards/accuracies`/`rewards/margins`)
quand elles sont présentes dans le run sélectionné, cartes et courbe
`chosen`/`rejected` incluses ; un run SFT ne les a jamais et n'affiche
donc jamais ces éléments (cf. AGENTS.md).

## Document suivant

Diagrammes : `docs/diagrams/04_etape3_dpo/` (séquence, paquets,
déploiement, à écrire ; seul le diagramme d'activité
`dpo_double_fonction_entrainement.puml` existe à ce jour). Puis un
`03_guide_implementation_pas_a_pas.md` (Étape 3), pour l'ordre concret
d'écriture des fichiers, une fois ce document de conception validé
comme référence stable : non encore écrit au moment de la rédaction de
ce chapitre.
