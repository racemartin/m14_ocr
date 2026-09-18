\newpage

# Étape 3 : Introduction aux concepts (DPO)

> **Statut réel (18/09/2026) : Étape 3 non implémentée.** Aucun script
> d'entraînement DPO n'existe dans `training/` à ce jour, et aucun
> port de dominio dédié n'existe dans `src/chsa_triage/domain/ports/`
> (vérifié par grep réel, cf. §4). Les données sont en revanche
> prêtes : 97 081 exemples DPO déjà extraits du dataset pivot anonymisé
> et déjà splittés train/val/test (`docs/references/livre_theorique_llm_triage_chsa.md`
> §"Vue d'ensemble", ligne "1. Données"), un sous-ensemble déjà publié
> comme jeu de données Hugging Face
> (`docs/02_etape1_donnees/dataset_card_dpo_hf.md`). Ce document est
> donc, comme `docs/03_etape2_sft/00_introduction_concepts.md` l'était
> avant l'écriture du code SFT, un document de **conception et de
> décisions**, pas d'implémentation : il tranche les ambiguïtés
> ouvertes pour que l'implémentation réelle (guide pas-à-pas, cas
> d'usage, adaptateur GPU) puisse suivre sans avoir à re-débattre ces
> choix.

Public visé : quelqu'un qui connaît le SFT (chapitre précédent) mais
n'a jamais fait d'alignement par préférences. Chaque concept est
expliqué avec le "quoi", le "pourquoi ici" (contexte POC : budget GPU
limité NF5, calendrier de 4 semaines cahier des charges §8) et un
renvoi vers l'endroit du roadmap où il intervient
(`docs/diagrams/00_vue_ensemble/vision_generale_etapes_v3_detaille.puml`,
partition "ETAPE 3, Semaine 3 : DPO").

## 0. D'où vient le point de départ : l'Étape 2 (checkpoint SFT-LoRA)

Le DPO ne part jamais de `Qwen3-1.7B-Base` tel quel : il continue
l'entraînement à partir du modèle déjà affiné par SFT. C'est le
cahier des charges qui l'impose explicitement ("SFT + LoRA, puis DPO",
§6 ; planning §8, "S3 : DPO sur checkpoint SFT"), et c'est aussi ce
que la dérivation théorique du DPO présuppose (§2 ci-dessous) : la
politique de référence `π_ref` doit déjà savoir "avoir une
conversation" dans le bon format avant qu'on lui apprenne à préférer
une réponse à une autre, sans quoi la comparaison chosen/rejected
porterait sur du bruit plutôt que sur un vrai signal de préférence.

Le point de départ réel n'est plus hypothétique : le premier
entraînement SFT-LoRA complet a été mené à son terme sur un GPU cloud
L4 (~20 minutes, 342 pas, 3 époques, **verdict de convergence SAINE**),
et les poids LoRA ont été publiés durablement sur
`mombasstic/chsa-triage-sft-lora`
(`docs/references/livre_theorique_llm_triage_chsa.md` §5.6, AGENTS.md).
Le DPO chargerait donc :

- `Qwen/Qwen3-1.7B-Base` (le même modèle de base, quantifié 4-bit NF4
  comme pour le SFT, cf. §5 ci-dessous) ;
- l'adaptateur LoRA `mombasstic/chsa-triage-sft-lora` appliqué
  par-dessus, comme point de départ de `π_θ` (la politique qu'on
  entraîne) ;
- une **copie gelée** du même point de départ comme `π_ref` (§2) : soit
  littéralement le même modèle base+LoRA-SFT rechargé une seconde fois
  sans dégeler ses poids, soit délégué à `trl` (`ref_model=None`, cf.
  §6.4 du livre théorique et §5 ci-dessous), qui dérive `π_ref` en
  interne à partir de `π_θ` avant le premier pas d'entraînement.

Ce point de départ n'a, à ce jour, **jamais été vérifié en pratique**
pour le DPO (aucun code ne l'a encore chargé) : c'est un point de
vigilance explicitement noté dans le livre théorique (§6.4) et repris
en §5 ci-dessous.

## 1. Qu'est-ce que le DPO (Direct Preference Optimization) ?

Le SFT (chapitre précédent) apprend au modèle à reproduire une seule
réponse "correcte" par exemple : c'est un entraînement **supervisé**
classique, chaque exemple a une vérité de terrain connue à l'avance.
Le **DPO** résout un problème différent : apprendre au modèle à
**préférer** une réponse à une autre, sans qu'aucune des deux ne soit
"la vérité absolue". Concrètement, chaque exemple DPO est un triplet
`(prompt, chosen, rejected)` : un contexte clinique, une réponse jugée
meilleure (`chosen`, notée `y_w` pour *winner*) et une réponse jugée
moins bonne (`rejected`, notée `y_l` pour *loser*). L'entraînement
pousse le modèle à augmenter la probabilité relative de `y_w` face à
`y_l`, sans jamais lui dicter *littéralement* le texte à produire mot
pour mot comme le fait le SFT.

C'est ce que le roadmap appelle l'**alignement** : après le SFT, qui
enseigne le format et le contenu médical général, le DPO affine le
**jugement clinique** du modèle (préférer une décision de triage sûre à
une décision dangereuse) sans nécessiter de nouvelles données
étiquetées "réponse correcte" : seulement des paires déjà comparées.
C'est directement pertinent ici parce que la seule source de données
DPO disponible pour ce projet (`UltraMedical-Preference`, cahier des
charges §5.1) est déjà sous cette forme comparative, pas sous forme de
réponses canoniques.

**Pourquoi DPO plutôt que RLHF/PPO classique** (l'approche historique
d'alignement par préférences) : DPO **contourne** l'étape la plus
coûteuse du RLHF, l'entraînement d'un modèle de récompense séparé
suivi d'une boucle de RL instable à régler (§2 ci-dessous détaille
pourquoi c'est mathématiquement possible). Pour un POC à budget GPU
contraint (NF5, un seul GPU cloud type L4) et un calendrier de 4
semaines qui ne laisse qu'une semaine pour cette étape (cahier des
charges §8, "S3"), entraîner et faire tenir en mémoire un second
modèle de récompense en plus de la politique serait hors de portée.
C'est aussi la technique explicitement imposée par le cahier des
charges (§6 : "Technique de fine-tuning imposée : SFT + LoRA, puis
DPO") : pas un choix technique fait en dehors de la mission.

## 2. La perte DPO : dérivation théorique déjà écrite, application pratique ici

La dérivation mathématique complète (RLHF → modèle de préférence de
Bradley-Terry → politique optimale sous contrainte KL → perte DPO en
forme close) est déjà écrite en détail dans
`docs/references/livre_theorique_llm_triage_chsa.md` §6 : ce document
**ne la reproduit pas**, il s'appuie dessus. Seul le résultat final
(§6.3 du livre théorique) est rappelé ici, pour fixer le vocabulaire
utilisé dans le reste de ce chapitre :

```
L_DPO(θ; π_ref) = -E_{(x,y_w,y_l)} [
    log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) )
]
```

Ce que le livre théorique établit et que ce document reprend comme
acquis : aucun modèle de récompense `r_θ` séparé n'est jamais entraîné,
la récompense reste **implicite**, portée par le rapport de
vraisemblance entre la politique en cours d'entraînement `π_θ` et une
copie gelée du modèle de référence `π_ref` (ici, le checkpoint SFT-LoRA
du §0). C'est un entraînement supervisé classique au sens calcul (une
perte scalaire par exemple, pas de rollout ni d'échantillonnage comme
en PPO), *off-policy* sur des paires de préférence fixées à l'avance :
propriété directement pertinente ici puisque `UltraMedical-Preference`
est exactement ça, des paires déjà figées, jamais générées par le
modèle en cours d'entraînement lui-même.

**Ce que la dérivation implique concrètement pour ce projet** (le
détail pratique que le livre théorique, volontairement, ne couvre
pas) :

- Le terme `β` n'est pas un simple hyperparamètre parmi d'autres : il
  contrôle littéralement, dans la formule ci-dessus, à quel point la
  politique entraînée est **autorisée à s'éloigner** de `π_ref` pour
  satisfaire la préférence. Un `β` élevé pénalise fortement tout
  écart par rapport au comportement post-SFT déjà validé (verdict
  SAINE, §0) ; un `β` faible laisse le modèle s'éloigner davantage
  pour mieux satisfaire chaque préférence individuelle, au risque de
  dégrader ce que le SFT avait déjà appris. Valeur par défaut
  réellement vérifiée dans `trl.DPOConfig` (trl==1.13.0,
  `beta=0.1`, même méthode de vérification sans GPU déjà utilisée pour
  `peft.LoraConfig`/`trl.SFTConfig` en Étape 2) : un point de départ
  raisonnable, jamais recalibré empiriquement sur ce projet (§5).
- La perte compare `π_θ` et `π_ref` **au même `x`, `y_w`, `y_l`** :
  contrairement au SFT (où `ExempleFormate` porte un seul texte ChatML
  complet), un exemple DPO doit exposer `prompt`, `chosen` et
  `rejected` comme trois entités **distinctes**, jamais concaténées à
  l'avance en un seul texte. C'est la contrainte qui motive
  directement la décision de conception du §4 (`ExemplePivot` porte
  déjà cette séparation nativement, `prompt`/`chosen`/`rejected`
  séparés, cf. `domain/model/exemple_pivot.py`).
- La condition implicite "π_ref doit déjà être une politique
  raisonnable" (sinon la comparaison de vraisemblance ne porte sur
  rien d'utile) est exactement pourquoi le DPO ne peut pas partir de
  `Qwen3-1.7B-Base` brut : c'est la dérivation elle-même, pas
  seulement le cahier des charges, qui impose le point de départ
  post-SFT du §0.

Diagramme complémentaire, illustrant la double fonction d'une seule
passe DPO (alignement clinique + apprentissage du format de sortie,
cf. §3) :
`docs/diagrams/04_etape3_dpo/activite/dpo_double_fonction_entrainement.puml`
(déjà écrit, marqué `[CONCEPTUEL/PRÉVU]`) : référencé ici, pas
reproduit.

## 3. Décision (tranchée) : le format de sortie cible via reformulation du dataset DPO

### 3.1. Rappel de la décision restée ouverte à l'Étape 2

`docs/03_etape2_sft/02_etapes_cas_usage.md` (§"Décision encore
ouverte") documentait un écart réel : le cahier des charges (F3-F4)
exige une sortie **JSON strict** (`niveau`, `categorie`,
`ressources_estimees`) précédée d'un bloc `<think>...</think>`, mais
**aucune** `completion`/`chosen` du dataset pivot actuel (MediQAl,
FrenchMedMCQA, MedQuAD, UltraMedical-Preference) n'est déjà dans ce
format. Trois options y étaient posées sans trancher :

1. Ne rien changer aux données pour le SFT, repousser le format à un
   prompt système few-shot (inférence) ou au DPO.
2. Construire un jeu de complétions synthétiques SFT reformulées.
3. Limiter le format strict à de futurs exemples de triage dédiés, non
   encore collectés.

### 3.2. Décision retenue : Option 1, appliquée spécifiquement au DPO

**Décision** : le dataset source du DPO (`UltraMedical-Preference`,
paires `chosen`/`rejected` déjà réelles et vérifiées dans le pivot,
format "*Explicit prompt*" de `trl` : `prompt` séparé,
`chosen`/`rejected` ne portant que le tour `assistant`, cf. §4) est
**reformulé** vers le format `<think>` + JSON cible via un prompt de
reformulation, appliqué à `chosen` uniquement.

**Ce que "reformuler uniquement `chosen`" veut dire concrètement**,
et pourquoi c'est délibéré plutôt qu'un oubli : le diagramme
`dpo_double_fonction_entrainement.puml` (§2 ci-dessus) montre
explicitement, dans son exemple illustratif, que `y_w` (chosen) est
"structuré (JSON + balises ChatML)" tandis que `y_l` (rejected) reste
"texte libre, non structuré". Reformuler `chosen` vers le format cible
et laisser `rejected` dans sa forme originale (texte libre, telle
qu'elle existe déjà dans `UltraMedical-Preference`) obtient cette
opposition **sans coût de reformulation supplémentaire** sur
`rejected` : une seule passe de reformulation par exemple plutôt que
deux, cohérent avec le budget/calendrier contraint (NF5, cahier des
charges §8). Le triplet `(prompt, chosen, rejected)` résultant fait
alors préférer, dans la **même** descente de gradient, une réponse à
la fois cliniquement meilleure *et* structurée, à une réponse restée
en texte libre : c'est très exactement la "fonction 2" du diagramme
("apprentissage du format de sortie contractuel" par la préférence
elle-même, jamais par une passe d'entraînement séparée).

**Pourquoi cette décision plutôt que les options 2/3, et pourquoi
spécifiquement au DPO** :

- **Le SFT est déjà fait.** Refaire ses 37 802 exemples avec des
  complétions synthétiques reformulées (option 2 de l'Étape 2) serait
  coûteux et hors délai : cela demanderait de relancer une passe
  d'entraînement complète déjà validée (verdict SAINE, §0) sur un
  jeu de données entièrement nouveau, sans garantie de faire mieux.
- **Collecter de nouvelles données de triage dédiées (option 3) ne
  rentre pas non plus dans le calendrier de 4 semaines** (cahier des
  charges §8) : la collecte, l'anonymisation RGPD (NF2) et la
  vérification qualité d'un nouveau corpus prendraient, à elles
  seules, une fraction significative du temps déjà consommé pour les
  quatre corpus existants (`docs/02_etape1_donnees/`).
- **Le DPO est la seule étape qui n'a encore reçu AUCUNE donnée
  d'entraînement.** C'est la fenêtre disponible restante : reformuler
  son dataset source enseigne, dans la même passe, (a) la préférence
  clinique correcte (le signal DPO natif) et (b) le format de sortie
  contractuel (via l'asymétrie structuré/non-structuré ci-dessus) :
  double fonction pour un seul entraînement, déjà documentée dans le
  diagramme fusionné référencé en §2, pas un travail de données
  distinct en plus du DPO lui-même.

**Limite honnête, à documenter plutôt qu'à ignorer** :
`UltraMedical-Preference` est un corpus de préférence médicale
générale (réponses comparées par qualité générale de réponse), pas un
corpus nativement étiqueté "triage ESI". Le signal de préférence
clinique d'origine (pourquoi `chosen` a été jugé meilleur que
`rejected` dans le corpus source) ne porte donc pas nécessairement,
tel quel, sur la correction d'un niveau ESI : le prompt de
reformulation devra projeter honnêtement le contenu clinique de
`chosen` vers le schéma `niveau`/`categorie`/`ressources_estimees`
(en déduisant un niveau ESI plausible du contenu, jamais en
l'inventant sans lien avec le texte source), et non prétendre que
cette reformulation transforme le corpus en un jeu de préférences de
triage ESI vérifié cliniquement. Ce point reste un axe de validation
humaine future (cf. `docs/02_etape1_donnees/01_rapport_rgpd.md` pour
le précédent de révision humaine déjà en place pour un autre type de
décision automatisée sur ce projet), pas une garantie acquise par la
seule reformulation.

**Conséquence sur le pipeline de données** : la reformulation opère
sur le dataset pivot DPO **après** son extraction/anonymisation
existante (`interfaces/cli/E1_05_03_extraire_sous_ensemble_dpo.py`,
déjà écrit, 97 081 exemples disponibles avant filtrage de publication,
cf. §0), comme une étape supplémentaire, spécifique au DPO, avant le
formatage ChatML. Elle n'implique aucune modification des fichiers
déjà écrits d'Étape 1 (`dataset_pivot.jsonl`,
`dataset_pivot_anonymise.jsonl` restent inchangés, même principe de
non-mutation déjà en place pour le pivot original, cf. AGENTS.md) : le
résultat de la reformulation est un nouveau champ/fichier dérivé, pas
une réécriture sur place. L'implémentation réelle (quel modèle
reformule, quel prompt exact, où le résultat est persisté) est
délibérément **hors de ce document** : c'est un cas d'usage à part
entière, à concevoir dans le guide d'implémentation pas-à-pas, une fois
ce document de conception validé.

## 4. Décision (tranchée) : un nouveau port de domaine `EntraineurPreference`, pas une généralisation de `EntraineurSupervise`

### 4.1. Pourquoi `EntraineurSupervise` ne convient pas tel quel

Vérifié par lecture directe du code (pas supposé) :
`domain/ports/entraineur_supervise.py` expose un unique port,
`EntraineurSupervise`, dont la méthode `entrainer(dataset_train,
dataset_validation, config_lora, hyperparametres) ->
ResultatEntrainementSFT` opère sur `Iterable[ExempleFormate]` : un
type qui ne porte qu'**un seul** champ de texte
(`ExempleFormate.texte`, le rendu ChatML complet prompt+completion
concaténé, cf. `domain/model/exemple_formate.py`). Un grep réel sur
`src/chsa_triage/domain/ports/` confirme qu'aucun port DPO n'existe
encore à ce jour.

Cette forme ne correspond pas à ce que la perte DPO (§2) exige : elle
compare `π_θ(y_w|x)` à `π_θ(y_l|x)` (et pareil pour `π_ref`), donc
**trois** quantités distinctes par exemple (`prompt`, `chosen`,
`rejected`), jamais une seule séquence concaténée. Vérifié
indépendamment côté `trl` (trl==1.13.0, installé temporairement pour
inspection puis désinstallé, même méthode que la vérification
`peft.LoraConfig`/`trl.SFTConfig` de l'Étape 2) : `DPOTrainer` attend
un jeu de données avec les colonnes `prompt`/`chosen`/`rejected`
(`trl.data_utils.is_conversational` reconnaît explicitement ces trois
clés, chacune une liste de messages `{role, content}`, exactement la
forme déjà portée par `ExemplePivot.prompt`/`chosen`/`rejected`, tuples
de `Message(role, contenu)`). Faire porter cette forme par
`EntraineurSupervise.entrainer()` obligerait soit à ajouter des
paramètres optionnels `chosen`/`rejected` à une méthode pensée pour un
seul texte (branches conditionnelles dans un port censé rester un
contrat simple), soit à détourner `ExempleFormate.texte` pour y
encoder trois textes concaténés avec un séparateur artificiel :
les deux cassent la promesse d'un port = une responsabilité claire,
déjà respectée par tous les autres ports du projet (`MoteurInference`,
`FormateurConversation`, `SuiviExperimentation`).

### 4.2. Décision : nouveau port, précédent déjà posé dans ce même projet

**Décision** : un port de domaine séparé, `EntraineurPreference`
(`domain/ports/entraineur_preference.py`, à écrire), même patron
`Protocol`/dataclasses pures que `EntraineurSupervise` (aucun objet
`torch`/`trl`/`peft` ne traverse la frontière domaine/application).

Ce n'est pas un choix arbitraire : c'est **exactement** le précédent
déjà posé ailleurs dans ce projet pour une situation structurellement
identique. `docs/03_etape2_sft/02_etapes_cas_usage.md` documente que
`domain/ports/formateur_invite_zero_shot.py::FormateurInviteZeroShot`
a été ajouté comme un **nouveau** port, séparé de
`FormateurConversation`, précisément parce que leurs contrats
diffèrent sur un point structurel (`add_generation_prompt`, et le fait
de ne jamais montrer `completion`) même si le **même** adaptateur
concret (`ChatMLFormateurAdapter`) implémente les deux (cf. AGENTS.md,
note du 15/09/2026). `EntraineurSupervise` vs `EntraineurPreference`
est le même type de situation : deux contrats dont la forme de
données diffère structurellement (un texte vs. un triplet), pas deux
variantes d'un même contrat.

Esquisse du port (illustratif, à écrire lors de l'implémentation
réelle, pas figé ici dans le détail) :

```python
class EntraineurPreference(Protocol):
    """Port generique : entrainer une politique par preference DPO,
    a partir d'un checkpoint SFT-LoRA de depart / reference."""

    def entrainer(
        self,
        dataset_train             : Iterable[ExempleFormatePreference],
        dataset_validation         : Iterable[ExempleFormatePreference],
        config_lora                  : ConfigurationLora,
        hyperparametres                : HyperparametresEntrainementDpo,
        chemin_checkpoint_politique_depart : str,  # le SFT-LoRA du §0
    ) -> ResultatEntrainementDPO: ...
```

### 4.3. `ExempleFormate` ne gagne pas de variante : `ExempleFormatePreference` en parallèle

Question explicitement posée par la tâche : `ExempleFormate` a-t-il
besoin d'un champ/variante DPO, ou faut-il un type parallèle ?
**Décision : un type parallèle**, `ExempleFormatePreference`
(`domain/model/exemple_formate_preference.py`, à écrire), plutôt que
d'ajouter des champs optionnels `chosen`/`rejected` à `ExempleFormate`.
Raisonnement : `ExempleFormate.texte` (un seul champ, non optionnel)
est déjà consommé tel quel par `EntrainerSftUseCase`,
`SauvegarderCheckpointSftUseCase` et `TrlSftEntraineurAdapter`, tous
écrits et testés (cf. `docs/03_etape2_sft/02_etapes_cas_usage.md`) : y
ajouter des champs optionnels DPO créerait un type dont la moitié des
champs ne serait jamais renseignée par le chemin SFT, et forcerait
chaque consommateur SFT existant à ignorer explicitement des champs
qui ne le concernent pas. `ExempleFormatePreference` reste, comme
`ExempleFormate`, une `dataclass(frozen=True, slots=True)` pure :

```python
@dataclass(frozen=True, slots=True)
class ExempleFormatePreference:
    """Rendu (ChatML ou equivalent) d'un ExemplePivot DPO, triplet
    prompt/chosen/rejected distinct, jamais concatene en un seul texte
    (cf. domain/ports/entraineur_preference.py, format DPOTrainer)."""

    identifiant   : str   # repris de ExemplePivot.identifiant
    texte_prompt   : str
    texte_chosen    : str
    texte_rejected   : str
```

Le rendu de chaque champ reste, comme pour `formater_invite_zero_shot`
déjà écrit (§0, cf. `ChatMLFormateurAdapter`), une question de detail
d'implémentation différée : `texte_prompt` suivrait vraisemblablement
le même chemin que `formater_invite_zero_shot`
(`add_generation_prompt=True`, prompt seul), `texte_chosen`/
`texte_rejected` porteraient chacun la réponse assistant correspondante
(potentiellement déjà reformatée en `<think>`+JSON pour `chosen`, cf.
§3). Un port séparé, `FormateurPreference.formater(exemple:
ExemplePivot) -> ExempleFormatePreference` (même précédent qu'en
§4.2, vraisemblablement implémenté par le même `ChatMLFormateurAdapter`
déjà écrit, réutilisant le tokenizer déjà chargé), plutôt qu'une
méthode supplémentaire ajoutée à `FormateurConversation`.

### 4.4. Réutilisation délibérée : `MetriquesEntrainement` et `evaluer_convergence`

Décision complémentaire, motivée par la même discipline anti-abstraction
prématurée déjà appliquée ailleurs dans ce projet : le `Résultat`
retourné par `EntraineurPreference.entrainer()` (proposé :
`ResultatEntrainementDPO(chemin_checkpoint, courbe_metriques:
tuple[MetriquesEntrainement, ...])`) réutilise **telle quelle** la
dataclass `MetriquesEntrainement` déjà écrite pour le SFT (`etape`,
`perte_train`, `perte_validation`, `norme_gradient`), plutôt que
d'en créer une variante DPO. Raison : `application/verdict_convergence.py
::evaluer_convergence()` n'exige rien de plus que ces quatre champs
génériques pour diagnostiquer SAINE/SURAPPRENTISSAGE/SOUS_APPRENTISSAGE/
INSTABLE (cf. `docs/03_etape2_sft/02_etapes_cas_usage.md` §4) : une
courbe de perte DPO se diagnostique avec la même logique qu'une courbe
de perte SFT, aucune spécificité DPO n'entre dans ce diagnostic. Cela
permettrait de réutiliser `evaluer_convergence()` sans aucune
modification pour le DPO. Les métriques réellement spécifiques au DPO
que `trl.DPOTrainer` journalise (`rewards/chosen`, `rewards/rejected`,
`rewards/accuracies`, `rewards/margins`, noms réels de l'API `trl`, non
couverts par `MetriquesEntrainement`) seraient relayées directement
vers `SuiviExperimentation.logger_metrique(...)` par le futur cas
d'usage d'orchestration (même patron que `EntrainerSftUseCase` relayant
`perte_train`/`perte_validation`/`norme_gradient`), sans passer par
`evaluer_convergence()` : une extension additive de la boucle de suivi,
pas une modification du diagnostic de convergence existant.

## 5. Hyperparamètres spécifiques au DPO

Comme pour le SFT (`recipes/sft_qwen3_lora.yaml`), une future recette
DPO (`recipes/dpo_qwen3_lora.yaml`, non créée par ce document, simple
esquisse ci-dessous à titre d'illustration) réutiliserait la même
structure `quantification:`/`lora:` que la recette SFT (même modèle
de base, même schéma QLoRA NF4, §0), en ajoutant une section
`entrainement:` propre au DPO :

```yaml
modele_base: Qwen/Qwen3-1.7B-Base
checkpoint_politique_depart: mombasstic/chsa-triage-sft-lora   # §0

quantification:   # identique a la recette SFT
  bits: 4
  type_quantification: nf4
  double_quantification: true
  dtype_calcul: bfloat16

lora:   # point de depart identique au SFT ; pourrait etre re-derive
  rang: 16
  alpha: 32
  dropout: 0.05
  modules_cibles: [q_proj, k_proj, v_proj, o_proj]

entrainement:
  beta: 0.1                 # cf. discussion §2 ; defaut reel trl.DPOConfig
  taux_apprentissage: 5.0e-6  # nettement < SFT (2.0e-4), cf. tableau ci-dessous
  nombre_epoques: 1            # ordre de grandeur usuel DPO, a valider empiriquement
  taille_lot: 4
  type_perte: sigmoid          # perte DPO "standard" ; cf. tableau ci-dessous
  precompute_ref_log_probs: false

suivi:
  backend: hf_dataset   # meme raison qu'en SFT (disque HF Jobs ephemere, cf. AGENTS.md)
  nom_experience: chsa-triage-dpo
```

Hyperparamètres/champs de `trl.DPOConfig` propres au DPO (sans
équivalent SFT), avec leurs valeurs par défaut **réellement vérifiées**
(trl==1.13.0, même méthode d'installation temporaire/désinstallation
que pour `peft.LoraConfig`/`trl.SFTConfig` en Étape 2, jamais
supposées) :

| Champ | Défaut réel vérifié | Rôle |
|---|---|---|
| `beta` | `0.1` | Poids de la régularisation KL implicite envers `π_ref` (cf. §2) : plus haut = plus proche du SFT, plus bas = préférence suivie plus agressivement. |
| `loss_type` | `["sigmoid"]` (perte DPO standard, §2) | `trl` supporte plusieurs variantes de perte de préférence (`sigmoid`, `ipo`, `hinge`, entre autres) ; `sigmoid` est la formule dérivée en §6.3 du livre théorique, celle décrite dans ce document. |
| `learning_rate` | `1e-06` | **Deux ordres de grandeur sous le taux SFT** (`2.0e-4` dans `recipes/sft_qwen3_lora.yaml`) : cohérent avec la mise en garde du livre théorique (§6.4, "nettement < SFT") et avec l'intuition du §2 (on affine une politique déjà bonne, pas on en construit une nouvelle). |
| `max_length` | `1024` | Longueur maximale du triplet `prompt`+réponse ; **remplace** les champs historiques séparés `max_prompt_length`/`max_completion_length` d'anciennes versions de `trl` (absents de trl==1.13.0, vérifié par introspection réelle de signature, pas supposé) : à garder à l'esprit si la documentation `trl` consultée date d'une version antérieure. |
| `precompute_ref_log_probs` | `False` | Mitigation directe du point de vigilance mémoire du livre théorique (§6.4, "second passage... à faire tenir en mémoire simultanément") : mis à `True`, `trl` précalcule et met en cache les log-probabilités de `π_ref` une seule fois avant l'entraînement, plutôt que de garder `π_ref` chargé en mémoire GPU à chaque pas. Non activé par défaut ; à mesurer une fois un GPU réel disponible, même prudence que les optimisations Unsloth/Liger/FlashAttention-2 non validées en Étape 2. |

**Quel dataset** : `dataset_dpo` (colonnes `prompt`/`chosen`/`rejected`,
§4) viendrait du split DPO déjà produit
(`E1_05_03_extraire_sous_ensemble_dpo.py`, 97 081 exemples disponibles
avant filtrage de publication, cf. §0 et §3.2 pour l'étape de
reformulation intercalée), filtré sur `split=train`/`val` de la même
façon que le SFT (`type_exemple == TypeExemple.DPO`, déjà correct
depuis la correction du bug de fuite SFT/DPO documentée dans
AGENTS.md) ; le `split=test` resterait réservé à l'évaluation finale
(cahier des charges §9, "Ne jamais mélanger données d'entraînement et
données d'évaluation").

**Pourquoi pas de grille d'hyperparamètres DPO dans ce document** :
`AjusterBoucleHyperparametresSftUseCase` (Étape 2) et
`application/grille_hyperparametres.py` sont déjà génériques sur
`HyperparametresEntrainement`/`ConfigurationLora` (cf.
`docs/03_etape2_sft/02_etapes_cas_usage.md` §5) ; une grille DPO
réutiliserait vraisemblablement le même mécanisme sur
`HyperparametresEntrainementDpo` (§4.2) plutôt que d'en inventer un
nouveau, mais trancher les axes concrets de cette grille (quels `beta`,
quels `taux_apprentissage` essayer) suppose d'observer d'abord une
vraie courbe DPO, exactement comme les seuils de
`evaluer_convergence()` restaient "provisoires" avant le premier run
SFT réel (§4 ci-dessus, §4 de `02_etapes_cas_usage.md`) : prématuré ici.

## Point de vigilance : deux décisions tranchées ici, une troisième qui reste ouverte

Ce document tranche deux décisions de conception qui bloquaient le
démarrage de l'implémentation DPO (§3, format de sortie via
reformulation ; §4, port `EntraineurPreference` séparé). Une troisième
reste délibérément **hors de ce document**, à trancher lors de
l'implémentation réelle plutôt que par anticipation :

- **Qui reformule, et avec quel prompt exact** (§3.2) : quel modèle
  effectue la reformulation `chosen` → `<think>`+JSON (un modèle plus
  grand via API, ou `Qwen3-1.7B-Base` lui-même en few-shot), et
  comment son résultat serait validé avant inclusion (cf. la limite
  honnête notée en §3.2 sur le transfert du signal de préférence
  clinique). Ce point conditionne directement un futur cas d'usage
  dédié (`E1_0X` ou `E2_0X`, convention `AGENTS.md`), pas encore
  nommé ni conçu.
- Le **mapping exact** entre `Message(role, contenu)` (déjà dans le
  pivot) et les colonnes `prompt`/`chosen`/`rejected` attendues par
  `trl.DPOTrainer` au niveau du **rendu texte** (`ExempleFormatePreference`,
  §4.3) : esquissé ici par analogie avec `formater_invite_zero_shot`,
  jamais vérifié contre un vrai appel `DPOTrainer.train()` (aucun GPU
  disponible au moment de l'écriture de ce document, même limite que
  `TrlSftEntraineurAdapter` avant son premier run réel, cf. AGENTS.md).

## Document suivant

`02_etapes_cas_usage.md` (Étape 3) : correspondance entre les
décisions ci-dessus et les fichiers réels à écrire
(`domain/ports/entraineur_preference.py`,
`domain/model/exemple_formate_preference.py`, le cas d'usage de
reformulation, `E3_0X_uc_*`), une fois ce document de conception
disponible comme référence stable. Document non encore écrit au
moment de la rédaction de ce chapitre.
