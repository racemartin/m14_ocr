\newpage

# Étape 2 : Introduction aux concepts (SFT + LoRA)

> Les trois phases sans GPU de l'Étape 2 (domaine, application, cas
> d'usage `E2_00` à `E2_03`, adaptateurs `ChatMLFormateurAdapter`,
> `MlflowSuiviExperimentation`, `TensorboardSuiviExperimentation`) sont
> écrites et testées (221 tests en vert au moment de l'écriture). Ce
> document reste toutefois majoritairement conceptuel : il explique
> le "quoi"/"pourquoi" des techniques (SFT, LoRA, quantification 4-bit,
> ChatML, packing, grad norm) plutôt que le code lui-même, cf.
> `02_etapes_cas_usage.md` pour la correspondance avec les fichiers
> réels. Ce qui reste non écrit et nécessite un GPU réel :
> `infrastructure/adapters/trl_sft_entraineur.py` et
> `training/sft_train.py` (aucun poids de `Qwen3-1.7B-Base` n'a encore
> été téléchargé, aucune bibliothèque d'entraînement GPU exécutée) :
> les commandes/journaux les concernant, montrés ici, restent des
> exemples illustratifs, pas des résultats mesurés.

Public visé : quelqu'un qui connaît le machine learning mais n'a
jamais fait de fine-tuning de grand modèle de langage. Chaque concept
est expliqué avec le "quoi", le "pourquoi ici" (contexte POC : budget
GPU limité, `Qwen3-1.7B-Base` imposé par le cahier des charges §6) et
un renvoi vers l'endroit du roadmap où il intervient
(`docs/diagrams/00_vue_ensemble/activite/roadmap_activite.puml`,
partition "ETAPE 2, Semaine 2 : SFT + LoRA").

## 0. D'où vient le point de départ : l'Étape 1bis (baseline)

Avant tout fine-tuning, le roadmap place une étape courte,
l'**Étape 1bis (Baseline)** : évaluer `Qwen3-1.7B-Base` tel quel
(zéro/few-shot, sans aucun entraînement) sur le jeu de test clinique
déjà isolé à l'Étape 1. Le but n'est pas de produire un bon agent de
triage à ce stade : c'est d'obtenir un chiffre de référence (exactitude
de classification ESI baseline) auquel comparer le modèle après SFT,
puis après DPO. Sans cette mesure, "le SFT a-t-il amélioré le modèle ?"
resterait une question sans réponse chiffrée. Cette étape n'est **pas**
traitée dans ce document (elle est hors périmètre de la planification
Étape 2) : elle est mentionnée ici uniquement parce que l'évaluation
décrite en §6 ci-dessous s'y compare.

## 1. Qu'est-ce que le SFT (Supervised Fine-Tuning) ?

Un modèle de langage "de base" (`Qwen3-1.7B-Base`) a été pré-entraîné
sur d'énormes quantités de texte générique pour prédire le mot suivant
: il ne sait pas nativement "avoir une conversation" ni "répondre à
une instruction" de façon utile et structurée. Le **SFT** est l'étape
qui lui apprend ce comportement : on continue l'entraînement, mais
cette fois sur un jeu de paires **instruction → réponse** de qualité
(ici, les ≈37 802 exemples SFT du dataset pivot, cf.
`docs/02_etape1_donnees/00_couverture_exigences_officielles.md`), pour
que le modèle apprenne à produire des réponses dans le style et le
format attendus plutôt que de simplement continuer un texte au hasard.

C'est un fine-tuning **supervisé** : chaque exemple a une réponse
correcte connue à l'avance (contrairement au DPO de l'Étape 3, qui
compare deux réponses entre elles sans qu'aucune ne soit "la vérité
absolue" : voir `docs/04_etape3_dpo/` une fois écrit).

## 2. Qu'est-ce que LoRA, et pourquoi pas un fine-tuning complet ?

Un fine-tuning **complet** ("full fine-tuning") met à jour **tous**
les paramètres du modèle. Pour `Qwen3-1.7B` (1,7 milliard de
paramètres), cela veut dire garder en mémoire GPU, pendant
l'entraînement : les poids (≈3,4 Go en bf16), leurs gradients (≈3,4 Go
de plus) et l'état de l'optimiseur Adam (deux moments par paramètre,
≈13,6 Go de plus) : de l'ordre de 20 Go rien que pour ces trois
éléments, avant même de compter les activations. C'est hors de portée
du budget GPU du POC (cahier des charges, exigence NF5 : "QLoRA 4-bit,
≤ budget d'un GPU cloud unique, ex. T4/A10/L4" : des GPU avec
typiquement 16-24 Go de VRAM).

**LoRA** (*Low-Rank Adaptation*) contourne le problème autrement :
les poids du modèle de base restent **gelés** (jamais mis à jour), et
on insère, à côté de certaines couches linéaires ciblées
(`target_modules` : typiquement les projections d'attention
`q_proj`/`k_proj`/`v_proj`/`o_proj`), une paire de petites matrices
entraînables de rang réduit `r` (ex. `r=16`), multipliées par un
facteur d'échelle `alpha`. Seules ces matrices, de l'ordre de 0,1 à
1 % du nombre total de paramètres, sont entraînées. Conséquence
directe : plus besoin de stocker un gradient ni un état d'optimiseur
pour les ≈1,7 milliard de poids gelés, seulement pour les quelques
millions de paramètres LoRA. C'est ce qui rend l'entraînement
faisable sur un seul GPU cloud modeste, dans le budget temps/argent
d'un POC de 4 semaines.

**Pourquoi LoRA plutôt qu'une autre technique d'adaptation légère**
(ex. adapters classiques, prompt tuning) : c'est la technique
explicitement imposée par le cahier des charges (§6, "Technique de
fine-tuning imposée : SFT + LoRA, puis DPO") et la mieux outillée côté
écosystème Hugging Face (`peft.LoraConfig`, intégration native dans
`trl.SFTTrainer`) : pas un choix technique fait en dehors de la
mission.

## 3. Qu'est-ce que la quantification 4-bit NF4, et pourquoi la combiner à LoRA ?

Charger les poids du modèle de base en 4-bit plutôt qu'en 16-bit
(bf16/fp16) réduit leur empreinte mémoire d'environ 4× (≈0,9 Go au
lieu de ≈3,4 Go pour `Qwen3-1.7B`). **NF4** (*4-bit NormalFloat*) est
un format de quantification conçu spécifiquement pour des poids dont
la distribution suit une loi normale (ce qui est le cas empiriquement
pour les poids d'un réseau de neurones entraîné) : il répartit ses 16
niveaux de quantification pour minimiser l'erreur sur cette
distribution précise, plutôt qu'un pas linéaire uniforme. C'est le
format utilisé par la technique **QLoRA** (Dettmers et al., 2023), qui
est exactement la combinaison "base gelée quantifiée 4-bit + adaptateurs
LoRA entraînés en précision plus haute (bf16)" décrite ici.

Les deux techniques sont complémentaires, pas redondantes : LoRA
réduit ce qu'il faut **entraîner** (gradients/optimiseur), la
quantification 4-bit réduit ce qu'il faut **stocker** pour la partie
gelée. Combinées, elles permettent de charger et d'entraîner un modèle
de 1,7 milliard de paramètres sur un GPU avec quelques Go de VRAM
disponibles : cohérent avec l'exigence NF5 et avec les flavors HF Jobs
cités dans `docs/01_environnement/00_guide_installation_environnement.md`
(`t4-small`, `a10g-small`).

Concrètement, ceci se configure via `transformers.BitsAndBytesConfig`
(`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`,
`bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16`)
: voir `scripts/check_env_gpu.py::verifier_chargement_4bit`, déjà
écrit et qui teste exactement ce chargement.

## 4. Qu'est-ce que ChatML, et qu'est-ce que l'`assistant_only_loss` ?

**ChatML** est un format de sérialisation d'une conversation
multi-tours (system/user/assistant) en un unique flux de texte, délimité
par des tokens de contrôle (`<|im_start|>role ... <|im_end|>`). C'est
le format que `Qwen3-1.7B-Base` reconnaît nativement via son
*chat template* (un gabarit Jinja embarqué dans le tokenizer,
appliqué par `tokenizer.apply_chat_template(...)`).

Le schéma pivot (`ExemplePivot`, cf.
`src/chsa_triage/domain/model/exemple_pivot.py`) stocke déjà `prompt`
et `completion` comme des tuples de `Message(role, contenu)` : c'est-à-
dire déjà dans une forme structurée prête à être rendue en ChatML par
`apply_chat_template`, sans transformation de schéma supplémentaire.
Voir §"Point de vigilance" ci-dessous pour ce qui manque malgré tout
avant le SFT.

Pendant l'**entraînement**, `add_generation_prompt=False` : on ne
demande pas au modèle de générer une suite, on lui montre une
conversation déjà complète (prompt **et** réponse) et on calcule une
perte sur cette réponse. À l'**inférence** (Étape 4), c'est l'inverse :
`add_generation_prompt=True`, on ne montre que le prompt et on
laisse le modèle générer la suite.

Par défaut, un entraînement causal calcule la perte sur **tous** les
tokens de la séquence, y compris ceux du prompt (system + user) : ce
qui entraînerait le modèle à "prédire" le prompt lui-même, un signal
inutile puisque le prompt n'est jamais quelque chose que le modèle
doit apprendre à produire. **`assistant_only_loss`** (option de
`trl.SFTConfig`) masque la perte sur tout ce qui n'est pas un token de
la réponse de l'assistant : seul le texte que le modèle est
effectivement censé apprendre à générer contribue au gradient. C'est
la même chat template qui porte l'information "quels tokens sont
`assistant`" : un prérequis d'implémentation à vérifier explicitement
plutôt qu'à supposer (voir `01_installation_configuration.md`).

## 5. Qu'est-ce que le *packing* ?

Les exemples SFT ont des longueurs très variables (une réponse
FrenchMedMCQA tient en une lettre, une réponse UltraMedical-Preference
peut faire plusieurs milliers de caractères : cf. les temps de
traitement mesurés par source dans
`docs/02_etape1_donnees/00_couverture_exigences_officielles.md`).
Sans *packing*, chaque exemple occupe une séquence de longueur fixe
(la plus longue autorisée), remplie de tokens de *padding* pour les
exemples plus courts : beaucoup de calcul GPU dépensé sur du
"remplissage" qui ne porte aucune information. Le **packing**
concatène plusieurs exemples courts dans une seule séquence
d'entraînement (jusqu'à la longueur maximale), ce qui réduit le
gaspillage de calcul et accélère l'entraînement à budget GPU égal :
un gain de throughput directement pertinent pour un POC à budget
temps/argent limité (`trl.SFTConfig(packing=True)`, roadmap Étape 2).

## 6. Qu'est-ce qu'un *grad norm*, et pourquoi le surveiller ?

La **norme du gradient** (*grad norm*) est la magnitude du vecteur
gradient calculé à chaque pas d'entraînement : un seul nombre qui
résume "à quel point les poids sont sur le point de changer". Elle
complète la courbe de perte (*loss*) pour diagnostiquer la santé de
l'entraînement :

- une norme qui **explose** (croît sans borne, ou `NaN`) signale une
  instabilité numérique (taux d'apprentissage trop élevé, par exemple)
  : l'entraînement doit être arrêté et le taux d'apprentissage revu ;
- une norme qui s'**effondre** vers zéro alors que la perte de
  validation ne baisse plus signale que le modèle a cessé d'apprendre
  (taux d'apprentissage trop faible, ou `r`/`alpha` LoRA trop
  restrictifs pour capter le signal).

C'est exactement l'un des deux signaux que le roadmap prévoit de
suivre via MLflow/TensorBoard ("Suivi : MLflow / TensorBoard (loss,
grad norm)") et qui alimente le point de décision "Convergence saine ?"
de l'Étape 2 : ni la perte seule, ni le grad norm seul, ne suffisent :
c'est leur lecture combinée qui distingue un entraînement sain d'un
sur-apprentissage (perte train qui continue de baisser, perte
validation qui remonte) ou d'un sous-apprentissage (les deux
stagnent).

## 7. Optimisations de calcul mentionnées au roadmap (Unsloth, Liger Kernel, FlashAttention-2, chunked cross-entropy)

Ces quatre éléments ne changent pas *ce que* le modèle apprend : ils
changent uniquement la **vitesse** et l'**empreinte mémoire** de
l'entraînement, ce qui, dans un POC à budget GPU contraint (NF5), a un
effet direct sur ce qu'il est possible d'exécuter dans le temps/budget
imparti :

- **Unsloth** : ré-implémentations optimisées (noyaux CUDA fusionnés)
  de la boucle d'entraînement QLoRA pour les architectures type
  Qwen/Llama/Mistral : gains de vitesse et de mémoire rapportés
  significatifs sur ce type de fine-tuning, à vérifier empiriquement
  sur `Qwen3-1.7B-Base` une fois l'environnement GPU disponible (déjà
  dans l'extra `remote` de `pyproject.toml`).
- **Liger Kernel** : bibliothèque de noyaux Triton pour les couches
  coûteuses (RMSNorm, RoPE, cross-entropy) des LLM, complémentaire
  d'Unsloth.
- **FlashAttention-2** : implémentation de l'attention qui évite de
  matérialiser la matrice d'attention complète en mémoire : réduit la
  mémoire consommée par l'attention de façon quadratique en longueur
  de séquence à linéaire.
- **Chunked cross-entropy** (`loss_type=chunked_nll` sur
  `trl.SFTTrainer`) : calcule la perte par morceaux de la séquence
  plutôt que de matérialiser d'un coup le tenseur de logits complet
  (taille = longueur de séquence × taille du vocabulaire, souvent le
  plus gros tenseur intermédiaire d'un forward pass LLM) : réduit le
  pic mémoire, pas un concept de qualité d'apprentissage.

Ces optimisations sont donc des leviers d'**infrastructure**
(`infrastructure/adapters/`, voir `02_etapes_cas_usage.md`), pas des
concepts qu'il faut comprendre pour raisonner sur la qualité du
modèle entraîné.

## Point de vigilance : le format de sortie cible n'existe pas encore dans les données

Le cahier des charges (§3, F3-F4) exige que l'agent final produise
une sortie **JSON strict** (`niveau`, `categorie`, `ressources_estimees`)
précédée d'un bloc de raisonnement explicite (`<think>...</think>`).
Le roadmap note ce besoin dans la case "Formater en ChatML" ("Prefilling
JSON / bloc `<think>` préparés"). Or les `completion`/`chosen` du
dataset pivot actuel (MediQAl, FrenchMedMCQA, MedQuAD) sont des
réponses en langage naturel issues des corpus sources : **aucune
n'est déjà au format `<think>` + JSON triage attendu**. C'est un écart
réel entre les données disponibles et le format de sortie final, pas
une supposition : il reste ouvert malgré l'écriture du formateur
ChatML (`E2_00_uc_formater_dataset_chatml.py`, cf.
`02_etapes_cas_usage.md` §"Décision à prendre").

## Document suivant

`01_installation_configuration.md` : ce qu'il faut installer et
configurer dans l'Environnement B avant de pouvoir exécuter quoi que
ce soit décrit ci-dessus.
