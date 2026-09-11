\newpage

# Étape 1 : Couverture des exigences officielles de la mission

> Texte ci-dessous : reprise littérale du descriptif officiel de
> l'Étape 1 tel que fourni par la mission OpenClassrooms, suivi de la
> couverture réelle apportée par ce que nous avons construit à date.

## Texte officiel de la mission

**Dans cette étape vous allez :**

- Collecter, nettoyer et structurer un corpus médical bilingue
  (français / anglais) destiné au fine-tuning et à l'alignement par
  préférences.
- Produire environ 5 000 paires instruction-réponse pour SFT et
  constituer un jeu de paires préférentielles (DPO) validées
  cliniquement.
- Anonymiser toutes les données et documenter le processus RGPD.
- Définir le schéma des métadonnées (symptômes, antécédents,
  constantes, source, niveau de confiance).
- Préparer jeux train / val / test et jeux d'évaluations cliniques
  séparés.

**Prérequis**

- Avoir réalisé un inventaire des sources de données disponibles
  (MediQA, FrenchMedMCQA, MedQuAD, UltraMedical-Preference, etc.).
- Avoir accès aux environnements de stockage et compute (espace
  disque, notebooks).

**Résultats attendus**

- Dataset médical bilingue anonymisé et versionné, prêt pour SFT
  (≈5 000 paires) et pour la constitution du jeu DPO.
- Schéma des métadonnées.
- Justification du processus RGPD suivi.

## Couverture réelle, exigence par exigence

Légende : [FAIT] fait et vérifié · [OUTILLAGE PRET] outillage/pipeline prêt, exécution
sur données réelles restant à faire · [A FAIRE] pas encore commencé.

| Exigence officielle | Statut | Où le trouver |
|---|---|---|
| Collecter le corpus bilingue | [FAIT] Les 6 fichiers réels (4 sources, MediQAl en 3 fichiers) sont téléchargés dans `data/raw/` (`E1_01_telecharger_corpus.py`) | `LecteurCorpusFichierLocal`, `LecteurCorpusHuggingFace`, `E1_01_telecharger_corpus.py` ; cahier des charges §5.1 |
| Nettoyer et structurer | [FAIT] `ProfilerCorpusUseCase` + adaptateur `ydata-profiling` **validés par smoke test réel** (voir section dédiée ci-dessous) ; les 6 fichiers réels sont profilés (`data/processed/rapports_profilage/`), y compris `ultramedical_preference.jsonl` (966 Mo) via l'option `--bloque` | `E1_02_profiler_corpus.py` ; diagramme d'activité Étape 1 |
| ≈5 000 paires SFT | [FAIT] Les 5 fichiers SFT (MediQAl-oeq/mcqu/mcqm, FrenchMedMCQA, MedQuAD) sont mappés et fusionnés dans `data/processed/dataset_pivot.jsonl` : **37 802 exemples SFT réels** (pivot régénéré le 08/09/2026 avec identifiants déterministes, 49 doublons exacts dédoublonnés sur les 37 851 d'origine; voir section dédiée), largement au-dessus des ≈5 000 attendus | `E1_03_00_construire_dataset_pivot.py`, `E1_03_01_mappers_corpus.py` |
| Paires DPO validées cliniquement | [OUTILLAGE PRET] Mapper technique prêt **et corrigé** (`mapper_ultramedical_preference` extrayait mal chosen/rejected, voir section dédiée) et exécuté sur données réelles : **97 081 paires DPO réelles** (pivot régénéré le 08/09/2026, 12 272 doublons exacts dédoublonnés sur les 109 353 d'origine; voir section dédiée) ; la validation clinique par un expert est hors du périmètre purement technique et reste à planifier avec le CHSA | même fichier ; objectifs §3.2-3.3 |
| Anonymisation + documentation RGPD | [FAIT] Adaptateur `PresidioAnonymiseur` **validé par smoke test réel** ; bug de performance O(n²) corrigé ; processus incrémental/reprenable (`--limite`, échantillonnage stratifié). **Refonte 08/09/2026** : identifiants déterministes (§ dédiée), dataset pivot **régénéré** avec dédoublonnage réel, et l'anonymisation écrit désormais dans un fichier **séparé** (`dataset_pivot_anonymise.jsonl`); le pivot original n'est plus jamais modifié. Chaque exécution génère automatiquement/fusionne un **rapport RGPD cumulé** (JSON + Markdown, `application/use_cases/E1_04_03_rapport_anonymisation.py`); remplace le calcul manuel ponctuel. Contrôle qualité désormais **automatisé** par comparaison de fichiers (`E1_04_02_controler_qualite_anonymisation.py`, regex + seconde opinion spaCy) plutôt qu'une relecture manuelle ponctuelle; voir § dédiée et `01_rapport_rgpd.md` | `AnonymiserDatasetUseCase` ; `ControlerQualiteAnonymisationUseCase` ; `docs/02_etape1_donnees/01_rapport_rgpd.md` ; cahier des charges NF2 |
| Schéma de métadonnées | [FAIT] Défini **et implémenté** comme entité de domaine (`ExemplePivot`, `ConstantesVitales`) ; `identifiant` déterministe + `identifiant_source_brute` (traçabilité vers le registre brut d'origine) ajoutés le 08/09/2026 | `domain/model/exemple_pivot.py` ; cahier des charges §5.2 ; diagramme de paquets Étape 1 |
| Splits train / val / test + éval clinique isolée | [OUTILLAGE PRET] `DecouperSplitsUseCase` implémenté et testé, découpage stratifié par (type_exemple, source), sous-échantillonnage optionnel (`--n`) ; opère désormais sur `dataset_pivot_anonymise.jsonl` (fichier de sortie séparé, cf. § dédiée) ; à relancer après chaque vague d'anonymisation supplémentaire | `E1_05_00_decouper_splits.py` ; `E1_05_01_verifier_repartition_splits.py` ; `tests/application/` |

## Couverture des prérequis

| Prérequis | Statut | Où le trouver |
|---|---|---|
| Inventaire des sources de données | [FAIT] Les 4 corpus identifiés, avec liens Hugging Face et rôle (SFT/DPO) | Cahier des charges §5.1 |
| Accès environnements stockage/compute | [FAIT] Environnement A (local, `uv`) et Environnement B (distant, HF Jobs + Spaces Dev Mode) documentés et scriptés | Guide d'installation ; `scripts/check_env_local.py`, `scripts/check_env_gpu.py`, `scripts/check_env_remote_hf.py` |

## Couverture des résultats attendus

| Résultat attendu | Statut |
|---|---|
| Dataset bilingue anonymisé et versionné (≈5 000 paires SFT + jeu DPO) | [OUTILLAGE PRET] Dataset pivot **régénéré** (identifiants déterministes, dédoublonnage réel) : **134 883 exemples** (37 802 SFT + 97 081 DPO, 12 321 doublons exacts écartés sur 147 204 registres bruts) ; **5 000 exemples anonymisés et découpés en splits (première vague sous le nouveau schéma, 08/09/2026)**, atteignant l'objectif chiffré ≈5 000 de la mission, écrits dans un fichier séparé (`dataset_pivot_anonymise.jsonl`, le pivot original n'est jamais modifié) ; 129 883 exemples restants prêts pour des vagues ultérieures (voir section dédiée) |
| Schéma des métadonnées | [FAIT] Livré (voir ci-dessus) ; `identifiant` déterministe + `identifiant_source_brute` (traçabilité) ajoutés le 08/09/2026 |
| Justification du processus RGPD suivi | [FAIT] Génération **automatique et reproductible** à chaque exécution (08/09/2026) : rapport RGPD cumulé (`data/processed/rapport_anonymisation_rgpd.{json,md}`, 64 667 entités détectées sur les 5 000 premiers exemples de la vague en cours, 90,6 % de taux de détection ≥1 entité) et contrôle qualité automatisé par comparaison de fichiers (`rapport_controle_qualite_anonymisation.{json,md}`, regex + seconde opinion spaCy, 200 exemples comparés, 0 PII résiduelle confirmée, 35 candidats en attente de révision humaine explicitement marqués comme tels); voir section dédiée ci-dessous et `01_rapport_rgpd.md` pour le contexte historique (revue manuelle ponctuelle sur l'ancien pivot, non remplacée rétroactivement) |

## Validation technique effectuée (smoke test d'intégration, 02/09/2026)

> Contrairement au reste de ce document qui distingue « outillage
> prêt » de « exécuté sur données réelles », cette section documente
> une **exécution réelle** du pipeline complet; sur des données
> **synthétiques** représentatives du schéma des 4 corpus (`data/raw`
> n'étant pas accessible depuis cet environnement, cf. contrainte
> réseau), mais avec les **vraies bibliothèques** (Presidio,
> ydata-profiling, spaCy) installées et exécutées pour de vrai, pas
> mockées.

**Pipeline exécuté de bout en bout avec succès :**
`E1_02_profiler_corpus.py` → `E1_03_00_construire_dataset_pivot.py` (2 corpus
fusionnés) → `E1_04_00_anonymiser_dataset.py` → `E1_05_00_decouper_splits.py`.

### Deux bugs réels découverts et corrigés à cette occasion

| Bug | Symptôme | Correction |
|---|---|---|
| `ydata-profiling` dépend de `pkg_resources`, retiré des versions récentes de `setuptools` | `ModuleNotFoundError: No module named 'pkg_resources'` | Épinglage `setuptools<81` ajouté à `pyproject.toml` (extra `local`) |
| `AnalyzerEngine()` de Presidio construit sans configuration ne supporte que l'anglais par défaut, et tentait de télécharger automatiquement `en_core_web_lg` (~400 Mo) | `ValueError: No matching recognizers were found` sur tout texte marqué `langue="fr"` | `PresidioAnonymiseur` configure désormais explicitement un `NlpEngineProvider` multi-langue (FR via `fr_core_news_md`, EN via `en_core_web_sm`) |

### Limite réelle observée (pas un bug, une confirmation)

Sur le texte anglais synthétique contenant le numéro `555-0142`
(format court, sans indicatif), **Presidio ne l'a pas détecté**,
alors que `Jean Dupont`, `Marie Lefevre`, `John Smith` et le numéro
français `01 23 45 67 89` ont bien été masqués. Ceci confirme
concrètement (et non plus seulement en théorie) l'exigence NF2 du
cahier des charges : un **contrôle qualité manuel par échantillonnage
est obligatoire**, l'anonymisation automatique seule ne suffit pas à
garantir 0 PII résiduelle.

## MediQAl : 3 configurations Hub, 2 schémas différents (03/09/2026)

`ANR-MALADES/MediQAl` expose 3 configurations sur le Hub, téléchargées
séparément dans `data/raw/` :

| Configuration | Split disponible | Fichier | Schéma réel observé |
|---|---|---|---|
| `oeq` (question ouverte) | `test` uniquement (pas de `train`) | `mediqal_oeq.jsonl` | `id`, `clinical_case`, `question`, `answer`, `medical_subject`, `question_type` |
| `mcqu` (QCM, 1 réponse) | `train`/`validation`/`test` | `mediqal_mcqu.jsonl` | `id`, `clinical_case`, `question`, `answer_a`..`answer_e`, `correct_answers` (1 lettre), `task="QCU"`, `medical_subject`, `question_type` |
| `mcqm` (QCM, réponses multiples) | `train`/`validation`/`test` | `mediqal_mcqm.jsonl` | identique à `mcqu`, mais `correct_answers` peut contenir plusieurs lettres (ex. `"C,D"`) |

Seul `oeq` a un champ `answer` direct. `mcqu` et `mcqm` ont un schéma
QCM (champs plats `answer_a`..`answer_e` + `correct_answers`),
structurellement proche de FrenchMedMCQA mais pas identique.

**Résolu le 07/09/2026** : mapper dédié `mapper_mediqal_qcm` écrit
pour `mcqu`/`mcqm`, routé via deux nouvelles clés `--corpus`
(`mediqal_mcqu`, `mediqal_mcqm` ; `mediqal_oeq` ajoutée en alias
explicite de `mediqal`). Traitement simple sans liste d'options dans le prompt
(comme `mapper_medquad`); prompt = `clinical_case` (si non nul)
concaténé avec `question`, completion = texte(s) de la (des)
réponse(s) correcte(s) résolu(s) via `correct_answers`, concaténés
avec un espace pour le cas `mcqm` multi-réponses. Exécuté sur les
fichiers réels : 10 113 exemples pour `mcqu`, 5 767 pour `mcqm`,
**0 rejet** dans les deux cas.

## Validation réelle sur les 6 fichiers téléchargés (03/09/2026)

Contrairement à la section précédente (données synthétiques), cette
section documente le profilage **réel** des 6 fichiers de
`data/raw/` (les 4 corpus, MediQAl en 3 configurations).

### Bug réel : OOM sur ultramedical_preference.jsonl (966 Mo)

`E1_02_profiler_corpus.py` sans option chargeait tout le fichier en memoire
(pandas DataFrame -> liste de dicts -> nouveau DataFrame pour
`ydata-profiling`) : sur cette WSL2 a 5.8 Go de RAM, le processus
etait tue par l'OOM killer Linux (confirme via `dmesg`, aucune trace
Python puisque le kill est externe au processus). Corrige par l'ajout
de `LecteurCorpusFichierLocal(taille_bloc=N)` (lecture pandas
`chunksize`) et de l'option `--bloque N` de `E1_02_profiler_corpus.py`, qui
produit un rapport `ydata-profiling` complet par bloc plutot qu'un
seul rapport sur la totalite. **Limite assumee et documentee** :
correlations et taux de doublons calcules par bloc, jamais sur la
totalite du corpus. Valide en reel : 109353 enregistrements profiles
en 11 blocs sans erreur ni OOM (45 min).

### Bug réel : taux de doublons plantait sur colonnes liste/dict

Repere en cours de route sur le meme fichier : `YdataProfileur`
plantait (`TypeError: unhashable type: 'list'`) au calcul du taux de
doublons, car `chosen`/`rejected` (listes de messages, format chat)
et `metadata` (dict) ne sont pas hachables par pandas. Corrige en
stringifiant une copie du DataFrame juste pour cette detection ; le
DataFrame original transmis a `ProfileReport` n'est pas modifie.

### Bug réel : mapper_ultramedical_preference serialisait la liste entiere

`mapper_ultramedical_preference` faisait `str(chosen)`/`str(rejected)`
sur la liste de messages entiere (format chat confirme sur les 109353
enregistrements : toujours 2 messages, le dernier toujours
`role=assistant`), au lieu d'extraire le texte de la reponse. Consequence
reelle : le prompt duplique (message `role=user`) et la syntaxe de
dict Python se seraient retrouves dans le contenu du `Message` pivot,
au lieu du texte de reponse; aurait corrompu silencieusement les
paires DPO. Corrige par `_extraire_reponse_assistant()`, qui prend le
dernier message de la liste.

### Nouveaux tests ajoutés

- `tests/interfaces/test_mappers_corpus.py` : 8 tests suite au smoke
  test synthetique du 02/09, +1 test (`..._format_chat_reel`) suite a
  la validation reelle du 03/09 ci-dessus.
- `tests/infrastructure/test_presidio_anonymiseur.py` : test
  d'intégration **réel** (pas mocké) qui aurait détecté immédiatement
  le bug de langue ci-dessus ; s'auto-ignore proprement si
  Presidio/spaCy ne sont pas installés.



1. ~~Télécharger les 6 fichiers réels dans `data/raw/`.~~ Fait le
   03/09/2026, refait le 07/09/2026 (`E1_01_telecharger_corpus.py`).
2. ~~Exécuter `E1_02_profiler_corpus.py` sur chacun (rapport ydata-profiling).~~
   Fait le 03/09/2026 pour les 6 fichiers (`ultramedical_preference.jsonl`
   via `--bloque`, voir section dédiée ci-dessus).
3. ~~Écrire le mapper QCM manquant pour MediQAl mcqu/mcqm et corriger
   le bug FrenchMedMCQA.~~ Fait le 07/09/2026 (voir sections dédiées
   ci-dessus et ci-dessous).
4. ~~Construire le dataset pivot fusionné sur les 6 fichiers réels.~~
   Fait le 07/09/2026 : 147 204 exemples, 0 rejet (voir section
   dédiée ci-dessous); **régénéré depuis le 08/09/2026 avec
   identifiants déterministes et dédoublonnage réel, nouveau total
   134 883 exemples, voir point 7**.
5. ~~Anonymiser le dataset pivot réel et le découper en splits.~~
   Décision produit appliquée le 08/09/2026 : processus incrémental
   via `--limite` (échantillonnage stratifié). Première vague initiale
   (5 000 exemples sur l'ancien pivot à identifiants aléatoires)
   **remplacée par une nouvelle première vague sur le pivot régénéré**
   (voir point 7); l'anonymisation écrit désormais dans un fichier
   séparé, le pivot original n'est plus jamais modifié.
6. ~~Rédiger le rapport de justification RGPD à partir des résultats
   réels d'anonymisation.~~ Fait le 09/09/2026 (voir `01_rapport_rgpd.md`
   et la section dédiée ci-dessus) : résultats quantitatifs réels par
   corpus (nouvelle instrumentation `AnonymiserDatasetUseCase.statistiques`)
   et contrôle qualité manuel assisté sur 170 enregistrements ;
   **remplacé le 08/09/2026 par une génération automatique reproductible
   à chaque exécution, voir point 7**.
7. ~~Rendre le pipeline reproductible : identifiants déterministes,
   pivot régénéré, anonymisation vers un fichier séparé, rapport RGPD
   et contrôle qualité générés automatiquement.~~ Fait le 08/09/2026
  ; voir les sections dédiées ci-dessus et
   ci-dessous, `01_rapport_rgpd.md`, et le README.

## FrenchMedMCQA : bug de mapper corrigé contre le schéma réel (07/09/2026)

Le mapper `mapper_frenchmedmcqa` original lisait
`enregistrement.get("options")` (un dict `{lettre: texte}` supposé
d'après la fiche Hugging Face) et utilisait `correct_answers`
directement comme texte de réponse. Sur le fichier réel téléchargé
(`nthngdy/frenchmedmcqa`, 1080 enregistrements sur les 3 splits), ce
champ `options` **n'existe pas** (les options sont des champs plats
`answer_a`..`answer_e`) et `correct_answers` est un **entier**, pas
une lettre; le mapper produisait donc un prompt sans aucune option et
une completion litéralement égale au chiffre (ex. `"4"`).

Codification réelle confirmée via `datasets.load_dataset(...).features`
(pas seulement la fiche Hugging Face) :
- `correct_answers` : `Value("int64")`, un index **0-based** dans
  `a`..`e` (confirmé aussi manuellement sur plusieurs enregistrements
  réels; ex. `correct_answers=4` → "e" pour une question sur les
  particules alpha, `correct_answers=0` → "a" pour une question sur la
  progestérone).
- `number_correct_answers` : `ClassLabel(names=["1","2","3","4","5"])`;
  l'index `0` signifie "1 réponse correcte".
- Sur les **1080 enregistrements réels** (train+validation+test),
  `number_correct_answers` vaut **toujours 0** (= 1 seule réponse
  correcte) : le cas multi-réponses suggéré par ce champ n'existe pas
  dans ce miroir Hugging Face du dataset, et `correct_answers` étant
  un entier unique, il ne pourrait de toute façon pas encoder
  plusieurs index simultanément. Pas de gestion multi-réponses ajoutée
  en conséquence; non observable, non representable par ce schéma.

Mapper corrigé pour résoudre l'index vers `answer_a`..`answer_e` et
appliquer le même traitement simple que les autres sources QCM
(pas de liste d'options dans le prompt).
Test existant (`test_mapper_frenchmedmcqa_valide`) réécrit contre ce
schéma réel (il était écrit contre le schéma synthétique erroné), test
supplémentaire ajouté pour un index non nul. Exécuté sur le fichier
réel : 595 exemples (split `train`), **0 rejet**.

## Construction du dataset pivot sur les 6 fichiers réels (07/09/2026)

Pipeline `E1_03_00_construire_dataset_pivot.py` exécuté pour de vrai sur les 6
fichiers de `data/raw/` (téléchargés dans cette session, comptes
identiques à ceux déjà documentés) :

| Fichier source | `--corpus` | Exemples pivot | Rejets |
|---|---|---:|---:|
| `mediqal_oeq.jsonl` | `mediqal_oeq` | 4 969 | 0 |
| `mediqal_mcqu.jsonl` | `mediqal_mcqu` | 10 113 | 0 |
| `mediqal_mcqm.jsonl` | `mediqal_mcqm` | 5 767 | 0 |
| `frenchmedmcqa.jsonl` | `frenchmedmcqa` | 595 | 0 |
| `medquad.jsonl` | `medquad` | 16 407 | 0 |
| `ultramedical_preference.jsonl` | `ultramedical_preference` | 109 353 | 0 |
| **Total** | | **147 204** | **0** |

Soit **37 851 exemples SFT** (MediQAl×3 + FrenchMedMCQA + MedQuAD) et
**109 353 paires DPO** (UltraMedical-Preference); largement au-dessus
des ≈5 000 paires SFT attendues par la mission. Aucun enregistrement
rejeté sur aucune des 6 sources : chaque mapper a reconnu 100% des
enregistrements de son fichier.

**Bug d'infrastructure réel découvert et corrigé au passage** : sans
lecture par blocs, `LecteurCorpusFichierLocal` charge tout le fichier
source en DataFrame pandas d'un coup; sur `ultramedical_preference.jsonl`
(966 Mo, 109 353 enregistrements), ceci a provoqué un épuisement
mémoire réel sur l'environnement de 5.8 Go de RAM disponible (déjà
documenté côté `E1_02_profiler_corpus.py --bloque`, mais pas encore côté
`E1_03_00_construire_dataset_pivot.py`). Option `--taille-bloc N` ajoutée à
`E1_03_00_construire_dataset_pivot.py`, même principe que `E1_02_profiler_corpus.py`
(lecture pandas par blocs de N lignes, mémoire de pointe bornée par la
taille du bloc). Utilisée avec succès (`--taille-bloc 5000`) pour
produire les 109 353 exemples DPO ci-dessus sans OOM.

## Identifiants déterministes + dédoublonnage réel + régénération du pivot (08/09/2026)

> **Cette section SUPERSEDE les chiffres de la section précédente**
> (« Construction du dataset pivot sur les 6 fichiers réels »,
> 07/09/2026, 147 204 exemples avec identifiants aléatoires). Le pivot
> a été entièrement régénéré : le total réel actuel du dataset pivot
> est **134 883 exemples**, pas 147 204. La section précédente reste
> comme trace historique de ce qui a été construit ce jour-là, mais ne
> reflète plus l'état courant.

**Contexte** : l'anonymisation
écrit dans un fichier **séparé**, sans jamais modifier le pivot
original; nécessaire pour (a) pouvoir régénérer le pivot sans perdre
le texte original d'exemples déjà anonymisés, et (b) un contrôle
qualité a posteriori par simple comparaison de fichiers. Ce design
exige de croiser les enregistrements entre fichiers régénérés par un
identifiant **stable**, or `ExemplePivot.nouvel_identifiant` générait
jusqu'ici un UUID aléatoire à chaque appel.

### Identifiant déterministe (`ExemplePivot.nouvel_identifiant`)

Remplacé par un hash SHA-256 tronqué de `(espace_noms, cle_naturelle)`,
même entrée, toujours le même identifiant. `espace_noms` n'est **pas**
toujours égal au champ `source` de l'exemple : il doit être assez fin
pour éviter toute collision entre fichiers qui partagent le même
espace de clés naturelles. Vérifié sur les données réelles (pas
supposé) :

| Source | Clé naturelle | Espace de noms | Particularité découverte |
|---|---|---|---|
| MediQAl-oeq | champ `id` | `mediqal_oeq` | N/A |
| MediQAl-mcqu | champ `id` | `mediqal_mcqu` (via `task="QCU"`) | Partage 1 492 valeurs de `id` avec oeq |
| MediQAl-mcqm | champ `id` | `mediqal_mcqm` (via `task="QCM"`) | Partage 1 280 valeurs de `id` avec oeq |
| FrenchMedMCQA | champ `id` | `frenchmedmcqa` | N/A |
| MedQuAD | hash(`Question`+`Answer`) | `medquad` | Aucun champ `id` brut |
| UltraMedical-Preference | `prompt_id`+`label_type`+`chosen`+`rejected` | `ultramedical_preference` | `prompt_id` seul n'est PAS unique |

Sans cette vérification sur les fichiers réels, un espace de noms
unique `"mediqal"` pour les 3 configurations MediQAl aurait produit
des collisions d'identifiant entre des registres **différents**
(1 492 + 1 280 cas réels); testé explicitement
(`tests/interfaces/test_mappers_corpus.py::test_mapper_mediqal_oeq_meme_id_que_mcqu_ne_collisionne_pas`
et l'équivalent mcqu/mcqm).

Le champ `ExemplePivot.identifiant_source_brute` a été ajouté en plus
(traçabilité) : il conserve la clé
naturelle "brute" telle qu'exposée par le mapper (le `id` d'origine
pour MediQAl/FrenchMedMCQA, le `prompt_id` pour UltraMedical-Preference,
le hash Question+Answer pour MedQuAD); utile pour retrouver le
registre brut d'origine dans `data/raw/*.jsonl` sans avoir à
recalculer `identifiant`. Il se propage automatiquement aux fichiers
dérivés (anonymisé, splits) puisque `dataclasses.replace(...)` ne
touche jamais ce champ.

### Dédoublonnage réel découvert (dédoublonner pour de vrai)

Conséquence directe de la déterminisme : deux registres bruts
strictement identiques sur les champs qui alimentent le pivot
produisent désormais le **même** identifiant. Vérifié sur les
fichiers réels (pas supposé) :

| Source | Registres bruts | Valeurs de clé uniques | Doublons exacts |
|---|---:|---:|---:|
| FrenchMedMCQA | 595 | 594 | **1** |
| MedQuAD | 16 407 | 16 359 | **48** |
| UltraMedical-Preference | 109 353 | 97 081 | **12 272** |
| **Total doublons** | | | **12 321** |

**`ConstruireDatasetPivotUseCase` dédoublonne réellement** (décision
prise le 08/09/2026, après évaluation; l'alternative
"index d'occurrence artificiel pour préserver le total 147 204" a été
écartée) : seul le premier exemple rencontré pour un identifiant donné
est conservé dans le pivot ; les registres écartés sont archivés (pas
silencieusement perdus) dans **`data/processed/doublons_supprimes.jsonl`**
(12 321 lignes, format `ExemplePivot`, un fichier partagé entre les 6
exécutions de `E1_03_00_construire_dataset_pivot.py`, mode ajout).

**Investigation légère sur la cause des doublons UltraMedical-Preference**
(hypothèse retenue : chevauchement entre/dans les datasets
source) : la distribution de multiplicité des clés dupliquées est
`{1: 84 809, 2: 12 272}`; **aucune clé n'apparaît plus de 2 fois**,
et le taux de duplication est réparti proportionnellement à travers
tous les préfixes de source visibles dans `prompt_id` (WikiInstruct,
MedMCQA, TextBookQA, ChatDoctor, MedQA, MedQA-Evol, MedQuad,
Medical-Instruct-120k, PubMedQA, MedInstruct-52k; tous représentés
dans les doublons à peu près à hauteur de leur poids réel dans le
corpus). Ce profil (toujours exactement 2 copies, proportionnel à
toutes les sources) est plus cohérent avec un **artefact d'export en
double** lors de la construction du corpus `UltraMedical-Preference`
lui-même (concaténation de deux exports qui se recouvrent, ou une
étape de dédoublonnage amont manquée) qu'avec un chevauchement de
questions **entre** sous-corpus différents (ce qui produirait des
`prompt_id` différents pour un contenu similaire, pas des quadruplets
`prompt_id`+`label_type`+`chosen`+`rejected` strictement identiques).
Investigation volontairement limitée à ce constat (pas de fouille plus
poussée dans la provenance du miroir Hugging Face); hors du périmètre
de cette tâche (identifiant + anonymisation + splits + contrôle
qualité), pas nécessaire pour trancher la décision de dédoublonnage.

### Régénération complète du pivot (08/09/2026)

Les 6 commandes de `E1_03_00_construire_dataset_pivot.py` (§ README) ont été
ré-exécutées depuis zéro sur les mêmes fichiers `data/raw/` (aucun
re-téléchargement), avec les mappers mis à jour (identifiant
déterministe) :

| Fichier source | `--corpus` | Registres bruts | Exemples pivot | Doublons écartés |
|---|---|---:|---:|---:|
| `mediqal_oeq.jsonl` | `mediqal_oeq` | 4 969 | 4 969 | 0 |
| `mediqal_mcqu.jsonl` | `mediqal_mcqu` | 10 113 | 10 113 | 0 |
| `mediqal_mcqm.jsonl` | `mediqal_mcqm` | 5 767 | 5 767 | 0 |
| `frenchmedmcqa.jsonl` | `frenchmedmcqa` | 595 | 594 | 1 |
| `medquad.jsonl` | `medquad` | 16 407 | 16 359 | 48 |
| `ultramedical_preference.jsonl` | `ultramedical_preference` | 109 353 | 97 081 | 12 272 |
| **Total** | | **147 204** | **134 883** | **12 321** |

Soit **37 802 exemples SFT** (MediQAl 20 849 + FrenchMedMCQA 594 +
MedQuAD 16 359, toujours largement au-dessus des ≈5 000 attendus par
la mission) et **97 081 paires DPO** (UltraMedical-Preference).
Vérification de cohérence effectuée : 0 collision d'identifiant
résiduelle dans le pivot final (134 883 identifiants tous uniques),
et le total par source (`MediQAl` 20 849, `FrenchMedMCQA` 594,
`MedQuAD` 16 359, `UltraMedical-Preference` 97 081) correspond
exactement à `registres bruts - doublons écartés` pour chaque source.

## Anonymisation vers fichier séparé + rapport RGPD et contrôle qualité automatiques (08/09/2026, exécution réelle)

Sur le pivot régénéré (134 883 exemples, § précédente), le nouveau
pipeline a été exécuté de bout en bout pour de vrai (pas un test) :

```bash
uv run python interfaces/cli/E1_04_00_anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --sortie data/processed/dataset_pivot_anonymise.jsonl --strategie replace --limite 5000
uv run python interfaces/cli/E1_04_02_controler_qualite_anonymisation.py --dataset data/processed/dataset_pivot.jsonl --anonymise data/processed/dataset_pivot_anonymise.jsonl --taille-echantillon 200
uv run python interfaces/cli/E1_05_00_decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl
uv run python interfaces/cli/E1_05_01_verifier_repartition_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl
```

**Anonymisation** (durée réelle mesurée : 57 min 48 s pour 5 000
exemples, cohérente avec les temps par source mesurés en 07-08/09,
voir § dédiée ci-dessous); rapport RGPD cumulé généré automatiquement
dans `data/processed/rapport_anonymisation_rgpd.{json,md}` :

| Source | Traités (cette vague) | Avec ≥1 entité | Taux |
|---|---:|---:|---:|
| FrenchMedMCQA | 22 | 7 | 31,8 % |
| MedQuAD | 606 | 528 | 87,1 % |
| MediQAl | 773 | 421 | 54,5 % |
| UltraMedical-Preference | 3 599 | 3 575 | 99,3 % |
| **Total** | **5 000** | **4 531** | **90,6 %** |

Soit **5 000/134 883 exemples anonymisés (3,7 % du dataset pivot,
compté réellement dans le fichier à l'exécution)**, **64 667 entités
détectées** au total. Portée explicitement indiquée dans le rapport
(cumulé sur 1 exécution à ce stade, tracé avec horodatage/stratégie/
limite/graine; voir `data/processed/rapport_anonymisation_rgpd.md`
§ « Executions ayant contribué »).

**Contrôle qualité automatique** (200 exemples comparés, échantillon
stratifié parmi les 5 000 disponibles, regex + seconde opinion spaCy,
rapport dans `data/processed/rapport_controle_qualite_anonymisation.{json,md}`) :

- **631 candidats de PII résiduelle** détectés par regex sur le texte
  anonymisé : **0 confirmés** (aucun match déterministe email/
  téléphone/URL/date non masqué, et aucun bigramme capitalisé confirmé
  entité nommée par spaCy), **596 écartés** par spaCy (bigrammes
  capitalisés type titres de section markdown `### Conclusion`,
  vocabulaire scientifique; cohérent avec les faux positifs déjà
  documentés dans la revue manuelle historique), **35 en attente de
  revision humaine** explicitement marqués comme tels (jamais une
  confirmation inventée).
- **18 candidats de faux positifs de masquage** (fragments originaux
  masqués sans confirmation spaCy qu'il s'agissait d'une entité
  nommée) : 2 sans aucune entité détectée, 16 avec une entité d'un
  type non tranchant (ex. plages temporelles `"30 minutes"`,
  `"hours to days"` masquées à tort comme `DATE_TIME`, âges
  `"30-year-old"`; sur-masquage inoffensif RGPD-wise mais qui dégrade
  la lisibilité, cohérent avec le phénomène déjà documenté dans la
  revue manuelle historique §4 de `01_rapport_rgpd.md`).
- Exemples réels (original → anonymisé) inspectables dans le rapport
  Markdown pour chacune des 4 sources.

**Exclusion des candidats en attente du décrédelage train/val/test
(décision du capitaine, 10/09/2026)** : un `ExemplePivot` portant au
moins un candidat de PII résiduelle encore SANS décision humaine
persistée (`en attente de revision humaine` ci-dessus) est exclu par
précaution du décrédelage `E1_05_00_decouper_splits.py`, plutôt que de bloquer
le pipeline en attendant qu'une personne tranche chaque candidat un
par un ; voir `DecouperSplitsUseCase.obtenir_identifiants_pii_en_attente`
et `ReviserPiiResiduelleUseCase.identifiants_en_attente`. Cet exemple
reste sans `split` (ni train, ni val, ni test) jusqu'à ce qu'une
décision humaine soit prise (`E1_04_01_reviser_pii_residuelle.py verify`) ou
que le candidat cesse d'exister après un reproces. Conséquence directe
sur la lecture de l'exigence NF2 : **« 0 PII résiduelle validée
manuellement »** ne couvre que les candidats ayant effectivement reçu
une décision explicite (`decisions_revision_humaine.jsonl`), pas
l'ensemble des candidats détectés par le contrôle qualité : les
candidats encore en attente ne sont ni confirmés ni infirmés, ils sont
simplement tenus à l'écart de l'entraînement/évaluation.

**Splits** (sur les 5 000 exemples anonymisés) : identique à la vague
historique (4 004 train / 498 val / 498 test), vérifié représentatif
par strate à ~80/10/10 dans chaque `(type_exemple, source)` (y compris
`FrenchMedMCQA`, 22 exemples, 18/2/2).

Cette exécution remplace la « première vague » historique (§
ci-dessous, ancien pivot à identifiants aléatoires) sous le nouveau
schéma reproductible. **142 204 exemples restaient dans l'ancien
schéma ; il en reste désormais 129 883** (134 883 − 5 000) à traiter
par des vagues ultérieures (`E1_04_00_anonymiser_dataset.py --limite N` plus
grand, ou `full`).

## Anonymisation réelle (ANCIEN schéma, ids aléatoires) : bug de performance O(n²) corrigé, decision produit appliquée, premiere vague exécutée (07-08/09/2026)

> **SUPERSEDÉ le 08/09/2026** par le design source/sortie séparés (cf.
> section dédiée plus bas) : la « première vague » décrite ci-dessous
> anonymisait le pivot **en place** (identifiants aléatoires, champ
> `anonymise` muté sur le fichier source lui-même); ce fichier n'existe
> plus, remplacé par le pivot régénéré (134 883 exemples,
> identifiants déterministes, jamais modifié) et une nouvelle première
> vague écrite dans `dataset_pivot_anonymise.jsonl` (fichier séparé).
> Les mesures de temps ci-dessous (durée par source, bug O(n²)
> corrigé) restent valables techniquement; le mécanisme
> d'anonymisation Presidio lui-même n'a pas changé, seul l'emplacement
> d'écriture du résultat a changé.

En préparant l'exécution de `E1_04_00_anonymiser_dataset.py` sur le dataset
pivot réel (147 204 exemples, 624 Mo), un second bug d'infrastructure
réel a été découvert : `AnonymiserDatasetUseCase.executer()` appelait
`self.repository.sauvegarder(exemple)` **à chaque itération** de la
boucle sur les 147 204 exemples. Or `JsonlDatasetRepository.sauvegarder`
relit et réécrit **tout le fichier JSONL** à chaque appel; sur 147 204
exemples cela revient à relire/réécrire un fichier de 624 Mo 147 204
fois, un O(n²) totalement infaisable (des heures rien que pour l'I/O
disque). Corrigé : la boucle accumule désormais les exemples
anonymisés puis appelle `sauvegarder_plusieurs(...)` **une seule fois**
à la fin (même méthode déjà utilisée par
`ConstruireDatasetPivotUseCase`), ramenant le coût I/O à une seule
lecture + une seule écriture du fichier, quel que soit le nombre
d'exemples traités.

Ce correctif rend l'exécution *faisable* mais ne suffit pas à la
rendre *rapide* : le coût dominant reste le calcul NLP (spaCy via
Presidio) sur chaque champ texte. Mesuré directement sur des extraits
réels du dataset pivot (`AnalyzerEngine.analyze` par champ,
échantillons de 20 à 30 enregistrements par source) :

| Source (offset dans le pivot) | Temps mesuré / enregistrement | Enregistrements | Temps estimé |
|---|---:|---:|---:|
| MediQAl-oeq (FR, court) | ~162 ms | 4 969 | ~13 min |
| MediQAl-mcqu (FR, moyen) | ~130 ms | 10 113 | ~22 min |
| MediQAl-mcqm (FR, moyen, non mesuré séparément) | ~130 ms (estimé) | 5 767 | ~12 min |
| FrenchMedMCQA (FR, court) | ~162 ms (estimé, non mesuré) | 595 | ~2 min |
| MedQuAD (EN, moyen) | ~300 ms | 16 407 | ~82 min |
| UltraMedical-Preference (EN, long, 2 x ~5000 car./enreg.) | ~548 ms | 109 353 | **~16h40** |
| **Total estimé** | | **147 204** | **~18h50** |

**Décision produit (08/09/2026)** : ne pas trancher une
fois pour toutes entre « échantillon » et « dataset complet ». Le
champ `ExemplePivot.anonymise` (deja présent dans le schéma pivot)
rend le processus nativement **incrémental et reprenable** :
`AnonymiserDatasetUseCase` ne retraite jamais un exemple déjà
`anonymise=True`. Un nouveau flag `--limite N` sur
`E1_04_00_anonymiser_dataset.py` (défaut **5000**, l'objectif chiffré de la
mission) anonymise à chaque appel un **échantillon stratifié** par
`(type_exemple, source)` de taille `N` parmi les exemples encore
`anonymise=False` (méthode du plus grand reste pour les quotas par
strate, tirage aléatoire seedé et reproductible) ; le reste du
dataset n'est pas touché et attend un appel ultérieur avec un `N` plus
grand ou `--limite full` (aucune limite). `E1_05_00_decouper_splits.py` a été
mis à jour en cohérence : découpage **stratifié** par
`(type_exemple, source)` plutôt qu'un shuffle global (une petite
source comme FrenchMedMCQA aurait pu se retrouver totalement absente
de train ou de test face à UltraMedical-Preference, ~180x plus
grande), et le même bug O(n²) corrigé (`sauvegarder_plusieurs` au lieu
de `sauvegarder` par itération).

**Première vague exécutée sur le dataset réel (08/09/2026)** :
`E1_04_00_anonymiser_dataset.py --limite 5000` (durée réelle mesurée :
**36 min 26 s**, cohérente avec l'estimation), puis
`E1_05_00_decouper_splits.py` sur les 5 000 exemples désormais anonymisés :

| Source | Anonymisés (sur cette vague) | Restants (`anonymise=False`) | Répartition split (train/val/test) |
|---|---:|---:|---|
| MediQAl (oeq+mcqu+mcqm confondus, même `source`) | 708 | 20 141 | 568/70/70 |
| FrenchMedMCQA | 20 | 575 | 16/2/2 |
| MedQuAD | 557 | 15 850 | 447/55/55 |
| UltraMedical-Preference | 3 715 | 105 638 | 2 973/371/371 |
| **Total** | **5 000** | **142 204** | **4 004/498/498** |

Chaque quota est proportionnel au poids réel de la source dans le
dataset (ex. MediQAl : 20 849/147 204 × 5 000 ≈ 708 ✓), et chaque
`(type_exemple, source)` est bien représenté dans train **et** val
**et** test, y compris la plus petite strate (FrenchMedMCQA, 20
exemples anonymisés). Le reste du dataset (142 204 exemples,
`anonymise=False`) est prêt à être traité par vagues ultérieures
(`--limite` plus grand, ou `full`) sans jamais retraiter ce qui est
déjà fait; décision explicitement laissée ouverte sur *quand* et
*avec quel N* lancer la vague suivante (à réévaluer avant le SFT/DPO
en fonction du risque de saturation/surapprentissage sur un
sous-échantillon trop petit, cf. § dédiée dans le README).

Vérification ajoutée après coup (`E1_05_01_verifier_repartition_splits.py`) :
chaque strate `(type_exemple, source)` respecte bien les proportions
80/10/10 demandées, **dans chaque split**, pas seulement au global :

| Strate | Total | train | val | test |
|---|---:|---|---|---|
| dpo/UltraMedical-Preference | 3 715 | 2 973 (80,0 %) | 371 (10,0 %) | 371 (10,0 %) |
| sft/FrenchMedMCQA | 20 | 16 (80,0 %) | 2 (10,0 %) | 2 (10,0 %) |
| sft/MedQuAD | 557 | 447 (80,3 %) | 55 (9,9 %) | 55 (9,9 %) |
| sft/MediQAl | 708 | 568 (80,2 %) | 70 (9,9 %) | 70 (9,9 %) |

## Rapport de justification RGPD complété avec des données réelles (09/09/2026)

> **SUPERSEDÉ le 08/09/2026** par la génération automatique du rapport
> RGPD (l'exigence est que le pipeline produise lui-même ses
> indicateurs, de façon reproductible, plutôt qu'un calcul manuel
> ponctuel comme celui décrit ci-dessous). Voir la section
> « Génération automatique du rapport RGPD + contrôle qualité par
> comparaison de fichiers (08/09/2026) » plus bas pour le nouveau
> mécanisme. L'échantillon de 5 000 exemples et l'instrumentation
> `AnonymiserDatasetUseCase.statistiques` décrits ci-dessous restent
> corrects comme description de principe, mais les chiffres exacts
> (5 000/147 204, 708/20/557/3 715 par source) portaient sur l'ANCIEN
> pivot (identifiants aléatoires, depuis régénéré) et ne reflètent
> plus l'état courant; voir `01_rapport_rgpd.md` pour la note de tête
> à jour.

`docs/02_etape1_donnees/01_rapport_rgpd.md` (auparavant un gabarit
`01_rapport_rgpd_template.md`) est désormais complété avec des chiffres
réels, pas des estimations :

- **Instrumentation ajoutée** : `AnonymiserDatasetUseCase` expose
  maintenant `self.statistiques: dict[str, StatistiquesSource]`
  (registres traités, registres avec ≥1 entité, détail des entités
  par type, par source), accumulée pendant l'anonymisation à partir du
  détail déjà renvoyé par `ResultatAnonymisation.entites_detectees`
  (auparavant calculé puis jeté à chaque champ anonymisé).
- **Section 3 (résultats quantitatifs)** : la stratégie `replace`
  utilisée pour la vague déjà fusionnée remplace toute entité par un
  jeton générique unique; le détail par type n'est donc pas
  récupérable a posteriori sur ce texte déjà anonymisé. Un nouvel
  échantillon stratifié de 5 000 exemples a été prélevé avec le même
  algorithme/graine/corpus (`echantillon_stratifie`, factorisé dans
  `chsa_triage.application.echantillonnage`); répartition par source
  vérifiée identique à celle de la vague déjà fusionnée (708/20/557/3 715) ;
  puis anonymisé pour de vrai (Presidio réel) avec l'instrumentation
  ci-dessus. Résultat : 65 914 entités détectées, 92,0 % des 5 000
  enregistrements avec ≥1 entité (détail par type et par corpus dans
  le rapport).
- **Section 4 (contrôle qualité manuel)** : 170 enregistrements relus
  (50 par source, ou tous les disponibles si moins de 50; 20 pour
  FrenchMedMCQA), avec comparaison texte original/texte anonymisé.
  **1 PII résiduelle réelle trouvée** (un prénom, `chsa-ultramedical-f87736240ce5`)
  et des faux positifs fréquents documentés (vocabulaire médical/
  scientifique capitalisé pris pour des entités nommées, sur-masquage
  systématique des sections bibliographiques d'UltraMedical-Preference).
  Cette revue constitue une **première passe de contrôle, pas une revue
  humaine indépendante**; le rapport recommande explicitement qu'un réviseur
  du domaine confirme avant tout usage clinique réel ;
  verdict retenu pour l'usage actuel (POC, fine-tuning
  expérimental) : dataset accepté en l'état, avec réserve sur le
  sur-masquage bibliographique d'UltraMedical-Preference.

## `--n` sur `E1_05_00_decouper_splits.py` et vérification de la représentativité par strate (09/09/2026)

Besoin produit : pouvoir demander un dataset d'entraînement de taille
N plutôt que de toujours repartir tout ce qui est anonymisé. `--n N`
(optionnel) sur `E1_05_00_decouper_splits.py`/`DecouperSplitsUseCase` prélève
d'abord un échantillon stratifié de taille N parmi les exemples
anonymisés (même algorithme du plus grand reste que `--limite`),
avant de répartir train/val/test dessus. Si N est omis ou ≥ au nombre
d'exemples anonymisés disponibles, comportement inchangé (tout est
reparti); un avertissement est tracé via LogTool, sans erreur, comme
pour les autres scripts Étape 1.

L'algorithme d'échantillonnage stratifié (méthode du plus grand reste),
auparavant privé à `AnonymiserDatasetUseCase`, a été extrait dans
`chsa_triage.application.echantillonnage.echantillon_stratifie` pour
être réutilisé par les deux cas d'usage sans duplication.

Un nouveau script `E1_05_01_verifier_repartition_splits.py` (et son cas d'usage
`VerifierRepartitionSplitsUseCase`) relit le dataset pivot déjà reparti
et affiche, par strate `(type_exemple, source)`, le décompte ET le
pourcentage par split; `E1_05_00_decouper_splits.py` n'affichait que le total
global, ce qui ne permettait pas de vérifier visuellement que la
stratification restait représentative dans **chaque** split (voir
tableau ci-dessus).

## Croissance stable de `E1_05_00_decouper_splits.py` : protection contre la fuite train/test à l'agrandissement (10/09/2026)

**Décision**, prise explicitement après comparaison de
deux alternatives. Constat : dans le comportement décrit à la section
précédente, `--n` prélevait un échantillon **frais** à chaque exécution
et **réassignait le split de TOUS les exemples considérés** (que ce
soit le sous-ensemble `--n` ou le dataset anonymisé entier). Relancer
`E1_05_00_decouper_splits.py` avec un `--n` différent (par exemple élargir de
5 000 à 10 000, ou omettre `--n` pour tout répartir) pouvait donc faire
passer un exemple déjà vu de `train` à `test`; ou l'inverse; d'une
exécution à l'autre. C'est une **fuite silencieuse** de données
d'entraînement dans le jeu d'évaluation, exactement le point de
vigilance explicite du cahier des charges de la mission : *« le jeu de
test ne doit jamais être réutilisé en entraînement »*. Un modèle évalué
sur un exemple qu'il a vu en entraînement lors d'une vague antérieure
donnerait des métriques de test artificiellement optimistes, sans
qu'aucune erreur ne le signale; le bug n'aurait été visible qu'en
comparant manuellement les identifiants de deux exécutions successives.

**Alternatives comparées** : (a) recalculer tout le découpage à chaque
exécution (comportement historique, simple mais sujet à la fuite
ci-dessus dès qu'on agrandit le dataset) vs (b) figer définitivement le
split de tout exemple déjà vu et ne répartir que les exemples
nouvellement candidats. Le choix retenu est (b).

**Nouveau comportement** (`DecouperSplitsUseCase.executer`, cf.
`src/chsa_triage/application/use_cases/E1_05_00_decouper_splits.py`) :
un exemple qui a déjà un `split` non nul (exécution antérieure) n'est
**plus jamais réassigné**, quel que soit le `--n` demandé ensuite.
`--n N` est désormais une taille **cible cumulée** : seuls
`N - (nombre déjà assigné)` exemples **nouveaux** (choisis parmi ceux
sans split, par le même échantillonnage stratifié qu'avant) reçoivent
un split lors de cette exécution. Omettre `--n` bascule le mode
« tout répartir » en mode « compléter ce qui manque »; changement de
comportement réel par rapport à la section précédente, documenté aussi
dans le README (§6/§7) et le docstring du module. Réduire un découpage
déjà fait (`--n` ≤ au nombre déjà assigné) n'est **pas supporté** : le
jeu ne peut que grandir, jamais rétrécir; un avertissement est tracé
via LogTool (pas une erreur) dans ce cas, la décision explicite étant
qu'encoger n'a pas de cas d'usage légitime ici (contrairement à
agrandir, qui correspond au besoin produit réel de vagues successives
d'anonymisation).

`E1_05_01_verifier_repartition_splits.py` n'a nécessité **aucune modification** :
il relit simplement les splits déjà présents dans le fichier de sortie,
quelle que soit la séquence d'exécutions `--n` qui les a produits; la
garantie de stabilité vit entièrement côté écriture
(`DecouperSplitsUseCase`), pas côté lecture.

Couvert par un test dédié
(`tests/application/test_decouper_splits.py`) : découpage avec `--n
5000`, relevé des identifiants et de leur split par identifiant, puis
second découpage avec `--n 10000` sur le même repository; vérification
que les 5 000 premiers identifiants conservent **exactement** le même
split qu'à la première exécution.
