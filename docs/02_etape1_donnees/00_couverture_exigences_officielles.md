\newpage

# Étape 1 — Couverture des exigences officielles de la mission

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
| Collecter le corpus bilingue | [FAIT] Les 4 corpus réels sont téléchargés dans `data/raw/` (`telecharger_corpus.py`), MediQAl en 3 fichiers (voir section dédiée ci-dessous) | `LecteurCorpusFichierLocal`, `LecteurCorpusHuggingFace`, `telecharger_corpus.py` ; cahier des charges §5.1 |
| Nettoyer et structurer | [FAIT] `ProfilerCorpusUseCase` + adaptateur `ydata-profiling` **validés par smoke test réel** (voir section dédiée ci-dessous) ; les 6 fichiers réels sont profilés (`data/processed/rapports_profilage/`), y compris `ultramedical_preference.jsonl` (966 Mo) via l'option `--bloque` | `profiler_corpus.py` ; diagramme d'activité Étape 1 |
| ≈5 000 paires SFT | [OUTILLAGE PRET] Mappers écrits et corrigés pour FrenchMedMCQA, MedQuAD et MediQAl-oeq (`mappers_corpus.py`) ; **mapper manquant pour MediQAl-mcqu/mcqm** (schéma QCM différent, voir section dédiée) ; execution de `construire_dataset_pivot.py` restant a faire | `construire_dataset_pivot.py` |
| Paires DPO validées cliniquement | [OUTILLAGE PRET] Mapper technique prêt **et corrigé** (`mapper_ultramedical_preference` extrayait mal chosen/rejected, voir section dédiée) ; la validation clinique par un expert est hors du périmètre purement technique et reste à planifier avec le CHSA | même fichier ; objectifs §3.2-3.3 |
| Anonymisation + documentation RGPD | [OUTILLAGE PRET] Adaptateur `PresidioAnonymiseur` **validé par smoke test réel** (bug de configuration multi-langue découvert et corrigé, cf. section dédiée) ; exécution sur données réelles + rapport de contrôle qualité RGPD restant à produire | `AnonymiserDatasetUseCase` ; cahier des charges NF2 |
| Schéma de métadonnées | [FAIT] Défini **et implémenté** comme entité de domaine (`ExemplePivot`, `ConstantesVitales`) | `domain/model/exemple_pivot.py` ; cahier des charges §5.2 ; diagramme de paquets Étape 1 |
| Splits train / val / test + éval clinique isolée | [FAIT] Implémenté et testé (`DecouperSplitsUseCase`), test clinique jamais réutilisé en entraînement | `decouper_splits.py` ; `tests/application/` |

## Couverture des prérequis

| Prérequis | Statut | Où le trouver |
|---|---|---|
| Inventaire des sources de données | [FAIT] Les 4 corpus identifiés, avec liens Hugging Face et rôle (SFT/DPO) | Cahier des charges §5.1 |
| Accès environnements stockage/compute | [FAIT] Environnement A (local, `uv`) et Environnement B (distant, HF Jobs + Spaces Dev Mode) documentés et scriptés | Guide d'installation ; `scripts/check_env_local.py`, `scripts/check_env_gpu.py`, `scripts/check_env_remote_hf.py` |

## Couverture des résultats attendus

| Résultat attendu | Statut |
|---|---|
| Dataset bilingue anonymisé et versionné (≈5 000 paires SFT + jeu DPO) | [A FAIRE] Pipeline complet prêt de bout en bout (voir diagramme d'activité), production réelle du dataset final restant à exécuter |
| Schéma des métadonnées | [FAIT] Livré (voir ci-dessus) |
| Justification du processus RGPD suivi | [OUTILLAGE PRET] Stratégie et outillage documentés (Presidio, 3 stratégies comparées) ; le rapport de justification final (taux de détection, contrôle qualité manuel) reste à rédiger une fois l'anonymisation exécutée sur données réelles |

## Validation technique effectuée (smoke test d'intégration, 02/09/2026)

> Contrairement au reste de ce document qui distingue « outillage
> prêt » de « exécuté sur données réelles », cette section documente
> une **exécution réelle** du pipeline complet — sur des données
> **synthétiques** représentatives du schéma des 4 corpus (`data/raw`
> n'étant pas accessible depuis cet environnement, cf. contrainte
> réseau), mais avec les **vraies bibliothèques** (Presidio,
> ydata-profiling, spaCy) installées et exécutées pour de vrai, pas
> mockées.

**Pipeline exécuté de bout en bout avec succès :**
`profiler_corpus.py` → `construire_dataset_pivot.py` (2 corpus
fusionnés) → `anonymiser_dataset.py` → `decouper_splits.py`.

### Deux bugs réels découverts et corrigés à cette occasion

| Bug | Symptôme | Correction |
|---|---|---|
| `ydata-profiling` dépend de `pkg_resources`, retiré des versions récentes de `setuptools` | `ModuleNotFoundError: No module named 'pkg_resources'` | Épinglage `setuptools<81` ajouté à `pyproject.toml` (extra `local`) |
| `AnalyzerEngine()` de Presidio construit sans configuration ne supporte que l'anglais par défaut, et tentait de télécharger automatiquement `en_core_web_lg` (~400 Mo) | `ValueError: No matching recognizers were found` sur tout texte marqué `langue="fr"` | `PresidioAnonymiseur` configure désormais explicitement un `NlpEngineProvider` multi-langue (FR via `fr_core_news_md`, EN via `en_core_web_sm`) |

### Limite réelle observée (pas un bug, une confirmation)

Sur le texte anglais synthétique contenant le numéro `555-0142`
(format court, sans indicatif), **Presidio ne l'a pas détecté** —
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

Seul `oeq` a un champ `answer` direct : c'est le seul que
`mapper_mediqal` (`mappers_corpus.py`) couvre aujourd'hui. `mcqu` et
`mcqm` ont un schéma QCM (options `answer_a`..`answer_e` +
`correct_answers`), structurellement proche de FrenchMedMCQA mais pas
identique (`mapper_frenchmedmcqa` attend un dict `options`, pas des
champs plats) : il n'existe pas encore de mapper pour ces deux
configurations. Les exécuter aujourd'hui via `construire_dataset_pivot.py
--corpus mediqal` ne provoquerait aucune erreur, mais produirait 0
exemple pivot (le mapper renvoie `None` pour chaque enregistrement,
faute de champ `answer`).

**Décision restant à prendre** (produit, pas seulement technique) :
comment traiter le cas `mcqm` à réponses multiples dans le mapper à
écrire — même traitement que `mcqu` (concaténer les réponses
correctes) ou traitement distinct ? Non tranché à ce jour.

## Validation réelle sur les 6 fichiers téléchargés (03/09/2026)

Contrairement à la section précédente (données synthétiques), cette
section documente le profilage **réel** des 6 fichiers de
`data/raw/` (les 4 corpus, MediQAl en 3 configurations).

### Bug réel : OOM sur ultramedical_preference.jsonl (966 Mo)

`profiler_corpus.py` sans option chargeait tout le fichier en memoire
(pandas DataFrame -> liste de dicts -> nouveau DataFrame pour
`ydata-profiling`) : sur cette WSL2 a 5.8 Go de RAM, le processus
etait tue par l'OOM killer Linux (confirme via `dmesg`, aucune trace
Python puisque le kill est externe au processus). Corrige par l'ajout
de `LecteurCorpusFichierLocal(taille_bloc=N)` (lecture pandas
`chunksize`) et de l'option `--bloque N` de `profiler_corpus.py`, qui
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
au lieu du texte de reponse — aurait corrompu silencieusement les
paires DPO. Corrige par `_extraire_reponse_assistant()`, qui prend le
dernier message de la liste.

### Nouveaux tests ajoutés

- `tests/interfaces/test_mappers_corpus.py` — 8 tests suite au smoke
  test synthetique du 02/09, +1 test (`..._format_chat_reel`) suite a
  la validation reelle du 03/09 ci-dessus.
- `tests/infrastructure/test_presidio_anonymiseur.py` — test
  d'intégration **réel** (pas mocké) qui aurait détecté immédiatement
  le bug de langue ci-dessus ; s'auto-ignore proprement si
  Presidio/spaCy ne sont pas installés.



1. ~~Télécharger les 4 corpus réels dans `data/raw/`.~~ Fait le
   03/09/2026 (`telecharger_corpus.py`) — MediQAl en 3 fichiers
   (`oeq`/`mcqu`/`mcqm`, voir section dédiée ci-dessus).
2. ~~Exécuter `profiler_corpus.py` sur chacun (rapport ydata-profiling).~~
   Fait le 03/09/2026 pour les 6 fichiers (`ultramedical_preference.jsonl`
   via `--bloque`, voir section dédiée ci-dessus).
3. Ajuster `mappers_corpus.py` aux noms de colonnes réels observés --
   **fait pour ultramedical_preference** (voir section dédiée
   ci-dessus) ; **decision produit encore requise** pour MediQAl :
   écrire le mapper QCM manquant pour
   `mediqal_mcqu.jsonl`/`mediqal_mcqm.jsonl` (voir section dédiée
   ci-dessus pour le traitement de `mcqm` à réponses multiples).
4. Enchaîner `construire_dataset_pivot.py` → `anonymiser_dataset.py` →
   `decouper_splits.py` pour chaque corpus (mediqal-oeq, frenchmedmcqa,
   medquad, ultramedical_preference des a present ; mediqal-mcqu/mcqm
   une fois le mapper QCM ecrit).
5. Rédiger le rapport de justification RGPD à partir des résultats
   réels d'anonymisation.
