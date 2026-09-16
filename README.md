# CHSA Triage : Agent IA de Triage Médical (POC)

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

## Séquence complète, de bout en bout

Index de navigation rapide vers les sections existantes, dans
l'ordre reel du pipeline. Chaque detail (commandes, options) vit
uniquement dans la section liee ; ne pas dupliquer ici.

1. [Préparation des données (Étape 1, sections 1 à 8)](#pipeline-étape-1-les-6-fichiers-sources-fusionnes-dans-le-meme-dataset-pivot)
2. [Extraction du sous-ensemble SFT publiable](#9-extraction-du-sous-ensemble-sft-5000-exemples-pour-publication-hugging-face)
3. [Extraction du sous-ensemble DPO publiable](#10-extraction-du-sous-ensemble-dpo-pour-publication-hugging-face)
4. [Évaluation baseline zero-shot (Étape 1bis)](#11-évaluation-baseline-zero-shot-étape-1bis-avant-sftdpo)
5. [Évaluation baseline zero-shot GPU, sur HF Jobs (Étape 1bis)](#12-évaluation-baseline-zero-shot-gpu-étape-1bis-sur-hf-jobs)
6. [Entraînement SFT réel](#suivi-dentrainement-en-vivo-étape-2--dashboard-streamlit)
7. **DPO (Étape 3) : non implémenté à ce jour.** Aucune commande ni
   étape n'existe encore dans le code pour cette phase ; voir
   `docs/04_etape3_dpo/` (à venir).

## Démarrage rapide (Étape 1 : données)

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

## Pipeline Étape 1 (les 6 fichiers sources, fusionnes dans le meme dataset pivot)

### 1. Telechargement (Hugging Face Hub -> data/raw/)

```bash
# NB : la configuration "oeq" de MediQAl n'a qu'un split "test" (pas de "train") ; --split explicite requis
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration oeq --split test --sortie data/raw/mediqal_oeq.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqu --sortie data/raw/mediqal_mcqu.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqm --sortie data/raw/mediqal_mcqm.jsonl

uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub nthngdy/frenchmedmcqa      --sortie data/raw/frenchmedmcqa.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub keivalya/MedQuad-MedicalQnADataset  --sortie data/raw/medquad.jsonl
uv run python interfaces/cli/E1_01_telecharger_corpus.py --identifiant-hub TsinghuaC3I/UltraMedical-Preference --sortie data/raw/ultramedical_preference.jsonl
```

### 2. Profilage individuel (un rapport ydata-profiling par corpus)

```bash
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_oeq.jsonl   --nom MediQAl-oeq
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_mcqu.jsonl  --nom MediQAl-mcqu
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/mediqal_mcqm.jsonl  --nom MediQAl-mcqm

uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/frenchmedmcqa.jsonl           --nom FrenchMedMCQA
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/medquad.jsonl                 --nom MedQuAD
uv run python interfaces/cli/E1_02_profiler_corpus.py --source data/raw/ultramedical_preference.jsonl --nom UltraMedicalPreference
```

### 3. Construction du dataset pivot (meme --sortie : fusionne les corpus par identifiant)

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

### 4. Anonymisation (incrementale/reprenable, cf. --limite), ecrit dans un fichier SEPARE

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

### 5. Controle qualite de l'anonymisation (comparaison de fichiers)

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
d'entites par type repris du rapport RGPD cumule (§4 ci-dessus, pas
recalcules). Voir `application/use_cases/E1_04_02_controler_qualite_anonymisation.py`.

**Muestreo INCREMENTAL** (09/09/2026, meme
patron que `--limite` ci-dessus) : `--registre-echantillons` (defaut
`data/processed/controle_qualite_identifiants_echantillonnes.jsonl`)
exclut du tirage les identifiants deja echantillonnes lors d'une
execution precedente, sur les deux strates ; chaque execution ne
compare que des identifiants NOUVEAUX. C'est ce qui rend les
decisions humaines de la section suivante cumulables entre
executions, au lieu d'un echantillon jete a chaque fois.

### 6. Revision humaine persistee des candidats de PII residuelle (NF2)

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
verdict `pendant_revision_humaine` du controle qualite (§5) etait un
cul-de-sac ; aucune decision de personne n'etait jamais persistee.
Chaque decision (`accepte` = confirme non-PII, `rejete` = PII reelle
confirmee) est identifiee par une cle stable
`(source_liste, identifiant, champ, type_motif, debut, fin)` et
persistee dans `data/processed/decisions_revision_humaine.jsonl`. Le
rapport de `E1_04_02_controler_qualite_anonymisation.py` (§5) relit ce fichier
pour annoter chaque candidat en attente de son statut de decision
(accepte/rejete/encore en attente). Voir
`application/use_cases/E1_04_01_reviser_pii_residuelle.py` et
`docs/02_etape1_donnees/01_rapport_rgpd.md` §7.5 pour la methodologie
complete.

### 7. Decoupage en splits (train / val / test, stratifie)

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
persistee (§6, `E1_04_01_reviser_pii_residuelle.py`) est exclu du decoupage de
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

### 8. Verification de la repartition des splits par strate

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

### 9. Extraction du sous-ensemble SFT (5000 exemples, pour publication Hugging Face)

Deux etapes : exporter les identifiants a exclure (candidats de PII
residuelle confirmes ou en attente), puis soustraire ce fichier du
pivot anonymise deja reparti en splits.

**Correction (12/09/2026) :** `ExtraireSousEnsembleSftUseCase` ne
filtrait auparavant que sur `split != null`, sans filtrer
`type_exemple` ; `DecouperSplitsUseCase` (§7) reparti A DESSEIN les
deux types (SFT et DPO) dans le meme fichier anonymise, donc des
exemples DPO (ex. UltraMedical-Preference, `chosen`/`rejected`
renseignes, `completion` vide) se retrouvaient dans le sous-ensemble
cense n'etre que du SFT (constate : un fichier de 5000 lignes etait
72% DPO). Le filtre `type_exemple == TypeExemple.SFT` est maintenant
explicite, avant tout comptage/exclusion/recoupe (idem pour
`FormaterDatasetChatMLUseCase`, §Etape 2). Consequence directe : la
taille reellement ecrite depend du nombre d'exemples PUREMENT SFT deja
repartis en split, pas du pool total (SFT+DPO) comme avant ; `--taille`
reste un PLAFOND, jamais un nombre garanti (cf. `manque` ci-dessous) :
si le pivot anonymise n'a encore couvert qu'une fraction du corpus
complet (vagues incrementales de `E1_04_00_anonymiser_dataset.py`,
§4), le pool SFT disponible peut etre plus petit que `--taille` et le
fichier ecrit contiendra alors HONNETEMENT moins de lignes (renommer
le fichier de sortie pour que son nom reflete ce compte reel, cf.
`chemin_sortie_defaut`). Sur le pivot COMPLETEMENT anonymise et
reparti (134 883 exemples, 37 802 SFT/97 081 DPO, §4/§7), le pool SFT
disponible large permet d'atteindre les 5000 demandes sans y toucher.

```bash
# 1. Export non interactif (lecture seule) des identifiants a exclure :
#    au moins un candidat VERDICT_CONFIRME (fuite non ambigue, jamais
#    soumise a decision humaine), ou au moins un candidat
#    VERDICT_REVISION_HUMAINE sans decision DECISION_ACCEPTE persistee
#    (candidat encore ouvert, ou explicitement rejete). Meme replay
#    deterministe que `E1_04_01_reviser_pii_residuelle.py verify` (§6), etendu
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
exemples du pivot, §4/§7 ; controle qualite §5 elargi a 1350
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
(`E1_04_00_anonymiser_dataset.py --limite <N>`, §4) puis le decoupage
des splits (`E1_05_00_decouper_splits.py --n <N>` ou sans `--n` pour
tout completer, §7) avant de relancer l'extraction ; elargir seulement
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
qualite (§5) reste partielle (voir la limite de couverture
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
echantillonne par le controle qualite (§5, 1350 identifiants cumules
sur 134 883 a ce jour, ~1%) ; un exemple jamais echantillonne peut
donc encore contenir une PII residuelle non detectee, meme apres
l'etape 1. Augmenter la couverture de
`E1_04_02_controler_qualite_anonymisation.py` (§5) avant publication reduit ce
risque, mais ne l'elimine pas completement sans audit exhaustif.

### 10. Extraction du sous-ensemble DPO (pour publication Hugging Face)

Meme patron exact que §9 (`ExtraireSousEnsembleSftUseCase`), en
filtrant `type_exemple == TypeExemple.DPO` au lieu de SFT : le cahier
des charges (`docs/00_cadrage/01_cahier_des_charges.md` §7, Livrable
1) exige "SFT ~5000 paires + DPO" dans le dataset publie, et
`E1_05_03_extraire_sous_ensemble_dpo.py` produit ce second sous-ensemble
a partir du meme pivot anonymise et du meme fichier d'exclusions PII
(reutilise tel quel, sans le regenerer). Meme fichier d'exclusions
(§9 etape 1) : un identifiant y figure independamment du type
d'exemple, l'export ne le regenere donc pas.

```bash
# Reutilise le meme fichier d'exclusions que §9 etape 1 (pas besoin de
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

Sortie attendue (meme forme que §9, `Strate` prefixee `dpo/` au lieu
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
37 802 SFT / 97 081 DPO au 12/09/2026, §4/§7), le pool DPO disponible
(uniquement `UltraMedical-Preference` a ce jour) est largement
suffisant pour atteindre les 5000 demandes sans y toucher ; comme en
§9, `--taille` reste un PLAFOND jamais garanti si le pivot anonymise
ne couvre encore qu'une fraction du corpus complet.

**Publication sur Hugging Face Hub.** Meme depot que §9 recommande une
publication SEPAREE (deux fichiers distincts dans le meme depot
prive), pour que le nom du fichier continue de refleter honnetement
son contenu et son compte reel :

```bash
# Le depot existe deja depuis §9 (creer une seule fois) :
hf repo create mombasstic/dataset_chsa_triage_sft_anonymise_5000 --repo-type dataset --private

# Publier le fichier DPO a cote du fichier SFT deja publie :
hf upload mombasstic/dataset_chsa_triage_sft_anonymise_5000 data/processed/dataset_chsa_triage_dpo_anonymise_5000.jsonl --repo-type dataset

# Publier/mettre a jour la dataset card (README.md du depot HF) :
hf upload mombasstic/dataset_chsa_triage_sft_anonymise_5000 docs/02_etape1_donnees/dataset_card_dpo_hf.md README.md --repo-type dataset

# Verifier le compte de lignes cote Hub (meme methode que §9, ne pas
# se fier au dataset viewer qui peut prendre du temps sur un depot prive) :
hf download mombasstic/dataset_chsa_triage_sft_anonymise_5000 dataset_chsa_triage_dpo_anonymise_5000.jsonl --repo-type dataset --local-dir /tmp/verificacion_hf_dpo
wc -l /tmp/verificacion_hf_dpo/dataset_chsa_triage_dpo_anonymise_5000.jsonl
```

Meme limite de couverture qu'en §9 : l'exclusion ne porte que sur ce
qui a deja ete audite par le controle qualite (§5).

### 11. Évaluation baseline zero-shot (Étape 1bis, avant SFT/DPO)

Mesure la performance de `Qwen/Qwen3-1.7B-Base` SANS entrainement sur
le split `test` DEJA EXISTANT du pivot anonymise (champ `split`, §7),
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

### 12. Évaluation baseline zero-shot GPU (Étape 1bis, sur HF Jobs)

Meme mesure que §11 (`Qwen/Qwen3-1.7B-Base` SANS entrainement, meme
sous-ensemble de 278 exemples `split=test`/`type_exemple=sft`), mais
sur GPU reel via `transformers` en pleine precision bf16
(`TransformersInferenceAdapter`), PAS sur GGUF quantifie Q4_K_M
(§11, `LlamaCppInferenceAdapter`) : pour comparer plus tard au modele
SFT/DPO (probablement lui aussi evalue via `transformers`/GPU) SANS
melanger l'effet de la quantification avec l'effet reel de
l'entrainement. `EvaluerBaselineZeroShotUseCase` (application) est
REUTILISE SANS MODIFICATION entre §11 et §12 : seul l'adaptateur
d'inference change.

Le dataset a evaluer vit sur un depot dataset HF PRIVE deja publie,
`mombasstic/chsa-triage-baseline-test` (fichier
`dataset_pivot_test_sft.jsonl`, 278 exemples, EXACTEMENT le meme
sous-ensemble que §11, pour que les deux resultats soient comparables) ;
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
(credentials HF reelles disponibles ici, contrairement a la section
"Suivi d'entrainement en vivo" ci-dessous). Comme prevu (aucun GPU
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

## Suivi d'entrainement en vivo (Étape 2 : dashboard Streamlit)

Pendant un run SFT-LoRA reel (Environnement B, GPU sur HF Jobs),
`training/E2_04_sft_train.py --suivi-hf-repo <repo>` (recette
`suivi.backend: hf_dataset`) publie la courbe de perte train/validation
en continu vers un dataset Hugging Face Hub, lu EN VIVO par un
dashboard Streamlit deploye a part sur un Space
(`monitoring/app_suivi_entrainement.py`). Aucune de ces commandes n'a
ete executee reellement (aucune credential HF disponible ici) : elles
sont documentees, verifiees dans leur syntaxe (`hf repo create --help`,
`hf upload --help`), mais **NON EXECUTEES/NON VERIFIEES en reseau
reel**.

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
#    cree a l'etape 1, pour que le Space ait des vraies donnees a lire :
uv run python training/E2_04_sft_train.py \
    --recette recipes/sft_qwen3_lora.yaml \
    --dataset data/processed/dataset_pivot_anonymise.jsonl \
    --suivi-hf-repo mombasstic/chsa-triage-sft-metrics
```

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

### Historique complet dans un MLflow local (importateur)

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

## Structure (architecture hexagonale)

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

## État d'avancement

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
      `data/processed/decisions_revision_humaine.jsonl`) ; voir §6
      ci-dessus et `docs/02_etape1_donnees/01_rapport_rgpd.md` §7.5.
      **Vagues ultérieures** : à relancer avec `--limite` plus grand
      (ou `full`) avant le SFT/DPO ; réévaluer d'abord le risque de
      saturation/surapprentissage d'un entraînement sur un
      sous-échantillon trop petit face au dataset complet (à étudier
      à ce moment-là, pas tranché ici)
- [ ] Étape 2 : SFT + LoRA
- [ ] Étape 3 : DPO
- [ ] Étape 4 : Déploiement (FastAPI + Streamlit + vLLM + CI/CD)
