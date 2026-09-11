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
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration oeq --split test --sortie data/raw/mediqal_oeq.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqu --sortie data/raw/mediqal_mcqu.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub ANR-MALADES/MediQAl --configuration mcqm --sortie data/raw/mediqal_mcqm.jsonl

uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub nthngdy/frenchmedmcqa      --sortie data/raw/frenchmedmcqa.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub keivalya/MedQuad-MedicalQnADataset  --sortie data/raw/medquad.jsonl
uv run python interfaces/cli/telecharger_corpus.py --identifiant-hub TsinghuaC3I/UltraMedical-Preference --sortie data/raw/ultramedical_preference.jsonl
```

### 2. Profilage individuel (un rapport ydata-profiling par corpus)

```bash
uv run python interfaces/cli/profiler_corpus.py --source data/raw/mediqal_oeq.jsonl   --nom MediQAl-oeq
uv run python interfaces/cli/profiler_corpus.py --source data/raw/mediqal_mcqu.jsonl  --nom MediQAl-mcqu
uv run python interfaces/cli/profiler_corpus.py --source data/raw/mediqal_mcqm.jsonl  --nom MediQAl-mcqm

uv run python interfaces/cli/profiler_corpus.py --source data/raw/frenchmedmcqa.jsonl           --nom FrenchMedMCQA
uv run python interfaces/cli/profiler_corpus.py --source data/raw/medquad.jsonl                 --nom MedQuAD
uv run python interfaces/cli/profiler_corpus.py --source data/raw/ultramedical_preference.jsonl --nom UltraMedicalPreference
```

### 3. Construction du dataset pivot (meme --sortie : fusionne les corpus par identifiant)

```bash
# NB : MediQAl a 3 configurations, avec 2 schemas differents : le
# mapper (donc la valeur --corpus) depend du schema, pas seulement de
# la source Hub. "oeq" (question/answer) -> mapper_mediqal ;
# "mcqu"/"mcqm" (QCM, answer_a..answer_e + correct_answers) ->
# mapper_mediqal_qcm. Voir interfaces/cli/mappers_corpus.py.
# NB : --taille-bloc requis sur ultramedical_preference.jsonl (966 Mo,
# 109353 enregistrements) ; sans lecture par blocs, OOM reel confirme
# sur 5.8 Go de RAM disponibles. Voir --taille-bloc dans
# interfaces/cli/construire_dataset_pivot.py.
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/mediqal_oeq.jsonl --corpus mediqal_oeq --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/mediqal_mcqu.jsonl --corpus mediqal_mcqu --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/mediqal_mcqm.jsonl --corpus mediqal_mcqm --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/frenchmedmcqa.jsonl --corpus frenchmedmcqa --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/medquad.jsonl --corpus medquad --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/ultramedical_preference.jsonl --corpus ultramedical_preference --sortie data/processed/dataset_pivot.jsonl --taille-bloc 5000
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
uv run python interfaces/cli/anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --sortie data/processed/dataset_pivot_anonymise.jsonl --strategie replace --limite 5000
```

```bash
# Pour enchainer les vagues successives sans relancer la commande a
# la main a chaque fois : scripts/anonymiser_par_lots.sh rappelle
# anonymiser_dataset.py en boucle jusqu'a couverture complete du
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
Voir `application/use_cases/uc_03_01_rapport_anonymisation.py`.

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
uv run python interfaces/cli/controler_qualite_anonymisation.py --dataset data/processed/dataset_pivot.jsonl --anonymise data/processed/dataset_pivot_anonymise.jsonl --taille-echantillon 200
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
recalcules). Voir `application/use_cases/uc_03_02_controler_qualite_anonymisation.py`.

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
uv run python interfaces/cli/reviser_pii_residuelle.py verify --dataset data/processed/dataset_pivot.jsonl --anonymise data/processed/dataset_pivot_anonymise.jsonl

# Corriger une decision deja prise (sans repasser par toute la liste) :
uv run python interfaces/cli/reviser_pii_residuelle.py modify --identifiant chsa-xxxxxxxx
```

Ferme l'ecart identifie sur l'exigence NF2 du cahier des charges
("anonymisation validee **manuellement**") : avant ce script, le
verdict `pendant_revision_humaine` du controle qualite (§5) etait un
cul-de-sac ; aucune decision de personne n'etait jamais persistee.
Chaque decision (`accepte` = confirme non-PII, `rejete` = PII reelle
confirmee) est identifiee par une cle stable
`(source_liste, identifiant, champ, type_motif, debut, fin)` et
persistee dans `data/processed/decisions_revision_humaine.jsonl`. Le
rapport de `controler_qualite_anonymisation.py` (§5) relit ce fichier
pour annoter chaque candidat en attente de son statut de decision
(accepte/rejete/encore en attente). Voir
`application/use_cases/uc_03_03_reviser_pii_residuelle.py` et
`docs/02_etape1_donnees/01_rapport_rgpd.md` §7.5 pour la methodologie
complete.

### 7. Decoupage en splits (train / val / test, stratifie)

```bash
# decouper_splits.py opere sur le fichier ANONYMISE (dataset_pivot_anonymise.jsonl),
# PAS sur le pivot original ; le relancer apres chaque nouvelle vague
# d'anonymisation.
uv run python interfaces/cli/decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl

# --n : taille CIBLE cumulee (pas la taille de cette seule execution).
uv run python interfaces/cli/decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl --n 5000
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
uv run python interfaces/cli/decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl --n 5000

# Deuxieme execution, plus tard : --n 10000 preleve N - (deja assignes)
# = 10000 - 5000 = 5000 NOUVEAUX exemples (echantillon stratifie parmi
# ceux qui n'ont pas encore de split) et leur assigne un split. Les
# 5000 PREMIERS exemples GARDENT exactement le split qui leur a ete
# assigne lors de la premiere execution ; aucun n'est deplace entre
# train/val/test.
uv run python interfaces/cli/decouper_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl --n 10000
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

**Exclusion des candidats PII en attente de revision humaine
(10/09/2026).** Par precaution, un exemple
portant au moins un candidat de PII residuelle SANS decision humaine
persistee (§6, `reviser_pii_residuelle.py`) est exclu du decoupage de
cette execution : il reste sans `split` jusqu'a ce qu'une decision
soit prise. `decouper_splits.py` accepte donc desormais les memes
adaptateurs que `reviser_pii_residuelle.py verify` pour recalculer cet
ensemble : `--original` (defaut `data/processed/dataset_pivot.jsonl`),
`--registre-echantillons` (defaut
`data/processed/controle_qualite_identifiants_echantillonnes.jsonl`),
`--decisions` (defaut `data/processed/decisions_revision_humaine.jsonl`)
et `--jeton-masque`. Le nombre d'exemples exclus pour cette raison
lors de cette execution est affiche en sortie.

### 8. Verification de la repartition des splits par strate

```bash
# decouper_splits.py n'affiche que le total global (train/val/test).
# verifier_repartition_splits.py relit le fichier anonymise deja
# reparti et affiche, pour chaque strate (type_exemple, source), le
# decompte ET le pourcentage par split, pour verifier visuellement
# que l'echantillonnage stratifie reste representatif DANS CHAQUE
# split (ex. une petite source comme FrenchMedMCQA doit rester
# ~80/10/10 comme les grosses sources, pas disparaitre de train ou de
# test). Fonctionne sans changement avec la croissance stable de
# decouper_splits.py ci-dessus : il relit simplement les splits deja
# presents dans le fichier, quelle que soit la sequence d'executions
# --n qui les a produits.
uv run python interfaces/cli/verifier_repartition_splits.py --dataset data/processed/dataset_pivot_anonymise.jsonl
```

`anonymiser_dataset.py` affiche une barre de progression `tqdm` pendant le traitement (peut durer plusieurs dizaines de minutes sur un gros dataset).

### 9. Extraction du sous-ensemble SFT (5000 exemples, pour publication Hugging Face)

```bash
jq -c 'select(.split != null)' data/processed/dataset_pivot_anonymise.jsonl > data/processed/dataset_sft_5000.jsonl
wc -l data/processed/dataset_sft_5000.jsonl   # doit afficher 5000
```

Le pivot anonymise complet contient bien plus d'exemples que les 5000
deja repartis en splits (§7-8 ci-dessus) : `decouper_splits.py --n 5000`
n'affecte un `split` (train/val/test) qu'a un echantillon stratifie de
5000 exemples, les autres restant a `split: null`. Filtrer sur
`split != null` recupere donc exactement ce sous-ensemble deja
stratifie, sans nouveau tirage. Publier uniquement ces 5000 exemples
plutot que les 134 883 du pivot complet reduit la surface d'exposition
publique de donnees issues des corpus sources. `jq` est prefere a
`grep` car il parse reellement le JSON plutot que de chercher un motif
texte, ce qui evite tout faux positif si la sous-chaine `"split"`
apparaissait ailleurs (par exemple dans un champ de texte libre).

Le pivot anonymise complet (`dataset_pivot_anonymise.jsonl`) reste
local sous `data/processed/` (deja exclu de Git, voir le commentaire
correspondant dans `.gitignore`) ; le fichier filtre de 5000 exemples
est autonome et reproductible a tout moment a partir du pivot complet
via la commande ci-dessus. La publication sur Hugging Face Hub (nom du
depot et visibilite encore a decider) reste une etape ulterieure, qui
sera documentee separement une fois ces choix arretes.

**Point de vigilance (11/09/2026) :** filtrer sur `split != null` ne
garantit PAS a lui seul que les 5000 exemples sont exempts de PII
residuelle. Seul un sous-ensemble du pivot a ete audite par le
controle qualite (§5) a ce jour, et des candidats confirmes ou en
attente de revision humaine peuvent se trouver parmi des exemples
deja repartis en split (l'exclusion decrite en §7 ne protege que les
repartitions futures, jamais retroactivement). Croiser ce fichier
avec le rapport de controle qualite (§5) et les decisions humaines
(§6) avant toute publication.

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
      `controler_qualite_anonymisation.py`) ; **première vague
      exécutée sur le pivot régénéré (5 000/134 883 exemples, 90,6 %
      avec ≥1 entité détectée, 64 667 entités) et découpée en splits
      (4 004/498/498, vérifiée représentative par strate)** ; 200
      exemples contrôlés automatiquement (0 PII résiduelle confirmée,
      35 candidats explicitement en attente de révision humaine) ;
      voir `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`.
      Muestreo du contrôle qualité rendu **incrémental** (09/09/2026,
      `--registre-echantillons`) et les 35 candidats en attente
      peuvent désormais être tranchés avec une décision humaine
      **persistée** (`reviser_pii_residuelle.py`,
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
