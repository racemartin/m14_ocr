---
license: other
tags:
  - chsa-triage
  - dpo
  - medical
  - anonymized
pretty_name: CHSA Triage DPO (sous-ensemble anonymise, publication HF)
---

# CHSA Triage DPO

Sous-ensemble DPO (preference pairs) du dataset pivot anonymise du POC
CHSA Triage, publie a cote du sous-ensemble SFT
(`dataset_chsa_triage_sft_anonymise_5000.jsonl`, meme depot, voir
README.md §9) dans le meme depot Hugging Face prive, conformement au
cahier des charges (`docs/00_cadrage/01_cahier_des_charges.md` §7,
Livrable 1 : "SFT ~5000 paires + DPO"). Produit par
`interfaces/cli/E1_05_03_extraire_sous_ensemble_dpo.py` (README.md
§10) : meme pivot anonymise, meme fichier d'exclusions PII, meme
algorithme de recoupe stratifie que le sous-ensemble SFT.

**Contenu clinique anonymise** (Presidio + spaCy, voir
`docs/02_etape1_donnees/01_rapport_rgpd.md`) : source unique a ce jour
`UltraMedical-Preference` (paires de preference generees par un modele,
pas des dossiers patients reels). Le nom du fichier reflete toujours le
compte reel de lignes qu'il contient (jamais fige a l'avance).

## Format

Une ligne JSON par exemple (`ExemplePivot` serialise, cf.
`src/chsa_triage/infrastructure/adapters/jsonl_dataset_repository.py::exemple_pivot_vers_dict`) :

| Champ | Type | Signification |
|---|---|---|
| `identifiant` | texte | Identifiant deterministe (sha256 de `espace_noms:cle_naturelle`, jamais aleatoire) |
| `source` | texte | Corpus d'origine (`UltraMedical-Preference` a ce jour) |
| `type_exemple` | texte | Toujours `"dpo"` dans ce fichier |
| `langue` | texte | `"fr"` ou `"en"` |
| `prompt` | liste de `{role, contenu}` | Tour(s) de dialogue formant la question/instruction |
| `chosen` | liste de `{role, contenu}` | Reponse preferee |
| `rejected` | liste de `{role, contenu}` | Reponse rejetee |
| `completion` | liste (vide) | Toujours vide pour un exemple DPO (`prompt`+`chosen`+`rejected` seuls renseignes, cf. `ExemplePivot.est_complet_pour_dpo`) |
| `anonymise` | booleen | Toujours `true` dans ce fichier (source du pivot anonymise) |
| `split` | texte | `"train"`, `"val"` ou `"test"` (deja stratifie, cf. README.md §7) |

Pas de metriques d'entrainement dans ce depot : contrairement au
dataset de suivi (`data/demos/README_dataset_card.md`), ce sont des
donnees d'ENTREE pour un futur entrainement DPO (Etape 3,
`docs/04_etape3_dpo/`), pas des courbes de sortie d'un run.

## Limite de couverture connue

Meme limite que le sous-ensemble SFT (README.md §9) : l'exclusion des
identifiants portant une PII residuelle ne porte que sur ce qui a deja
ete audite par le controle qualite
(`interfaces/cli/E1_04_02_controler_qualite_anonymisation.py`, une
fraction du pivot complet a ce jour). Un exemple jamais echantillonne
peut donc encore contenir une PII residuelle non detectee.
