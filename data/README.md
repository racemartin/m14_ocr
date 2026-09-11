# data/ : Organisation et schéma

Ce dossier est volontairement **vide dans le dépôt versionné**
(`.gitkeep` uniquement, `.gitignore` exclut le contenu réel). Cible
long terme (Livrable 1) : versionner le dataset anonymisé sur Hugging
Face Hub. **Pas encore fait** : attendre que
le contrôle qualité PII (§ ci-dessous) soit intégralement clos avant
toute publication, même privée. En attendant, origine et licence des
4 sources sont documentées dans le dépôt ; voir
`docs/00_cadrage/02_sources_donnees_licences.md`.

## Structure

```
data/
├── raw/          corpus bruts tels que téléchargés (JSONL)
├── processed/    dataset pivot, dataset anonymisé, doublons écartés, rapports
└── splits/       (réservé : les splits sont actuellement stockés
                   comme un champ `split` sur chaque ExemplePivot
                   dans processed/, pas des fichiers séparés)
```

## `data/raw/` : corpus bruts attendus

| Fichier attendu | Corpus source | Format |
|---|---|---|
| `mediqal_oeq.jsonl`, `mediqal_mcqu.jsonl`, `mediqal_mcqm.jsonl` | MediQAl (3 configurations Hub, 2 schémas) | `oeq` : colonnes `question`/`answer` ; `mcqu`/`mcqm` : QCM `answer_a`..`answer_e` + `correct_answers` |
| `frenchmedmcqa.jsonl` | FrenchMedMCQA | colonnes `question`, `answer_a`..`answer_e`, `correct_answers` |
| `medquad.jsonl` | MedQuAD | colonnes `Question`/`Answer` |
| `ultramedical_preference.jsonl` | UltraMedical-Preference | colonnes `prompt`, `chosen`, `rejected` |

Noms de colonnes vérifiés contre le contenu réel téléchargé et
implémentés dans `interfaces/cli/E1_03_01_mappers_corpus.py`. Détail par
source (schéma, config Hub) : `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`.

## `data/processed/` : schéma pivot (`ExemplePivot`)

Chaque ligne de `dataset_pivot.jsonl` correspond à l'entité de domaine
`ExemplePivot` (`src/chsa_triage/domain/model/exemple_pivot.py`) :

```json
{
  "identifiant": "chsa-<espace_noms>-<hash>",
  "identifiant_source_brute": "cle naturelle du registre brut d'origine",
  "source": "MediQAl | FrenchMedMCQA | MedQuAD | UltraMedical-Preference",
  "type_exemple": "sft | dpo",
  "langue": "fr | en",
  "symptomes": "texte libre",
  "antecedents": "texte libre ou null",
  "constantes_vitales": {"pression_arterielle": null, "frequence_cardiaque": null, "saturation_o2": null, "frequence_respiratoire": null},
  "prompt": [{"role": "user", "contenu": "..."}],
  "completion": [{"role": "assistant", "contenu": "..."}],
  "chosen": [],
  "rejected": [],
  "niveau_confiance": "haute | moyenne | basse",
  "anonymise": true,
  "split": "train | val | test"
}
```

Référence complète du schéma, y compris l'écart assumé avec l'énoncé
initial de la mission : cahier des charges §5.2
(`docs/00_cadrage/01_cahier_des_charges.md`).

### Fichiers produits dans `data/processed/`

| Fichier | Contenu |
|---|---|
| `dataset_pivot.jsonl` | Pivot consolidé, **jamais modifié** une fois construit (source de vérité immuable) |
| `dataset_pivot_anonymise.jsonl` | Sortie **séparée** de l'anonymisation : mêmes identifiants que le pivot, champs texte masqués, `anonymise=true` |
| `doublons_supprimes.jsonl` | Doublons exacts écartés à la construction du pivot (même `identifiant`), archivés, jamais perdus |
| `rapport_anonymisation_rgpd.{json,md}` | Rapport RGPD cumulé (registres traités, entités détectées par type, historique des exécutions) |
| `rapport_controle_qualite_anonymisation.{json,md}` | Contrôle qualité par comparaison pivot original / anonymisé (PII résiduelle, sur-masquage) |
| `rapports_profilage/` | Rapports HTML `ydata-profiling`, un par fichier source, non versionnés (volumineux, régénérables) |

Le dataset pivot n'est **jamais modifié en place** par l'anonymisation :
`E1_04_00_anonymiser_dataset.py` lit `dataset_pivot.jsonl` et écrit dans
`dataset_pivot_anonymise.jsonl` (fichiers séparés depuis le
08/09/2026 ; permet de relancer le contrôle qualité ou
une nouvelle vague d'anonymisation sans jamais perdre l'original). La
déduplication est réelle (pas seulement documentée) : `identifiant`
est déterministe (hash de `espace_noms:cle_naturelle`), donc un même
registre brut produit toujours le même identifiant, et
`ConstruireDatasetPivotUseCase` n'en garde qu'un seul exemplaire par
identifiant ; les doublons écartés atterrissent dans
`doublons_supprimes.jsonl`.

## Pipeline de génération

Séquence complète (7 étapes : téléchargement, profilage, construction
du pivot, anonymisation, contrôle qualité, découpage en splits,
vérification de la répartition), avec commandes exactes et détail de
chaque option (`--taille-bloc`, `--limite`, `--n`, etc.) : voir le
README principal du projet (`../README.md`, section « Pipeline Étape
1 »), pour ne pas maintenir une copie divergente ici.
