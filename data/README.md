# data/ — Organisation et schéma

Ce dossier est volontairement **vide dans le dépôt versionné**
(`.gitkeep` uniquement, `.gitignore` exclut le contenu réel) : les
données ne sont jamais commitées, seulement versionnées sur Hugging
Face Hub une fois anonymisées (Livrable 1).

## Structure

```
data/
├── raw/          corpus bruts tels que téléchargés (JSONL/CSV)
├── processed/    dataset pivot consolidé + rapports de profilage
└── splits/       (réservé — les splits sont actuellement stockés
                   comme un champ `split` sur chaque ExemplePivot
                   dans processed/, pas des fichiers séparés)
```

## `data/raw/` — corpus bruts attendus

| Fichier attendu | Corpus source | Format |
|---|---|---|
| `mediqal_*.jsonl` ou `.csv` | MediQAl | colonnes `question`/`answer` (à confirmer par profilage réel) |
| `frenchmedmcqa_*.jsonl` | FrenchMedMCQA | colonnes `question`, `options`, `correct_answers` |
| `medquad_*.jsonl` | MedQuAD | colonnes `Question`/`Answer` |
| `ultramedical_preference_*.jsonl` | UltraMedical-Preference | colonnes `prompt`, `chosen`, `rejected` |

Ces noms de colonnes sont ceux documentés par les fiches Hugging Face
de chaque dataset et implémentés dans `interfaces/cli/mappers_corpus.py`.
**Ils doivent être vérifiés** contre le vrai contenu téléchargé — voir
`profiler_corpus.py`.

## `data/processed/` — schéma pivot (`ExemplePivot`)

Chaque ligne de `dataset_pivot.jsonl` correspond à l'entité de domaine
`ExemplePivot` (`src/chsa_triage/domain/model/exemple_pivot.py`) :

```json
{
  "identifiant": "chsa-<source>-<uuid>",
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

Référence complète du schéma : cahier des charges §5.2
(`docs/00_cadrage/01_cahier_des_charges.md`).

## `data/processed/rapports_profilage/`

Rapports HTML `ydata-profiling`, un par corpus, générés par
`profiler_corpus.py`. Non versionnés (volumineux, régénérables à
tout moment).

## Pipeline de génération

```bash
uv run python interfaces/cli/profiler_corpus.py --source data/raw/<corpus>.jsonl --nom <Corpus>
uv run python interfaces/cli/construire_dataset_pivot.py --source data/raw/<corpus>.jsonl --corpus <corpus> --sortie data/processed/dataset_pivot.jsonl
uv run python interfaces/cli/anonymiser_dataset.py --dataset data/processed/dataset_pivot.jsonl --strategie replace
uv run python interfaces/cli/decouper_splits.py --dataset data/processed/dataset_pivot.jsonl
```

Répéter les deux premières commandes pour chacun des 4 corpus (chaque
appel de `construire_dataset_pivot.py` fusionne dans le même fichier
de sortie), puis anonymiser et découper une seule fois sur l'ensemble
consolidé.
