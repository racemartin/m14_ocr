---
license: other
tags:
  - chsa-triage
  - sft
  - lora
  - training-metrics
pretty_name: CHSA Triage SFT Metrics (dataset de suivi d'entrainement)
---

# CHSA Triage SFT Metrics

Dataset de suivi d'entrainement pour le POC CHSA Triage (Étape 2 :
SFT-LoRA), lu en direct par le dashboard Streamlit
(`monitoring/app_suivi_entrainement.py`) pendant un run GPU sur HF
Jobs. Ne contient ni donnees patient ni texte clinique : uniquement des
courbes de metriques numeriques (perte, gradient) par etape d'entrainement.

**Contenu actuel du depot : uniquement le run `demo_datos_ficticios`**,
des chiffres **FACTICES** (generes, pas un vrai entrainement) servant a
verifier que le dashboard fonctionne de bout en bout sans attendre une
session GPU payante. Voir `data/demos/chsa-triage-sft-metrics-fake.json`
dans le depot de code pour la source de ce fichier et
`README.md` §"Suivi d'entrainement en vivo" pour la commande de
publication exacte. Un vrai run remplacera/completera ce contenu par un
dossier nomme d'apres `NOM_RUN_PAR_DEFAUT` (`sft-lora`,
`application/use_cases/E2_01_uc_entrainer_sft.py`).

## Format

Un fichier `<nom_run>/metriques.jsonl` par run d'entrainement, format
JSON Lines (une ligne JSON par metrique par etape), ecrit par
`HfDatasetSuiviExperimentation.logger_metrique()`
(`src/chsa_triage/infrastructure/adapters/hf_dataset_suivi_experimentation.py`) :

| Champ | Type | Signification |
|---|---|---|
| `etape` | entier | Le "global step" de `transformers.Trainer` (pas de batch, cumule sur toutes les epoques, pas remis a zero a chaque epoque) |
| `nom` | texte | Nom de la metrique : `perte_train`, `perte_validation` ou `norme_gradient` |
| `valeur` | nombre | Valeur mesuree a cette etape |
| `horodatage` | nombre | `time.time()` Unix, au moment ou `logger_metrique()` a ecrit la ligne (PAS le moment ou la metrique a ete calculee cote GPU, cf. limite connue plus bas) |

Une etape n'a pas forcement les trois metriques : `perte_validation`
n'apparait qu'aux etapes ou une evaluation a eu lieu (voir plus bas).
Le dashboard pivote ce format long vers une table large par etape
(`monitoring/logica_suivi_entrainement.py::pivoter_par_etape`).

## Les trois metriques

### `perte_train` (train loss)

La perte d'entrainement (entropie croisee moyenne sur les tokens NON
masques de l'exemple, cf. `docs/03_etape2_sft/` pour la theorie du
masquage) mesuree sur le dernier lot (batch) d'entrainement traite.

- **Origine reelle** : cle `"loss"` de `transformers.Trainer.state.log_history`
  (`trl.SFTTrainer` en herite sans la surcharger), enregistree tous les
  `logging_steps` pas (valeur par defaut de `transformers.TrainingArguments`,
  non surchargee explicitement dans `TrlSftEntraineurAdapter` : a
  verifier sur le premier run reel, un petit dataset peut produire tres
  peu de points si `logging_steps` est grand face au nombre total de pas).
- **Capturee dans le code** : `_courbe_depuis_log_history()`
  (`infrastructure/adapters/trl_sft_entraineur.py`), qui lit
  `entree["loss"]` pour chaque entree du `log_history` portant une cle
  `"step"`.
- **Publiee vers ce dataset par** : `EntrainerSftUseCase.entrainer()`
  (`application/use_cases/E2_01_uc_entrainer_sft.py`), boucle
  `self.suivi.logger_metrique("perte_train", point.perte_train, point.etape)`.
- **Tendance attendue** : decroissante au fil des etapes. Une baisse
  relative trop faible entre la premiere et la derniere etape
  (`SEUIL_BAISSE_TRAIN_RELATIVE_MINIMALE = 0.05`, soit 5 %,
  `application/verdict_convergence.py`) signale un
  `SOUS_APPRENTISSAGE` (le modele n'apprend quasiment rien).

### `perte_validation` (validation/eval loss)

Meme perte, mais mesuree sur le jeu de validation (jamais vu en
entrainement), a intervalle regulier.

- **Origine reelle** : cle `"eval_loss"` du `log_history`, produite par
  `SFTTrainer` a chaque evaluation. `TrlSftEntraineurAdapter` configure
  `eval_strategy="epoch"` (`sft_config = SFTConfig(...)`) : une
  evaluation par EPOQUE, pas par pas fixe, donc au plus
  `hyperparametres.nombre_epoques` points de validation par run (3 avec
  la recette actuelle, `recipes/sft_qwen3_lora.yaml`).
- **Capturee dans le code** : meme fonction
  `_courbe_depuis_log_history()`, dictionnaire intermediaire
  `pertes_validation_par_etape` associant chaque `step` d'evaluation a
  son `eval_loss`, recroise ensuite avec les etapes d'entrainement.
- **Publiee vers ce dataset par** : meme boucle que `perte_train`, mais
  seulement `if point.perte_validation is not None` (n'apparait donc
  pas a toutes les etapes dans ce fichier).
- **Tendance attendue** : decroissante en parallele de `perte_train`.
  Si elle REMONTE sur les 3 derniers points mesures
  (`FENETRE_PAS_VALIDATION = 3`) de plus de 0.05 en absolu
  (`SEUIL_HAUSSE_VALIDATION_SURAPPRENTISSAGE`) pendant que `perte_train`
  continue de baisser, c'est un `SURAPPRENTISSAGE` (le modele memorise
  l'entrainement au lieu de generaliser).

### `norme_gradient` (gradient norm)

La norme (L2) du gradient calcule lors de la retropropagation, avant
l'etape d'optimisation. Indicateur de stabilite numerique de
l'entrainement, independant du niveau de la perte elle-meme.

- **Origine reelle** : cle `"grad_norm"` du `log_history` (meme cadence
  que `"loss"`, tous les `logging_steps` pas).
- **Capturee dans le code** : `entree.get("grad_norm", 0.0)` dans
  `_courbe_depuis_log_history()` (`0.0` par defaut si l'entree ne porte
  pas cette cle, ce qui ne devrait arriver que pour des entrees qui ne
  sont pas des pas d'entrainement).
- **Publiee vers ce dataset par** : meme boucle, toujours (jamais
  `None`, contrairement a `perte_validation`).
- **Tendance attendue** : stable ou en leger declin apres les premiers
  pas. Une divergence (ratio dernier/premier pas au-dela de
  `SEUIL_RATIO_DIVERGENCE_GRADIENT = 10.0`) COMBINEE a une `perte_train`
  qui remonte signale un run `INSTABLE` (explosion du gradient) ; une
  valeur NaN/infinie sur N'IMPORTE laquelle des trois metriques, a
  n'importe quelle etape, est aussi classee `INSTABLE` immediatement
  (verifie en premier, avant tout autre diagnostic).

## Relation entre les trois metriques (le verdict de convergence)

Le dashboard calcule un verdict en direct
(`monitoring/logica_suivi_entrainement.py::evaluer_convergence_en_vivo`,
qui reutilise telle quelle la logique de domaine
`application/verdict_convergence.py::evaluer_convergence`) en
combinant les trois metriques, dans cet ordre de priorite :

1. **`INSTABLE`** : NaN/infini n'importe ou, OU norme de gradient qui
   explose pendant que la perte d'entrainement remonte.
2. **`SOUS_APPRENTISSAGE`** : `perte_train` quasiment plate du debut a
   la fin de la courbe (le modele n'apprend pas assez).
3. **`SURAPPRENTISSAGE`** : `perte_validation` remonte alors que
   `perte_train` continue de baisser (le modele memorise au lieu de
   generaliser ; c'est la comparaison ENTRE `perte_train` et
   `perte_validation`, jamais l'une des deux seule, qui revele ce cas).
4. **`SAINE`** : aucun des cas ci-dessus.

`norme_gradient` sert donc surtout a detecter une divergence NUMERIQUE
(instabilite), tandis que la comparaison `perte_train` vs
`perte_validation` sert a detecter une divergence de GENERALISATION
(sur/sous-apprentissage). Ces deux questions sont independantes : un
run peut avoir un gradient parfaitement stable et quand meme
sur-apprendre.

Ces seuils numeriques (`SEUIL_*` ci-dessus) sont explicitement
documentes comme PROVISOIRES dans le code (aucune vraie courbe
d'entrainement observee au moment ou ils ont ete choisis) : a
recalibrer une fois un premier run reel disponible.

## Limites connues de ce mecanisme de publication (honnete, pas encore corrige)

- **Publication en un seul lot en fin de run, pas vraiment pas-a-pas** :
  `EntrainerSftUseCase.entrainer()` n'appelle `logger_metrique()` qu'
  APRES que `self.entraineur.entrainer(...)` soit revenu, c'est-a-dire
  apres que `trainer.train()` (TRL) ait fini TOUT le run. La courbe
  complete est donc relayee vers ce dataset en rafale a la fin de
  l'entrainement, pas progressivement pendant qu'il tourne : le
  dashboard verra la courbe complete apparaitre d'un coup, pas grandir
  metrique par metrique en temps reel. Pour un vrai suivi pas-a-pas
  pendant un long run GPU, il faudrait un `TrainerCallback` personnalise
  appelant `logger_metrique()` depuis `on_log`/`on_evaluate` : pas
  implemente a ce jour.
- **`horodatage` mesure l'ecriture, pas le calcul** : consequence directe
  du point ci-dessus, toutes les lignes d'un meme run auront des
  horodatages tres proches les uns des autres (ecrites en boucle rapide),
  pas espacees dans le temps comme l'aurait ete un vrai suivi pas-a-pas.
- **`perte_validation` rare par construction** : `eval_strategy="epoch"`
  signifie au plus `nombre_epoques` points de validation par run (3
  aujourd'hui), jamais un point de validation a chaque `logging_steps`.
