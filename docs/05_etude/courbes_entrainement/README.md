# Courbes d'entrainement et d'evaluation (export PNG statique)

Figures generees par `monitoring/exporter_courbes_png.py` (G7 de l'audit
`m14-ocr-nfr-latencia-robustez-trazabilidad`) depuis les metriques deja
publiees sur les depots dataset HF correspondants, aucun nouveau run
necessaire. Commande utilisee pour chaque dossier, a titre de
reproduction :

```
uv run python monitoring/exporter_courbes_png.py \
    --repo-id <depot> --nom-run <run> \
    --sortie-dir docs/05_etude/courbes_entrainement
```

## `sft-lora-16092026-reconstruit/courbe_pertes.png`

Depot `mombasstic/chsa-triage-sft-metrics`. Premier run SFT-LoRA reel
(`assistant_only_loss=False`, perte pleine sequence), courbe
reconstruite a posteriori depuis le log brut du job
(`monitoring/reconstruire_courbe_sft_depuis_log.py`, cf. `AGENTS.md`) :
la publication en direct avait ete perdue par un bug de configuration
(`suivi.backend: mlflow` au lieu de `hf_dataset`, corrige depuis).
Perte train decroissante sur 342 pas (3 epoques), 3 points de perte
validation.

## `sft-lora-essai-1/courbe_pertes.png`

Depot `mombasstic/chsa-triage-sft-metrics-assistant-only-loss`. Run SFT
de cette session (`assistant_only_loss=True`, perte masquee aux tours
assistant). **Les 3 points de perte validation ont ete repares dans
cette tache** : ils avaient ete reellement calcules par le job
(`6abec439404719ba3761a42c`, confirme via `hf jobs logs`,
`eval_loss` 1.7024/1.6537/1.6403 aux pas 113/226/339) mais jamais
publies, a cause d'un bug reel dans
`TrlSftEntraineurAdapter._courbe_depuis_log_history` : `eval_strategy=
"epoch"` declenche l'evaluation aux bornes d'epoque, qui ne coincident
jamais exactement avec un pas multiple de `logging_steps=10` (ce
meme phenomene, deja documente dans `AGENTS.md` pour le tout premier
run SFT, bornes 114/228/342), et la fonction exigeait une egalite
exacte de `step` pour rattacher une perte de validation a un point de
la courbe. **Corrige dans le code** (rattache desormais chaque
`eval_loss` au pas d'entrainement publie le plus proche, cf.
`tests/infrastructure/test_trl_sft_entraineur_courbe.py`) pour que les
prochains runs ne perdent plus cette donnee ; les 3 valeurs reelles de
ce run ont ete rattachees retroactivement sur le depot HF (pas
110/230/330, les plus proches des bornes reelles 113/226/339,
cf. `parametres.json` du run pour la note de reparation complete).

## `evaluation-post-sft/metriques_evaluation.png` et `evaluation-post-dpo/metriques_evaluation.png`

Depot `mombasstic/chsa-triage-baseline-metrics`. Evaluation batch
(`EvaluerBaselineZeroShotUseCase`, 278 exemples du split test) du
checkpoint SFT-LoRA puis du checkpoint DPO-LoRA. Panneau gauche :
`exact_match`/`f1_moyen` (0-1). Panneau droit : `latence_ms_moyenne`.
F1 moyen quasi identique entre les deux (post-SFT 0.1117, post-DPO
0.1095) : le DPO n'a pas degrade la qualite mesuree par cette metrique
stricte (deja documente en texte dans `AGENTS.md`/README, ici visualise
pour la premiere fois). Latence ~11.2-11.6s par generation sur ces deux
runs : mesure batch hors-ligne (GPU L4, HF Jobs), PAS une mesure contre
l'endpoint API deploye (cf. Phase 2 de cette meme tache pour cette
derniere).

## Run DPO d'entrainement : aucune figure (gap reel, non corrige ici)

`mombasstic/chsa-triage-dpo-metrics/dpo-lora/metriques.jsonl` est vide
(0 octet) sur le Hub, bien que `parametres.json` du meme run porte les
vrais hyperparametres DPO (`beta=0.3`, checkpoint de depart
`mombasstic/chsa-triage-sft-lora`, cf. le fichier lui-meme) : le DPO a
bien tourne reellement (poids publies, evalue ci-dessus) mais sa
courbe de perte/recompenses d'entrainement n'a jamais ete publiee. Le
script `exporter_courbes_png.py` ne produit donc aucun fichier pour ce
run (`Aucune metrique a exporter`). Racine probable : meme famille de
bug que celui corrige ci-dessus (`_courbe_depuis_log_history`, reutilise
tel quel par `TrlDpoEntraineurAdapter`), mais non investiguee en detail
ici : hors du perimetre explicite de cette tache (qui ne demandait
d'investiguer que le run SFT assistant-only-loss), notee pour une tache
future plutot que de rallonger celle-ci.
