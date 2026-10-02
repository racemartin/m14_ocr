# CHSA Triage Pre-vérificatios

Étape 3 · assistant_only_loss + NFR latence/robustesse/traçabilité · 02‑10‑2026

Déployé et sain

F1 post‑SFT (config. actuelle)

0.112

inchangé cette nuit

F1 assistant_only_loss=true

0.039

testé et écarté

Latence diagnostic

13–16 s

Space réel, 3 scénarios

Déploiement

200 OK

vérifié en direct par moi

## 01Ce qui a été testé

La variante `assistant_only_loss=true` du SFT a été réellement implémentée et entraînée — l'amélioration prévue pour cette nuit. Le résultat mesuré est une régression nette, pas une amélioration : **F1 0.039**, en dessous même du zero‑shot (0.043), très loin du 0.112 déjà obtenu. La variante a été écartée et la configuration par défaut restaurée, avec la raison documentée directement dans le fichier de recette.

| Configuration | F1 (token) | Latence moyenne |
| --- | --- | --- |
| Zero‑shot GPU (sans entraînement) | 0.04338 | 7 295 ms |
| SFT, assistant_only_loss=true | 0.03919 | 12 182 ms |
| SFT, assistant_only_loss=false (actuel) | 0.11167 | 11 553 ms |
| Post‑DPO (modèle en production) | 0.10951 | 11 227 ms |

278/278 exemples évalués à chaque fois, 0 échec d'inférence. Source : `mombasstic/chsa-triage-baseline-metrics`.

## 02Ce qui a vraiment été comblé

La mission demandait des tests de latence, de robustesse et des audits de traçabilité "en conditions réalistes". Avant cette nuit, ce manque existait même dans la documentation interne du projet. Il y a désormais des chiffres réels contre le Space déployé : temps de réponse mesurés, trois sessions concurrentes sans échec, échecs d'inférence correctement journalisés dans l'audit, et des graphiques réels exportés des courbes SFT/DPO.

Valeurs exactes des courbes d'entraînement (les PNG du dépôt sont visuels ; voici les chiffres bruts derrière) :

| Run | Pas | Perte train | Perte validation |
| --- | --- | --- | --- |
| SFT baseline (actuel) | 10 | 2.5687 | — |
| 340 (final) | 1.5289 | 1.6365 |  |
| SFT assistant_only_loss=true | 10 | 2.8193 | — |
| 330 (final) | 1.5391 | 1.6403 |  |

Table complète pas à pas (33‑34 points par run) : `mombasstic/chsa-triage-sft-metrics` et `-assistant-only-loss`, dataset HF, téléchargeable via `monitoring/hf_dataset_runs.py`.

## 03Constat clinique à mentionner

Sous‑triage réel observé Cas de douleur thoracique urgente (62 ans, antécédent d'infarctus) classé **ESI 4** alors que le tableau clinique attendait **ESI 2**. Confirme, avec un exemple réel et non abstrait, pourquoi le garde‑fou de sécurité clinique reste une décision produit en suspens — ce POC ne doit pas être utilisé sans supervision humaine.

## 04Ce qu'il reste

- Rien de bloquant sur ce projet.
- Ouvert, non urgent : pourquoi le masquage de la perte a autant régressé (hypothèses documentées, non investiguées davantage).

Matériel complet : `docs/05_etude/courbes_entrainement/` (graphiques) · `docs/05_etude/benchmark_latence_robustesse/` (rapport + captures des 3 cas cliniques)