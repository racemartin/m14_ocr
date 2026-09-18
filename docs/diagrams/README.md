# Index des diagrammes

Structure : un sous-dossier par étape du projet (aligné sur les
dossiers de `docs/`), et à l'intérieur, un sous-dossier par type de
diagramme UML/PlantUML. Chaque diagramme est fourni en 4 formats :
`.puml` (source), `.png`, `.svg`, `.pdf`.

```
docs/diagrams/
├── _common/estilo.iuml          style PlantUML partagé (!include)
├── 00_vue_ensemble/
│   ├── vision_generale_etapes.puml  V1 : pitch visuel simplifié (5 cases)
│   │                                  du projet, étapes 1 à 3 en ordre,
│   │                                  avec entrée/sortie de chacune (réel,
│   │                                  DPO marqué comme futur), pour une
│   │                                  première lecture rapide
│   ├── vision_generale_etapes_v3_detaille.puml  V3 : flux technique
│   │                                  détaillé (mêmes 5 étapes que la V1,
│   │                                  développées), avec les scripts CLI,
│   │                                  adaptateurs et dépôts Hugging Face
│   │                                  réels (noms exacts vérifiés contre
│   │                                  le code) ; recrée en PlantUML éditable
│   │                                  le contenu de vision_generale_etapes_V2.png
│   │                                  (PNG fixe créé manuellement dans un
│   │                                  outil externe, sans source éditable,
│   │                                  laissé intact) en corrigeant l'étape
│   │                                  DPO pour la marquer [CONCEPTUEL]/
│   │                                  [PRÉVU] au lieu de la présenter avec
│   │                                  le même style que les étapes réelles,
│   │                                  pour un lecteur qui veut le détail
│   │                                  technique
│   └── activite/                 roadmap complet du projet (4 semaines)
├── 01_environnement/
│   └── paquets/                  architecture hexagonale (classes réelles)
├── 02_etape1_donnees/
│   ├── activite/                  pipeline de données bout-en-bout ;
│   │                                anonymisation Presidio + contrôle qualité PII résiduelle ;
│   │                                échantillon_stratifie() (méthode du plus grand reste,
│   │                                partagée entre --limite et --n)
│   ├── sequence/                   les 6 scripts CLI (telecharger, profiler, construire le pivot, anonymiser, decouper, verifier la repartition)
│   │                                + vue frontiere generique/specifique du pivot (E1_03_01_mappers_corpus.py)
│   ├── paquets/                     classes réelles de l'Étape 1 ;
│   │                                 paquets en jeu pour anonymisation/rapport/QC ;
│   │                                 paquets en jeu pour decouper_splits/verifier_repartition_splits ;
│   │                                 PAS ENCORE mis à jour (11/09/2026, ajout jugé mineur pour ce
│   │                                 diagramme) avec l'extraction du sous-ensemble DPO
│   │                                 (E1_05_03_extraire_sous_ensemble_dpo.py), le nouveau port
│   │                                 FormateurInviteZeroShot, ou l'évaluation baseline zero-shot
│   │                                 (E1_06_00_evaluer_baseline_zero_shot.py + implémentation réelle
│   │                                 de LlamaCppInferenceAdapter) : voir AGENTS.md pour ces ajouts
│   └── deploiement/                 environnement local (WSL2, uv)
├── 03_etape2_sft/
│   ├── activite/                  pipeline SFT (reel, avec le point de
│   │                                decision "convergence saine ?")
│   ├── sequence/                   entrainement SFT : script training/E2_04_sft_train.py,
│   │                                cas d'usage et adaptateurs reels (reel)
│   ├── paquets/                     classes/ports reels du SFT-LoRA, y compris le
│   │                                 paquet monitoring/ (dashboard + importateur
│   │                                 MLflow local), et leur relation aux paquets
│   │                                 reels de l'Etape 1 (reel)
│   └── deploiement/                HF Jobs (entrainement) + Space Streamlit (suivi
│                                      en vivo) + machine locale (historique
│                                      MLflow/SQLite) (reel)
├── 04_etape3_dpo/
│   ├── activite/                  double fonction en une seule passe DPO
│   │                                (`dpo_double_fonction_entrainement.puml`,
│   │                                conceptuel : Étape 3 non implémentée)
│   └── deploiement/                infrastructure HF Jobs (conceptuel, à faire)
└── 05_etape4_deploiement/
    ├── activite/                  pipeline CI/CD (conceptuel)
    ├── sequence/                   cas d'usage en production (conceptuel)
    └── deploiement/                 architecture complète (conceptuel)
```

## État de couverture

| Étape | Activité | Séquence | Paquets | Déploiement |
|---|---|---|---|---|
| 00 : Vue d'ensemble | [FAIT] | N/A | N/A | N/A |
| 00 : Vue d'ensemble : pitch V1 (`vision_generale_etapes.puml`) | [FAIT] (réel) | N/A | N/A | N/A |
| 00 : Vue d'ensemble : détail technique V3 (`vision_generale_etapes_v3_detaille.puml`) | [FAIT] (réel, étape DPO marquée [CONCEPTUEL]) | N/A | N/A | N/A |
| 01 : Environnement | N/A | N/A | [FAIT] (réel) | N/A |
| 02 : Étape 1 (données) | [FAIT] (réel) | [FAIT] (réel) | [FAIT] (réel) | [FAIT] (réel) |
| 03 : Étape 2 (SFT) | [FAIT] (réel) | [FAIT] (réel) | [FAIT] (réel) | [FAIT] (réel) |
| 04 : Étape 3 (DPO) | [FAIT] (conceptuel) | [A FAIRE] | [A FAIRE] | [A FAIRE] |
| 05 : Étape 4 (déploiement) | [FAIT] (conceptuel) | [FAIT] (conceptuel) | [A FAIRE] | [FAIT] (conceptuel) |

**« réel »** = généré à partir du code effectivement écrit
(`src/chsa_triage/`, `interfaces/cli/`, `training/`, `monitoring/`).
**« conceptuel »** = anticipe une architecture qui n'est pas encore
codée (`interfaces/api/`, `interfaces/web/`), à mettre à jour dès que
le code correspondant existe. Les diagrammes de l'Étape 2 (SFT) sont
passés de « conceptuel » à « réel » le 14/09/2026 une fois
`training/E2_04_sft_train.py` et `TrlSftEntraineurAdapter` effectivement
écrits (vérifiés SANS GPU, jamais exécutés sur une vraie session GPU à
ce moment-là) et le paquet `monitoring/` (dashboard Streamlit +
importateur MLflow local) ajouté au diagramme de paquets. Mis à jour le
17/09/2026 (diagrammes de classes, de paquets et de déploiement de
l'Étape 2) pour refléter le premier entraînement SFT-LoRA réellement
exécuté avec succès sur GPU L4 (verdict SAINE, poids publiés), les
trois adaptateurs `MoteurInference` réels, la publication réelle des
poids/métriques sur HF Hub, et les deux nouveaux CLI
(`interfaces/cli/E2_00_formater_dataset_chatml.py`,
`interfaces/cli/E2_05_evaluer_post_sft.py`) : cf. AGENTS.md pour le
détail complet.
**[A FAIRE]** = pas encore produit : les diagrammes de séquence, de
paquets et de déploiement du DPO (Étape 3) restent à faire, aucune
proposition de conception équivalente n'ayant encore été documentée
pour cette étape ; le diagramme de paquets de l'Étape 4 de même, une
fois `interfaces/api` et `interfaces/web` implémentés (ou une
proposition de conception documentée pour eux, sur le même principe
que l'Étape 2). Le premier diagramme d'activité de l'Étape 3
(`04_etape3_dpo/activite/dpo_double_fonction_entrainement.puml`,
18/09/2026) est conceptuel par nature (aucun script DPO n'existe dans
le code) : il illustre comment une seule passe d'entraînement DPO
enseignerait à la fois la préférence clinique et le format de sortie
JSON contractuel (F3), complémentaire du diagramme simplifié déjà
présent dans le livre théorique (§6).

## Vue d'ensemble : V1 vs V3 (et le PNG V2)

`00_vue_ensemble/` contient trois vues du même pipeline, pour des
publics distincts : **V1** (`vision_generale_etapes.puml`, référencée
dans `README.md`) reste la version simple à 5 cases pour une première
lecture ; **V2** (`vision_generale_etapes_V2.png`) est un PNG plat créé
manuellement dans un outil externe, sans fichier source éditable dans
le dépôt, laissé intact ; **V3** (`vision_generale_etapes_v3_detaille.puml`,
18/09/2026) recrée
en PlantUML éditable le contenu technique de la V2 (mêmes 5 étapes,
mêmes noms de scripts/adaptateurs/dépôts HF) tout en corrigeant un
problème d'honnêteté visuelle de la V2 : celle-ci dessinait l'Étape 3
(DPO) avec exactement le même style que les 4 étapes réelles, alors
qu'aucun script DPO n'existe dans le code. La V3 marque cette case
**[CONCEPTUEL] / [PRÉVU]** (fond distinct, même convention que
`dpo_double_fonction_entrainement.puml`), précise que « dpo-lora » et
le « checkpoint aligné » sont des noms prédits par convention jamais
vérifiés, et renvoie vers le diagramme DPO dédié pour le détail de la
double fonction plutôt que de le dupliquer.

## Régénérer les diagrammes

```bash
# Un seul fichier (invocations séparées : combiner -tpng -tsvg -tpdf dans
# un seul appel ne génère silencieusement que le dernier format demandé,
# cf. AGENTS.md pour les contournements spécifiques à cette machine)
cd docs/diagrams/<etape>/<type>
plantuml -tpng <nom>.puml
plantuml -tsvg <nom>.puml
plantuml -tpdf <nom>.puml

# Tous les diagrammes du projet
cd docs/diagrams
find . -name "*.puml" -not -path "./_common/*" | while read -r f; do
  dir=$(dirname "$f"); base=$(basename "$f")
  (cd "$dir" && plantuml -tpng "$base" && plantuml -tsvg "$base" && plantuml -tpdf "$base")
done
```

## Convention de style

Tous les diagrammes incluent `../../_common/estilo.iuml` (adapter le
nombre de `..` selon la profondeur) pour une identité visuelle
cohérente : palette teal/crème/ambre reprise de la première session
de diagrammes du projet.
