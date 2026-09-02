# Index des diagrammes

Structure : un sous-dossier par étape du projet (aligné sur les
dossiers de `docs/`), et à l'intérieur, un sous-dossier par type de
diagramme UML/PlantUML. Chaque diagramme est fourni en 4 formats :
`.puml` (source), `.png`, `.svg`, `.pdf`.

```
docs/diagrams/
├── _common/estilo.iuml          style PlantUML partagé (!include)
├── 00_vue_ensemble/
│   └── activite/                 roadmap complet du projet (4 semaines)
├── 01_environnement/
│   └── paquets/                  architecture hexagonale (classes réelles)
├── 02_etape1_donnees/
│   ├── activite/                  pipeline de données bout-en-bout
│   ├── sequence/                   construction du dataset pivot
│   ├── paquets/                     classes réelles de l'Étape 1
│   └── deploiement/                 environnement local (WSL2, uv)
├── 03_etape2_sft/
│   ├── activite/                  pipeline SFT (conceptuel)
│   └── deploiement/                infrastructure HF Jobs / Dev Mode (conceptuel)
├── 04_etape3_dpo/
│   ├── activite/                  pipeline DPO (conceptuel)
│   └── deploiement/                infrastructure HF Jobs (conceptuel)
└── 05_etape4_deploiement/
    ├── activite/                  pipeline CI/CD (conceptuel)
    ├── sequence/                   cas d'usage en production (conceptuel)
    └── deploiement/                 architecture complète (conceptuel)
```

## État de couverture

| Étape | Activité | Séquence | Paquets | Déploiement |
|---|---|---|---|---|
| 00 — Vue d'ensemble | [FAIT] | — | — | — |
| 01 — Environnement | — | — | [FAIT] (réel) | — |
| 02 — Étape 1 (données) | [FAIT] (réel) | [FAIT] (réel) | [FAIT] (réel) | [FAIT] (réel) |
| 03 — Étape 2 (SFT) | [FAIT] (conceptuel) | [A FAIRE] | [A FAIRE] | [FAIT] (conceptuel) |
| 04 — Étape 3 (DPO) | [FAIT] (conceptuel) | [A FAIRE] | [A FAIRE] | [FAIT] (conceptuel) |
| 05 — Étape 4 (déploiement) | [FAIT] (conceptuel) | [FAIT] (conceptuel) | [A FAIRE] | [FAIT] (conceptuel) |

**« réel »** = généré à partir du code effectivement écrit
(`src/chsa_triage/`, `interfaces/cli/`).
**« conceptuel »** = anticipe une architecture qui n'est pas encore
codée (`training/*.py`, `interfaces/api/`, `interfaces/web/`), à
mettre à jour dès que le code correspondant existe.
**[A FAIRE]** = pas encore produit : les diagrammes de séquence et de
paquets du SFT/DPO n'ont de sens qu'une fois `training/sft_train.py`
et `training/dpo_train.py` écrits (sinon il n'y a pas encore de
classes/appels réels à documenter) ; le diagramme de paquets de
l'Étape 4 de même, une fois `interfaces/api` et `interfaces/web`
implémentés.

## Régénérer les diagrammes

```bash
# Un seul fichier
cd docs/diagrams/<etape>/<type>
plantuml -tpng -tsvg -tpdf <nom>.puml

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
