# Index de la documentation

La documentation est numérotée selon la séquence du projet. Lire les
dossiers dans l'ordre, et à l'intérieur de chaque dossier, lire les
fichiers dans l'ordre numérique de leur préfixe.

```
docs/
├── 00_cadrage/                  Étape 0 — cadrage du projet
│   ├── 00_objectifs_du_projet.md
│   └── 01_cahier_des_charges.md
├── 01_environnement/            Étape 0 — mise en place technique
│   ├── 00_guide_installation_environnement.md
│   └── 01_architecture_hexagonale.md
├── 02_etape1_donnees/           Étape 1 — préparation des données (Semaine 1)
├── 03_etape2_sft/               Étape 2 — SFT + LoRA (Semaine 2)
├── 04_etape3_dpo/               Étape 3 — alignement DPO (Semaine 3)
├── 05_etape4_deploiement/       Étape 4 — déploiement et évaluation (Semaine 4)
└── diagrams/                    Diagrammes UML, un sous-dossier par étape
    ├── README.md                 index détaillé + état de couverture
    ├── 00_vue_ensemble/activite/
    ├── 01_environnement/paquets/
    ├── 02_etape1_donnees/{activite,sequence,paquets,deploiement}/
    ├── 03_etape2_sft/{activite,deploiement}/
    ├── 04_etape3_dpo/{activite,deploiement}/
    └── 05_etape4_deploiement/{activite,sequence,deploiement}/
```

Voir `diagrams/README.md` pour le détail de chaque diagramme (type,
formats disponibles, statut réel/conceptuel).

## Convention de nommage

- Le préfixe du **dossier** correspond à l'étape du projet (`00` =
  cadrage, `01` = environnement, `02` = Étape 1 données, etc. —
  aligné sur `00_cadrage/00_objectifs_du_projet.md`).
- Le préfixe du **fichier** correspond à l'ordre de lecture à
  l'intérieur de l'étape.
- Chaque document se termine par un renvoi explicite vers le document
  suivant à lire.

## Dossiers encore vides

`02_etape1_donnees/`, `03_etape2_sft/`, `04_etape3_dpo/` et
`05_etape4_deploiement/` sont prêts à recevoir la documentation
spécifique à chaque étape au fur et à mesure de son avancement (par
exemple : rapport de profilage des corpus, notes de configuration
LoRA, résultats DPO, procédure de déploiement vLLM...).
