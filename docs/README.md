# Index de la documentation

La documentation est numérotée selon la séquence du projet. Lire les
dossiers dans l'ordre, et à l'intérieur de chaque dossier, lire les
fichiers dans l'ordre numérique de leur préfixe.

```
docs/
├── 00_cadrage/                  Étape 0 : cadrage du projet
│   ├── 01_cahier_des_charges.md
│   └── 02_sources_donnees_licences.md
├── 01_environnement/            Étape 0 : mise en place technique
│   ├── 00_guide_installation_environnement.md
│   └── 01_architecture_hexagonale.md
├── 02_etape1_donnees/           Étape 1 : préparation des données (FAIT)
│   ├── 00_couverture_exigences_officielles.md
│   ├── 01_rapport_rgpd.md
│   └── dataset_card_dpo_hf.md
├── 03_etape2_sft/               Étape 2 : SFT + LoRA (FAIT)
│   ├── 00_introduction_concepts.md
│   ├── 01_installation_configuration.md
│   ├── 02_etapes_cas_usage.md
│   └── 03_guide_implementation_pas_a_pas.md
├── 04_etape3_dpo/               Étape 3 : alignement DPO (FAIT)
│   ├── 00_introduction_concepts.md
│   ├── 02_etapes_cas_usage.md
│   └── 03_guide_implementation_pas_a_pas.md
├── 05_etude/                    Livrable 3 : rapport technique final
│   └── M14_Rapport_technique_CHSA_Triage.docx (généré par generer_rapport_technique.py)
├── diagrams/                    Diagrammes UML, un sous-dossier par étape
│   └── README.md                 index détaillé + état de couverture
├── images/, references/        Assets et sources externes (logos, PDF/pptx
│                                fournis, livre théorique) ; pas de lecture
│                                séquentielle, consultés par renvoi depuis
│                                les documents ci-dessus.
```

L'Étape 4 (déploiement, endpoint vLLM+LoRA, API/frontend, CI/CD) n'a
pas son propre dossier `docs/` : sa procédure réelle et son historique
de débogage vivent directement dans le `README.md` racine (§4, table
de dépannage) plutôt que d'être dupliqués ici ; ses diagrammes restent
dans `diagrams/05_etape4_deploiement/`.

Voir `diagrams/README.md` pour le détail de chaque diagramme (type,
formats disponibles, statut réel/conceptuel).

## Convention de nommage

- Le préfixe du **dossier** correspond à l'étape du projet (`00` =
  cadrage, `01` = environnement, `02` = Étape 1 données, etc.).
- Le préfixe du **fichier** correspond à l'ordre de lecture à
  l'intérieur de l'étape.
- Chaque document se termine par un renvoi explicite vers le document
  suivant à lire.
