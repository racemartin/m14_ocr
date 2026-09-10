# Origine, licences et citations des sources de données

Documentation minimale exigée par la mission comme alternative à la
publication du dataset sur Hugging Face Hub (cf. §7 « Livrables »,
`01_cahier_des_charges.md` — Livrable 1 : « Dataset ... versionné »).
La publication (même privée) sur HF
Hub est différée jusqu'à ce que le contrôle qualité PII soit intégralement clos
(voir `data/README.md` et
`docs/02_etape1_donnees/01_rapport_rgpd.md`) ; en attendant, cette
documentation tient lieu de traçabilité pour les 4 sources listées au
§5.1 du cahier des charges.

Vérifié le 08/09/2026 contre la fiche Hugging Face réelle de chaque
dataset (métadonnées `license` exposées par l'API HF, dataset card).
Aucune valeur ci-dessous n'est supposée : là où la fiche HF ne
déclare pas de licence, c'est dit explicitement plutôt que de deviner.

## MediQAl

- **Origine** : https://huggingface.co/datasets/ANR-MALADES/MediQAl
- **Licence** : **CC-BY-4.0** — déclarée explicitement dans les
  métadonnées de la fiche HF.
- **Citation demandée** :
  ```bibtex
  @article{bazoge2026mediqal,
    title={MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation},
    author={Bazoge, Adrien},
    journal={Scientific Data},
    year={2026},
    publisher={Nature Publishing Group UK London}
  }
  ```

## FrenchMedMCQA

- **Origine** : https://huggingface.co/datasets/nthngdy/frenchmedmcqa
- **Licence** : **incertaine, à surveiller.** La fiche Hugging Face
  ne déclare **aucune licence** dans ses métadonnées (`license: null`
  via l'API HF au 08/09/2026). Le dépôt GitHub original du dataset
  (`qanastek/FrenchMedMCQA`, auteurs de la publication associée)
  affiche un badge **Apache-2.0** dans son README, mais aucun fichier
  `LICENSE` n'a pu être confirmé directement à cette date. À
  reconfirmer avant toute publication ou usage commercial du dataset
  pivot dérivé.
- **Citation** : publication associée à `qanastek/FrenchMedMCQA`
  (Labrak et al., FrenchMedMCQA : *A French Multiple-Choice Question
  Answering Dataset for Medical domain*) — à citer par prudence même
  si la fiche HF ne l'exige pas explicitement.

## MedQuAD

- **Origine** : https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset
- **Licence** : **incertaine, à surveiller.** La fiche Hugging Face
  (miroir communautaire `keivalya/MedQuad-MedicalQnADataset`) ne
  déclare **aucune licence** dans ses métadonnées (`license: null`
  via l'API HF au 08/09/2026). Le dépôt source original
  (`abachaa/MedQuAD` sur GitHub, auteurs académiques) indique une
  licence **CC-BY-4.0** globale, **mais** précise que les réponses de
  3 sous-sources (A.D.A.M. Medical Encyclopedia, MedlinePlus Drug
  information, MedlinePlus Herbal medicine and supplement
  information) ont été **retirées** du jeu de données pour respecter
  des droits d'auteur tiers (seules les métadonnées/URLs sont
  conservées pour ces 3 sous-sources). Comme le fichier utilisé ici
  provient d'un miroir HF tiers, il n'est pas garanti que cette
  exclusion ait été respectée dans le fichier réellement téléchargé —
  **à vérifier avant publication**.
- **Citation demandée** (papier source) : Asma Ben Abacha, Dina
  Demner-Fushman, *« A Question-Entailment Approach to Question
  Answering »*, BMC Bioinformatics, 2019.

## UltraMedical-Preference

- **Origine** : https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference
- **Licence** : **MIT** — déclarée explicitement dans les métadonnées
  de la fiche HF.
- **Citation demandée** :
  ```bibtex
  @misc{zhang2024ultramedical,
        title={UltraMedical: Building Specialized Generalists in Biomedicine},
        author={Kaiyan Zhang and Sihang Zeng and Ermo Hua and Ning Ding and Zhang-Ren Chen and Zhiyuan Ma and Haoxin Li and Ganqu Cui and Biqing Qi and Xuekai Zhu and Xingtai Lv and Hu Jinfang and Zhiyuan Liu and Bowen Zhou},
        year={2024},
        eprint={2406.03949},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
  }
  ```

## Synthèse

| Source | Licence | Statut de vérification |
|---|---|---|
| MediQAl | CC-BY-4.0 | Confirmée (métadonnées HF) |
| FrenchMedMCQA | Apache-2.0 (probable, non confirmée sur HF) | À reconfirmer |
| MedQuAD | CC-BY-4.0 en amont (source académique), non redéclarée sur le miroir HF utilisé | À reconfirmer, notamment l'exclusion des 3 sous-sources sous droits tiers |
| UltraMedical-Preference | MIT | Confirmée (métadonnées HF) |

Toute publication future du dataset pivot dérivé (Livrable 1, HF Hub)
devra reprendre ce tableau et, a minima, citer les 4 travaux
ci-dessus dans la fiche du dataset publié.
