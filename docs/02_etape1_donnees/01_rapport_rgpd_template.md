# Rapport de justification RGPD — Anonymisation du dataset CHSA

> **Statut : gabarit à remplir.** Ce document a la structure et la
> méthodologie prêtes ; les sections marquées `[À REMPLIR]` doivent
> être complétées une fois `anonymiser_dataset.py` exécuté sur les 4
> corpus réels (`data/raw/`), conformément au résultat attendu
> officiel : *« Justification du processus RGPD suivi »*.

## 1. Cadre légal et périmètre

Le dataset source combine des extraits de conversations et de
questions-réponses médicales (MediQAl, FrenchMedMCQA, MedQuAD,
UltraMedical-Preference). Bien que ces corpus soient publics et
académiques, la mission impose une anonymisation systématique avant
tout usage en fine-tuning, par précaution et conformité RGPD
(art. 5, minimisation des données ; art. 25, protection des données
dès la conception).

## 2. Outil et méthode

- **Outil** : Microsoft Presidio (`presidio-analyzer` +
  `presidio-anonymizer`), recommandation explicite de la mission.
- **Modèles NLP** : `fr_core_news_md` (français), `en_core_web_sm`
  (anglais) — configuration multi-langue explicite (voir note
  technique §5).
- **Entités ciblées** : noms de personnes, numéros de téléphone,
  adresses e-mail, lieux, dates de naissance, identifiants médicaux
  le cas échéant.
- **Stratégies comparées** :

| Stratégie | Comportement | Cas d'usage retenu |
|---|---|---|
| `replace` | Remplace l'entité par un jeton `<INFO_MASQUEE>` | **Retenue par défaut** — préserve la lisibilité de la phrase pour le SFT |
| `mask` | Remplace par des `*` | Alternative si le jeton `<INFO_MASQUEE>` s'avère lui-même trop informatif statistiquement |
| `redact` | Supprime purement l'entité | Non retenue — casse la syntaxe de la phrase, dégraderait la qualité SFT |

## 3. Résultats quantitatifs [À REMPLIR après exécution réelle]

| Corpus | Enregistrements traités | Entités détectées (total) | Entités par type | Taux d'enregistrements avec ≥1 entité détectée |
|---|---|---|---|---|
| MediQAl | — | — | — | — |
| FrenchMedMCQA | — | — | — | — |
| MedQuAD | — | — | — | — |
| UltraMedical-Preference | — | — | — | — |

## 4. Contrôle qualité manuel [À REMPLIR]

Conformément à l'exigence NF2 du cahier des charges (0 PII résiduelle
sur échantillon de contrôle), un échantillon aléatoire d'au moins
**50 enregistrements anonymisés par corpus** doit être relu
manuellement.

- Taille de l'échantillon contrôlé : `[À REMPLIR]`
- Nombre de faux négatifs trouvés (PII non détectée) : `[À REMPLIR]`
- Nombre de faux positifs trouvés (masquage abusif de terme médical
  non-personnel) : `[À REMPLIR]`
- Décision : `[À REMPLIR]` (dataset accepté / itération de filtrage
  supplémentaire nécessaire)

## 5. Note technique — limite connue documentée

Un smoke test d'intégration (02/09/2026, données synthétiques) a
révélé que Presidio, mal configuré, ne supporte l'anglais **que** par
défaut — corrigé via un `NlpEngineProvider` multi-langue explicite
(voir `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`
§Validation technique).

Le même test a montré qu'un numéro de téléphone au format court
(sans indicatif, ex. `555-0142`) n'est **pas** détecté par le
reconnaisseur par défaut de Presidio en anglais. **Ce n'est pas un
bug corrigible côté code applicatif** — c'est une limite connue des
modèles de reconnaissance d'entités pré-entraînés. Elle justifie à
elle seule l'obligation du contrôle qualité manuel (§4) : l'automatisation
réduit drastiquement le travail humain, elle ne le remplace pas
totalement pour un cas d'usage médical sensible.

## 6. Conclusion [À REMPLIR]

`[À REMPLIR une fois les sections 3 et 4 complétées : le dataset
anonymisé est-il jugé conforme pour un usage en fine-tuning ?]`
