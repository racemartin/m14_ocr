# Cahier des charges — Agent IA de Triage Médical (CHSA)

Référence : Mission OpenClassrooms *« Développez le POC d'un agent de
triage médical »*, Dr. Marie Dubois. Voir `00_objectifs_du_projet.md` (meme dossier)
pour la justification détaillée de chaque choix technique.

## 1. Objet et périmètre

Développer, en 4 semaines, un **Proof of Concept** démontrant la
faisabilité technique et la valeur clinique d'un agent IA de triage
pour le service des urgences du CHSA, basé sur `Qwen3-1.7B` affiné par
SFT + LoRA puis aligné par DPO.

**Hors périmètre** (explicitement, cf. Phase 3 de la mission) : modèles
32B+, intégration SIH réelle, RAG/base vectorielle, RLVR/RL classique,
monitoring de drift en production, MLOps complet (BentoML, Airflow),
protocole MCP.

## 2. Parties prenantes

| Rôle | Personne | Responsabilité |
|---|---|---|
| Commanditaire / évaluatrice | Dr. Marie Dubois — Directrice Innovation Médicale | Validation clinique, soutenance |
| IA Engineer junior | Rafael Cerezo Martín | Conception, développement, livraison |

## 3. Exigences fonctionnelles

| ID | Exigence | Source |
|---|---|---|
| F1 | Collecter les symptômes du patient via un dialogue structuré | Mission — brief Dr. Dubois |
| F2 | Classer la priorité clinique selon l'échelle ESI (Niveaux 1-5) | Manuel SFT §7.1 |
| F3 | Produire une sortie JSON strict (niveau, catégorie, ressources estimées) | Manuel SFT §7.1 |
| F4 | Exposer un raisonnement clinique explicite (bloc `<think>`) | Mission — « explications claires » |
| F5 | Répondre en français et en anglais | Mission — dataset bilingue |
| F6 | Tracer chaque interaction (horodatage, entrée, sortie, version modèle) | Mission — « auditabilité » |
| F7 | Exposer le modèle via une API de démonstration | Mission — Livrable 4 |

## 4. Exigences non fonctionnelles

| ID | Exigence | Cible / seuil |
|---|---|---|
| NF1 | Latence d'inférence par requête | Mesurée et documentée (vLLM), seuil à définir avec baseline |
| NF2 | Conformité RGPD des données d'entraînement | Anonymisation Presidio validée manuellement, 0 PII résiduelle sur échantillon de contrôle |
| NF3 | Reproductibilité de l'entraînement | Seeds fixées, checkpoints et logs conservés |
| NF4 | Garde-fou de sécurité clinique | Toute réponse jugée `safety < 4/7` par le juge LLM → score global forcé à 0 (rejet) |
| NF5 | Empreinte GPU maîtrisée | QLoRA 4-bit, ≤ budget d'un GPU cloud unique (ex. T4/A10/L4) |
| NF6 | Documentation et auditabilité | Chaque transformation de données tracée, README par livrable |

## 5. Sources de données et schéma pivot

### 5.1 Corpus utilisés

| Corpus | Contenu | Langue | Rôle |
|---|---|---|---|
| [MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | Q/R médicales | FR | SFT |
| [FrenchMedMCQA](https://huggingface.co/datasets/nthngdy/frenchmedmcqa) | QCM pharmacie/médecine | FR | SFT (raisonnement) |
| [MedQuAD](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | Q/R médicales | EN | SFT |
| [UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | Paires chosen/rejected | EN (majoritaire) | DPO |

### 5.2 Schéma pivot (métadonnées communes)

```json
{
  "id": "chsa-<source>-<uuid>",
  "langue": "fr | en",
  "source": "MediQAl | FrenchMedMCQA | MedQuAD | UltraMedical-Preference",
  "type": "sft | dpo",
  "symptomes": "texte libre ou structuré",
  "antecedents": "texte libre, optionnel",
  "constantes_vitales": {"pa": null, "fc": null, "spo2": null, "fr": null},
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "..."}],
  "chosen": null,
  "rejected": null,
  "niveau_confiance": "haute | moyenne | basse",
  "anonymise": true,
  "split": "train | val | test"
}
```

### 5.3 Volumétrie cible

- **SFT** : ≈ 5 000 paires instruction-réponse (agrégées et dédupliquées
  depuis MediQAl, FrenchMedMCQA, MedQuAD).
- **DPO** : jeu de paires préférentielles issu de UltraMedical-Preference,
  filtré/adapté au domaine du triage.
- **Split** : train / val / test disjoints, test clinique isolé dès
  l'Étape 1 et jamais réutilisé en entraînement (cf. §Points de
  vigilance mission).

## 6. Contraintes techniques

- Environnement local : WSL2, 5 Go RAM → **pas d'entraînement local**
  (cf. `../01_environnement/` `00_guide_installation_environnement.md`).
- Modèle imposé : `Qwen3-1.7B-Base` → `Qwen3-1.7B` (SFT LoRA + DPO).
- Technique de fine-tuning imposée : SFT + LoRA, puis DPO.
- Moteur d'inférence imposé : vLLM (PagedAttention).
- API imposée : FastAPI, conteneurisation Docker.
- CI/CD imposé : GitHub Actions.
- Anonymisation imposée : Microsoft Presidio.

## 7. Livrables (rappel contractuel)

| # | Livrable | Format | Contenu |
|---|---|---|---|
| 1 | Dataset | HF Datasets / JSONL, versionné | Bilingue, anonymisé, SFT ≈5000 paires + DPO |
| 2 | Modèle | Poids + adaptateurs | Qwen3-1.7B SFT-LoRA + DPO |
| 3 | Rapport technique | PDF, ≤ 20 pages | Méthodologie, métriques, analyse, roadmap |
| 4 | Endpoint + CI/CD | Cloud + GitHub Actions | vLLM, Docker, FastAPI, tests automatisés |

Nommage du dépôt final : `Titre_du_projet_Nom_Prenom.zip`, chaque
livrable nommé `Nom_Prenom_n°_Nom_du_livrable_mmaaaa`.

## 8. Planning (4 semaines)

| Semaine | Objectif | Livrable associé |
|---|---|---|
| S1 | Données : collecte, unification, anonymisation, split | Livrable 1 |
| S2 | SFT + LoRA sur `Qwen3-1.7B-Base` | Checkpoint intermédiaire |
| S3 | DPO sur checkpoint SFT | Livrable 2 |
| S4 | Déploiement vLLM/Docker/CI-CD + évaluation + rapport | Livrables 3 & 4 |

## 9. Critères d'acceptation du POC

- Le modèle produit un JSON valide (schéma Pydantic) sur ≥ 95 % des
  requêtes de test.
- L'accuracy de classification ESI sur le jeu de test clinique dépasse
  la baseline zéro-shot (Phase 1) de façon mesurable.
- Aucune réponse retenue en évaluation finale n'a de score de sécurité
  < 4/7 (garde-fou NF4).
- Endpoint vLLM opérationnel avec latence documentée.
- Pipeline CI/CD exécute les tests automatiquement à chaque push.

## 10. Points de vigilance (repris de la mission)

- Ne jamais mélanger données d'entraînement et données d'évaluation.
- Conserver une trace de chaque transformation de données.
- Protéger les clés/secrets et l'accès aux endpoints.
- Prévoir des procédures de surveillance après déploiement.
- Documenter clairement les limites d'usage pour les utilisateurs.

*(document suivant : `../01_environnement/` `00_guide_installation_environnement.md`)*
