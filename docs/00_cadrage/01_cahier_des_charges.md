# Cahier des charges : Agent IA de Triage Médical (CHSA)

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
| Commanditaire / évaluatrice | Dr. Marie Dubois, Directrice Innovation Médicale | Validation clinique, soutenance |
| IA Engineer junior | Rafael Cerezo Martín | Conception, développement, livraison |

## 3. Exigences fonctionnelles

| ID | Exigence | Source |
|---|---|---|
| F1 | Collecter les symptômes du patient via un questionnaire intelligent adaptatif | Mission (texte exact, brief Dr. Dubois) |
| F2 | Classer la priorité clinique selon l'échelle ESI (Niveaux 1-5) | Manuel SFT §7.1 |
| F3 | Produire une sortie JSON strict (niveau, catégorie, ressources estimées) | Manuel SFT §7.1 (absent du texte de mission lui-même) |
| F4 | Exposer un raisonnement clinique explicite (bloc `<think>`) | Mission : « explications claires » (le bloc `<think>` precis est un choix d'implementation, pas une exigence litterale) |
| F5 | Répondre en français et en anglais | Mission : dataset bilingue |
| F6 | Tracer chaque interaction (horodatage, entrée, sortie, version modèle) | Mission : « auditabilité » |
| F7 | Exposer le modèle via une API de démonstration | Mission : Livrable 4 |

> **Note (implémentation, 23/09/2026)** : relecture du texte de mission
> original (`MISION!_Finetunez votre propre LLM - OpenClassrooms.pdf`,
> ce même dossier) apres les premiers runs DPO reels. Deux
> clarifications importantes, qui reconfigurent comment F1/F3/F4 sont
> satisfaites :
>
> - **F1** n'est demandee QUE dans le brief narratif de la mission
>   (email Dr. Dubois) ; aucune des 4 etapes concretes de la mission
>   (donnees / SFT+DPO / deploiement / rapport) ne demande d'entrainer
>   cette capacite specifiquement, et les 4 corpus sources imposes
>   (MediQAl, FrenchMedMCQA, MedQuAD, UltraMedical-Preference) sont
>   tous des paires a UN SEUL tour (Q/R, QCM, preference), jamais des
>   entretiens multi-tours. La seule mention connexe est la phase 2 du
>   brief, "simulation d'inference en conditions quasi-reelles" -
>   rattachee au deploiement (Etape 4), pas a l'entrainement. F1 est
>   donc traitee comme une exigence de **conception du prompting au
>   moment de l'inference** (systeme de prompt + historique de
>   conversation multi-tours envoye au modele deja SFT+DPO), pas comme
>   une exigence de donnees d'entrainement supplementaires.
> - **F3/F4** (JSON strict + bloc `<think>`) ne viennent PAS du texte
>   de mission OpenClassrooms lui-meme (aucune occurrence de "JSON" ni
>   de `<think>` dans le PDF source) : ce sont des exigences
>   auto-imposees, ajoutees via le Manuel SFT interne au projet. La
>   tentative d'enseigner ce format PENDANT le DPO (reformulation
>   `chosen -> <think>+JSON`, cf. `docs/04_etape3_dpo/`) a echoue
>   empiriquement sur 6 runs reels (le modele 1.7B ne suit pas la
>   consigne de reformulation de facon fiable) ; decision prise de
>   **decoupler** : le DPO s'entraine desormais sur les paires
>   `chosen`/`rejected` d'origine (`--skip-reformulation`), et le
>   respect du format JSON est repousse au **prompting/contrainte de
>   generation a l'inference (Etape 4)**, plutot que d'etre re-appris
>   pendant l'alignement. Cout reel mesure de ce choix : le F1 token
>   post-DPO **regresse** par rapport au post-SFT (0,112 -> 0,049,
>   README §3), et 0/278 sorties du checkpoint DPO brut sont un JSON
>   valide - preuve que le format doit bien etre repris a la couche
>   inference, il ne survit pas tel quel a travers le DPO actuel.
>   Details et chiffres complets : `README.md` §3 "DPO".

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
  "identifiant": "chsa-<espace_noms>-<hash>",
  "identifiant_source_brute": "cle naturelle du registre brut d'origine",
  "source": "MediQAl | FrenchMedMCQA | MedQuAD | UltraMedical-Preference",
  "type_exemple": "sft | dpo",
  "langue": "fr | en",
  "symptomes": "texte libre ou structuré",
  "antecedents": "texte libre, optionnel",
  "constantes_vitales": {"pression_arterielle": null, "frequence_cardiaque": null, "saturation_o2": null, "frequence_respiratoire": null},
  "prompt": [{"role": "system", "contenu": "..."}, {"role": "user", "contenu": "..."}],
  "completion": [{"role": "assistant", "contenu": "..."}],
  "chosen": [],
  "rejected": [],
  "niveau_confiance": "haute | moyenne | basse",
  "anonymise": true,
  "split": "train | val | test"
}
```

> **Note (implémentation, 08/09/2026)** : les noms de champ ci-dessus
> sont ceux **réellement implémentés** (`ExemplePivot`,
> `ConstantesVitales`, `src/chsa_triage/domain/model/exemple_pivot.py`),
> pas ceux de l'énoncé initial de la mission. Ils ont été adaptés au
> français/vocabulaire du domaine pendant l'implémentation, pour
> rester cohérents avec le reste du code (100 % en français) : `id` →
> `identifiant`, `type` → `type_exemple`, `pa/fc/spo2/fr` →
> `pression_arterielle/frequence_cardiaque/saturation_o2/frequence_respiratoire`,
> et la clé `content` des messages `prompt`/`completion` →
> `contenu`. `chosen`/`rejected` valent un tuple vide `()` (`[]` en
> JSON) quand ils ne s'appliquent pas, jamais `null`. Le champ
> `identifiant_source_brute` a été ajouté a posteriori (08/09/2026,
> il ne figurait pas dans l'énoncé initial) pour
> tracer chaque `ExemplePivot` jusqu'au registre brut d'origine dans
> `data/raw/*.jsonl` (exigence RGPD d'auditabilité, cf. NF6). Les
> valeurs des énumérations (`fr`/`en`, `sft`/`dpo`,
> `haute`/`moyenne`/`basse`, `train`/`val`/`test`) sont, elles,
> inchangées par rapport à l'énoncé. Le code fait foi : cette section
> documente l'implémentation réelle, elle ne la prescrit plus.

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
  requêtes de test. **Statut réel (23/09/2026)** : non atteint par le
  checkpoint DPO brut évalué directement (0/278, cf. note §3
  ci-dessus) — attendu, puisque ce critère est désormais visé au
  niveau de l'**endpoint déployé** (prompting/contrainte de génération,
  Étape 4), pas au niveau du checkpoint SFT+DPO nu. À re-mesurer une
  fois le prompting de production en place, pas avant.
- L'accuracy de classification ESI sur le jeu de test clinique dépasse
  la baseline zéro-shot (Phase 1) de façon mesurable. **Statut réel** :
  non mesurable tel quel sur le checkpoint DPO brut (0 paire JSON
  comparable, même cause que ci-dessus) ; le F1 token, lui, régresse
  par rapport au post-SFT (0,112 -> 0,049, README §3) — signal à
  surveiller une fois le format restauré côté inférence, pas à ignorer.
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
