\newpage

# Étape 2 : Étapes du pipeline et cas d'usage proposés

> Document de conception : aucun des cas d'usage, ports ou adaptateurs
> nommés ci-dessous n'est encore écrit. Les noms suivent le patron
> `uc_NN_verbe_objet` initialement utilisé par l'Étape 1 avant son
> renommage en `E1_NN[_MM]_verbe_objet.py` (voir
> `src/chsa_triage/application/use_cases/`) ; cette proposition pour
> l'Étape 2 continue la numérotation globale héritée (`uc_05_...`) et
> n'a pas encore été retouchée pour suivre la nouvelle convention.
> Statut, pour chaque ligne : **[CONCEPTION]**
> = proposé et motivé ici, aucun code écrit ; il n'y a pas encore de
> **[FAIT]** possible pour cette étape (cf. légende de
> `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`,
> réservée aux exécutions réelles).

## Vue d'ensemble

| Étape du roadmap | Cas d'usage / module proposé | Port(s) | Adaptateur(s) proposé(s) | Statut |
|---|---|---|---|---|
| Charger `Qwen3-1.7B-Base` en 4-bit NF4 | *(pas de cas d'usage dédié : voir §1 Décision)* | `EntraineurSupervise` | `TrlSftEntraineurAdapter` (charge le modèle dans son constructeur) | [CONCEPTION] |
| Formater en ChatML + `assistant_only_loss` | `uc_05_00_formater_dataset_chatml.py` → `FormaterDatasetChatMLUseCase` | `FormateurConversation`, `RepositoryLectureEcriture` (×2) | `ChatMLFormateurAdapter`, `JsonlDatasetRepository` (réutilisé) | [CONCEPTION] |
| Configurer `LoraConfig` | *(pas de cas d'usage dédié, assemblage de configuration : voir §1 Décision)* | (aucun) | `recipes/sft_qwen3_lora.yaml` → `ConfigurationLora` | [CONCEPTION] |
| Lancer `SFTTrainer` | `uc_05_01_entrainer_sft.py` → `EntrainerSftUseCase` | `EntraineurSupervise`, `SuiviExperimentation` | `TrlSftEntraineurAdapter`, `MlflowSuiviExperimentation` (ou `TensorboardSuiviExperimentation`) | [CONCEPTION] |
| Évaluer sur validation intermédiaire | fonction pure `application/verdict_convergence.py::evaluer_convergence` | (aucun port : lit la courbe déjà produite par `entrainer()`) | (aucun) | [CONCEPTION] |
| Boucle d'ajustement d'hyperparamètres | `uc_05_02_ajuster_boucle_hyperparametres_sft.py` → `AjusterBoucleHyperparametresSftUseCase` | `EntraineurSupervise`, `SuiviExperimentation` | (réutilise ceux ci-dessus) | [CONCEPTION] |
| Sauvegarder le checkpoint SFT-LoRA | `uc_05_03_sauvegarder_checkpoint_sft.py` → `SauvegarderCheckpointSftUseCase` | `RepositoryLectureEcriture` | `JsonlDatasetRepository` (réutilisé, 3ᵉ type d'entité) | [CONCEPTION] |

Diagrammes correspondants :
`docs/diagrams/03_etape2_sft/activite/pipeline_sft_lora.puml` (vue
d'ensemble du flux, y compris le point de décision "Convergence
saine ?"), `docs/diagrams/03_etape2_sft/sequence/entrainer_sft.puml`
(interaction script → cas d'usage → adaptateurs) et
`docs/diagrams/03_etape2_sft/paquets/sft_paquets.puml` (classes
proposées et leurs dépendances).

## 1. Charger le modèle quantifié + configurer LoRA : pas de cas d'usage dédié

**Décision de conception** : contrairement à ce que suggère le
découpage du roadmap (deux boîtes distinctes "Charger en 4-bit" et
"Configurer LoraConfig"), aucun cas d'usage séparé n'est proposé pour
ces deux étapes. Elles deviennent des responsabilités internes de
`TrlSftEntraineurAdapter` (constructeur) et de l'assemblage de
configuration fait par le point d'entrée `training/sft_train.py` à
partir de `recipes/sft_qwen3_lora.yaml`.

**Pourquoi** : le domaine ne peut pas représenter un modèle chargé
(poids `torch`, tokenizer, éventuel wrapper PEFT) comme une simple
`dataclass` : contrairement à `ExemplePivot`, ce n'est pas une donnée,
c'est un objet vivant d'un framework externe. Le précédent déjà posé
par `domain/ports/moteur_inference.py` (Étape 4) résout exactement ce
problème : le port `MoteurInference` n'expose que `generer(messages) ->
ReponseModele` : l'adaptateur (`LlamacppInferenceAdapter`,
`VllmEndpointInferenceAdapter`) possède et charge le modèle en interne,
l'application n'en voit jamais la représentation concrète. `PresidioAnonymiseur`
suit le même principe (`AnalyzerEngine`/`AnonymizerEngine` construits
dans son `__init__`, jamais exposés). `EntraineurSupervise` reprend ce
patron : `TrlSftEntraineurAdapter.__init__(identifiant_modele_base,
configuration_quantification)` charge `BitsAndBytesConfig` + le modèle
+ le tokenizer une fois, `entrainer(...)` reçoit une `ConfigurationLora`
en paramètre (donnée pure, sérialisable, testable sans GPU) et
applique `peft.LoraConfig` + `trl.SFTTrainer` en interne. Aucun objet
`torch`/`transformers` ne traverse la frontière `application/domain`.

## 2. `uc_05_00_formater_dataset_chatml.py` : `FormaterDatasetChatMLUseCase`

Lit les `ExemplePivot` d'un split donné (`RepositoryLectureEcriture`,
réutilisation directe du port déjà utilisé partout en Étape 1 : c'est
la preuve concrète de la promesse faite dans
`docs/01_environnement/01_architecture_hexagonale.md` : "un port
générique peut servir à n'importe quel type d'entité"), appelle
`FormateurConversation.formater(exemple)` pour chacun, et persiste le
résultat via une **seconde** instance du même port générique, paramétrée
cette fois sur `ExempleFormate` plutôt que sur `ExemplePivot`.

Port proposé (`domain/ports/formateur_conversation.py`) :

```python
@dataclass(frozen=True, slots=True)
class ExempleFormate:
    identifiant : str    # repris de ExemplePivot.identifiant, jamais régénéré
    texte        : str    # rendu ChatML complet (system+user+assistant)

class FormateurConversation(Protocol):
    def formater(self, exemple: ExemplePivot) -> ExempleFormate:
        """Rend un ExemplePivot en ChatML via apply_chat_template
        (add_generation_prompt=False : exemple d'entrainement complet,
        pas une invite a generer)."""
        ...
```

`ChatMLFormateurAdapter` (proposé) enveloppe
`AutoTokenizer.apply_chat_template` : c'est un des rares éléments de
l'Étape 2 qui ne nécessite **pas** de GPU (juste le tokenizer,
téléchargeable et exécutable en Environnement A) : voir
`03_guide_implementation_pas_a_pas.md` pour la conséquence sur le plan
de test (ce cas d'usage peut être testé en intégration réelle dès
l'Environnement A, avant tout accès GPU).

**Décision à prendre avant d'écrire ce cas d'usage** (écart identifié
en `00_introduction_concepts.md` §"Point de vigilance") : le format de
sortie cible du projet (`<think>...</think>` + JSON strict
`niveau`/`categorie`/`ressources_estimees`, cahier des charges F3-F4)
n'existe dans aucune `completion`/`chosen` du dataset pivot actuel.
Trois options, non tranchées ici :

1. **Ne rien changer aux données pour le SFT**, et repousser
   l'apprentissage du format `<think>`+JSON à un prompt système
   few-shot appliqué uniquement à l'inférence (Étape 4) ou à l'Étape 3
   (DPO, où `UltraMedical-Preference` pourrait être adapté au format
   cible via un prompt de reformulation). Risque : le SFT n'entraîne
   alors que le contenu médical, pas le format de sortie contractuel.
2. **Construire un jeu de complétions synthétiques** respectant le
   format cible (ex. un modèle plus grand reformule chaque
   `completion` existante en `<think>`+JSON), à valider avant
   inclusion : travail de données supplémentaire, hors périmètre de ce
   document.
3. **Limiter le format `<think>`+JSON strict aux futurs exemples
   spécifiquement écrits pour le triage** (non encore collectés), et
   traiter les 37 802 exemples SFT actuels comme un pré-entraînement
   de connaissance médicale générale, le format venant d'une passe
   ultérieure.

Cette décision conditionne directement l'implémentation de
`ChatMLFormateurAdapter` (le `system` prompt du gabarit ChatML change
selon l'option retenue) : à trancher avant d'écrire uc_05_00, pas
pendant.

## 3. `uc_05_01_entrainer_sft.py` : `EntrainerSftUseCase`

Orchestration d'un run d'entraînement unique : appelle
`SuiviExperimentation.demarrer_run(...)`, délègue l'entraînement à
`EntraineurSupervise.entrainer(dataset_train, dataset_validation,
config_lora, hyperparametres)`, relaie chaque métrique de la courbe
retournée vers `SuiviExperimentation.logger_metrique(...)`, puis
`terminer_run()`.

Port proposé (`domain/ports/entraineur_supervise.py`) :

```python
@dataclass(frozen=True, slots=True)
class MetriquesEntrainement:
    etape             : int
    perte_train        : float
    perte_validation    : float | None
    norme_gradient       : float

@dataclass(frozen=True, slots=True)
class ResultatEntrainementSFT:
    chemin_checkpoint : str
    courbe_metriques   : tuple[MetriquesEntrainement, ...]

class EntraineurSupervise(Protocol):
    def entrainer(
        self,
        dataset_train      : Iterable[ExempleFormate],
        dataset_validation  : Iterable[ExempleFormate],
        config_lora          : ConfigurationLora,
        hyperparametres       : HyperparametresEntrainement,
    ) -> ResultatEntrainementSFT: ...
```

`TrlSftEntraineurAdapter` (proposé,
`infrastructure/adapters/trl_sft_entraineur.py`) est le seul point du
projet qui importe `trl`, `peft`, `bitsandbytes`, potentiellement
`unsloth`/`liger_kernel` : cohérent avec la règle "seule
`infrastructure/adapters` importe des bibliothèques externes"
(`docs/01_environnement/01_architecture_hexagonale.md` §2).

## 4. Évaluation de convergence : fonction pure, pas un cas d'usage

`application/verdict_convergence.py::evaluer_convergence(courbe:
tuple[MetriquesEntrainement, ...]) -> VerdictConvergence` lit la
courbe déjà produite par `entrainer()` (pas d'appel de port
supplémentaire) et retourne un verdict :

```python
class VerdictConvergence(str, Enum):
    SAINE               = "saine"
    SURAPPRENTISSAGE      = "surapprentissage"   # perte train baisse, perte val remonte
    SOUS_APPRENTISSAGE     = "sous_apprentissage"  # les deux stagnent
    INSTABLE                = "instable"            # norme de gradient diverge/NaN
```

Ce module ne dépend d'aucun port, exactement comme
`application/echantillonnage.py` et
`application/detection_pii_residuelle.py` en Étape 1 : logique pure,
testable sans aucun adaptateur ni GPU, en Environnement A. Le seuil
exact séparant "sain" de "surapprentissage" (ex. delta de perte
validation sur les N derniers pas) est un paramètre à calibrer une
fois une vraie courbe observée, pas à figer ici.

## 5. `uc_05_02_ajuster_boucle_hyperparametres_sft.py` : `AjusterBoucleHyperparametresSftUseCase`

Boucle bornée : appelle `EntrainerSftUseCase`, passe la courbe
résultante à `evaluer_convergence`, et si le verdict n'est pas
`SAINE`, tire le prochain jeu d'hyperparamètres depuis
`application/grille_hyperparametres.py::candidat_suivant(grille,
historique)` (grille restreinte définie dans
`recipes/sft_qwen3_lora.yaml`, cf. `01_installation_configuration.md`
§6) et relance. S'arrête soit sur `SAINE`, soit sur épuisement de la
grille (retourne alors le meilleur run observé, jamais une erreur
silencieuse).

**Décision (pourquoi une grille restreinte et pas Optuna)** : le
roadmap écarte explicitement Optuna pour ce POC ("Optuna ecarte du
POC (Phase 3)"), cohérent avec le cahier des charges (§1, "Phase 3"
hors périmètre). Une recherche d'hyperparamètres bayésienne suppose un
budget de plusieurs dizaines de runs complets pour apporter un gain
réel : hors de portée du budget GPU d'un POC de 4 semaines (NF5). Une
grille de quelques combinaisons `(taux_apprentissage, rang)` choisies
à la main, déclenchée seulement si le premier run ne converge pas, est
proportionnée au budget disponible.

## 6. `uc_05_03_sauvegarder_checkpoint_sft.py` : `SauvegarderCheckpointSftUseCase`

Les poids de l'adaptateur LoRA lui-même sont écrits sur disque par
`peft`/`trl` directement (`chemin_checkpoint` retourné par
`entrainer()`) : ce cas d'usage ne les manipule pas. Son rôle est de
persister un **enregistrement de métadonnées** décrivant ce
checkpoint, via une troisième instance de
`RepositoryLectureEcriture` (après `ExemplePivot` et `ExempleFormate`
: même port générique, troisième type d'entité).

Domaine proposé (`domain/model/checkpoint_entraine.py`) :

```python
@dataclass(frozen=True, slots=True)
class CheckpointEntraine:
    identifiant           : str    # ex. hash(recette + horodatage)
    chemin                  : str
    modele_base              : str
    configuration_lora        : ConfigurationLora
    hyperparametres            : HyperparametresEntrainement
    metriques_finales           : MetriquesEntrainement
    verdict_convergence          : VerdictConvergence
    horodatage                    : str
```

Sérialisé en JSONL (même mécanisme que
`data/processed/dataset_pivot.jsonl`), un enregistrement ressemblerait
à :

```json
{
  "identifiant": "chsa-sft-lora-8f2c1a9b4e6d3f01",
  "chemin": "checkpoints/sft-lora/2026-XX-XX/",
  "modele_base": "Qwen/Qwen3-1.7B-Base",
  "configuration_lora": {"rang": 16, "alpha": 32, "dropout": 0.05, "modules_cibles": ["q_proj", "k_proj", "v_proj", "o_proj"]},
  "hyperparametres": {"taux_apprentissage": 0.0002, "nombre_epoques": 3, "taille_lot": 4, "packing": true, "type_perte": "chunked_nll"},
  "metriques_finales": {"etape": 1200, "perte_train": 0.83, "perte_validation": 0.91, "norme_gradient": 1.4},
  "verdict_convergence": "saine",
  "horodatage": "2026-XX-XXTXX:XX:XXZ"
}
```

C'est ce fichier (`data/processed/checkpoints_sft.jsonl`, proposé) que
l'Étape 3 (DPO, "Charger checkpoint SFT-LoRA comme politique de
référence") et le rapport technique final (Livrable 3) liraient pour
retrouver le meilleur checkpoint sans avoir à re-parcourir les logs
MLflow/TensorBoard bruts : même logique que
`data/processed/rapport_anonymisation_rgpd.json` en Étape 1
(indicateurs structurés générés automatiquement plutôt que recalculés
à la main).

## 7. Suivi d'expérimentation : port partagé par toutes les étapes ci-dessus

```python
class SuiviExperimentation(Protocol):
    def demarrer_run(self, nom: str, parametres: dict) -> None: ...
    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None: ...
    def terminer_run(self) -> None: ...
```

Deux adaptateurs proposés, un par backend mentionné au roadmap
(`MlflowSuiviExperimentation`, `TensorboardSuiviExperimentation`) :
`training/sft_train.py` en injecte un des deux selon
`recipes/sft_qwen3_lora.yaml::suivi.backend`, sans que
`EntrainerSftUseCase` ni `AjusterBoucleHyperparametresSftUseCase` n'aient
à connaître lequel. Un adaptateur composite (loggant vers les deux à
la fois) resterait possible sans changer le port, si le besoin se
présente à l'implémentation.

## Document suivant

Diagrammes : `docs/diagrams/03_etape2_sft/` (activité, séquence,
paquets, déploiement). Puis `03_guide_implementation_pas_a_pas.md`
pour l'ordre concret d'écriture des fichiers.
