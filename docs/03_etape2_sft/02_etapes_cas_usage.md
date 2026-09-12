\newpage

# Étape 2 : Étapes du pipeline et cas d'usage

> Les cas d'usage, ports et fonctions pures décrits ci-dessous sont
> écrits et testés (221 tests en vert), à l'exception de
> `TrlSftEntraineurAdapter` (§1) qui reste à écrire : c'est le seul
> élément de ce document qui nécessite un GPU réel. Les fichiers de
> cas d'usage suivent la convention `E2_NN_uc_verbe_objet.py` (`NN`
> = ordre du pipeline, marqueur `_uc_` réservé aux classes de cas
> d'usage, jamais présent dans les noms de classe eux-mêmes ; cf.
> `AGENTS.md`). Statut, pour chaque ligne : **[FAIT]** = écrit et
> couvert par des tests réels (avec de faux adaptateurs en mémoire
> pour les cas d'usage qui dépendent d'un port GPU) · **[CONCEPTION]**
> = proposé ici, pas encore écrit, réservé à `TrlSftEntraineurAdapter`
> et à `training/E2_04_sft_train.py` (cf. légende complète de
> `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`).

## Vue d'ensemble

| Étape du roadmap | Cas d'usage / module | Port(s) | Adaptateur(s) | Statut |
|---|---|---|---|---|
| Charger `Qwen3-1.7B-Base` en 4-bit NF4 | *(pas de cas d'usage dédié : voir §1 Décision)* | `EntraineurSupervise` | `TrlSftEntraineurAdapter` (charge le modèle dans son constructeur) | [CONCEPTION] |
| Formater en ChatML + `assistant_only_loss` | `E2_00_uc_formater_dataset_chatml.py` → `FormaterDatasetChatMLUseCase` | `FormateurConversation`, `RepositoryLectureEcriture` (×2) | `ChatMLFormateurAdapter`, `JsonlDatasetRepository` (réutilisé) | [FAIT] |
| Configurer `LoraConfig` | *(pas de cas d'usage dédié, assemblage de configuration : voir §1 Décision)* | (aucun) | `recipes/sft_qwen3_lora.yaml` → `ConfigurationLora` | [FAIT] la dataclass (`domain/model/configuration_entrainement.py`) · [CONCEPTION] l'assemblage depuis la recette YAML (`recipes/` n'existe pas encore) |
| Lancer `SFTTrainer` | `E2_01_uc_entrainer_sft.py` → `EntrainerSftUseCase` | `EntraineurSupervise`, `SuiviExperimentation` | `TrlSftEntraineurAdapter`, `MlflowSuiviExperimentation` (ou `TensorboardSuiviExperimentation`) | [FAIT] le cas d'usage (testé avec un faux `EntraineurSupervise`) · [CONCEPTION] `TrlSftEntraineurAdapter`, seul adaptateur GPU manquant |
| Évaluer sur validation intermédiaire | fonction pure `application/verdict_convergence.py::evaluer_convergence` | (aucun port : lit la courbe déjà produite par `entrainer()`) | (aucun) | [FAIT] |
| Boucle d'ajustement d'hyperparamètres | `E2_02_uc_ajuster_boucle_hyperparametres_sft.py` → `AjusterBoucleHyperparametresSftUseCase` | `EntraineurSupervise`, `SuiviExperimentation` | (réutilise ceux ci-dessus) | [FAIT] le cas d'usage · [CONCEPTION] dépend du même adaptateur GPU manquant que la ligne précédente |
| Sauvegarder le checkpoint SFT-LoRA | `E2_03_uc_sauvegarder_checkpoint_sft.py` → `SauvegarderCheckpointSftUseCase` | `RepositoryLectureEcriture` | `JsonlDatasetRepository` (réutilisé, 3ᵉ type d'entité) | [FAIT] |

Diagrammes correspondants :
`docs/diagrams/03_etape2_sft/activite/pipeline_sft_lora.puml` (vue
d'ensemble du flux, y compris le point de décision "Convergence
saine ?"), `docs/diagrams/03_etape2_sft/sequence/entrainer_sft.puml`
(interaction script → cas d'usage → adaptateurs),
`docs/diagrams/03_etape2_sft/paquets/sft_paquets.puml` (modules et
leurs dépendances) et `docs/diagrams/03_etape2_sft/paquets/classes_etape2.puml`
(attributs/méthodes des classes réelles).

## 1. Charger le modèle quantifié + configurer LoRA : pas de cas d'usage dédié

**Décision de conception** : contrairement à ce que suggère le
découpage du roadmap (deux boîtes distinctes "Charger en 4-bit" et
"Configurer LoraConfig"), aucun cas d'usage séparé n'existe pour ces
deux étapes. Elles restent des responsabilités internes de
`TrlSftEntraineurAdapter` (constructeur, **pas encore écrit**, seul
élément de ce document qui nécessite un GPU réel) et de l'assemblage de
configuration que fera le point d'entrée `training/E2_04_sft_train.py`
(également pas encore écrit) à partir de `recipes/sft_qwen3_lora.yaml`
(pas encore créé).

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
dans son `__init__`, jamais exposés). `EntraineurSupervise` (le port,
déjà écrit dans `domain/ports/entraineur_supervise.py`) reprend ce
patron : `TrlSftEntraineurAdapter.__init__(identifiant_modele_base,
configuration_quantification)` chargera `BitsAndBytesConfig` + le
modèle + le tokenizer une fois, `entrainer(...)` reçoit une
`ConfigurationLora` en paramètre (donnée pure, déjà écrite,
sérialisable, testable sans GPU) et appliquera `peft.LoraConfig` +
`trl.SFTTrainer` en interne. Aucun objet `torch`/`transformers` ne
traversera la frontière `application/domain`.

## 2. `E2_00_uc_formater_dataset_chatml.py` : `FormaterDatasetChatMLUseCase`

`executer(self, split: TypeSplit) -> int` lit les `ExemplePivot` du
`split` demandé, filtrés aussi sur `type_exemple == TypeExemple.SFT`
(`repository_pivot.lister(filtre={"split": split, "type_exemple": TypeExemple.SFT})`,
correction du 12/09/2026 : ce cas d'usage prépare des données
d'entraînement SFT, un `ExemplePivot` DPO n'a pas de `completion`, cf.
AGENTS.md et README §9 pour le même trou trouvé dans
`ExtraireSousEnsembleSftUseCase`. Réutilisation directe du port déjà
utilisé partout en Étape 1 : c'est la preuve concrète de la promesse
faite dans `docs/01_environnement/01_architecture_hexagonale.md` :
"un port générique peut servir à n'importe quel type d'entité"), appelle
`FormateurConversation.formater(exemple)` pour chacun, et persiste le
résultat en un seul appel `sauvegarder_plusieurs(...)` (jamais un
`sauvegarder()` par item, cf. `AGENTS.md` sur le coût O(n²)) via une
**seconde** instance du même port générique, paramétrée cette fois sur
`ExempleFormate` plutôt que sur `ExemplePivot`. Retourne le nombre
d'exemples formatés.

Port (`domain/ports/formateur_conversation.py`, écrit) :

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

`ChatMLFormateurAdapter` (`infrastructure/adapters/chatml_formateur_adapter.py`,
écrit) enveloppe `AutoTokenizer.apply_chat_template` : c'est un des
rares éléments de l'Étape 2 qui ne nécessite **pas** de GPU (juste le
tokenizer, téléchargeable et exécutable en Environnement A) : voir
`03_guide_implementation_pas_a_pas.md` pour la conséquence sur le plan
de test (ce cas d'usage est testé en intégration réelle dès
l'Environnement A, `tests/infrastructure/test_chatml_formateur_adapter.py`,
avant tout accès GPU). Le tokenizer est chargé paresseusement (au
premier `formater()`, pas à la construction de l'adaptateur) ; le
modèle par défaut est `Qwen/Qwen3-1.7B-Base` (`nom_modele`,
configurable) ; `formater()` concatène les messages `prompt` puis
`completion` de l'`ExemplePivot` avant de les rendre via le chat
template natif du tokenizer.

**Décision encore ouverte** (écart identifié
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
selon l'option retenue) : `ChatMLFormateurAdapter` est déjà écrit,
mais ne tranche encore aucune de ces trois options (il rend tel quel
le contenu `prompt`/`completion` existant, sans injecter de `system`
prompt ni de format `<think>`+JSON) : reste à trancher avant que le
formatage ChatML produise des exemples réellement alignés sur le
format de sortie cible.

## 3. `E2_01_uc_entrainer_sft.py` : `EntrainerSftUseCase`

Orchestration d'un run d'entraînement unique, via la méthode
`entrainer(dataset_train, dataset_validation, config_lora,
hyperparametres, nom_run=NOM_RUN_PAR_DEFAUT)` (nommée `entrainer`, pas
`executer` comme les autres cas d'usage de cette étape, pour rester
proche du vocabulaire du port qu'elle orchestre) : appelle
`SuiviExperimentation.demarrer_run(nom_run, parametres_run)` (les
paramètres du run sont `{**asdict(config_lora),
**asdict(hyperparametres)}`), délègue l'entraînement à
`EntraineurSupervise.entrainer(dataset_train, dataset_validation,
config_lora, hyperparametres)`, relaie chaque métrique de la courbe
retournée vers `SuiviExperimentation.logger_metrique(...)` (une
entrée `perte_train` et `norme_gradient` par point, plus
`perte_validation` quand elle n'est pas `None`), puis `terminer_run()`.
Retourne le `ResultatEntrainementSFT` inchangé.

Port (`domain/ports/entraineur_supervise.py`, écrit) :

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

`TrlSftEntraineurAdapter` (**pas encore écrit**,
`infrastructure/adapters/trl_sft_entraineur.py`) sera le seul point du
projet qui importe `trl`, `peft`, `bitsandbytes`, potentiellement
`unsloth`/`liger_kernel` : cohérent avec la règle "seule
`infrastructure/adapters` importe des bibliothèques externes"
(`docs/01_environnement/01_architecture_hexagonale.md` §2). En
attendant, `EntrainerSftUseCase` est testé
(`tests/application/test_E2_01_uc_entrainer_sft.py`) avec un faux
`EntraineurSupervise` en mémoire : toute la logique d'orchestration
(démarrage/clôture du run, relais des métriques) est donc déjà
validée indépendamment de ce futur adaptateur GPU.

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
écrite et testée sans aucun adaptateur ni GPU, en Environnement A
(`tests/application/test_verdict_convergence.py`). Ordre de priorité
réel du diagnostic : `INSTABLE` (perte/norme de gradient NaN ou
infinie n'importe où dans la courbe, ou ratio de norme de gradient
dernier/premier pas au-delà de `SEUIL_RATIO_DIVERGENCE_GRADIENT`
combiné à une perte d'entraînement qui remonte) prime sur
`SOUS_APPRENTISSAGE` (baisse relative de la perte d'entraînement
entre premier et dernier pas sous
`SEUIL_BAISSE_TRAIN_RELATIVE_MINIMALE`), qui prime sur
`SURAPPRENTISSAGE` (hausse de la perte de validation au-delà de
`SEUIL_HAUSSE_VALIDATION_SURAPPRENTISSAGE` sur les
`FENETRE_PAS_VALIDATION` derniers points mesurés) ; sinon `SAINE`. Les
quatre constantes de seuil sont documentées dans le module comme
**provisoires** : aucune vraie courbe d'entraînement n'a encore été
observée (`TrlSftEntraineurAdapter` n'existe pas encore), elles
devront être recalibrées après un premier run réel.

## 5. `E2_02_uc_ajuster_boucle_hyperparametres_sft.py` : `AjusterBoucleHyperparametresSftUseCase`

Boucle bornée, `executer(dataset_train, dataset_validation,
config_lora, hyperparametres_initiaux, nom_run=NOM_RUN_PAR_DEFAUT)` :
matérialise `dataset_train`/`dataset_validation` en listes (un
`Iterable` épuisable ne survivrait pas à un second passage), puis pour
chaque essai (le premier avec `hyperparametres_initiaux`) appelle
`EntrainerSftUseCase.entrainer(...)`, passe la courbe résultante à
`evaluer_convergence`, et si le verdict n'est pas `SAINE`, tire le
prochain jeu d'hyperparamètres depuis
`application/grille_hyperparametres.py::candidat_suivant(grille,
historique)` (grille restreinte définie dans
`recipes/sft_qwen3_lora.yaml`, cf. `01_installation_configuration.md`
§6) et relance. S'arrête soit sur `SAINE`, soit sur épuisement de la
grille (`candidat_suivant` retourne `None`), et retourne alors, dans
tous les cas, un `ResultatBoucleAjustement(meilleur_essai, essais)` :
`essais` trace tous les essais effectués (`EssaiHyperparametres`,
un triplet hyperparamètres/résultat/verdict), `meilleur_essai` est
choisi par `_cle_classement` : un essai `SAINE` est **toujours**
préféré à un essai non `SAINE`, quelle que soit sa perte finale ; à
égalité de statut, la perte du dernier point de la courbe la plus
basse gagne (perte de validation si disponible à ce point, sinon
perte d'entraînement). Jamais d'erreur silencieuse même si la grille
s'épuise sans qu'aucun essai ne converge.

`application/grille_hyperparametres.py::candidat_suivant(grille:
Sequence[HyperparametresEntrainement], historique:
Sequence[HyperparametresEntrainement]) -> HyperparametresEntrainement | None`
retourne le premier élément de `grille` absent de `historique`
(comparaison par égalité de valeur, `HyperparametresEntrainement`
étant `frozen=True`), ou `None` si la grille est épuisée. **Précision
de conception** (le docstring du module la documente explicitement,
non fixée par ce document au moment où il a été écrit) : `grille` est
une séquence déjà entièrement étalée (le produit cartésien des axes
de `recipes/sft_qwen3_lora.yaml::grille_hyperparametres` est calculé
en amont par l'appelant, pas par `candidat_suivant`). Point resté
ouvert : l'axe `rang` de la grille pilote `ConfigurationLora`, pas
`HyperparametresEntrainement` ; `AjusterBoucleHyperparametresSftUseCase`
reçoit un `config_lora` fixe et ne le fait pas varier d'un essai à
l'autre dans cette phase (faire varier `rang` en boucle suppose de
faire varier `config_lora` en parallèle de `grille`, ce qui reste à
trancher).

**Décision (pourquoi une grille restreinte et pas Optuna)** : le
roadmap écarte explicitement Optuna pour ce POC ("Optuna ecarte du
POC (Phase 3)"), cohérent avec le cahier des charges (§1, "Phase 3"
hors périmètre). Une recherche d'hyperparamètres bayésienne suppose un
budget de plusieurs dizaines de runs complets pour apporter un gain
réel : hors de portée du budget GPU d'un POC de 4 semaines (NF5). Une
grille de quelques combinaisons `(taux_apprentissage, rang)` choisies
à la main, déclenchée seulement si le premier run ne converge pas, est
proportionnée au budget disponible.

## 6. `E2_03_uc_sauvegarder_checkpoint_sft.py` : `SauvegarderCheckpointSftUseCase`

Les poids de l'adaptateur LoRA lui-même seront écrits sur disque par
`peft`/`trl` directement (`chemin_checkpoint` retourné par
`entrainer()`) : ce cas d'usage ne les manipule pas. Son rôle est de
persister un **enregistrement de métadonnées** décrivant ce
checkpoint, via `executer(identifiant, chemin, modele_base,
configuration_lora, hyperparametres, metriques_finales,
verdict_convergence)`, qui construit un `CheckpointEntraine`
(l'horodatage vient d'une `horloge: Callable[[], str]` injectable,
`datetime.now(timezone.utc).isoformat()` par défaut, pour rester
testable sans dépendre de l'heure réelle) et le persiste via une
troisième instance de `RepositoryLectureEcriture` (après `ExemplePivot`
et `ExempleFormate` : même port générique, troisième type d'entité).
Retourne le `CheckpointEntraine` construit, pour usage immédiat par
l'appelant.

Domaine (`domain/model/checkpoint_entraine.py`, écrit) :

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

Deux adaptateurs écrits, un par backend mentionné au roadmap.
`MlflowSuiviExperimentation(uri_tracking)` appelle `mlflow.set_tracking_uri`
dans `__post_init__`, puis `mlflow.start_run`/`log_params`,
`mlflow.log_metric(..., step=etape)`, `mlflow.end_run()` (MLflow ≥ 3
refuse le backend fichier brut par défaut : utiliser un backend
`sqlite:///chemin/mlflow.db` ou exporter `MLFLOW_ALLOW_FILE_STORE=true`,
cf. le docstring du module). `TensorboardSuiviExperimentation(repertoire_logs)`
ouvre un `SummaryWriter` distinct par run sous
`repertoire_logs/<nom>` (`add_scalar`/`add_text`, `close()` à la
clôture). Tous deux sont testables en intégration réelle dès
l'Environnement A (`mlflow`/`tensorboard` sont dans l'extra `local`,
pas seulement `remote`) : `tests/infrastructure/test_mlflow_suivi_experimentation.py`,
`tests/infrastructure/test_tensorboard_suivi_experimentation.py`.
`training/E2_04_sft_train.py` (pas encore écrit) injectera l'un des deux
selon `recipes/sft_qwen3_lora.yaml::suivi.backend`, sans que
`EntrainerSftUseCase` ni `AjusterBoucleHyperparametresSftUseCase` n'aient
à connaître lequel. Un adaptateur composite (loggant vers les deux à
la fois) resterait possible sans changer le port, si le besoin se
présente à l'implémentation de `training/E2_04_sft_train.py`.

## Document suivant

Diagrammes : `docs/diagrams/03_etape2_sft/` (activité, séquence,
paquets, déploiement). Puis `03_guide_implementation_pas_a_pas.md`
pour l'ordre concret d'écriture des fichiers.
