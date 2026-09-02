# Architecture hexagonale — Agent IA de Triage Médical (CHSA)

## 1. Pourquoi hexagonal ici

Le projet a deux points de variation forts qui bougeront pendant les 4
semaines :
1. **La source d'inférence** : moteur local (`llama.cpp`/GGUF, gratuit,
   utilisé pendant tout le développement) vs. endpoint distant vLLM
   (payant, utilisé pour la démonstration/soutenance et la mesure de
   latence réelle).
2. **La source de stockage des données** : fichiers JSONL locaux
   pendant l'Étape 1, `datasets` Hugging Face une fois versionnés.

L'architecture hexagonale (ports & adaptateurs) isole le **domaine**
et l'**application** de ces choix d'infrastructure, afin qu'un
changement d'infrastructure (passer de JSONL local à HF Datasets, ou
de `llama.cpp` à vLLM) ne touche **aucune** ligne de code métier —
seul un nouvel adaptateur est écrit.

## 2. Les couches

```
┌─────────────────────────────────────────────────────────────────┐
│  interfaces/  (adaptateurs PRIMAIRES — pilotent l'application)   │
│  cli/   api/ (FastAPI)   web/ (Streamlit)                        │
└───────────────────────────┬───────────────────────────────────────┘
                             │ appelle
┌───────────────────────────▼───────────────────────────────────────┐
│  application/  (cas d'usage — orchestrent les ports)               │
│  use_cases/ : ImporterCorpusUseCase, ProfilerCorpusUseCase, ...    │
└───────────────────────────┬───────────────────────────────────────┘
                             │ dépend de (interfaces uniquement)
┌───────────────────────────▼───────────────────────────────────────┐
│  domain/  (coeur métier — ZÉRO dépendance externe)                 │
│  model/ : ExemplePivot, CorpusSource, NiveauConfiance, ...         │
│  ports/  : DatasetRepository, CorpusReader, Anonymiseur, ...       │
└───────────────────────────▲───────────────────────────────────────┘
                             │ implémente
┌───────────────────────────┴───────────────────────────────────────┐
│  infrastructure/  (adaptateurs SECONDAIRES — implémentent les ports)│
│  adapters/ : JsonlDatasetRepository, PresidioAnonymiseur, ...       │
└─────────────────────────────────────────────────────────────────┘
```

- **`domain/model`** : les entités métier. Ici, oui, le vocabulaire
  **est** médical (`ExemplePivot` a des champs `symptomes`,
  `antecedents`, `constantes_vitales`...) — c'est le rôle du domaine
  de modéliser le métier. Zéro import de `pandas`, `presidio`,
  `datasets` ou quoi que ce soit d'externe : uniquement des
  `dataclasses`/`Enum` Python purs.
- **`domain/ports`** : les interfaces (contrats) que l'infrastructure
  doit respecter. **Ici, les noms de méthode restent génériques**
  (`save`, `find_by_id`, `list`, `read_raw_records`) — jamais
  `get_symptomes_patient()` ou `charger_corpus_mediqa()`. Un port
  générique peut ainsi servir à n'importe quel type d'entité
  (`Repository[ExemplePivot]` aujourd'hui, `Repository[AutreChose]`
  demain) sans être réécrit. C'est ce découplage qui permet de changer
  de technologie de stockage sans toucher à l'application.
- **`application/use_cases`** : la logique d'orchestration
  (« importer un corpus, le profiler, le convertir en schéma pivot,
  l'anonymiser, le découper en splits »). Cette couche **connaît** le
  domaine (elle manipule des `ExemplePivot`), mais ne connaît
  **aucune** techno d'infrastructure — elle appelle des ports, jamais
  directement `ydata_profiling` ou `presidio_analyzer`.
- **`infrastructure/adapters`** : implémentations concrètes des ports.
  C'est la **seule** couche qui importe des bibliothèques externes
  (pandas, Presidio, `datasets`, `llama-cpp-python`, client HTTP
  vLLM...).
- **`interfaces/`** : adaptateurs primaires — ce qui *déclenche*
  l'application (CLI pour les scripts d'Étape 1, API FastAPI et UI
  Streamlit pour le chat de l'Étape 4).

## 3. Règle de dépendance

```
interfaces  →  application  →  domain  ←  infrastructure
```

Les flèches vont **toutes vers `domain`**, jamais depuis `domain` vers
l'extérieur. `domain` ne dépend de rien ; tout le reste dépend de
`domain`.

## 4. Exemple concret — port générique vs. adaptateur spécifique

Port (générique, dans `domain/ports/dataset_repository.py`) :

```python
class RepositoryLectureEcriture(Protocol[T]):
    def sauvegarder(self, item: T) -> None: ...
    def sauvegarder_plusieurs(self, items: Iterable[T]) -> None: ...
    def trouver_par_id(self, identifiant: str) -> T | None: ...
    def lister(self, filtre: dict | None = None) -> Iterator[T]: ...
```

Adaptateur local (`infrastructure/adapters/jsonl_dataset_repository.py`) :
implémente ce port en lisant/écrivant des fichiers `.jsonl`.

Adaptateur distant (à écrire plus tard) : implémenterait le **même**
port en s'appuyant sur `datasets.load_dataset` / `push_to_hub`.

`application/use_cases/construire_dataset_pivot.py` ne sait *jamais*
lequel des deux est branché — il appelle `repository.sauvegarder(...)`.
L'injection de l'adaptateur concret se fait dans `interfaces/cli/`
(le point d'entrée), pas dans le domaine ni l'application.

## 5. Application au module d'inférence (Étape 4, préparé dès maintenant)

Même logique pour le futur chat Streamlit/FastAPI :

- Port `domain/ports/moteur_inference.py` : `generer(prompt: str) -> ReponseModele`
  — générique, ne sait pas si le moteur est local ou distant.
- Adaptateur `infrastructure/adapters/llamacpp_inference_adapter.py`
  (local, GGUF, gratuit).
- Adaptateur `infrastructure/adapters/vllm_endpoint_inference_adapter.py`
  (distant, HTTP vers l'endpoint vLLM payant).
- Le backend FastAPI (`interfaces/api/`) choisit l'adaptateur à
  l'injection (variable d'environnement `MOTEUR_INFERENCE=local|distant`),
  sans que l'application ni le domaine n'aient besoin de le savoir.

## 6. Ce qui est déjà implémenté vs. prévu

| Élément | Statut |
|---|---|
| `domain/model` (entités Étape 1) | Implémenté |
| `domain/ports` (repository, corpus reader, anonymiseur, profileur) | Implémenté |
| `application/use_cases` (Étape 1) | Implémenté |
| `infrastructure/adapters` JSONL, Presidio, ydata-profiling | Implémenté |
| `interfaces/cli` (Étape 1) | Implémenté |
| `domain/ports/moteur_inference.py` | Implémenté (port), adaptateurs non branchés |
| `*_inference_adapter.py` | Squelette (Étape 4) |
| `interfaces/api` (FastAPI) | Squelette (Étape 4) |
| `interfaces/web` (Streamlit) | Squelette (Étape 4) |
| `docker/` (frontend/backend) | Squelette (Étape 4) |

*(document suivant : le code lui-même, voir `../../src/chsa_triage/` ;
et le diagramme de roadmap complet dans `../diagrams/roadmap_activite.puml`)*
