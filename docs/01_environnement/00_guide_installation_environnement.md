# Guide d'installation de l'environnement de développement

> Ce guide remplace la version précédente. Deux changements majeurs :
> 1. Le projet suit une **architecture hexagonale** (ports & adaptateurs) —
>    voir `01_architecture_hexagonale.md` (meme dossier) pour le détail des couches.
> 2. **Environnement A (local)** utilise **`uv`** (et non plus
>    `venv`/`pip` bruts). **Environnement B (distant)** utilise
>    **Hugging Face en compte payant** — `HF Jobs` pour l'entraînement
>    batch, `HF Spaces Dev Mode` pour le développement interactif en
>    SSH/VSCode.

---

## 0. Vue d'ensemble des deux environnements

| | Environnement A — Local | Environnement B — Distant (HF payant) |
|---|---|---|
| Machine | WSL2, 5 Go RAM, pas de GPU | Infrastructure Hugging Face (GPU à la demande) |
| Gestionnaire de paquets | `uv` | `uv` (le même, exécuté à distance) |
| Rôle | Domaine, application, adaptateurs testables sans GPU (JSONL, profiling, anonymisation), inférence locale GGUF | Entraînement SFT/DPO, développement interactif nécessitant GPU |
| Outils HF | `huggingface-cli` (auth, download) | `hf jobs` (batch), `hf` Dev Mode (interactif, SSH/VSCode) |
| Coût | 0 € | Pay-as-you-go — voir §3.4 pour limiter la facture |

---

## 1. Environnement A — Local (WSL2 + `uv`)

### 1.1 Installer `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # ou redémarrer le shell
uv --version
```

`uv` remplace `pip` + `venv` + `pip-tools` : il résout et installe les
dépendances beaucoup plus vite, et gère l'environnement virtuel de
façon transparente via `pyproject.toml`.

### 1.2 Initialiser le projet

```bash
cd ~/chsa-triage   # racine du dépôt (voir arborescence §4)
uv sync --extra local
```

`uv sync` lit `pyproject.toml` (fourni dans le zip, voir §5) et crée
`.venv/` avec exactement les dépendances du groupe `local` : pandas,
`datasets`, `ydata-profiling`, `presidio-analyzer/anonymizer`, `spacy`,
`pydantic`, `transformers` (CPU), `huggingface_hub`, `pytest`, `ruff`,
`mlflow`. **Aucune dépendance GPU** (pas de `torch` CUDA, pas
`unsloth`) — cohérent avec la contrainte des 5 Go RAM.

```bash
# Activer l'environnement pour une session shell interactive
source .venv/bin/activate

# Ou, sans activer, exécuter directement une commande dans l'environnement :
uv run pytest tests/
uv run python scripts/check_env_local.py
```

### 1.3 Modèles spaCy (Presidio)

```bash
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_sm
```

### 1.4 Authentification Hugging Face

```bash
uv run huggingface-cli login
# Colle ton token (rôle "write" nécessaire pour push dataset/modèle et lancer des Jobs)
```

Variables sensibles dans `.env` (jamais commité, voir `.env.example`
fourni) :

```bash
HF_TOKEN=hf_xxx
MLFLOW_TRACKING_URI=<uri_partagee_avec_environnement_B>
```

### 1.5 Vérification

```bash
uv run python scripts/check_env_local.py
```

---

## 2. Pourquoi un compte Hugging Face payant ?

Le choix retenu (cf. échange précédent) : **HF plutôt que Kaggle/Colab
pour les runs qui comptent**, car l'intégration avec le Hub où sont
déjà versionnés le dataset et le modèle est native, et parce que tout
le pipeline est écrit en `.py` — HF Jobs exécute des scripts sans
aucun notebook. Le compte payant (crédits pay-as-you-go) débloque :
- **HF Jobs** : exécution facturée à la seconde, GPU à la demande,
  aucune machine à gérer.
- **Spaces Dev Mode** : SSH + VSCode Remote sur un Space avec GPU,
  pour le développement interactif (débogage, itération rapide) —
  fonctionnalité réservée aux comptes PRO/Team/Enterprise.

**Règle de discipline pour limiter la facture (cf. §3.4) : Dev Mode
sert à développer et déboguer, jamais à laisser tourner un
entraînement complet en arrière-plan sans surveillance.**

---

## 3. Environnement B — Distant (HF payant)

### 3.1 Installer le CLI `hf` en local

```bash
uv tool install huggingface_hub[cli]
hf auth login
```

### 3.2 HF Jobs — entraînement batch (SFT, DPO)

Aucune installation côté "serveur" : le script `.py` (avec ses
dépendances déclarées en en-tête façon `uv script`, ou via
`--with`) est envoyé tel quel.

```bash
# Test rapide de connectivité GPU
hf jobs uv run --flavor t4-small \
  python -c "import torch; print(torch.cuda.get_device_name())"

# Lancement réel du SFT (script fourni dans training/sft_train.py)
hf jobs uv run \
  --flavor a10g-small \
  --timeout 6h \
  --with trl --with peft --with bitsandbytes --with unsloth \
  --secrets HF_TOKEN \
  training/sft_train.py --config recipes/sft_qwen3_lora.yaml

# Suivi
hf jobs ps
hf jobs stats <job_id>
hf jobs logs <job_id>
```

### 3.3 Spaces Dev Mode — développement interactif SSH/VSCode

**Étape 1 — Créer le Space :**

```bash
hf repo create chsa-triage-dev --type space --space_sdk docker
```

**Étape 2 — Activer Dev Mode et choisir le matériel :**

Dans Settings du Space (interface web) :
1. Sélectionner un hardware GPU (ex. `T4 small` pour du débogage léger,
   `A10G` si tu dois reproduire un vrai pas d'entraînement).
2. Cliquer sur **« Enable Dev Mode »**. Le Space redémarre en mode
   développement : un serveur SSH et un serveur VSCode démarrent en
   tâche de fond à l'intérieur du conteneur.

**Étape 3 — Récupérer les instructions de connexion :**

La modale "Dev Mode" du Space affiche une commande SSH prête à copier,
du type :

```bash
ssh -p <port> <namespace>-chsa-triage-dev@ssh.hf.space
```

Ajouter au `~/.ssh/config` local pour simplifier :

```
Host chsa-dev
  HostName ssh.hf.space
  User <namespace>-chsa-triage-dev
  Port <port>
```

Puis simplement :

```bash
ssh chsa-dev
```

**Étape 4 — VSCode Remote :**

1. Installer l'extension **Remote - SSH** dans VSCode.
2. `Cmd/Ctrl+Shift+P` → *Remote-SSH: Connect to Host* → choisir `chsa-dev`.
3. Ouvrir le dossier du Space (`/app` ou équivalent selon l'image).
4. Le terminal intégré VSCode est alors un terminal **sur la machine
   GPU distante** — installer `uv` comme en local (§1.1) et faire
   `uv sync --extra remote` pour installer torch/CUDA, `trl`, `peft`,
   `unsloth`, `vllm`, etc.

**Étape 5 — Persister le travail :**

Le conteneur Dev Mode n'est pas permanent : committer régulièrement
(`git add . && git commit && git push`) vers le dépôt du Space, ou
pousser directement les checkpoints vers un autre repo HF (`datasets`
ou `models`) avec `huggingface_hub.upload_folder`.

> Les requirements sont **volontairement absents de l'image Docker
> de base** — comme documenté par Hugging Face, il faut les installer
> manuellement (`uv sync`) à chaque nouvelle session Dev Mode, sauf
> si tu construis une image Docker custom pour le Space qui les
> embarque déjà (recommandé une fois la liste stabilisée, pour éviter
> de payer du temps GPU en réinstallation).

### 3.4 Discipline de facturation

- **Jamais de Dev Mode GPU ouvert sans surveillance** — désactiver
  (Settings → Disable Dev Mode) ou changer le hardware vers CPU dès
  la session de débogage terminée.
- Utiliser **HF Jobs** (facturé à la seconde, pas d'oubli possible
  puisqu'il se termine seul) pour tout ce qui est *batch* et
  reproductible : entraînement complet, évaluation, benchmark de
  latence.
- Réserver **Dev Mode** aux moments où il faut vraiment interagir en
  direct (déboguer un crash CUDA, vérifier un tokenizer pas à pas).
- Suivre la consommation via `hf jobs stats` et le tableau de
  facturation du compte HF chaque semaine.

---

## 4. Arborescence du dépôt (architecture hexagonale)

```
chsa-triage/
├── docs/                          # toute la documentation du projet
├── scripts/                       # scripts opérationnels (vérif. env, CLI ad hoc)
├── src/chsa_triage/
│   ├── domain/                    # coeur métier — aucune dépendance externe
│   │   ├── model/                 # entités (ExemplePivot, CorpusSource, ...)
│   │   └── ports/                 # interfaces génériques (Protocol)
│   ├── application/               # cas d'usage, orchestrent les ports
│   │   └── use_cases/
│   └── infrastructure/            # adaptateurs concrets (implémentent les ports)
│       └── adapters/
├── interfaces/                    # adaptateurs "primaires" (pilotent l'appli)
│   ├── cli/                       # entrées en ligne de commande (Étape 1)
│   ├── api/                       # backend FastAPI (Étape 4, chat/inférence)
│   └── web/                       # frontend Streamlit (Étape 4, chat/inférence)
├── training/                      # scripts exécutés via HF Jobs (SFT, DPO)
├── recipes/                       # fichiers YAML de configuration d'entraînement
├── docker/                        # Dockerfiles + docker-compose (frontend/backend)
├── data/{raw,processed,splits}/
├── tests/{domain,application,infrastructure}/
├── pyproject.toml                 # dépendances gérées par uv (groupes local/remote/dev)
├── .env.example
└── README.md
```

Voir `01_architecture_hexagonale.md` (meme dossier) pour la justification détaillée
de chaque couche, et pourquoi les **ports** utilisent des noms de
méthode génériques (`save`, `find_by_id`, `list`) plutôt que des noms
liés au domaine médical (`get_symptomes_patient`).

---

## 5. Checklist de démarrage

- [ ] `uv` installé en local, `uv sync --extra local` exécuté sans erreur.
- [ ] `check_env_local.py` passe.
- [ ] Compte HF payant actif, `hf auth login` effectué.
- [ ] Test `hf jobs uv run --flavor t4-small ...` réussi (connectivité GPU).
- [ ] Space Dev Mode créé, connexion SSH testée, VSCode Remote-SSH connecté.
- [ ] `uv sync --extra remote` exécuté dans la session Dev Mode.
- [ ] `.env` local rempli à partir de `.env.example` (jamais commité).

*(document suivant : `01_architecture_hexagonale.md`, meme dossier)*
