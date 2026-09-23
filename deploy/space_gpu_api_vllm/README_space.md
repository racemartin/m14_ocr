---
title: CHSA Triage API GPU
emoji: 🚑
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

Space Docker/GPU, couteux, a n'allumer que pendant les tests. Seule
piece deployee sur HF Spaces (architecture a 2 pieces, decision
produit du 24/09/2026, cf. `interfaces/web/README_space.md` pour le
frontend Streamlit compagnon, execute en LOCAL par l'operateur humain,
jamais comme Space separe). Sert l'API FastAPI (`interfaces/api/`) ET
le serveur vLLM (LoRA DPO, jamais fusionne avec la base) dans le MEME
conteneur, via `Dockerfile` + `demarrer.sh` de ce dossier.

`GET /sante` (`interfaces/api/app.py`) reste HTTP 200 des que uvicorn
demarre, meme si vLLM met plusieurs minutes a charger le modele : c'est
ce que le frontend Streamlit local sonde en boucle pour detecter
automatiquement la disponibilite reelle du modele, sans synchronisation
manuelle des deux demarrages.

Secrets du Space a definir avant publication :
- `CHSA_CLE_API_DEMO` : cle attendue en en-tete `X-API-Key`.
- `HF_TOKEN` : necessaire des que `CHSA_JOURNAL_AUDIT=hf_dataset` (voir
  ci-dessous) pousse vers un depot dataset prive ; deja necessaire
  ailleurs dans ce projet pour le meme mecanisme (suivi d'experimentation,
  cf. AGENTS.md/README §2.6).

Variables optionnelles pour le journal d'audit F6 (traçabilite) : par
defaut (`JsonlJournalAudit`), les entrees sont ecrites dans un fichier
local du conteneur, **perdu a chaque redemarrage du Space** (filesystem
ephemere, aucun stockage persistant HF payant active). Pour persister
gratuitement via un dataset HF Hub (`HfDatasetJournalAudit`, meme
mecanisme `huggingface_hub.CommitScheduler` que le suivi d'experimentation) :
- `CHSA_JOURNAL_AUDIT=hf_dataset`
- `CHSA_JOURNAL_AUDIT_REPO=mombasstic/chsa-triage-audit-journal`

Depot dataset a creer une seule fois au prealable (necessite `HF_TOKEN`) :

```bash
hf repo create mombasstic/chsa-triage-audit-journal --repo-type dataset --private
```

Publication (jamais realisee dans cette tache, effet externe reel,
GPU payant, a autoriser explicitement, action reservee a l'operateur
humain) : uploader
`deploy/space_gpu_api_vllm/Dockerfile` (renomme `Dockerfile` a la
racine du Space), `deploy/space_gpu_api_vllm/demarrer.sh`, ce fichier
(renomme `README.md` a la racine du Space), ainsi que `src/`,
`interfaces/`, `pyproject.toml`, `uv.lock` (memes chemins que le
`Dockerfile` racine du depot, cf. sa propre note de conception).
