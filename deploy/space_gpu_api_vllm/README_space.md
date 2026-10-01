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
piece deployee sur HF Spaces. Sert, dans le MEME conteneur, le serveur
vLLM (LoRA DPO, jamais fusionne avec la base), l'API FastAPI
(`interfaces/api/`) ET l'interface Streamlit de test
(`interfaces/web/`), via `Dockerfile` + `demarrer.sh` de ce dossier.

Decision produit du 01/10/2026 : un Space Docker n'expose qu'un seul
port public (`app_port` ci-dessous) -- vLLM et l'API restent internes
au conteneur, et c'est Streamlit qui est branche sur ce port public,
donc visible directement a l'URL du Space, sans rien a lancer en local
(avant cette date, Streamlit tournait uniquement EN LOCAL sur la
machine de l'operateur humain ; cf. historique git de
`interfaces/web/README_space.md` pour cette ancienne architecture a 2
pieces).

`GET /sante` (`interfaces/api/app.py`) reste HTTP 200 des que uvicorn
demarre, meme si vLLM met plusieurs minutes a charger le modele : c'est
ce que Streamlit sonde en boucle pour detecter automatiquement la
disponibilite reelle du modele, sans synchronisation manuelle des
demarrages.

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

Publication : automatique via `.github/workflows/deploy.yml` des que
`CI` reussit sur `main` (effet externe reel, GPU payant, autorise
explicitement par le capitaine le 01/10/2026). Uploade
`deploy/space_gpu_api_vllm/Dockerfile` (renomme `Dockerfile` a la
racine du Space), `deploy/space_gpu_api_vllm/demarrer.sh`, ce fichier
(renomme `README.md` a la racine du Space), ainsi que `src/`,
`interfaces/` (donc `interfaces/web/` desormais aussi), `training/`,
`monitoring/`.
