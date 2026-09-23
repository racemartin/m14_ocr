"""
Point d'entree ASGI reel de l'API (`uvicorn interfaces.api.main:app`),
la seule couche qui lit des variables d'environnement (jamais de cle
en dur, cf. `securite.py`). `creer_application()` (`app.py`) reste
testable sans ces variables.

Variables d'environnement :
- `CHSA_CLE_API_DEMO` (obligatoire) : cle attendue en en-tete
  `X-API-Key` par les clients de cette API.
- `CHSA_MOTEUR_INFERENCE` = `distant` (defaut, vLLM, cible mission) ou
  `local` (llama.cpp, dev sans GPU ; meme adaptateur/patron que
  `interfaces/cli/E1_06_00_evaluer_baseline.py`).
- `CHSA_URL_MOTEUR_INFERENCE` : URL du serveur d'inference (defaut
  `http://127.0.0.1:8000` en mode `distant`, `http://127.0.0.1:8080`
  en mode `local`).
- `CHSA_CLE_API_VLLM` (optionnelle, mode `distant`) : transmise en
  `Authorization: Bearer ...` au serveur vLLM (`vllm serve --api-key`).
- `CHSA_NOM_MODELE_VLLM` (mode `distant`, defaut `dpo`) : nom donne a
  l'adaptateur LoRA via `--lora-modules` (decision Etape 4 : LoRA
  JAMAIS fusionne avec la base, cf. AGENTS.md/roadmap).
- `CHSA_VERSION_MODELE` (defaut `mombasstic/chsa-triage-dpo-lora`) :
  consignee telle quelle au journal d'audit (F6), texte libre.
- `CHSA_CHEMIN_JOURNAL_AUDIT` (defaut `data/processed/journal_audit.jsonl`,
  mode `jsonl`, defaut).
- `CHSA_JOURNAL_AUDIT` = `jsonl` (defaut, fichier local, PERDU a chaque
  redemarrage du Space Docker/GPU, filesystem ephemere) ou `hf_dataset`
  (persistance gratuite via un dataset HF Hub, `HfDatasetJournalAudit`,
  meme mecanisme que `HfDatasetSuiviExperimentation`, cf. AGENTS.md).
- `CHSA_JOURNAL_AUDIT_REPO` (obligatoire si `CHSA_JOURNAL_AUDIT=hf_dataset`,
  ex. `mombasstic/chsa-triage-audit-journal`) : depot dataset HF, doit
  deja exister (`hf repo create ... --repo-type dataset --private`) et
  `HF_TOKEN` doit etre defini (secret du Space).
- `CHSA_JOURNAL_AUDIT_REPERTOIRE_LOCAL` (mode `hf_dataset`, defaut
  `data/processed/suivi_hf_dataset_audit`) : dossier de travail local
  surveille par le `CommitScheduler`.
"""

from __future__ import annotations

import os

from chsa_triage.domain.ports.journal_audit import JournalAudit
from chsa_triage.domain.ports.moteur_inference import MoteurInference
from chsa_triage.infrastructure.adapters.hf_dataset_journal_audit import (
    HfDatasetJournalAudit,
)
from chsa_triage.infrastructure.adapters.jsonl_journal_audit import (
    JsonlJournalAudit,
)
from chsa_triage.infrastructure.adapters.llamacpp_inference_adapter import (
    LlamaCppInferenceAdapter,
)
from chsa_triage.infrastructure.adapters.vllm_endpoint_inference_adapter import (
    VllmEndpointInferenceAdapter,
)
from interfaces.api.app import creer_application

URL_VLLM_PAR_DEFAUT = "http://127.0.0.1:8000"
URL_LLAMACPP_PAR_DEFAUT = "http://127.0.0.1:8080"
CHEMIN_JOURNAL_AUDIT_PAR_DEFAUT = "data/processed/journal_audit.jsonl"
REPERTOIRE_JOURNAL_AUDIT_HF_LOCAL_PAR_DEFAUT = "data/processed/suivi_hf_dataset_audit"


def _construire_moteur_inference() -> MoteurInference:
    mode = os.environ.get("CHSA_MOTEUR_INFERENCE", "distant")

    if mode == "local":
        url = os.environ.get("CHSA_URL_MOTEUR_INFERENCE", URL_LLAMACPP_PAR_DEFAUT)
        return LlamaCppInferenceAdapter(url_serveur_local=url)

    if mode == "distant":
        url = os.environ.get("CHSA_URL_MOTEUR_INFERENCE", URL_VLLM_PAR_DEFAUT)
        cle_api_vllm = os.environ.get("CHSA_CLE_API_VLLM")
        nom_modele = os.environ.get("CHSA_NOM_MODELE_VLLM", "dpo")
        return VllmEndpointInferenceAdapter(url_endpoint=url, cle_api=cle_api_vllm, nom_modele=nom_modele)

    raise ValueError(f"CHSA_MOTEUR_INFERENCE invalide : {mode!r} (attendu 'local' ou 'distant')")


def _construire_journal_audit() -> JournalAudit:
    mode = os.environ.get("CHSA_JOURNAL_AUDIT", "jsonl")

    if mode == "jsonl":
        return JsonlJournalAudit(os.environ.get("CHSA_CHEMIN_JOURNAL_AUDIT", CHEMIN_JOURNAL_AUDIT_PAR_DEFAUT))

    if mode == "hf_dataset":
        repo_id = os.environ.get("CHSA_JOURNAL_AUDIT_REPO")
        if not repo_id:
            raise RuntimeError(
                "CHSA_JOURNAL_AUDIT_REPO doit etre definie quand CHSA_JOURNAL_AUDIT=hf_dataset "
                "(ex. mombasstic/chsa-triage-audit-journal)."
            )
        repertoire_local = os.environ.get(
            "CHSA_JOURNAL_AUDIT_REPERTOIRE_LOCAL", REPERTOIRE_JOURNAL_AUDIT_HF_LOCAL_PAR_DEFAUT
        )
        return HfDatasetJournalAudit(repo_id=repo_id, repertoire_local=repertoire_local)

    raise ValueError(f"CHSA_JOURNAL_AUDIT invalide : {mode!r} (attendu 'jsonl' ou 'hf_dataset')")


def _cle_api_demo() -> str:
    cle = os.environ.get("CHSA_CLE_API_DEMO")
    if not cle:
        raise RuntimeError(
            "CHSA_CLE_API_DEMO doit etre definie (auth par cle de l'API de demonstration, "
            "jamais de defaut permissif)."
        )
    return cle


_moteur_inference = _construire_moteur_inference()

app = creer_application(
    moteur_inference=_moteur_inference,
    journal_audit=_construire_journal_audit(),
    cle_api=_cle_api_demo(),
    version_modele=os.environ.get("CHSA_VERSION_MODELE", "mombasstic/chsa-triage-dpo-lora"),
    # Seul VllmEndpointInferenceAdapter (mode `distant`) expose une
    # verification de sante reelle (`/health` vLLM) ; LlamaCppInferenceAdapter
    # (mode `local`, dev sans GPU) n'a pas d'equivalent branche ici, `/sante`
    # retombe alors sur le defaut "toujours disponible" de `creer_application()`.
    verificateur_sante_moteur=getattr(_moteur_inference, "verifier_sante", None),
)
