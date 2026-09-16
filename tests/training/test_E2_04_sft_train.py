"""
Tests de `training/E2_04_sft_train.py::_publier_checkpoint_hf` : la
frontiere reseau HF Hub elle-meme est remplacee (monkeypatch de
`HfApi` importe dans le module), aucun reseau reel. Confirme que le
depot est cree (prive, `exist_ok=True`) puis que le DOSSIER LOCAL du
meilleur checkpoint est publie tel quel, jamais un essai intermediaire
rejete (cf. AVERTISSEMENT du module).
"""

from __future__ import annotations

import sys
from pathlib import Path

# training/ est un paquet installable (cf. pyproject.toml), mais pas
# forcement installe pendant les tests locaux ; ajouter la racine du
# depot au chemin de recherche, meme patron que tests/interfaces/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import E2_04_sft_train


class _HfApiFactice:
    def __init__(self) -> None:
        self.appels_create_repo: list[dict] = []
        self.appels_upload_folder: list[dict] = []

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        self.appels_create_repo.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private, "exist_ok": exist_ok}
        )

    def upload_folder(self, *, repo_id, folder_path, repo_type):
        self.appels_upload_folder.append(
            {"repo_id": repo_id, "folder_path": folder_path, "repo_type": repo_type}
        )


def test_publier_checkpoint_hf_cree_le_depot_prive_puis_publie_le_dossier(monkeypatch):
    api_factice = _HfApiFactice()
    monkeypatch.setattr(E2_04_sft_train, "HfApi", lambda: api_factice)

    E2_04_sft_train._publier_checkpoint_hf("outputs/sft-lora/run-20260916", "mombasstic/chsa-triage-sft-lora")

    assert api_factice.appels_create_repo == [
        {
            "repo_id": "mombasstic/chsa-triage-sft-lora",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api_factice.appels_upload_folder == [
        {
            "repo_id": "mombasstic/chsa-triage-sft-lora",
            "folder_path": "outputs/sft-lora/run-20260916",
            "repo_type": "model",
        }
    ]
