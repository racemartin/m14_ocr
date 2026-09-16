"""
Tests de `training/E2_04_sft_train.py::_publier_checkpoint_hf` : la
frontiere reseau HF Hub elle-meme est remplacee (monkeypatch de
`HfApi` importe dans le module), aucun reseau reel. Confirme que le
depot est cree (prive, `exist_ok=True`) puis que le DOSSIER LOCAL du
meilleur checkpoint est publie tel quel, jamais un essai intermediaire
rejete (cf. AVERTISSEMENT du module).

Teste aussi `_verifier_type_perte_valide` (garde-fou early sur
`entrainement.type_perte`, cf. AVERTISSEMENT en tete de module et
AGENTS.md : un job GPU L4 facture reel a echoue avec
`type_perte=chunked_nll` car la dependance `unsloth` de l'extra
`remote` plafonne `trl` a 0.24.0 des qu'une resolution fraiche a lieu,
et ce trl ne connait pas `chunked_nll`). Pas de dependance a `trl`
reel ici : ce test verifie l'ensemble de valeurs sures cablees dans le
module (`VALEURS_TYPE_PERTE_VALIDES`), pas un appel reel a
`trl.SFTConfig`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("type_perte", ["nll", "dft"])
def test_verifier_type_perte_valide_accepte_les_valeurs_sures(type_perte):
    E2_04_sft_train._verifier_type_perte_valide(type_perte)


def test_verifier_type_perte_valide_rejette_chunked_nll():
    """
    `chunked_nll` est le cas reel qui a fait echouer un job GPU facture
    (cf. AVERTISSEMENT du module) : trl le supporte bien depuis trl>=1.12,
    mais la dependance `unsloth` de l'extra `remote` plafonne `trl` a
    0.24.0 des qu'une resolution fraiche a lieu (HF Jobs), un trl
    anterieur a `chunked_nll`.
    """
    with pytest.raises(SystemExit, match="chunked_nll"):
        E2_04_sft_train._verifier_type_perte_valide("chunked_nll")


def test_verifier_type_perte_valide_rejette_toute_valeur_inconnue():
    with pytest.raises(SystemExit):
        E2_04_sft_train._verifier_type_perte_valide("une_valeur_qui_nexiste_pas")
