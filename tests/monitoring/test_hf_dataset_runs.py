"""
Tests de `monitoring/hf_dataset_runs.py` : la frontiere reseau HF Hub
elle-meme est remplacee (monkeypatch des symboles `HfApi`/
`hf_hub_download` importes dans ce module), aucun reseau reel. Confirme
que les bons arguments sont transmis et que le contenu telecharge est
relu tel quel (et que `parametres.json` absent -> `{}`, pas une
exception).
"""

from __future__ import annotations

import json

from huggingface_hub.errors import EntryNotFoundError

from monitoring import hf_dataset_runs


class _HfApiFactice:
    def __init__(self, chemins: list[str]) -> None:
        self._chemins = chemins

    def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
        assert repo_type == "dataset"
        return self._chemins


def test_lister_runs_delegue_a_hf_api(monkeypatch):
    monkeypatch.setattr(hf_dataset_runs, "HfApi", lambda: _HfApiFactice(["essai-1/metriques.jsonl", "essai-2/metriques.jsonl"]))

    assert hf_dataset_runs.lister_runs("un-depot") == ["essai-1", "essai-2"]


def test_telecharger_texte_metriques_relit_le_fichier_local(monkeypatch, tmp_path):
    chemin = tmp_path / "metriques.jsonl"
    chemin.write_text('{"etape": 0, "nom": "perte_train", "valeur": 1.0, "horodatage": 1.0}\n', encoding="utf-8")
    appels = []

    def hf_hub_download_factice(repo_id, repo_type, filename):
        appels.append((repo_id, repo_type, filename))
        return str(chemin)

    monkeypatch.setattr(hf_dataset_runs, "hf_hub_download", hf_hub_download_factice)

    texte = hf_dataset_runs.telecharger_texte_metriques("un-depot", "essai-1")

    assert texte == chemin.read_text(encoding="utf-8")
    assert appels == [("un-depot", "dataset", "essai-1/metriques.jsonl")]


def test_telecharger_parametres_lit_le_json(monkeypatch, tmp_path):
    chemin = tmp_path / "parametres.json"
    chemin.write_text(json.dumps({"rang_lora": 8}), encoding="utf-8")
    monkeypatch.setattr(hf_dataset_runs, "hf_hub_download", lambda repo_id, repo_type, filename: str(chemin))

    assert hf_dataset_runs.telecharger_parametres("un-depot", "essai-1") == {"rang_lora": 8}


def test_telecharger_parametres_absent_retourne_dict_vide(monkeypatch):
    def leve_entry_not_found(repo_id, repo_type, filename):
        raise EntryNotFoundError("absent")

    monkeypatch.setattr(hf_dataset_runs, "hf_hub_download", leve_entry_not_found)

    assert hf_dataset_runs.telecharger_parametres("un-depot", "essai-1") == {}
