"""
Tests de `training/E3_03_dpo_train.py` : les deux gardes de demarrage
(`_verifier_type_perte_dpo_valide`, `_verifier_suivi_hf_repo_coherent`
reutilisee TELLE QUELLE de `training.E2_04_sft_train`) et `_construire_suivi`,
meme patron que `tests/training/test_E2_04_sft_train.py`. Aucune
dependance a `trl`/GPU/reseau reel ici.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import E2_04_sft_train, E3_03_dpo_train


@pytest.mark.parametrize("type_perte", ["sigmoid"])
def test_verifier_type_perte_dpo_valide_accepte_les_valeurs_sures(type_perte):
    E3_03_dpo_train._verifier_type_perte_dpo_valide(type_perte)


@pytest.mark.parametrize("type_perte", ["nll", "dft", "chunked_nll", "hinge", "ipo", "une_valeur_qui_nexiste_pas"])
def test_verifier_type_perte_dpo_valide_rejette_le_reste(type_perte):
    """
    `nll`/`dft`/`chunked_nll` sont le vocabulaire SFT (trl.SFTConfig.loss_type),
    pas celui du DPO : confirme que ce guard DPO-specifique ne reutilise
    pas par erreur le vocabulaire SFT (cf. DEVIATION documentee en tete
    de training/E3_03_dpo_train.py). `hinge`/`ipo` sont des valeurs
    reelles de trl.DPOConfig.loss_type mais jamais verifiees ni
    utilisees par ce projet : rejetees par prudence, pas par erreur.
    """
    with pytest.raises(SystemExit, match="sigmoid"):
        E3_03_dpo_train._verifier_type_perte_dpo_valide(type_perte)


def test_verifier_suivi_hf_repo_coherent_est_bien_celle_de_e2_04_sft_train():
    """
    Reutilisation TELLE QUELLE (pas une copie) : meme fonction, meme
    objet, importee directement, cf. AGENTS.md et l'en-tete du module.
    """
    assert E3_03_dpo_train._verifier_suivi_hf_repo_coherent is E2_04_sft_train._verifier_suivi_hf_repo_coherent


def test_construire_suivi_refuse_suivi_hf_repo_sans_backend_hf_dataset():
    arguments = argparse.Namespace(
        suivi_hf_repo="mombasstic/chsa-triage-dpo-metrics",
        suivi_uri="sqlite:///:memory:",
        suivi_repertoire="/tmp/tb",
        suivi_hf_repertoire_local="/tmp/hf",
    )
    with pytest.raises(SystemExit, match="suivi-hf-repo"):
        E3_03_dpo_train._construire_suivi({"backend": "mlflow"}, arguments)


def test_construire_suivi_mlflow_sans_suivi_hf_repo():
    arguments = argparse.Namespace(
        suivi_hf_repo=None,
        suivi_uri="sqlite:///:memory:",
        suivi_repertoire="/tmp/tb",
        suivi_hf_repertoire_local="/tmp/hf",
    )
    suivi = E3_03_dpo_train._construire_suivi({"backend": "mlflow"}, arguments)
    assert suivi.__class__.__name__ == "MlflowSuiviExperimentation"


def test_identifiant_checkpoint_deterministe_et_distinct_du_sft():
    """
    Prefixe `dpo:` (cf. `_identifiant_checkpoint`) : deux checkpoints
    (SFT et DPO) partageant le meme `modele_base`/horodatage ne
    collisionnent jamais dans `data/processed/checkpoints_sft.jsonl`
    (fichier partage, cf. docs/04_etape3_dpo/02_etapes_cas_usage.md §7).
    """
    id_dpo = E3_03_dpo_train._identifiant_checkpoint("Qwen/Qwen3-1.7B-Base", "outputs/dpo-lora/run-1")
    id_sft = E2_04_sft_train._identifiant_checkpoint("Qwen/Qwen3-1.7B-Base", "outputs/dpo-lora/run-1")
    assert id_dpo != id_sft
    # Deterministe : meme entree, meme sortie.
    assert id_dpo == E3_03_dpo_train._identifiant_checkpoint("Qwen/Qwen3-1.7B-Base", "outputs/dpo-lora/run-1")
