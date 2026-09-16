"""
Tests de `monitoring/reconstruire_courbe_sft_depuis_log.py` : logique
pure de parsing/reconstruction seulement (aucun reseau reel, aucun
appel HF Hub). Le fragment de log utilise reproduit fidelement le
format reel observe dans le log sauvegarde du job
`6aaab9a95527934177eeaac8` (barres tqdm "X/Y [MM:SS<remaining,
R.RRs/it]", dictionnaires Python litteraux `{'loss': ...}`/
`{'eval_loss': ...}`, messages `[E2_04_sft_train]`), mais avec des
valeurs et un nombre de pas reduits pour rester lisible.
"""

from __future__ import annotations

import json
import subprocess
from calendar import timegm
from datetime import datetime

import pytest

from monitoring import reconstruire_courbe_sft_depuis_log as reconstruire

# 20 pas au total, 2 epoques (pas_par_epoque = 10) ; horodatage
# d'ancrage encode dans le nom de repertoire de checkpoint
# ("run-20260101T000000Z").
TEXTE_LOG_FACTICE = """
2026-09-16 15:47:07,292 [E2_04_sft_train]   Entrainement SFT-LoRA + boucle d'ajustement (AjusterBoucleHyperparametresSftUseCase)
   Building chsa-triage @ git+https://github.com/racemartin/m14_ocr.git@abcdef0123456789abcdef0123456789abcdef01
  10%|          | 2/20 [00:06<00:56,  3.15s/it][A
{'loss': 2.0, 'grad_norm': 0.5, 'learning_rate': 0.0002, 'entropy': 2.1, 'num_tokens': 100.0, 'mean_token_accuracy': 0.5, 'epoch': 0.2}
  50%|          | 10/20 [00:32<00:32,  3.24s/it][A
{'eval_loss': 1.5, 'eval_runtime': 5.0, 'eval_samples_per_second': 1.0, 'eval_steps_per_second': 1.0, 'eval_entropy': 1.6, 'eval_num_tokens': 500.0, 'eval_mean_token_accuracy': 0.6, 'epoch': 1.0}
  60%|          | 12/20 [00:39<00:26,  3.25s/it][A
{'loss': 1.8, 'grad_norm': 0.4, 'learning_rate': 0.0001, 'entropy': 1.9, 'num_tokens': 600.0, 'mean_token_accuracy': 0.62, 'epoch': 1.2}
100%|          | 20/20 [01:05<00:00,  3.25s/it][A
{'eval_loss': 1.3, 'eval_runtime': 5.0, 'eval_samples_per_second': 1.0, 'eval_steps_per_second': 1.0, 'eval_entropy': 1.4, 'eval_num_tokens': 1000.0, 'eval_mean_token_accuracy': 0.7, 'epoch': 2.0}
{'train_runtime': 65.0, 'train_samples_per_second': 1.0, 'train_steps_per_second': 0.3, 'train_loss': 1.75, 'epoch': 2.0}
2026-09-16 16:06:57,053 [E2_04_sft_train]           exemples train formates.....: 50
2026-09-16 16:06:57,053 [E2_04_sft_train]           exemples validation formates: 10
2026-09-16 16:06:57,053 [E2_04_sft_train]           verdict retenu..............: saine
2026-09-16 16:06:57,053 [E2_04_sft_train]           nombre d'essais.............: 1
2026-09-16 16:06:57,053 [E2_04_sft_train]           checkpoint retenu...........: outputs/sft-lora/run-20260101T000000Z
2026-09-16 16:06:57,054 [E2_04_sft_train]     Publication des poids du meilleur checkpoint sur HF Hub (mombasstic/chsa-triage-sft-lora)
"""

HORODATAGE_ANCRAGE = float(timegm(datetime(2026, 1, 1, 0, 0, 0).timetuple()))


def test_extraire_barres_entrainement_isole_le_bon_denominateur():
    total_pas, elapsed_par_etape = reconstruire.extraire_barres_entrainement(TEXTE_LOG_FACTICE)

    assert total_pas == 20
    assert elapsed_par_etape == {2: 6, 10: 32, 12: 39, 20: 65}


def test_extraire_horodatage_ancrage_lit_le_nom_du_checkpoint():
    assert reconstruire.extraire_horodatage_ancrage(TEXTE_LOG_FACTICE) == HORODATAGE_ANCRAGE


def test_extraire_points_metriques_associe_bonnes_etapes_et_horodatages():
    points = reconstruire.extraire_points_metriques(TEXTE_LOG_FACTICE)
    par_cle = {(p.etape, p.nom): p.valeur for p in points}
    horodatages = {(p.etape, p.nom): p.horodatage for p in points}

    # Pas d'entrainement : etape = dernier pas tqdm vu avant la ligne.
    assert par_cle[(2, "perte_train")] == 2.0
    assert par_cle[(2, "norme_gradient")] == 0.5
    assert par_cle[(12, "perte_train")] == 1.8
    assert horodatages[(2, "perte_train")] == HORODATAGE_ANCRAGE + 6
    assert horodatages[(12, "perte_train")] == HORODATAGE_ANCRAGE + 39

    # Pas d'evaluation : etape deduite de epoch * pas_par_epoque (10).
    assert par_cle[(10, "perte_validation")] == 1.5
    assert par_cle[(20, "perte_validation")] == 1.3
    assert horodatages[(10, "perte_validation")] == HORODATAGE_ANCRAGE + 32
    assert horodatages[(20, "perte_validation")] == HORODATAGE_ANCRAGE + 65

    # Metriques additionnelles mappees telles quelles.
    assert par_cle[(2, "learning_rate")] == 0.0002
    assert par_cle[(10, "eval_mean_token_accuracy")] == 0.6

    # 'num_tokens'/'eval_runtime' ne font pas partie du mapping : ignores.
    assert (2, "num_tokens") not in par_cle
    assert (10, "eval_runtime") not in par_cle


def test_extraire_points_metriques_leve_si_pas_total_non_divisible_par_epoques():
    texte = TEXTE_LOG_FACTICE.replace("'epoch': 2.0}\n{'train_runtime'", "'epoch': 3.0}\n{'train_runtime'")
    with pytest.raises(ValueError, match="non divisible"):
        reconstruire.extraire_points_metriques(texte)


def test_points_vers_jsonl_produit_une_ligne_json_par_point():
    points = reconstruire.extraire_points_metriques(TEXTE_LOG_FACTICE)
    texte_jsonl = reconstruire.points_vers_jsonl(points)
    lignes = [json.loads(l) for l in texte_jsonl.splitlines() if l.strip()]

    assert len(lignes) == len(points)
    assert {"etape", "nom", "valeur", "horodatage"} <= lignes[0].keys()
    assert any(l["nom"] == "perte_train" and l["etape"] == 2 for l in lignes)


def test_points_vers_jsonl_chaine_vide_si_aucun_point():
    assert reconstruire.points_vers_jsonl([]) == ""


def test_extraire_metadonnees_log_lit_les_faits_du_log():
    metadonnees = reconstruire.extraire_metadonnees_log(TEXTE_LOG_FACTICE)

    assert metadonnees["nombre_exemples_train"] == 50
    assert metadonnees["nombre_exemples_validation"] == 10
    assert metadonnees["verdict_convergence"] == "saine"
    assert metadonnees["nombre_essais"] == 1
    assert metadonnees["checkpoint_local"] == "outputs/sft-lora/run-20260101T000000Z"
    assert metadonnees["checkpoint_hf_repo"] == "mombasstic/chsa-triage-sft-lora"
    assert metadonnees["train_runtime_s"] == 65.0
    assert metadonnees["train_loss_final"] == 1.75
    assert metadonnees["nombre_pas_total"] == 20


def test_extraire_recette_au_commit_appelle_git_show_avec_le_bon_sha(monkeypatch):
    appels = []

    def subprocess_run_factice(commande, capture_output, text, check):
        appels.append(commande)
        return subprocess.CompletedProcess(commande, 0, stdout="lora:\n  rang: 16\n", stderr="")

    monkeypatch.setattr(reconstruire.subprocess, "run", subprocess_run_factice)

    recette, sha = reconstruire.extraire_recette_au_commit(TEXTE_LOG_FACTICE)

    assert sha == "abcdef0123456789abcdef0123456789abcdef01"
    assert recette == {"lora": {"rang": 16}}
    assert appels == [["git", "show", f"{sha}:recipes/sft_qwen3_lora.yaml"]]


def test_extraire_recette_au_commit_absent_retourne_dict_vide():
    recette, sha = reconstruire.extraire_recette_au_commit("aucun commit ici")

    assert recette == {}
    assert sha is None


def test_construire_parametres_marque_explicitement_la_reconstruction(monkeypatch):
    monkeypatch.setattr(
        reconstruire.subprocess,
        "run",
        lambda commande, capture_output, text, check: subprocess.CompletedProcess(commande, 0, stdout="lora:\n  rang: 16\n", stderr=""),
    )

    parametres = reconstruire.construire_parametres(TEXTE_LOG_FACTICE, "un-job-id")

    assert parametres["reconstruit_depuis_log"] is True
    assert parametres["job_id"] == "un-job-id"
    assert parametres["commit_git"] == "abcdef0123456789abcdef0123456789abcdef01"
    assert parametres["recette"] == {"lora": {"rang": 16}}
    assert parametres["verdict_convergence"] == "saine"
    assert "horodatage_note" in parametres
