"""
Tests de `monitoring/exporter_courbes_png.py` : zero reseau, rejoue un
run deja telecharge (`data/demos/sft-lora-16092026-reconstruit/metriques.jsonl`,
un vrai run SFT reconstruit, cf. AGENTS.md) et une fixture DPO
synthetique (aucun run DPO publie dans `data/demos/`, les colonnes de
recompense sont donc construites ici), meme esprit que
`monitoring/importer_mlflow_local.py` (rejoue sans serveur distant).
"""

from __future__ import annotations

import json
from pathlib import Path

from monitoring.exporter_courbes_png import (
    exporter_cartes_recompenses_dpo_png,
    exporter_courbe_pertes_png,
    exporter_courbe_recompenses_dpo_png,
    exporter_metriques_evaluation_png,
    exporter_toutes_les_courbes,
)
from monitoring.logica_suivi_entrainement import (
    analyser_jsonl_metriques,
    pivoter_par_etape,
)

CHEMIN_FIXTURE_RUN_SFT_REEL = Path(
    "data/demos/sft-lora-16092026-reconstruit/metriques.jsonl"
)


def _tableau_depuis_fixture(chemin: Path) -> list[dict]:
    texte = chemin.read_text(encoding="utf-8")
    return pivoter_par_etape(analyser_jsonl_metriques(texte))


def _ligne_jsonl(etape: int, nom: str, valeur: float) -> str:
    return json.dumps({"etape": etape, "nom": nom, "valeur": valeur, "horodatage": 1.0})


def test_exporter_courbe_pertes_png_sur_un_run_sft_reel_deja_telecharge(tmp_path):
    tableau_large = _tableau_depuis_fixture(CHEMIN_FIXTURE_RUN_SFT_REEL)
    chemin_sortie = tmp_path / "courbe_pertes.png"

    ecrit = exporter_courbe_pertes_png(tableau_large, chemin_sortie, titre="Test SFT reel")

    assert ecrit is True
    assert chemin_sortie.exists()
    assert chemin_sortie.stat().st_size > 0


def test_exporter_courbe_pertes_png_tableau_vide_n_ecrit_rien(tmp_path):
    chemin_sortie = tmp_path / "courbe_pertes.png"

    ecrit = exporter_courbe_pertes_png([], chemin_sortie, titre="vide")

    assert ecrit is False
    assert not chemin_sortie.exists()


def test_exporter_courbe_recompenses_dpo_png_absent_sur_un_run_sft(tmp_path):
    """Un run SFT reel n'a jamais les colonnes `rewards/*` : rien ne doit etre ecrit."""
    tableau_large = _tableau_depuis_fixture(CHEMIN_FIXTURE_RUN_SFT_REEL)
    chemin_sortie = tmp_path / "courbe_recompenses.png"

    ecrit = exporter_courbe_recompenses_dpo_png(
        tableau_large, chemin_sortie, titre="Test SFT reel"
    )

    assert ecrit is False
    assert not chemin_sortie.exists()


def test_exporter_courbe_recompenses_dpo_png_present_sur_une_fixture_dpo(tmp_path):
    texte_dpo = "\n".join(
        [
            _ligne_jsonl(0, "perte_train", 0.9),
            _ligne_jsonl(0, "rewards/chosen", 0.1),
            _ligne_jsonl(0, "rewards/rejected", -0.2),
            _ligne_jsonl(0, "rewards/accuracies", 0.6),
            _ligne_jsonl(0, "rewards/margins", 0.3),
            _ligne_jsonl(1, "perte_train", 0.8),
            _ligne_jsonl(1, "rewards/chosen", 0.3),
            _ligne_jsonl(1, "rewards/rejected", -0.4),
            _ligne_jsonl(1, "rewards/accuracies", 0.7),
            _ligne_jsonl(1, "rewards/margins", 0.7),
        ]
    )
    tableau_large = pivoter_par_etape(analyser_jsonl_metriques(texte_dpo))
    chemin_courbe = tmp_path / "courbe_recompenses.png"
    chemin_cartes = tmp_path / "cartes_recompenses.png"

    ecrit_courbe = exporter_courbe_recompenses_dpo_png(
        tableau_large, chemin_courbe, titre="Test DPO"
    )
    ecrit_cartes = exporter_cartes_recompenses_dpo_png(
        tableau_large, chemin_cartes, titre="Test DPO cartes"
    )

    assert ecrit_courbe is True
    assert chemin_courbe.exists()
    assert ecrit_cartes is True
    assert chemin_cartes.exists()


def test_exporter_metriques_evaluation_png_absent_sur_un_run_sft(tmp_path):
    """Un run SFT reel n'a jamais exact_match/f1_moyen/latence_ms_moyenne :
    rien ne doit etre ecrit."""
    tableau_large = _tableau_depuis_fixture(CHEMIN_FIXTURE_RUN_SFT_REEL)
    chemin_sortie = tmp_path / "metriques_evaluation.png"

    ecrit = exporter_metriques_evaluation_png(
        tableau_large, chemin_sortie, titre="Test SFT reel"
    )

    assert ecrit is False
    assert not chemin_sortie.exists()


def test_exporter_metriques_evaluation_png_present_sur_une_fixture_evaluation(tmp_path):
    """Forme reelle d'un run EvaluerBaselineZeroShotUseCase (evaluation-post-sft/
    -post-dpo) : un seul point a etape 0, exact_match/f1_moyen/latence_ms_moyenne/
    nombre_echecs_inference."""
    texte_evaluation = "\n".join(
        [
            _ligne_jsonl(0, "exact_match", 0.0),
            _ligne_jsonl(0, "f1_moyen", 0.1117),
            _ligne_jsonl(0, "latence_ms_moyenne", 11553.37),
            _ligne_jsonl(0, "nombre_echecs_inference", 0),
        ]
    )
    tableau_large = pivoter_par_etape(analyser_jsonl_metriques(texte_evaluation))
    chemin_sortie = tmp_path / "metriques_evaluation.png"

    ecrit = exporter_metriques_evaluation_png(
        tableau_large, chemin_sortie, titre="Test evaluation-post-sft"
    )

    assert ecrit is True
    assert chemin_sortie.exists()
    assert chemin_sortie.stat().st_size > 0


def test_exporter_toutes_les_courbes_sur_un_run_sft_reel_n_ecrit_que_la_courbe_de_pertes(
    tmp_path,
):
    tableau_large = _tableau_depuis_fixture(CHEMIN_FIXTURE_RUN_SFT_REEL)

    chemins = exporter_toutes_les_courbes(
        tableau_large, tmp_path, nom_run="sft-lora-16092026-reconstruit"
    )

    assert len(chemins) == 1
    assert chemins[0].name == "courbe_pertes.png"
    assert chemins[0].exists()
