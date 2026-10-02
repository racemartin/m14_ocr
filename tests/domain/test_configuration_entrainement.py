"""
Tests du domaine : verifie que les valeurs de
`recipes/sft_qwen3_lora.yaml` se deserialisent correctement vers
`ConfigurationQuantification`, `ConfigurationLora` et
`HyperparametresEntrainement`. Aucune dependance externe hormis
`pyyaml` (deja resolu en transitif par `huggingface_hub`/`transformers`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainement,
)

CHEMIN_RECETTE = Path(__file__).parents[2] / "recipes" / "sft_qwen3_lora.yaml"


def _charger_recette() -> dict:
    with CHEMIN_RECETTE.open(encoding="utf-8") as fichier:
        return yaml.safe_load(fichier)


def test_recette_existe():
    assert CHEMIN_RECETTE.is_file()


def test_quantification_se_deserialise_depuis_la_recette():
    recette = _charger_recette()
    quantification = ConfigurationQuantification(**recette["quantification"])

    assert quantification.bits == 4
    assert quantification.type_quantification == "nf4"
    assert quantification.double_quantification is True
    assert quantification.dtype_calcul == "bfloat16"


def test_lora_se_deserialise_depuis_la_recette():
    recette = _charger_recette()
    section_lora = dict(recette["lora"])
    section_lora["modules_cibles"] = tuple(section_lora["modules_cibles"])
    lora = ConfigurationLora(**section_lora)

    assert lora.rang == 16
    assert lora.alpha == 32
    assert lora.dropout == pytest.approx(0.05)
    assert lora.modules_cibles == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_hyperparametres_se_deserialisent_depuis_la_recette():
    """`entrainement:` se deserialise integralement, `assistant_only_loss`
    inclus (champ de `HyperparametresEntrainement` depuis la resolution
    du backlog `m14-ocr-assistant-only-loss-estructurado`). Valeur
    attendue `False` : un run GPU reel (01-02/10/2026) a mesure
    `assistant_only_loss=True` en regression (F1 0.039 contre 0.112),
    la recette a ete repassee a `False`, cf. le commit qui l'a fait."""
    recette = _charger_recette()
    section = recette["entrainement"]
    hyperparametres = HyperparametresEntrainement(
        taux_apprentissage=section["taux_apprentissage"],
        nombre_epoques=section["nombre_epoques"],
        taille_lot=section["taille_lot"],
        packing=section["packing"],
        type_perte=section["type_perte"],
        assistant_only_loss=section["assistant_only_loss"],
    )

    assert hyperparametres.taux_apprentissage == pytest.approx(2.0e-4)
    assert hyperparametres.nombre_epoques == 3
    assert hyperparametres.taille_lot == 4
    assert hyperparametres.packing is True
    assert hyperparametres.type_perte == "nll"
    assert hyperparametres.assistant_only_loss is False


def test_suivi_backend_vaut_hf_dataset_par_defaut_dans_la_recette():
    """
    Regression : un premier entrainement SFT-LoRA reel lance sur HF Jobs
    (GPU L4, verdict "saine", poids publies avec succes) a perdu TOUTE
    sa courbe d'entrainement parce que cette recette valait encore
    `suivi.backend: mlflow` alors que `--suivi-hf-repo` etait passe sur
    la ligne de commande : `mlflow` ecrit dans un SQLite local, perdu
    avec le conteneur ephemere du job (le disque ne survit pas au job).
    `training/E2_04_sft_train.py::_verifier_suivi_hf_repo_coherent`
    refuse desormais de demarrer dans cette combinaison, mais cette
    recette doit aussi rester correcte par defaut : cf. AVERTISSEMENT
    dans training/E2_04_sft_train.py et AGENTS.md.
    """
    recette = _charger_recette()

    assert recette["suivi"]["backend"] == "hf_dataset"
