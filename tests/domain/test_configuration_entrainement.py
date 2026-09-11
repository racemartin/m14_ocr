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
    """
    `entrainement:` contient aussi `assistant_only_loss`, hors du champ
    de HyperparametresEntrainement (pilote trl.SFTConfig, pas cette
    dataclass, cf. docstring de configuration_entrainement.py) : on
    selectionne explicitement les cles couvertes plutot qu'un
    depaquetage direct du dictionnaire.
    """
    recette = _charger_recette()
    section = recette["entrainement"]
    hyperparametres = HyperparametresEntrainement(
        taux_apprentissage=section["taux_apprentissage"],
        nombre_epoques=section["nombre_epoques"],
        taille_lot=section["taille_lot"],
        packing=section["packing"],
        type_perte=section["type_perte"],
    )

    assert hyperparametres.taux_apprentissage == pytest.approx(2.0e-4)
    assert hyperparametres.nombre_epoques == 3
    assert hyperparametres.taille_lot == 4
    assert hyperparametres.packing is True
    assert hyperparametres.type_perte == "chunked_nll"
