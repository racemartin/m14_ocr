"""
Tests du domaine : (de)serialisation JSON de CheckpointEntraine, sur le
modele de l'exemple reel donne en
docs/03_etape2_sft/02_etapes_cas_usage.md paragraphe 6. Aucun
adaptateur de persistance n'existe encore pour cette entite (prevu a
uc_05_03_sauvegarder_checkpoint_sft, hors perimetre de cette phase) :
la (de)serialisation est faite ici a la main, meme mecanisme que
`data/processed/dataset_pivot.jsonl`.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from chsa_triage.domain.model.checkpoint_entraine import (
    CheckpointEntraine,
    VerdictConvergence,
)
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement

CHECKPOINT_EXEMPLE = CheckpointEntraine(
    identifiant="chsa-sft-lora-8f2c1a9b4e6d3f01",
    chemin="checkpoints/sft-lora/2026-09-11/",
    modele_base="Qwen/Qwen3-1.7B-Base",
    configuration_lora=ConfigurationLora(
        rang=16, alpha=32, dropout=0.05,
        modules_cibles=("q_proj", "k_proj", "v_proj", "o_proj"),
    ),
    hyperparametres=HyperparametresEntrainement(
        taux_apprentissage=0.0002, nombre_epoques=3, taille_lot=4,
        packing=True, type_perte="chunked_nll",
    ),
    metriques_finales=MetriquesEntrainement(
        etape=1200, perte_train=0.83, perte_validation=0.91, norme_gradient=1.4,
    ),
    verdict_convergence=VerdictConvergence.SAINE,
    horodatage="2026-09-11T10:00:00Z",
)


def _vers_json(checkpoint: CheckpointEntraine) -> str:
    donnees = asdict(checkpoint)
    donnees["configuration_lora"]["modules_cibles"] = list(donnees["configuration_lora"]["modules_cibles"])
    donnees["verdict_convergence"] = checkpoint.verdict_convergence.value
    return json.dumps(donnees)


def _depuis_json(texte: str) -> CheckpointEntraine:
    donnees = json.loads(texte)
    donnees["configuration_lora"]["modules_cibles"] = tuple(donnees["configuration_lora"]["modules_cibles"])
    return CheckpointEntraine(
        identifiant=donnees["identifiant"],
        chemin=donnees["chemin"],
        modele_base=donnees["modele_base"],
        configuration_lora=ConfigurationLora(**donnees["configuration_lora"]),
        hyperparametres=HyperparametresEntrainement(**donnees["hyperparametres"]),
        metriques_finales=MetriquesEntrainement(**donnees["metriques_finales"]),
        verdict_convergence=VerdictConvergence(donnees["verdict_convergence"]),
        horodatage=donnees["horodatage"],
    )


def test_serialisation_json_produit_le_schema_attendu():
    donnees = json.loads(_vers_json(CHECKPOINT_EXEMPLE))

    assert donnees["identifiant"] == "chsa-sft-lora-8f2c1a9b4e6d3f01"
    assert donnees["configuration_lora"] == {
        "rang": 16, "alpha": 32, "dropout": 0.05,
        "modules_cibles": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    assert donnees["hyperparametres"] == {
        "taux_apprentissage": 0.0002, "nombre_epoques": 3, "taille_lot": 4,
        "packing": True, "type_perte": "chunked_nll",
    }
    assert donnees["metriques_finales"] == {
        "etape": 1200, "perte_train": 0.83, "perte_validation": 0.91, "norme_gradient": 1.4,
    }
    assert donnees["verdict_convergence"] == "saine"
    assert donnees["horodatage"] == "2026-09-11T10:00:00Z"


def test_aller_retour_json_preserve_l_egalite():
    reconstruit = _depuis_json(_vers_json(CHECKPOINT_EXEMPLE))

    assert reconstruit == CHECKPOINT_EXEMPLE
