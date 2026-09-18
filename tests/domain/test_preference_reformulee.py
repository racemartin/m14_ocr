"""
Tests du domaine : (de)serialisation JSON de ChosenReformule, meme
patron que `tests/domain/test_checkpoint_entraine.py`. Aucun
adaptateur de persistance dedie n'est ecrit ici (le cas d'usage
`ReformulerPreferenceDpoUseCase` reutilise le port generique
`RepositoryLectureEcriture`) : la (de)serialisation est faite a la
main, meme mecanisme que pour `CheckpointEntraine`.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.model.preference_reformulee import ChosenReformule

CHOSEN_REFORMULE_EXEMPLE = ChosenReformule(
    identifiant="chsa-ultramedical-preference-8f2c1a9b4e6d3f01",
    chosen_reformule=(
        Message(
            role="assistant",
            contenu='<think>Douleur thoracique aigue, risque cardiaque.</think>'
            '{"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG, troponine"}',
        ),
    ),
    horodatage="2026-09-19T10:00:00Z",
)


def _vers_json(chosen_reformule: ChosenReformule) -> str:
    donnees = asdict(chosen_reformule)
    donnees["chosen_reformule"] = list(donnees["chosen_reformule"])
    return json.dumps(donnees)


def _depuis_json(texte: str) -> ChosenReformule:
    donnees = json.loads(texte)
    return ChosenReformule(
        identifiant=donnees["identifiant"],
        chosen_reformule=tuple(Message(**message) for message in donnees["chosen_reformule"]),
        horodatage=donnees["horodatage"],
    )


def test_serialisation_json_produit_le_schema_attendu():
    donnees = json.loads(_vers_json(CHOSEN_REFORMULE_EXEMPLE))

    assert donnees["identifiant"] == "chsa-ultramedical-preference-8f2c1a9b4e6d3f01"
    assert donnees["chosen_reformule"] == [
        {
            "role": "assistant",
            "contenu": '<think>Douleur thoracique aigue, risque cardiaque.</think>'
            '{"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG, troponine"}',
        }
    ]
    assert donnees["horodatage"] == "2026-09-19T10:00:00Z"


def test_aller_retour_json_preserve_l_egalite():
    reconstruit = _depuis_json(_vers_json(CHOSEN_REFORMULE_EXEMPLE))

    assert reconstruit == CHOSEN_REFORMULE_EXEMPLE
