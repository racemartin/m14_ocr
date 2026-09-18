"""
Tests de `parser_reformulation_stricte`, sur des chaines construites a
la main, sur le modele de tests/application/test_E1_04_02_detection_pii_residuelle.py
(fonction pure, aucune dependance a un GPU ni a MoteurInference).
"""

from __future__ import annotations

import json

from chsa_triage.application.validation_reformulation_dpo import parser_reformulation_stricte
from chsa_triage.domain.model.exemple_pivot import Message

JSON_VALIDE = json.dumps({"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG, troponine"})
TEXTE_VALIDE = f"<think>Douleur thoracique aigue, risque cardiaque eleve.</think>{JSON_VALIDE}"


def test_texte_valide_retourne_un_seul_tour_assistant():
    resultat = parser_reformulation_stricte(TEXTE_VALIDE)

    assert resultat == (Message(role="assistant", contenu=TEXTE_VALIDE),)


def test_texte_valide_avec_espaces_autour_est_accepte():
    resultat = parser_reformulation_stricte(f"  \n{TEXTE_VALIDE}\n  ")

    assert resultat is not None


def test_json_invalide_retourne_none():
    texte = "<think>raisonnement</think>{niveau: 2, pas du json valide}"

    assert parser_reformulation_stricte(texte) is None


def test_cle_manquante_retourne_none():
    json_incomplet = json.dumps({"niveau": 2, "categorie": "cardio-vasculaire"})
    texte = f"<think>raisonnement</think>{json_incomplet}"

    assert parser_reformulation_stricte(texte) is None


def test_cle_en_trop_retourne_none():
    json_en_trop = json.dumps(
        {"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG", "extra": "non attendu"}
    )
    texte = f"<think>raisonnement</think>{json_en_trop}"

    assert parser_reformulation_stricte(texte) is None


def test_bloc_think_absent_retourne_none():
    assert parser_reformulation_stricte(JSON_VALIDE) is None


def test_bloc_think_vide_retourne_none():
    texte = f"<think>   </think>{JSON_VALIDE}"

    assert parser_reformulation_stricte(texte) is None


def test_json_n_est_pas_un_objet_retourne_none():
    texte = "<think>raisonnement</think>[1, 2, 3]"

    assert parser_reformulation_stricte(texte) is None


def test_texte_vide_retourne_none():
    assert parser_reformulation_stricte("") is None


def test_texte_sans_json_apres_think_retourne_none():
    texte = "<think>raisonnement</think>Reponse en texte libre, pas de JSON."

    assert parser_reformulation_stricte(texte) is None
