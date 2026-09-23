"""
Tests de `parser_diagnostic_strict`, meme patron que
tests/application/test_validation_reformulation_dpo.py (fonction pure,
aucune dependance a un GPU ni a MoteurInference).
"""

from __future__ import annotations

import json

from chsa_triage.application.validation_diagnostic import parser_diagnostic_strict
from chsa_triage.domain.model.diagnostic_clinique import DiagnosticClinique

JSON_VALIDE = json.dumps({"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG, troponine"})
TEXTE_VALIDE = f"<think>Douleur thoracique aigue, risque cardiaque eleve.</think>{JSON_VALIDE}"


def test_texte_valide_retourne_un_diagnostic_structure():
    resultat = parser_diagnostic_strict(TEXTE_VALIDE)

    assert resultat == DiagnosticClinique(
        raisonnement="Douleur thoracique aigue, risque cardiaque eleve.",
        niveau=2,
        categorie="cardio-vasculaire",
        ressources_estimees="ECG, troponine",
    )


def test_texte_valide_avec_espaces_autour_est_accepte():
    assert parser_diagnostic_strict(f"  \n{TEXTE_VALIDE}\n  ") is not None


def test_json_invalide_retourne_none():
    texte = "<think>raisonnement</think>{niveau: 2, pas du json valide}"

    assert parser_diagnostic_strict(texte) is None


def test_cle_manquante_retourne_none():
    json_incomplet = json.dumps({"niveau": 2, "categorie": "cardio-vasculaire"})
    texte = f"<think>raisonnement</think>{json_incomplet}"

    assert parser_diagnostic_strict(texte) is None


def test_cle_en_trop_retourne_none():
    json_en_trop = json.dumps(
        {"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG", "extra": "non attendu"}
    )
    texte = f"<think>raisonnement</think>{json_en_trop}"

    assert parser_diagnostic_strict(texte) is None


def test_bloc_think_absent_retourne_none():
    assert parser_diagnostic_strict(JSON_VALIDE) is None


def test_bloc_think_vide_retourne_none():
    texte = f"<think>   </think>{JSON_VALIDE}"

    assert parser_diagnostic_strict(texte) is None


def test_json_n_est_pas_un_objet_retourne_none():
    texte = "<think>raisonnement</think>[1, 2, 3]"

    assert parser_diagnostic_strict(texte) is None


def test_texte_vide_retourne_none():
    assert parser_diagnostic_strict("") is None


def test_niveau_non_entier_retourne_none():
    json_niveau_texte = json.dumps(
        {"niveau": "deux", "categorie": "cardio-vasculaire", "ressources_estimees": "ECG"}
    )
    texte = f"<think>raisonnement</think>{json_niveau_texte}"

    assert parser_diagnostic_strict(texte) is None


def test_niveau_booleen_retourne_none():
    """bool est une sous-classe d'int en Python : rejete explicitement."""
    json_niveau_bool = json.dumps(
        {"niveau": True, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG"}
    )
    texte = f"<think>raisonnement</think>{json_niveau_bool}"

    assert parser_diagnostic_strict(texte) is None


def test_categorie_non_chaine_retourne_none():
    json_categorie_int = json.dumps({"niveau": 2, "categorie": 42, "ressources_estimees": "ECG"})
    texte = f"<think>raisonnement</think>{json_categorie_int}"

    assert parser_diagnostic_strict(texte) is None
