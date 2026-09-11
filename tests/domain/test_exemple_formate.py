"""Tests du domaine : aucune dependance externe, s'executent partout."""

from chsa_triage.domain.model.exemple_formate import ExempleFormate


def test_construction_et_egalite():
    exemple_1 = ExempleFormate(identifiant="chsa-test-abc123", texte="<|im_start|>system\n...")
    exemple_2 = ExempleFormate(identifiant="chsa-test-abc123", texte="<|im_start|>system\n...")
    exemple_3 = ExempleFormate(identifiant="chsa-test-autre", texte="<|im_start|>system\n...")

    assert exemple_1 == exemple_2
    assert exemple_1 != exemple_3
    assert exemple_1.identifiant == "chsa-test-abc123"
