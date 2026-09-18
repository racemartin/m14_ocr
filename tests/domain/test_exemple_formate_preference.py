"""Tests du domaine : aucune dependance externe, s'executent partout."""

from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference


def test_construction_et_egalite():
    exemple_1 = ExempleFormatePreference(
        identifiant="chsa-test-abc123",
        texte_prompt="<|im_start|>user\n...",
        texte_chosen="<|im_start|>assistant\nchosen...",
        texte_rejected="<|im_start|>assistant\nrejected...",
    )
    exemple_2 = ExempleFormatePreference(
        identifiant="chsa-test-abc123",
        texte_prompt="<|im_start|>user\n...",
        texte_chosen="<|im_start|>assistant\nchosen...",
        texte_rejected="<|im_start|>assistant\nrejected...",
    )
    exemple_3 = ExempleFormatePreference(
        identifiant="chsa-test-autre",
        texte_prompt="<|im_start|>user\n...",
        texte_chosen="<|im_start|>assistant\nchosen...",
        texte_rejected="<|im_start|>assistant\nrejected...",
    )

    assert exemple_1 == exemple_2
    assert exemple_1 != exemple_3
    assert exemple_1.identifiant == "chsa-test-abc123"
    assert exemple_1.texte_chosen != exemple_1.texte_rejected
