"""Tests du domaine — aucune dependance externe, s'executent partout."""

from chsa_triage.domain.model import (
    ExemplePivot,
    Langue,
    Message,
    TypeExemple,
)


def test_exemple_sft_complet_est_valide():
    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test"),
        source="test",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes="fievre",
        prompt=(Message(role="user", contenu="J'ai de la fievre"),),
        completion=(Message(role="assistant", contenu="Depuis combien de temps ?"),),
    )
    assert exemple.est_complet_pour_sft()
    assert not exemple.est_complet_pour_dpo()


def test_exemple_dpo_incomplet_sans_rejected():
    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test"),
        source="test",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="chest pain"),),
        chosen=(Message(role="assistant", contenu="call emergency"),),
    )
    assert not exemple.est_complet_pour_dpo()


def test_identifiant_genere_est_unique():
    id_1 = ExemplePivot.nouvel_identifiant("mediqal")
    id_2 = ExemplePivot.nouvel_identifiant("mediqal")
    assert id_1 != id_2
    assert id_1.startswith("chsa-mediqal-")
