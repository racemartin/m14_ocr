"""Tests du domaine — aucune dependance externe, s'executent partout."""

from chsa_triage.domain.model import (
    ExemplePivot,
    Langue,
    Message,
    TypeExemple,
)


def test_exemple_sft_complet_est_valide():
    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test", "1"),
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
        identifiant=ExemplePivot.nouvel_identifiant("test", "1"),
        source="test",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="chest pain"),),
        chosen=(Message(role="assistant", contenu="call emergency"),),
    )
    assert not exemple.est_complet_pour_dpo()


def test_identifiant_genere_est_deterministe_pour_la_meme_cle_naturelle():
    id_1 = ExemplePivot.nouvel_identifiant("mediqal_oeq", "1234")
    id_2 = ExemplePivot.nouvel_identifiant("mediqal_oeq", "1234")
    assert id_1 == id_2
    assert id_1.startswith("chsa-mediqal_oeq-")


def test_identifiant_genere_differe_selon_la_cle_naturelle():
    id_1 = ExemplePivot.nouvel_identifiant("mediqal_oeq", "1234")
    id_2 = ExemplePivot.nouvel_identifiant("mediqal_oeq", "5678")
    assert id_1 != id_2


def test_identifiant_genere_differe_selon_l_espace_de_noms_meme_cle_naturelle():
    """
    Cas reel rencontre sur les donnees brutes : mediqal_oeq et
    mediqal_mcqu partagent des valeurs de champ `id` identiques bien
    qu'ils decrivent des registres differents -- l'espace de noms doit
    donc suffire, a lui seul, a eviter la collision.
    """
    id_1 = ExemplePivot.nouvel_identifiant("mediqal_oeq", "1234")
    id_2 = ExemplePivot.nouvel_identifiant("mediqal_mcqu", "1234")
    assert id_1 != id_2
