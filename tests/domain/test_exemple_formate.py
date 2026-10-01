"""Tests du domaine : aucune dependance externe, s'executent partout."""

from chsa_triage.domain.model.exemple_formate import ExempleFormate, LimiteTour


def test_construction_et_egalite():
    exemple_1 = ExempleFormate(
        identifiant="chsa-test-abc123", texte="<|im_start|>system\n..."
    )
    exemple_2 = ExempleFormate(
        identifiant="chsa-test-abc123", texte="<|im_start|>system\n..."
    )
    exemple_3 = ExempleFormate(
        identifiant="chsa-test-autre", texte="<|im_start|>system\n..."
    )

    assert exemple_1 == exemple_2
    assert exemple_1 != exemple_3
    assert exemple_1.identifiant == "chsa-test-abc123"


def test_tours_par_defaut_est_vide():
    """Un ExempleFormate construit sans `tours` (comme avant cette
    fonctionnalite) a des bornes inconnues, pas des tours vides par
    construction : le champ reste `()`, ce qui signale a
    `_tokeniser_exemple_masque` de retomber sur la perte pleine
    sequence plutot que de masquer a tort."""
    exemple = ExempleFormate(identifiant="chsa-test-abc123", texte="peu importe")
    assert exemple.tours == ()
    assert exemple.textes_assistant() == ()


def test_textes_assistant_extrait_les_bons_segments():
    texte = (
        "<|im_start|>user\nQuestion ?<|im_end|>\n"
        "<|im_start|>assistant\nReponse.<|im_end|>\n"
    )
    borne_user = len("<|im_start|>user\nQuestion ?<|im_end|>\n")
    tours = (
        LimiteTour(role="user", debut=0, fin=borne_user),
        LimiteTour(role="assistant", debut=borne_user, fin=len(texte)),
    )
    exemple = ExempleFormate(
        identifiant="chsa-test-abc123", texte=texte, tours=tours
    )

    assert exemple.textes_assistant() == (texte[borne_user:],)
    assert "Reponse." in exemple.textes_assistant()[0]
    assert "Question ?" not in exemple.textes_assistant()[0]


def test_textes_assistant_concatene_plusieurs_tours_assistant():
    tours = (
        LimiteTour(role="system", debut=0, fin=3),
        LimiteTour(role="assistant", debut=3, fin=6),
        LimiteTour(role="user", debut=6, fin=9),
        LimiteTour(role="assistant", debut=9, fin=12),
    )
    exemple = ExempleFormate(
        identifiant="chsa-test-abc123", texte="SYSASSUSRASS", tours=tours
    )

    assert exemple.textes_assistant() == ("ASS", "ASS")
