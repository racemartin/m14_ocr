"""
Tests de la detection heuristique par regex de candidats de PII
residuelle (aucune dependance NLP ici ; le tri confirme/faux positif
est fait ailleurs par la seconde opinion spaCy, cf.
`test_controler_qualite_anonymisation.py`).
"""

from __future__ import annotations

from chsa_triage.application.detection_pii_residuelle import (
    CATEGORIES_DETERMINISTES,
    detecter_candidats,
)


def test_detecte_un_email():
    candidats = detecter_candidats("Contactez jean.dupont@example.com pour un avis.")
    types = {c.type_motif for c in candidats}
    assert "email" in types


def test_detecte_une_url():
    candidats = detecter_candidats("Voir https://example.com/article pour plus de details.")
    types = {c.type_motif for c in candidats}
    assert "url" in types


def test_detecte_une_date():
    candidats = detecter_candidats("Rendez-vous le 15/03/2024 a l'hopital.")
    types = {c.type_motif for c in candidats}
    assert "date" in types


def test_detecte_une_date_en_toutes_lettres():
    candidats = detecter_candidats("Consultation prevue le 15 mars 2024.")
    types = {c.type_motif for c in candidats}
    assert "date" in types


def test_detecte_un_telephone():
    candidats = detecter_candidats("Appelez le 06 12 34 56 78 pour confirmer.")
    types = {c.type_motif for c in candidats}
    assert "telephone" in types


def test_detecte_un_bigramme_capitalise():
    candidats = detecter_candidats("John Smith a ete admis hier soir.")
    bigrammes = [c for c in candidats if c.type_motif == "bigramme_capitalise"]
    assert any("John Smith" in c.passage for c in bigrammes)


def test_texte_vide_ne_produit_aucun_candidat():
    assert detecter_candidats("") == []


def test_texte_sans_pii_ne_produit_aucun_candidat():
    candidats = detecter_candidats("le patient presente une fievre et une toux depuis trois jours.")
    assert candidats == []


def test_passage_contient_le_contexte_autour_du_match():
    texte = "Avant. " + "x" * 5 + "jean.dupont@example.com" + "y" * 5 + " Apres."
    candidats = detecter_candidats(texte)
    (candidat,) = [c for c in candidats if c.type_motif == "email"]
    assert "jean.dupont@example.com" in candidat.passage
    # Le passage contient bien du contexte avant/apres, pas seulement le match brut.
    assert len(candidat.passage) > len("jean.dupont@example.com")


def test_categories_deterministes_couvre_email_telephone_url_date():
    assert CATEGORIES_DETERMINISTES == {"email", "telephone", "url", "date"}
    assert "bigramme_capitalise" not in CATEGORIES_DETERMINISTES
