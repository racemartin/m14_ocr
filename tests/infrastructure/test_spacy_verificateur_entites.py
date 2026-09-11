"""
Test d'integration reel de l'adaptateur spaCy de seconde opinion --
pas un mock. Meme convention que `test_presidio_anonymiseur.py` : se
saute automatiquement si spaCy/les modeles ne sont pas installes.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")

from chsa_triage.domain.ports.verificateur_entites import VerdictEntiteNommee
from chsa_triage.infrastructure.adapters.spacy_verificateur_entites import (
    SpacyVerificateurEntitesNommees,
)


def _modeles_spacy_disponibles() -> bool:
    try:
        spacy.load("fr_core_news_md")
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _modeles_spacy_disponibles(),
    reason="Modeles spaCy fr_core_news_md/en_core_web_sm non installes "
    "(python -m spacy download ...)",
)


def test_reconnait_un_nom_francais_comme_entite_pertinente():
    verificateur = SpacyVerificateurEntitesNommees()
    texte = "Jean Dupont est venu hier."
    debut, fin = texte.index("Jean Dupont"), texte.index("Jean Dupont") + len("Jean Dupont")
    assert verificateur.verifier(texte, "fr", debut, fin) is VerdictEntiteNommee.ENTITE_PERTINENTE


def test_reconnait_un_nom_anglais_comme_entite_pertinente():
    verificateur = SpacyVerificateurEntitesNommees()
    texte = "John Smith called yesterday."
    debut, fin = texte.index("John Smith"), texte.index("John Smith") + len("John Smith")
    assert verificateur.verifier(texte, "en", debut, fin) is VerdictEntiteNommee.ENTITE_PERTINENTE


def test_ecarte_un_terme_medical_capitalise_sans_entite():
    verificateur = SpacyVerificateurEntitesNommees()
    texte = "The patient reports Chronic Pain syndrome for two weeks."
    debut = texte.index("Chronic Pain")
    fin = debut + len("Chronic Pain")
    assert verificateur.verifier(texte, "en", debut, fin) is VerdictEntiteNommee.AUCUNE_ENTITE


def test_langue_inconnue_retombe_sur_anglais():
    verificateur = SpacyVerificateurEntitesNommees()
    texte = "John Smith called yesterday."
    debut, fin = texte.index("John Smith"), texte.index("John Smith") + len("John Smith")
    assert verificateur.verifier(texte, "de", debut, fin) is VerdictEntiteNommee.ENTITE_PERTINENTE


def test_reutilise_le_meme_doc_spacy_pour_le_meme_texte_et_langue():
    verificateur = SpacyVerificateurEntitesNommees()
    texte = "John Smith met Chronic Pain patient yesterday."
    debut1, fin1 = texte.index("John Smith"), texte.index("John Smith") + len("John Smith")
    debut2, fin2 = texte.index("Chronic Pain"), texte.index("Chronic Pain") + len("Chronic Pain")

    verificateur.verifier(texte, "en", debut1, fin1)
    doc_premier_appel = verificateur._cache_docs[(texte, "en")]

    verificateur.verifier(texte, "en", debut2, fin2)
    doc_second_appel = verificateur._cache_docs[(texte, "en")]

    assert doc_premier_appel is doc_second_appel
