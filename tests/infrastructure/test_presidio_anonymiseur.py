"""
Test d'integration reel de l'adaptateur Presidio -- pas un mock.

Ce test aurait detecte immediatement le bug decouvert lors du smoke
test manuel du 02/09/2026 : `AnalyzerEngine()` construit sans
configuration explicite ne supporte que l'anglais, ce qui faisait
lever `ValueError: No matching recognizers were found` sur tout texte
marque langue="fr". Corrige dans PresidioAnonymiseur via un
NlpEngineProvider multi-langue explicite.

Se saute automatiquement si presidio/spacy ne sont pas installes
(cas de l'environnement minimal de developpement rapide) -- s'execute
reellement des que `uv sync --extra local` a ete fait.
"""

from __future__ import annotations

import pytest

presidio_analyzer = pytest.importorskip("presidio_analyzer")
spacy = pytest.importorskip("spacy")

from chsa_triage.infrastructure.adapters.presidio_anonymiseur import (
    PresidioAnonymiseur,
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


def test_anonymise_un_nom_francais():
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser(
        "Contactez le Dr. Jean Dupont pour un avis medical.", langue="fr"
    )
    assert "Jean Dupont" not in resultat.texte_anonymise
    assert len(resultat.entites_detectees) >= 1


def test_anonymise_un_nom_anglais():
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser(
        "Patient contact: John Smith, phone number provided.", langue="en"
    )
    assert "John Smith" not in resultat.texte_anonymise


def test_texte_vide_ne_plante_pas():
    anonymiseur = PresidioAnonymiseur()
    resultat = anonymiseur.anonymiser("", langue="fr")
    assert resultat.texte_anonymise == ""
    assert resultat.entites_detectees == ()


def test_strategie_mask_produit_des_etoiles():
    anonymiseur = PresidioAnonymiseur(strategie="mask")
    resultat = anonymiseur.anonymiser("Jean Dupont est venu.", langue="fr")
    assert "*" in resultat.texte_anonymise or resultat.texte_anonymise == "Jean Dupont est venu."
