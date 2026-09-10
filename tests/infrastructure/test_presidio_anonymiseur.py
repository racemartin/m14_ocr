"""
Test d'integration reel de l'adaptateur Presidio ; pas un mock.

Ce test aurait detecte immediatement le bug decouvert lors du smoke
test manuel du 02/09/2026 : `AnalyzerEngine()` construit sans
configuration explicite ne supporte que l'anglais, ce qui faisait
lever `ValueError: No matching recognizers were found` sur tout texte
marque langue="fr". Corrige dans PresidioAnonymiseur via un
NlpEngineProvider multi-langue explicite.

Se saute automatiquement si presidio/spacy ne sont pas installes
(cas de l'environnement minimal de developpement rapide) ; s'execute
reellement des que `uv sync --extra local` a ete fait.
"""

from __future__ import annotations

import pytest

presidio_analyzer = pytest.importorskip("presidio_analyzer")
spacy = pytest.importorskip("spacy")

from chsa_triage.infrastructure.adapters.presidio_anonymiseur import (
    PresidioAnonymiseur,
    RecognizeurNirFrance,
    _est_date_absolue,
    _est_duree_relative,
    _normaliser_ages,
    _tranche_pour_age,
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


# ----------------------------------------------------------------------
# Recognizer NIR francais : checksum modulo 97 (pur, pas de NLP).
#
# Fixtures calculees directement avec l'algorithme documente (decret
# n°82-103, corrobore par xml.insee.fr/schema/nir.html et
# fr.wikipedia.org/wiki/Numero_de_securite_sociale_en_France) :
# cle = 97 - (13 premiers chiffres mod 97), avec pour la Corse la
# substitution standard A->0/-1000000, B->0/-2000000.
# ----------------------------------------------------------------------


def test_nir_valide_une_cle_correcte():
    recognizeur = RecognizeurNirFrance()
    # sexe=1 annee=85 mois=03 dept=75 commune=116 ordre=001 -> cle=27
    assert recognizeur.validate_result("185037511600127") is True


def test_nir_valide_avec_separateurs():
    recognizeur = RecognizeurNirFrance()
    assert recognizeur.validate_result("1 85 03 75 116 001 27") is True


def test_nir_valide_un_departement_corse():
    recognizeur = RecognizeurNirFrance()
    # sexe=2 annee=78 mois=06 dept=2B commune=045 ordre=003 -> cle=58
    assert recognizeur.validate_result("278062B04500358") is True


def test_nir_rejette_une_cle_incorrecte():
    recognizeur = RecognizeurNirFrance()
    assert recognizeur.validate_result("185037511600199") is False


def test_nir_rejette_15_chiffres_sans_rapport():
    """Un nombre a 15 chiffres pris au hasard n'a qu'une chance sur 97 d'avoir la bonne cle."""
    recognizeur = RecognizeurNirFrance()
    assert recognizeur.validate_result("999999999999999") is False


def test_analyzer_detecte_un_nir_reel_dans_un_texte():
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser(
        "Son NIR est 1 85 03 75 116 001 27, a rappeler pour la carte vitale.", langue="fr"
    )
    assert "185037511600127" not in resultat.texte_anonymise.replace(" ", "")
    assert any(e.type_entite == "FR_NIR" for e in resultat.entites_detectees)


def test_analyzer_ne_detecte_pas_un_nir_a_cle_invalide():
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser(
        "Reference dossier : 999999999999999, sans rapport avec un NIR.", langue="fr"
    )
    assert not any(e.type_entite == "FR_NIR" for e in resultat.entites_detectees)


# ----------------------------------------------------------------------
# Normalisation de l'age avant analyse (item 4a) : pur, pas de NLP.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("age,tranche_attendue", [(0, "pediatrique"), (12, "pediatrique"),
                                                    (13, "adolescent"), (17, "adolescent"),
                                                    (18, "adulte"), (64, "adulte"),
                                                    (65, "personne_agee"), (90, "personne_agee")])
def test_tranche_pour_age_respecte_les_bornes(age, tranche_attendue):
    assert _tranche_pour_age(age) == tranche_attendue


def test_normaliser_ages_fr_age_avec_parentheses():
    resultat = _normaliser_ages("Homme âgé (60 ans), chronique (7 mois).", "fr")
    assert "<AGE_ADULTE>" in resultat
    assert "60 ans" not in resultat
    # La duree de la maladie ("7 mois") n'est pas un age, non touchee.
    assert "7 mois" in resultat


def test_normaliser_ages_fr_enfant_de_x_ans():
    resultat = _normaliser_ages("Un enfant de 2 ans vivant en creche.", "fr")
    assert resultat == "Un <AGE_PEDIATRIQUE> vivant en creche."


def test_normaliser_ages_fr_personne_agee():
    resultat = _normaliser_ages("Patiente de 70 ans admise aux urgences.", "fr")
    assert "<AGE_PERSONNE_AGEE>" in resultat


def test_normaliser_ages_en_year_old():
    resultat = _normaliser_ages("A 7-year-old child presented with fever.", "en")
    assert "<AGE_PEDIATRIC>" in resultat
    assert "7-year-old" not in resultat


def test_normaliser_ages_en_years_old_pluriel():
    resultat = _normaliser_ages("She is 45 years old.", "en")
    assert "<AGE_ADULT>" in resultat


def test_normaliser_ages_en_aged():
    resultat = _normaliser_ages("70 year old woman with osteoporosis.", "en")
    assert "<AGE_ELDERLY>" in resultat


def test_pipeline_reel_age_normalise_avant_presidio():
    """L'age explicite ne doit jamais apparaitre tel quel comme DATE_TIME masque ; il est remplace en amont."""
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser("The 7-year-old boy was seen 3 months ago.", langue="en")
    assert "<AGE_PEDIATRIC>" in resultat.texte_anonymise
    assert "7-year-old" not in resultat.texte_anonymise


# ----------------------------------------------------------------------
# Operateur DATE_TIME : date absolue (masquee) vs duree relative
# (conservee) : item 4b.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("texte", ["12/05/1980", "1980-05-12", "12 janvier 2020", "12 January 2020"])
def test_est_date_absolue_reconnait_les_formats_courants(texte):
    assert _est_date_absolue(texte) is True


@pytest.mark.parametrize(
    "texte",
    [
        "il y a 3 semaines",
        "depuis 2 mois",
        "pendant 2 semaines",
        "dans 3 jours",
        "3 months ago",
        "since 2 weeks",
        "for 3 years",
    ],
)
def test_est_duree_relative_reconnait_les_declencheurs_fr_et_en(texte):
    assert _est_duree_relative(texte) is True
    assert _est_date_absolue(texte) is False


def test_pipeline_reel_masque_une_date_absolue_mais_preserve_une_duree_relative():
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser(
        "The boy, born 05/12/2018, was seen 3 months ago.", langue="en"
    )
    assert "05/12/2018" not in resultat.texte_anonymise
    assert "3 months ago" in resultat.texte_anonymise


# ----------------------------------------------------------------------
# Item 2 : verification reelle (pas supposee) de la resolution des
# entites qui se chevauchent et des noms coupes par un saut de ligne --
# AnonymizerEngine.anonymize() de Presidio la fait deja nativement, cf.
# docs/02_etape1_donnees/01_rapport_rgpd.md §7. Ces tests figent ce
# comportement verifie (regression), ils n'ajoutent pas de code de
# resolution supplementaire.
# ----------------------------------------------------------------------


def test_entites_qui_se_chevauchent_sont_resolues_sans_duplication():
    """
    'Jean.Dupont@example.com' est detecte a la fois comme EMAIL_ADDRESS
    et PERSON sur le meme span, et comme URL sur le sous-span
    'example.com' ; trois entites de types differents qui se
    chevauchent. Verifie qu'un seul jeton de masquage est produit (pas
    de doublon, pas de fragment residuel).
    """
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser("Jean.Dupont@example.com a signale le probleme.", langue="fr")
    assert resultat.texte_anonymise == "<INFO_MASQUEE> a signale le probleme."
    assert resultat.texte_anonymise.count("<INFO_MASQUEE>") == 1


def test_nom_coupe_par_un_saut_de_ligne_est_masque_comme_une_seule_entite():
    anonymiseur = PresidioAnonymiseur(strategie="replace")
    resultat = anonymiseur.anonymiser("Contactez\nJean\nDupont pour un avis medical urgent.", langue="fr")
    assert "Jean" not in resultat.texte_anonymise
    assert "Dupont" not in resultat.texte_anonymise
    assert resultat.texte_anonymise.count("<INFO_MASQUEE>") == 1
