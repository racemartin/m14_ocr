"""
Tests des metriques pures d'evaluation baseline
(`application/metriques_evaluation_baseline.py`), donnees synthetiques
uniquement : aucun reseau, aucun GPU, meme principe que
`tests/application/test_verdict_convergence.py`.
"""

from __future__ import annotations

from chsa_triage.application.metriques_evaluation_baseline import (
    correspondance_exacte,
    exactitude_classification_niveau,
    extraire_json,
    extraire_niveau_triage,
    f1_moyen,
    normaliser_texte,
    score_f1_tokens,
    taux_exact_match,
)

# ---------------------------------------------------------------------------
# extraire_json / extraire_niveau_triage
# ---------------------------------------------------------------------------


def test_extraire_json_texte_json_pur():
    assert extraire_json('{"niveau": 3, "categorie": "urgence"}') == {"niveau": 3, "categorie": "urgence"}


def test_extraire_json_json_precede_de_texte_libre():
    texte = 'Voici mon analyse : <think>le patient est stable</think> {"niveau": 2}'
    assert extraire_json(texte) == {"niveau": 2}


def test_extraire_json_retourne_none_si_pas_de_json():
    assert extraire_json("Le patient doit consulter un medecin rapidement.") is None


def test_extraire_json_retourne_none_si_json_invalide():
    assert extraire_json("{niveau: 3 sans guillemets}") is None


def test_extraire_niveau_triage_champ_present():
    assert extraire_niveau_triage('{"niveau": 3, "categorie": "x"}') == "3"


def test_extraire_niveau_triage_tolere_niveau_texte():
    assert extraire_niveau_triage('{"niveau": "ESI-2"}') == "ESI-2"


def test_extraire_niveau_triage_none_si_champ_absent():
    assert extraire_niveau_triage('{"categorie": "urgence"}') is None


def test_extraire_niveau_triage_none_si_pas_de_json_du_tout():
    """Cas reel attendu sur le dataset actuel (voir docstring du module) : reponse en langage naturel."""
    assert extraire_niveau_triage("Consultez un medecin generaliste sous 48h.") is None


# ---------------------------------------------------------------------------
# exactitude_classification_niveau
# ---------------------------------------------------------------------------


def test_exactitude_classification_niveau_toutes_comparables_et_correctes():
    paires = [
        ('{"niveau": 1}', '{"niveau": 1}'),
        ('{"niveau": 3}', '{"niveau": 3}'),
    ]
    resultat = exactitude_classification_niveau(paires)
    assert resultat.nombre_paires == 2
    assert resultat.nombre_comparables == 2
    assert resultat.nombre_corrects == 2
    assert resultat.exactitude == 1.0


def test_exactitude_classification_niveau_partiellement_correcte():
    paires = [
        ('{"niveau": 1}', '{"niveau": 1}'),
        ('{"niveau": 2}', '{"niveau": 3}'),
    ]
    resultat = exactitude_classification_niveau(paires)
    assert resultat.nombre_comparables == 2
    assert resultat.nombre_corrects == 1
    assert resultat.exactitude == 0.5


def test_exactitude_classification_niveau_exclut_les_paires_non_comparables():
    """Une paire ou l'un des deux textes n'a pas de niveau extractible ne compte pas dans l'exactitude."""
    paires = [
        ('{"niveau": 1}', '{"niveau": 1}'),
        ("reponse en langage naturel, pas de JSON", '{"niveau": 2}'),
    ]
    resultat = exactitude_classification_niveau(paires)
    assert resultat.nombre_paires == 2
    assert resultat.nombre_comparables == 1
    assert resultat.nombre_corrects == 1
    assert resultat.exactitude == 1.0


def test_exactitude_classification_niveau_none_si_aucune_paire_comparable():
    """None (pas 0.0) : distingue explicitement "pas de donnee" de "0% de bonnes reponses"."""
    paires = [("texte libre 1", "texte libre 2"), ("autre texte", "encore un autre")]
    resultat = exactitude_classification_niveau(paires)
    assert resultat.nombre_comparables == 0
    assert resultat.exactitude is None


# ---------------------------------------------------------------------------
# normaliser_texte / correspondance_exacte / score_f1_tokens
# ---------------------------------------------------------------------------


def test_normaliser_texte_minuscules_ponctuation_espaces():
    assert normaliser_texte("  Bonjour,   Docteur !!") == "bonjour docteur"


def test_correspondance_exacte_vrai_apres_normalisation():
    assert correspondance_exacte("Bonjour, Docteur !", "bonjour   docteur") is True


def test_correspondance_exacte_faux_si_contenu_different():
    assert correspondance_exacte("Niveau 1", "Niveau 2") is False


def test_score_f1_tokens_identique_vaut_1():
    assert score_f1_tokens("fievre et toux seche", "fievre et toux seche") == 1.0


def test_score_f1_tokens_disjoint_vaut_0():
    assert score_f1_tokens("fievre et toux", "douleur abdominale") == 0.0


def test_score_f1_tokens_partiel_entre_0_et_1():
    score = score_f1_tokens("fievre et toux seche", "fievre et douleurs")
    assert 0.0 < score < 1.0


def test_score_f1_tokens_deux_textes_vides_vaut_1():
    assert score_f1_tokens("   ", "!!!") == 1.0


def test_score_f1_tokens_un_seul_vide_vaut_0():
    assert score_f1_tokens("fievre", "   ") == 0.0


# ---------------------------------------------------------------------------
# taux_exact_match / f1_moyen (agregats)
# ---------------------------------------------------------------------------


def test_taux_exact_match_paires_vide_vaut_0():
    assert taux_exact_match([]) == 0.0


def test_taux_exact_match_calcule_la_proportion():
    paires = [("a b", "a b"), ("a b", "c d"), ("x y", "x y")]
    assert taux_exact_match(paires) == 2 / 3


def test_f1_moyen_paires_vide_vaut_0():
    assert f1_moyen([]) == 0.0


def test_f1_moyen_moyenne_les_scores_individuels():
    paires = [("a b", "a b"), ("a b", "c d")]
    assert f1_moyen(paires) == (1.0 + 0.0) / 2
