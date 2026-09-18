"""
Tests de la logique pure du dashboard de suivi d'entrainement
(`monitoring/logica_suivi_entrainement.py`) : aucun Streamlit, aucun
reseau ici (cf. AGENTS.md, meme principe que les autres tests
`application.*` sans port).
"""

from __future__ import annotations

import pytest

from chsa_triage.domain.model.checkpoint_entraine import VerdictConvergence
from monitoring.logica_suivi_entrainement import (
    COLONNES_RECOMPENSE_DPO,
    NOMBRE_MINIMAL_ETAPES_POUR_VERDICT,
    LigneMetrique,
    analyser_jsonl_metriques,
    construire_courbe_convergence,
    evaluer_convergence_en_vivo,
    extraire_noms_runs,
    filtrer_colonnes_presentes,
    pivoter_par_etape,
)


def _ligne_jsonl(etape: int, nom: str, valeur: float, horodatage: float = 1.0) -> str:
    import json

    return json.dumps({"etape": etape, "nom": nom, "valeur": valeur, "horodatage": horodatage})


def test_analyser_jsonl_metriques_ignore_les_lignes_vides():
    texte = "\n".join(
        [
            _ligne_jsonl(0, "perte_train", 1.2),
            "",
            "   ",
            _ligne_jsonl(0, "norme_gradient", 0.9),
        ]
    )

    lignes = analyser_jsonl_metriques(texte)

    assert lignes == [
        LigneMetrique(etape=0, nom="perte_train", valeur=1.2, horodatage=1.0),
        LigneMetrique(etape=0, nom="norme_gradient", valeur=0.9, horodatage=1.0),
    ]


def test_analyser_jsonl_metriques_texte_vide_retourne_liste_vide():
    assert analyser_jsonl_metriques("") == []


def test_pivoter_par_etape_regroupe_et_trie():
    lignes = [
        LigneMetrique(etape=1, nom="perte_train", valeur=0.8, horodatage=2.0),
        LigneMetrique(etape=0, nom="perte_train", valeur=1.2, horodatage=1.0),
        LigneMetrique(etape=0, nom="perte_validation", valeur=1.5, horodatage=1.1),
    ]

    tableau = pivoter_par_etape(lignes)

    assert tableau == [
        {"etape": 0, "perte_train": 1.2, "perte_validation": 1.5},
        {"etape": 1, "perte_train": 0.8},
    ]


def test_pivoter_par_etape_table_vide():
    assert pivoter_par_etape([]) == []


def test_construire_courbe_convergence_ignore_etapes_incompletes():
    tableau = [
        {"etape": 0, "perte_train": 1.2, "norme_gradient": 0.5},
        {"etape": 1, "perte_train": 0.8},  # norme_gradient manquante : ignoree
        {"etape": 2, "perte_train": 0.6, "perte_validation": 0.7, "norme_gradient": 0.4},
    ]

    courbe = construire_courbe_convergence(tableau)

    assert len(courbe) == 2
    assert courbe[0].etape == 0
    assert courbe[0].perte_validation is None
    assert courbe[1].etape == 2
    assert courbe[1].perte_validation == 0.7


def test_evaluer_convergence_en_vivo_pas_assez_de_donnees():
    tableau = [{"etape": 0, "perte_train": 1.2, "norme_gradient": 0.5}]

    verdict, message = evaluer_convergence_en_vivo(tableau)

    assert verdict is None
    assert "pas encore assez de donnees" in message
    assert f"1/{NOMBRE_MINIMAL_ETAPES_POUR_VERDICT}" in message


def test_evaluer_convergence_en_vivo_table_vide():
    verdict, message = evaluer_convergence_en_vivo([])

    assert verdict is None
    assert "0/" in message


def test_evaluer_convergence_en_vivo_sans_perte_validation_signale_limite():
    tableau = [
        {"etape": e, "perte_train": 1.0 - e * 0.2, "norme_gradient": 0.5}
        for e in range(5)
    ]

    verdict, message = evaluer_convergence_en_vivo(tableau)

    assert verdict == VerdictConvergence.SAINE
    assert "perte_validation absente" in message


def test_evaluer_convergence_en_vivo_courbe_saine_avec_validation():
    tableau = [
        {"etape": e, "perte_train": 2.0 - e * 0.3, "perte_validation": 1.8 - e * 0.2, "norme_gradient": 1.0}
        for e in range(6)
    ]

    verdict, message = evaluer_convergence_en_vivo(tableau)

    assert verdict == VerdictConvergence.SAINE
    assert "perte_validation absente" not in message


def test_evaluer_convergence_en_vivo_detecte_instabilite():
    tableau = [
        {"etape": 0, "perte_train": 1.0, "norme_gradient": 1.0},
        {"etape": 1, "perte_train": 1.5, "norme_gradient": 50.0},
    ]

    verdict, _ = evaluer_convergence_en_vivo(tableau)

    assert verdict == VerdictConvergence.INSTABLE


def test_filtrer_colonnes_presentes_run_sft_sans_recompenses():
    """Un run SFT ne journalise jamais les colonnes de recompense DPO : la liste retournee doit rester vide."""
    tableau_sft = [
        {"etape": 0, "perte_train": 1.2, "norme_gradient": 0.5},
        {"etape": 1, "perte_train": 0.9, "perte_validation": 1.1, "norme_gradient": 0.4},
    ]

    assert filtrer_colonnes_presentes(tableau_sft, COLONNES_RECOMPENSE_DPO) == []


def test_filtrer_colonnes_presentes_run_dpo_avec_recompenses():
    """Un run DPO journalise les 4 metriques de recompense `trl.DPOTrainer` : toutes doivent etre detectees, dans l'ordre de `COLONNES_RECOMPENSE_DPO`."""
    tableau_dpo = [
        {
            "etape": 0,
            "perte_train": 0.6,
            "norme_gradient": 0.3,
            "rewards/chosen": 0.12,
            "rewards/rejected": -0.05,
            "rewards/accuracies": 0.8,
            "rewards/margins": 0.17,
        },
    ]

    assert filtrer_colonnes_presentes(tableau_dpo, COLONNES_RECOMPENSE_DPO) == list(COLONNES_RECOMPENSE_DPO)


def test_filtrer_colonnes_presentes_detecte_meme_si_une_seule_etape_les_porte():
    """Une colonne presente sur une seule etape (ex. la derniere, en cours de journalisation) doit tout de meme etre detectee."""
    tableau = [
        {"etape": 0, "perte_train": 0.6, "norme_gradient": 0.3},
        {"etape": 1, "perte_train": 0.5, "norme_gradient": 0.25, "rewards/chosen": 0.2},
    ]

    assert filtrer_colonnes_presentes(tableau, COLONNES_RECOMPENSE_DPO) == ["rewards/chosen"]


def test_filtrer_colonnes_presentes_table_vide():
    assert filtrer_colonnes_presentes([], COLONNES_RECOMPENSE_DPO) == []


@pytest.mark.parametrize(
    ("chemins", "attendu"),
    [
        ([], []),
        (["essai-1/metriques.jsonl"], ["essai-1"]),
        (
            ["essai-2/metriques.jsonl", "essai-1/metriques.jsonl", "essai-1/parametres.json"],
            ["essai-1", "essai-2"],
        ),
        (["README.md", "essai-1/autre_fichier.txt"], []),
    ],
)
def test_extraire_noms_runs(chemins, attendu):
    assert extraire_noms_runs(chemins) == attendu
