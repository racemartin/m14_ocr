"""
Tests du rapport RGPD cumule : fusion entre executions, serialisation
JSON, presentation console/Markdown. Aucune I/O ici (le module ne fait
ni lecture ni ecriture de fichier), donc aucun fichier temporaire
necessaire dans ces tests.
"""

from __future__ import annotations

from chsa_triage.application.use_cases.anonymiser_dataset import StatistiquesSource
from chsa_triage.application.use_cases.rapport_anonymisation import (
    ExecutionAnonymisation,
    RapportAnonymisationCumule,
    formater_rapport_markdown,
    formater_resume_console,
    fusionner_execution,
    rapport_depuis_dict,
    rapport_vers_dict,
)


def _execution(source: str, traites: int, avec_entite: int, entites_par_type: dict, horodatage="2026-09-08T10:00:00+00:00", nombre_traites=None) -> ExecutionAnonymisation:
    return ExecutionAnonymisation(
        horodatage=horodatage,
        dataset="data/processed/dataset_pivot.jsonl",
        strategie="replace",
        limite="5000",
        graine_aleatoire=42,
        nombre_traites=nombre_traites if nombre_traites is not None else traites,
        statistiques_par_source={
            source: StatistiquesSource(
                registres_traites=traites,
                registres_avec_entite=avec_entite,
                entites_par_type=dict(entites_par_type),
            )
        },
    )


def test_fusionner_execution_sur_rapport_vide_initialise_les_cumules():
    rapport = fusionner_execution(RapportAnonymisationCumule(), _execution("MediQAl", 10, 6, {"PERSON": 4}))

    assert len(rapport.executions) == 1
    assert rapport.statistiques_cumulees["MediQAl"].registres_traites == 10
    assert rapport.statistiques_cumulees["MediQAl"].registres_avec_entite == 6
    assert rapport.statistiques_cumulees["MediQAl"].entites_par_type == {"PERSON": 4}


def test_fusionner_execution_additionne_sur_deux_executions_de_la_meme_source():
    rapport = fusionner_execution(RapportAnonymisationCumule(), _execution("MediQAl", 10, 6, {"PERSON": 4}))
    rapport = fusionner_execution(rapport, _execution("MediQAl", 5, 3, {"PERSON": 2, "LOCATION": 1}, horodatage="2026-09-09T10:00:00+00:00"))

    assert len(rapport.executions) == 2
    cumul = rapport.statistiques_cumulees["MediQAl"]
    assert cumul.registres_traites == 15
    assert cumul.registres_avec_entite == 9
    assert cumul.entites_par_type == {"PERSON": 6, "LOCATION": 1}


def test_fusionner_execution_garde_les_sources_distinctes():
    rapport = fusionner_execution(RapportAnonymisationCumule(), _execution("MediQAl", 10, 6, {"PERSON": 4}))
    rapport = fusionner_execution(rapport, _execution("MedQuAD", 7, 5, {"ORGANIZATION": 3}, horodatage="2026-09-09T10:00:00+00:00"))

    assert set(rapport.statistiques_cumulees) == {"MediQAl", "MedQuAD"}
    assert rapport.statistiques_cumulees["MedQuAD"].registres_traites == 7


def test_fusionner_execution_ne_mute_pas_le_rapport_original():
    rapport_initial = RapportAnonymisationCumule()
    fusionner_execution(rapport_initial, _execution("MediQAl", 10, 6, {"PERSON": 4}))

    assert rapport_initial.executions == []
    assert rapport_initial.statistiques_cumulees == {}


def test_serialisation_json_round_trip():
    rapport = fusionner_execution(RapportAnonymisationCumule(), _execution("MediQAl", 10, 6, {"PERSON": 4}))
    rapport = fusionner_execution(rapport, _execution("MedQuAD", 7, 5, {"ORGANIZATION": 3}, horodatage="2026-09-09T10:00:00+00:00"))

    reconstruit = rapport_depuis_dict(rapport_vers_dict(rapport))

    assert len(reconstruit.executions) == 2
    assert reconstruit.executions[0].horodatage == "2026-09-08T10:00:00+00:00"
    assert reconstruit.statistiques_cumulees["MediQAl"].registres_traites == 10
    assert reconstruit.statistiques_cumulees["MedQuAD"].entites_par_type == {"ORGANIZATION": 3}


def test_resume_console_indique_la_proportion_reelle_sur_le_total_dataset():
    rapport = fusionner_execution(RapportAnonymisationCumule(), _execution("MediQAl", 10, 6, {"PERSON": 4}))

    resume = formater_resume_console(rapport, total_dataset=100)

    assert "10/100" in resume
    assert "10.0%" in resume
    assert "6/10" in resume


def test_rapport_markdown_contient_la_portee_cumulee_et_le_detail_par_source():
    rapport = fusionner_execution(RapportAnonymisationCumule(), _execution("MediQAl", 10, 6, {"PERSON": 4}))
    rapport = fusionner_execution(rapport, _execution("MedQuAD", 7, 5, {"ORGANIZATION": 3}, horodatage="2026-09-09T10:00:00+00:00"))

    markdown = formater_rapport_markdown(rapport, total_dataset=1000)

    assert "cumules sur toutes les executions" in markdown
    assert "1000 exemples" in markdown
    assert "MediQAl" in markdown
    assert "MedQuAD" in markdown
    assert "2026-09-08T10:00:00+00:00" in markdown
    assert "2026-09-09T10:00:00+00:00" in markdown


def test_rapport_markdown_sur_rapport_vide_ne_plante_pas():
    markdown = formater_rapport_markdown(RapportAnonymisationCumule(), total_dataset=0)
    assert "Dataset pivot" in markdown
