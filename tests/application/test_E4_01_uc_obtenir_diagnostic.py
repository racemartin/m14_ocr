"""
Tests de `ObtenirDiagnosticUseCase`, avec un faux `MoteurInference` et un
faux `JournalAudit` en memoire, meme patron que
tests/application/test_E3_00_uc_reformuler_preference_dpo.py. Aucun
GPU/reseau necessaire.
"""

from __future__ import annotations

import json

from chsa_triage.application.use_cases.E4_01_uc_obtenir_diagnostic import (
    PROMPT_DIAGNOSTIC,
    REPETITION_PENALTY_DEFAUT,
    ObtenirDiagnosticUseCase,
)
from chsa_triage.domain.model.diagnostic_clinique import DiagnosticClinique
from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.ports.moteur_inference import ReponseModele

JSON_CIBLE = json.dumps({"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG, troponine"})
TEXTE_DIAGNOSTIC_VALIDE = f"<think>Douleur thoracique aigue.</think>{JSON_CIBLE}"

HISTORIQUE_EXEMPLE = (
    Message(role="user", contenu="Douleur thoracique depuis ce matin."),
    Message(role="assistant", contenu="Avez-vous des antecedents cardiaques ?"),
    Message(role="user", contenu="Oui, un infarctus il y a 2 ans."),
)


class FauxMoteurInference:
    def __init__(self, texte_reponse: str = TEXTE_DIAGNOSTIC_VALIDE) -> None:
        self.texte_reponse = texte_reponse
        self.appels: list[list[dict]] = []
        self.parametres_appels: list[dict] = []

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        self.appels.append(messages)
        self.parametres_appels.append(dict(parametres or {}))
        return ReponseModele(texte=self.texte_reponse, nombre_tokens_entree=30, nombre_tokens_sortie=40)


class FauxJournalAudit:
    def __init__(self) -> None:
        self.entrees: list[EntreeAudit] = []

    def consigner(self, entree: EntreeAudit) -> None:
        self.entrees.append(entree)


def test_diagnostic_bien_forme_est_parse():
    cas_usage = ObtenirDiagnosticUseCase(moteur=FauxMoteurInference(), journal=FauxJournalAudit())

    resultat = cas_usage.executer("conv-1", HISTORIQUE_EXEMPLE)

    assert resultat.format_respecte is True
    assert resultat.diagnostic == DiagnosticClinique(
        raisonnement="Douleur thoracique aigue.",
        niveau=2,
        categorie="cardio-vasculaire",
        ressources_estimees="ECG, troponine",
    )
    assert resultat.texte_brut == TEXTE_DIAGNOSTIC_VALIDE


def test_diagnostic_mal_forme_retourne_none_sans_lever():
    moteur = FauxMoteurInference(texte_reponse="Reponse en texte libre, pas de format attendu.")
    cas_usage = ObtenirDiagnosticUseCase(moteur=moteur, journal=FauxJournalAudit())

    resultat = cas_usage.executer("conv-1", HISTORIQUE_EXEMPLE)

    assert resultat.diagnostic is None
    assert resultat.format_respecte is False
    assert resultat.texte_brut == "Reponse en texte libre, pas de format attendu."


def test_messages_envoyes_au_modele_incluent_l_historique_puis_le_prompt_diagnostic():
    moteur = FauxMoteurInference()
    cas_usage = ObtenirDiagnosticUseCase(moteur=moteur, journal=FauxJournalAudit())

    cas_usage.executer("conv-1", HISTORIQUE_EXEMPLE)

    messages_envoyes = moteur.appels[0]
    assert len(messages_envoyes) == len(HISTORIQUE_EXEMPLE) + 1
    assert messages_envoyes[-1] == {"role": "user", "content": PROMPT_DIAGNOSTIC}
    assert {m["role"] for m in messages_envoyes} == {"user", "assistant"}


def test_consigne_une_entree_d_audit_meme_si_le_format_est_invalide():
    moteur = FauxMoteurInference(texte_reponse="pas de format")
    journal = FauxJournalAudit()
    cas_usage = ObtenirDiagnosticUseCase(
        moteur=moteur, journal=journal, version_modele="mombasstic/chsa-triage-dpo-lora", horloge=lambda: "T0"
    )

    cas_usage.executer("conv-42", HISTORIQUE_EXEMPLE)

    assert len(journal.entrees) == 1
    entree = journal.entrees[0]
    assert entree.conversation_id == "conv-42"
    assert entree.type_evenement == "diagnostic"
    assert entree.sortie == "pas de format"
    assert entree.version_modele == "mombasstic/chsa-triage-dpo-lora"
    assert entree.metadonnees == {"format_respecte": False}
    assert json.loads(entree.entree) == [
        {"role": m.role, "contenu": m.contenu} for m in HISTORIQUE_EXEMPLE
    ]


def test_repetition_penalty_est_toujours_transmise_au_moteur():
    moteur = FauxMoteurInference()
    cas_usage = ObtenirDiagnosticUseCase(moteur=moteur, journal=FauxJournalAudit())

    cas_usage.executer("conv-1", HISTORIQUE_EXEMPLE)

    assert moteur.parametres_appels[0]["repetition_penalty"] == REPETITION_PENALTY_DEFAUT == 1.2
