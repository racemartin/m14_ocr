"""
Tests de `ReformulerDiagnosticJsonUseCase`, avec un faux
`MoteurInference` et un faux `JournalAudit` en memoire, meme patron que
tests/application/test_E4_01_uc_obtenir_diagnostic.py. Aucun GPU/reseau
necessaire.
"""

from __future__ import annotations

import json

from chsa_triage.application.use_cases.E3_00_uc_reformuler_preference_dpo import (
    PROMPT_REFORMULATION_CHOSEN,
)
from chsa_triage.application.use_cases.E4_01_uc_obtenir_diagnostic import (
    PATRON_DIAGNOSTIC_REGEX,
    REPETITION_PENALTY_DEFAUT,
)
from chsa_triage.application.use_cases.E4_02_uc_reformuler_diagnostic_json import (
    ReformulerDiagnosticJsonUseCase,
)
from chsa_triage.domain.model.diagnostic_clinique import DiagnosticClinique
from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.ports.moteur_inference import ReponseModele

JSON_CIBLE = json.dumps(
    {
        "niveau": 2,
        "categorie": "cardio-vasculaire",
        "ressources_estimees": "ECG, troponine",
    }
)
TEXTE_DIAGNOSTIC_VALIDE = (
    f"<think>Douleur thoracique aigue.</think>{JSON_CIBLE}"
)
TEXTE_BRUT_MAL_FORME = "Reponse en texte libre, pas de format attendu."


class FauxMoteurInference:
    def __init__(self, texte_reponse: str = TEXTE_DIAGNOSTIC_VALIDE) -> None:
        self.texte_reponse = texte_reponse
        self.appels: list[list[dict]] = []
        self.parametres_appels: list[dict] = []

    def generer(
        self, messages: list[dict], parametres: dict | None = None
    ) -> ReponseModele:
        self.appels.append(messages)
        self.parametres_appels.append(dict(parametres or {}))
        return ReponseModele(
            texte=self.texte_reponse,
            nombre_tokens_entree=30,
            nombre_tokens_sortie=40,
        )


class FauxJournalAudit:
    def __init__(self) -> None:
        self.entrees: list[EntreeAudit] = []

    def consigner(self, entree: EntreeAudit) -> None:
        self.entrees.append(entree)


def test_reformulation_reussie_est_parsee():
    cas_usage = ReformulerDiagnosticJsonUseCase(
        moteur=FauxMoteurInference(), journal=FauxJournalAudit()
    )

    resultat = cas_usage.executer("conv-1", TEXTE_BRUT_MAL_FORME)

    assert resultat.format_respecte is True
    assert resultat.diagnostic == DiagnosticClinique(
        raisonnement="Douleur thoracique aigue.",
        niveau=2,
        categorie="cardio-vasculaire",
        ressources_estimees="ECG, troponine",
    )


def test_deuxieme_echec_retourne_none_sans_lever():
    moteur = FauxMoteurInference(texte_reponse="toujours pas de format")
    cas_usage = ReformulerDiagnosticJsonUseCase(
        moteur=moteur, journal=FauxJournalAudit()
    )

    resultat = cas_usage.executer("conv-1", TEXTE_BRUT_MAL_FORME)

    assert resultat.diagnostic is None
    assert resultat.format_respecte is False
    assert resultat.texte_brut == "toujours pas de format"


def test_message_envoye_reutilise_le_prompt_de_reformulation_e3_00():
    moteur = FauxMoteurInference()
    cas_usage = ReformulerDiagnosticJsonUseCase(
        moteur=moteur, journal=FauxJournalAudit()
    )

    cas_usage.executer("conv-1", TEXTE_BRUT_MAL_FORME)

    messages_envoyes = moteur.appels[0]
    assert len(messages_envoyes) == 1
    assert messages_envoyes[0]["role"] == "user"
    assert PROMPT_REFORMULATION_CHOSEN in messages_envoyes[0]["content"]
    assert TEXTE_BRUT_MAL_FORME in messages_envoyes[0]["content"]


def test_consigne_une_entree_d_audit_distincte_meme_si_le_format_est_invalide():
    moteur = FauxMoteurInference(texte_reponse="pas de format")
    journal = FauxJournalAudit()
    cas_usage = ReformulerDiagnosticJsonUseCase(
        moteur=moteur,
        journal=journal,
        version_modele="mombasstic/chsa-triage-dpo-lora",
        horloge=lambda: "T0",
    )

    cas_usage.executer("conv-42", TEXTE_BRUT_MAL_FORME)

    assert len(journal.entrees) == 1
    entree = journal.entrees[0]
    assert entree.conversation_id == "conv-42"
    assert entree.type_evenement == "diagnostic_reformule"
    assert entree.entree == TEXTE_BRUT_MAL_FORME
    assert entree.sortie == "pas de format"
    assert entree.version_modele == "mombasstic/chsa-triage-dpo-lora"
    assert entree.metadonnees["format_respecte"] is False
    assert entree.metadonnees["latence_ms"] == 0.0
    assert entree.metadonnees["nombre_tokens_sortie"] == 40


def test_repetition_penalty_et_structured_outputs_sont_transmis_au_moteur():
    """Incident reel (01/10/2026) : sans ces deux garde-fous, un essai
    reel a produit une longue digression hors sujet au lieu d'une
    reformulation JSON (meme classe de bug que E4_01, memes
    constantes reutilisees)."""
    moteur = FauxMoteurInference()
    cas_usage = ReformulerDiagnosticJsonUseCase(
        moteur=moteur, journal=FauxJournalAudit()
    )

    cas_usage.executer("conv-1", TEXTE_BRUT_MAL_FORME)

    assert (
        moteur.parametres_appels[0]["repetition_penalty"]
        == REPETITION_PENALTY_DEFAUT
        == 1.2
    )
    assert moteur.parametres_appels[0]["structured_outputs"] == {
        "regex": PATRON_DIAGNOSTIC_REGEX
    }
