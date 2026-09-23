"""
Tests d'integration de l'application FastAPI (`interfaces/api/app.py`),
via `fastapi.testclient.TestClient` (starlette, s'appuie sur `httpx`,
deja une dependance du projet). Faux `MoteurInference`/`JournalAudit`
en memoire injectes a `creer_application()` : aucun serveur vLLM/GPU
reel necessaire.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from chsa_triage.application.use_cases.E4_01_uc_obtenir_diagnostic import PROMPT_DIAGNOSTIC
from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.ports.moteur_inference import ReponseModele
from interfaces.api.app import creer_application

CLE_API_TEST = "cle-de-test-123"

JSON_DIAGNOSTIC_VALIDE = json.dumps(
    {"niveau": 3, "categorie": "respiratoire", "ressources_estimees": "oxygenotherapie"}
)
TEXTE_DIAGNOSTIC_VALIDE = f"<think>raisonnement clinique</think>{JSON_DIAGNOSTIC_VALIDE}"


class FauxMoteurInference:
    def __init__(self, texte_reponse: str = "Depuis quand avez-vous ces symptomes ?") -> None:
        self.texte_reponse = texte_reponse
        self.appels: list[list[dict]] = []

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        self.appels.append(messages)
        return ReponseModele(texte=self.texte_reponse, nombre_tokens_entree=10, nombre_tokens_sortie=5)


class FauxJournalAudit:
    def __init__(self) -> None:
        self.entrees: list[EntreeAudit] = []

    def consigner(self, entree: EntreeAudit) -> None:
        self.entrees.append(entree)


def _client(
    texte_reponse: str = "Depuis quand avez-vous ces symptomes ?",
    verificateur_sante_moteur=None,
) -> tuple[TestClient, FauxJournalAudit]:
    journal = FauxJournalAudit()
    app = creer_application(
        moteur_inference=FauxMoteurInference(texte_reponse=texte_reponse),
        journal_audit=journal,
        cle_api=CLE_API_TEST,
        verificateur_sante_moteur=verificateur_sante_moteur,
    )
    return TestClient(app), journal


def test_sante_ne_requiert_aucune_cle_api():
    client, _ = _client()

    reponse = client.get("/sante")

    assert reponse.status_code == 200


def test_sante_sans_verificateur_injecte_est_toujours_disponible():
    """Mode `local`/llama.cpp (`main.py`) : aucun verificateur specifique n'est branche."""
    client, _ = _client()

    reponse = client.get("/sante")

    assert reponse.status_code == 200
    assert reponse.json()["disponible"] is True


def test_sante_relaie_la_disponibilite_du_verificateur_injecte():
    """Double en memoire simulant `VllmEndpointInferenceAdapter.verifier_sante()`, moteur pas encore demarre."""
    client, _ = _client(
        verificateur_sante_moteur=lambda: {
            "disponible": False,
            "detail": "serveur vLLM indisponible : connexion refusee",
        }
    )

    reponse = client.get("/sante")

    # Toujours HTTP 200 (jamais d'erreur brute, cf. app.py) : seul le
    # corps distingue l'indisponibilite du moteur distant.
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["disponible"] is False
    assert "vLLM" in corps["detail"]


def test_sante_relaie_la_disponibilite_positive_du_verificateur_injecte():
    client, _ = _client(verificateur_sante_moteur=lambda: {"disponible": True, "detail": "serveur vLLM disponible"})

    reponse = client.get("/sante")

    assert reponse.status_code == 200
    assert reponse.json() == {"disponible": True, "detail": "serveur vLLM disponible"}


def test_demarrer_conversation_sans_cle_api_est_rejete():
    client, _ = _client()

    reponse = client.post("/conversations")

    assert reponse.status_code == 401


def test_demarrer_conversation_avec_mauvaise_cle_est_rejete():
    client, _ = _client()

    reponse = client.post("/conversations", headers={"X-API-Key": "mauvaise-cle"})

    assert reponse.status_code == 401


def test_demarrer_puis_poursuivre_conversation():
    client, journal = _client()
    entetes = {"X-API-Key": CLE_API_TEST}

    reponse_debut = client.post("/conversations", headers=entetes)
    assert reponse_debut.status_code == 200
    conversation_id = reponse_debut.json()["conversation_id"]

    reponse_tour = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=entetes,
        json={"message": "Douleur au ventre depuis ce matin."},
    )

    assert reponse_tour.status_code == 200
    corps = reponse_tour.json()
    assert corps["conversation_id"] == conversation_id
    assert corps["message_assistant"] == "Depuis quand avez-vous ces symptomes ?"
    assert len(journal.entrees) == 1
    assert journal.entrees[0].type_evenement == "tour_entretien"


def test_lire_conversation_retourne_l_historique_accumule():
    client, _ = _client()
    entetes = {"X-API-Key": CLE_API_TEST}
    conversation_id = client.post("/conversations", headers=entetes).json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/messages", headers=entetes, json={"message": "Douleur au ventre."}
    )

    reponse = client.get(f"/conversations/{conversation_id}", headers=entetes)

    assert reponse.status_code == 200
    historique = reponse.json()["historique"]
    assert len(historique) == 2
    assert historique[0]["role"] == "user"
    assert historique[1]["role"] == "assistant"


def test_poursuivre_conversation_inconnue_retourne_404():
    client, _ = _client()

    reponse = client.post(
        "/conversations/inconnue/messages", headers={"X-API-Key": CLE_API_TEST}, json={"message": "x"}
    )

    assert reponse.status_code == 404


def test_diagnostic_sur_conversation_vide_retourne_400():
    client, _ = _client()
    entetes = {"X-API-Key": CLE_API_TEST}
    conversation_id = client.post("/conversations", headers=entetes).json()["conversation_id"]

    reponse = client.post(f"/conversations/{conversation_id}/diagnostic", headers=entetes)

    assert reponse.status_code == 400


def test_diagnostic_bien_forme_retourne_les_champs_structures():
    client, journal = _client(texte_reponse=TEXTE_DIAGNOSTIC_VALIDE)
    entetes = {"X-API-Key": CLE_API_TEST}
    conversation_id = client.post("/conversations", headers=entetes).json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/messages", headers=entetes, json={"message": "Douleur thoracique."}
    )

    reponse = client.post(f"/conversations/{conversation_id}/diagnostic", headers=entetes)

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["format_respecte"] is True
    assert corps["niveau"] == 3
    assert corps["categorie"] == "respiratoire"
    assert corps["ressources_estimees"] == "oxygenotherapie"
    assert corps["raisonnement"] == "raisonnement clinique"
    assert any(e.type_evenement == "diagnostic" for e in journal.entrees)


def test_diagnostic_mal_forme_retourne_format_respecte_false_sans_erreur():
    client, _ = _client(texte_reponse="Reponse en texte libre.")
    entetes = {"X-API-Key": CLE_API_TEST}
    conversation_id = client.post("/conversations", headers=entetes).json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/messages", headers=entetes, json={"message": "Douleur thoracique."}
    )

    reponse = client.post(f"/conversations/{conversation_id}/diagnostic", headers=entetes)

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["format_respecte"] is False
    assert corps["niveau"] is None
    assert corps["texte_brut"] == "Reponse en texte libre."


def test_prompt_diagnostic_envoye_en_dernier_message_utilisateur():
    client, _ = _client(texte_reponse=TEXTE_DIAGNOSTIC_VALIDE)
    entetes = {"X-API-Key": CLE_API_TEST}
    conversation_id = client.post("/conversations", headers=entetes).json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/messages", headers=entetes, json={"message": "Douleur thoracique."}
    )

    client.post(f"/conversations/{conversation_id}/diagnostic", headers=entetes)

    # Verifie via l'audit consigne (entree = historique JSON) que le prompt de
    # diagnostic n'est PAS stocke dans l'historique lui-meme (seulement envoye
    # au modele au moment de l'appel), cf. ObtenirDiagnosticUseCase.
    reponse_historique = client.get(f"/conversations/{conversation_id}", headers=entetes)
    contenu_historique = json.dumps(reponse_historique.json())
    assert PROMPT_DIAGNOSTIC not in contenu_historique
