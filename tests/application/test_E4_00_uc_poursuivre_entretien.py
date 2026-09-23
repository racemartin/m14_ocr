"""
Tests de `PoursuivreEntretienUseCase`, avec un faux `MoteurInference` et
un faux `JournalAudit` en memoire, meme patron que
tests/application/test_E3_00_uc_reformuler_preference_dpo.py. Aucun
GPU/reseau necessaire.
"""

from __future__ import annotations

from chsa_triage.application.use_cases.E4_00_uc_poursuivre_entretien import (
    PROMPT_ENTRETIEN,
    REPETITION_PENALTY_DEFAUT,
    PoursuivreEntretienUseCase,
)
from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.ports.moteur_inference import ReponseModele


class FauxMoteurInference:
    def __init__(self, texte_reponse: str = "Depuis quand avez-vous ces symptomes ?") -> None:
        self.texte_reponse = texte_reponse
        self.appels: list[list[dict]] = []
        self.parametres_appels: list[dict] = []

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        self.appels.append(messages)
        self.parametres_appels.append(dict(parametres or {}))
        return ReponseModele(texte=self.texte_reponse, nombre_tokens_entree=10, nombre_tokens_sortie=5)


class FauxJournalAudit:
    def __init__(self) -> None:
        self.entrees: list[EntreeAudit] = []

    def consigner(self, entree: EntreeAudit) -> None:
        self.entrees.append(entree)


def test_premier_tour_prefixe_le_prompt_entretien_au_message_stocke():
    moteur = FauxMoteurInference()
    cas_usage = PoursuivreEntretienUseCase(moteur=moteur, journal=FauxJournalAudit())

    resultat = cas_usage.executer("conv-1", historique=(), message_infirmier="Douleur au ventre depuis ce matin.")

    assert resultat.message_utilisateur.role == "user"
    assert resultat.message_utilisateur.contenu.startswith(PROMPT_ENTRETIEN)
    assert "Douleur au ventre depuis ce matin." in resultat.message_utilisateur.contenu
    assert resultat.message_assistant == Message(role="assistant", contenu=moteur.texte_reponse)


def test_tour_suivant_n_est_pas_reprefixe():
    moteur = FauxMoteurInference()
    historique = (
        Message(role="user", contenu=f"{PROMPT_ENTRETIEN}\n\nDouleur au ventre."),
        Message(role="assistant", contenu="Depuis quand ?"),
    )
    cas_usage = PoursuivreEntretienUseCase(moteur=moteur, journal=FauxJournalAudit())

    resultat = cas_usage.executer("conv-1", historique=historique, message_infirmier="Depuis ce matin.")

    assert resultat.message_utilisateur == Message(role="user", contenu="Depuis ce matin.")


def test_messages_envoyes_au_modele_incluent_tout_l_historique_sans_role_system():
    moteur = FauxMoteurInference()
    historique = (
        Message(role="user", contenu=f"{PROMPT_ENTRETIEN}\n\nDouleur au ventre."),
        Message(role="assistant", contenu="Depuis quand ?"),
    )
    cas_usage = PoursuivreEntretienUseCase(moteur=moteur, journal=FauxJournalAudit())

    cas_usage.executer("conv-1", historique=historique, message_infirmier="Depuis ce matin.")

    roles_envoyes = {m["role"] for m in moteur.appels[0]}
    assert roles_envoyes == {"user", "assistant"}
    assert moteur.appels[0][-1] == {"role": "user", "content": "Depuis ce matin."}


def test_consigne_une_entree_d_audit_avec_entree_sortie_et_version_modele():
    moteur = FauxMoteurInference(texte_reponse="Avez-vous de la fievre ?")
    journal = FauxJournalAudit()
    cas_usage = PoursuivreEntretienUseCase(
        moteur=moteur, journal=journal, version_modele="mombasstic/chsa-triage-dpo-lora", horloge=lambda: "T0"
    )

    cas_usage.executer("conv-42", historique=(), message_infirmier="Douleur au ventre.")

    assert len(journal.entrees) == 1
    entree = journal.entrees[0]
    assert entree.conversation_id == "conv-42"
    assert entree.type_evenement == "tour_entretien"
    assert entree.entree == "Douleur au ventre."
    assert entree.sortie == "Avez-vous de la fievre ?"
    assert entree.version_modele == "mombasstic/chsa-triage-dpo-lora"
    assert entree.horodatage == "T0"


def test_repetition_penalty_est_toujours_transmise_au_moteur():
    moteur = FauxMoteurInference()
    cas_usage = PoursuivreEntretienUseCase(moteur=moteur, journal=FauxJournalAudit())

    cas_usage.executer("conv-1", historique=(), message_infirmier="Douleur au ventre.")

    assert moteur.parametres_appels[0]["repetition_penalty"] == REPETITION_PENALTY_DEFAUT == 1.2
