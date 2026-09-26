"""
Cas d'usage : poursuivre l'entretien clinique adaptatif (F1). F1 n'est
PAS une exigence de donnees d'entrainement supplementaires, mais de
prompting au moment de l'inference : le modele SFT+DPO recoit
l'historique et un prompt systeme d'entretien, et propose la question
suivante ; c'est l'infirmier (humain), jamais le modele, qui declenche
le diagnostic final (cf. `E4_01_uc_obtenir_diagnostic.py`).

"Demarrer" un entretien est juste "poursuivre" avec un historique vide,
pas de cas d'usage separe. Jamais de tour `role: "system"` envoye (ce
checkpoint n'en a jamais vu a l'entrainement, ce qui causait une sortie
vide, meme decision qu'en Etape 3) : le prompt d'entretien est prefixe
au PREMIER message utilisateur et stocke ainsi dans l'historique, pour
que le modele le revoie a chaque appel HTTP sans etat.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.ports.journal_audit import JournalAudit
from chsa_triage.domain.ports.moteur_inference import MoteurInference

PROMPT_ENTRETIEN = (
    "Tu es un assistant de triage aux urgences qui aide un infirmier a "
    "recueillir les informations d'un patient. Pose UNE SEULE question "
    "a la fois, courte et cliniquement pertinente, pour mieux cerner les "
    "symptomes, les antecedents ou les constantes vitales du patient. Ne "
    "propose ni diagnostic ni niveau de priorite a ce stade : "
    "l'infirmier declenchera lui-meme le diagnostic final quand il aura "
    "juge l'entretien suffisant."
)

# Constante de module (jamais un parametre optionnel oubliable par
# l'appelant) : garde-fou de securite clinique (NF4), pas un simple
# reglage de style. Sans repetition_penalty, un test reel a produit un
# changement de langue aleatoire (EN/FR -> JA/ZH/AR) et une reponse
# dangereuse sur un cas de douleur thoracique ; 1.2 est la valeur
# validee lors de l'evaluation post-DPO.
REPETITION_PENALTY_DEFAUT = 1.2

# Bug reel trouve en deploiement : sans temperature explicite, vLLM
# retombait sur son propre defaut (echantillonnage aleatoire), jamais
# teste avec repetition_penalty=1.2 lors de l'evaluation post-DPO.
# Sans temperature=0.0, repetition_penalty seul ne suffit pas a eviter
# la degenerescence (constate en conditions reelles, L4 et T4).
TEMPERATURE_DEFAUT = 0.0

NOMBRE_TOKENS_GENERES_ENTRETIEN = 128


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ResultatTourEntretien:
    """Les deux nouveaux tours produits par `executer()`, a ajouter par l'appelant a son historique."""

    message_utilisateur: Message
    message_assistant: Message


@dataclass(slots=True)
class PoursuivreEntretienUseCase:
    """Orchestre un tour d'entretien : historique + nouveau message -> question suivante du modele."""

    moteur: MoteurInference
    journal: JournalAudit
    version_modele: str = "mombasstic/chsa-triage-dpo-lora"
    horloge: Callable[[], str] = _horodatage_utc_iso

    def executer(
        self,
        conversation_id: str,
        historique: Sequence[Message],
        message_infirmier: str,
    ) -> ResultatTourEntretien:
        """Construit le message utilisateur (prefixe de `PROMPT_ENTRETIEN`
        si `historique` est vide), l'ajoute a l'historique et appelle le
        moteur. Une erreur d'inference se propage plutot que d'etre
        avalee (contrairement aux cas d'usage par lot : ici une seule
        requete pour un seul utilisateur, pas de lot a proteger)."""
        if not historique:
            contenu_utilisateur = f"{PROMPT_ENTRETIEN}\n\n{message_infirmier}"
        else:
            contenu_utilisateur = message_infirmier

        message_utilisateur = Message(role="user", contenu=contenu_utilisateur)
        messages_pour_modele = [
            {"role": m.role, "content": m.contenu}
            for m in (*historique, message_utilisateur)
        ]

        reponse = self.moteur.generer(
            messages_pour_modele,
            {
                "n_predict": NOMBRE_TOKENS_GENERES_ENTRETIEN,
                "repetition_penalty": REPETITION_PENALTY_DEFAUT,
                "temperature": TEMPERATURE_DEFAUT,
            },
        )
        message_assistant = Message(role="assistant", contenu=reponse.texte)

        self.journal.consigner(
            EntreeAudit(
                horodatage=self.horloge(),
                type_evenement="tour_entretien",
                conversation_id=conversation_id,
                entree=message_infirmier,
                sortie=reponse.texte,
                version_modele=self.version_modele,
            )
        )

        return ResultatTourEntretien(
            message_utilisateur=message_utilisateur,
            message_assistant=message_assistant,
        )
