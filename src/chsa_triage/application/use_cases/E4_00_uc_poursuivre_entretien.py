"""
Cas d'usage : poursuivre l'entretien clinique adaptatif avec le
patient/infirmier (F1). Decision de conception (cahier des charges
§3, note du 23/09/2026) : F1 n'est PAS une exigence de donnees
d'entrainement supplementaires, mais une exigence de **prompting au
moment de l'inference** - le modele deja SFT+DPO recoit l'historique
de conversation accumule et un prompt systeme d'entretien, et propose
la question suivante ; c'est l'infirmier (humain), jamais le modele
de facon autonome, qui decide quand declencher le diagnostic final
(cf. `E4_01_uc_obtenir_diagnostic.py`).

Reutilise `MoteurInference` sans modification (cinquieme reemploi du
projet, apres les deux baselines zero-shot, l'evaluation post-SFT et
la reformulation DPO). "Demarrer" un entretien n'est qu'un cas
particulier de "poursuivre" avec un historique vide : aucun cas
d'usage separe pour cela (pas de logique metier propre, juste une
liste vide - l'API cree l'identifiant de conversation et appelle ce
meme cas d'usage).

**Jamais de tour `role: "system"` envoye au modele** (meme decision
que `E3_00_uc_reformuler_preference_dpo.py`, cf. AGENTS.md : verifie
empiriquement que ce checkpoint n'a jamais vu de tour `system` pendant
son propre entrainement, un role absent du fine-tuning ayant deja
cause une degenerescence en sortie vide lors des runs DPO reels). Le
prompt d'entretien est donc prefixe au PREMIER message utilisateur
(stocke ainsi dans l'historique, pas seulement au moment de l'envoi) :
comme chaque appel HTTP vers vLLM est sans etat (l'historique complet
est renvoye a chaque tour), le modele doit revoir cette instruction a
chaque appel ulterieur, et la stocker dans l'historique est la facon
la plus simple de le garantir sans re-detecter "est-ce le premier
tour" a chaque appel.
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

NOMBRE_TOKENS_GENERES_ENTRETIEN = 128


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ResultatTourEntretien:
    """Les deux nouveaux tours produits par `executer()`, a ajouter par l'appelant a son historique."""

    message_utilisateur : Message
    message_assistant     : Message


@dataclass(slots=True)
class PoursuivreEntretienUseCase:
    """Orchestre un tour d'entretien : historique + nouveau message -> question suivante du modele."""

    moteur           : MoteurInference
    journal            : JournalAudit
    version_modele    : str = "mombasstic/chsa-triage-dpo-lora"
    horloge             : Callable[[], str] = _horodatage_utc_iso

    def executer(
        self, conversation_id: str, historique: Sequence[Message], message_infirmier: str
    ) -> ResultatTourEntretien:
        """
        Construit le message utilisateur a stocker (prefixe de
        `PROMPT_ENTRETIEN` si `historique` est vide, tel quel sinon),
        l'ajoute a l'historique existant et appelle `self.moteur.generer()`.
        Consigne systematiquement un `EntreeAudit` (F6), y compris si
        l'appel au moteur echoue (le domaine ne connait pas le type
        d'exception concret de l'adaptateur : laissee se propager, jamais
        avalee silencieusement, contrairement a
        `EvaluerBaselineZeroShotUseCase`/`ReformulerPreferenceDpoUseCase`
        ou un echec individuel est tolere dans un lot ; ici, une seule
        requete HTTP echoue pour un seul utilisateur, il n'y a pas de lot
        a proteger).
        """
        if not historique:
            contenu_utilisateur = f"{PROMPT_ENTRETIEN}\n\n{message_infirmier}"
        else:
            contenu_utilisateur = message_infirmier

        message_utilisateur = Message(role="user", contenu=contenu_utilisateur)
        messages_pour_modele = [
            {"role": m.role, "content": m.contenu} for m in (*historique, message_utilisateur)
        ]

        reponse = self.moteur.generer(
            messages_pour_modele, {"n_predict": NOMBRE_TOKENS_GENERES_ENTRETIEN}
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

        return ResultatTourEntretien(message_utilisateur=message_utilisateur, message_assistant=message_assistant)
