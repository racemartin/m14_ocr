"""
Cas d'usage : "obtenir le diagnostic" (F2/F3/F4). Declenche par un
bouton explicite cote infirmier, jamais par le modele de facon autonome
(cf. `E4_00_uc_poursuivre_entretien.py`) : envoie tout l'historique plus
un prompt demandant le format cible (`<think>` + JSON), en UN SEUL
appel. Orchestre generation + validation + audit, aucune logique de
parsing propre (reutilise `parser_diagnostic_strict`).

TODO NF4 (garde-fou de securite clinique, decision produit encore
ouverte) : le cahier des charges exige qu'une reponse jugee peu sure
par un futur "juge LLM" voie son score force a 0 (rejet). Ce controle
N'EST PAS implemente ici ; le resultat brut est retourne tel quel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from chsa_triage.application.validation_diagnostic import (
    parser_diagnostic_strict,
)
from chsa_triage.domain.model.diagnostic_clinique import DiagnosticClinique
from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.ports.journal_audit import JournalAudit
from chsa_triage.domain.ports.moteur_inference import MoteurInference

PROMPT_DIAGNOSTIC = (
    "Sur la base de l'entretien ci-dessus, produis maintenant le "
    "diagnostic de triage. Reponds STRICTEMENT sous cette forme, rien "
    "d'autre avant ou apres :\n"
    "<think>quelques phrases de raisonnement clinique s'appuyant sur "
    "l'entretien</think>"
    '{"niveau": <un entier d\'echelle ESI 1-5>, "categorie": "<categorie '
    'clinique courte>", "ressources_estimees": "<ressources/examens '
    'vraisemblables>"}'
)

# Garde-fou de securite clinique (NF4), pas un reglage de style : voir
# E4_00_uc_poursuivre_entretien.py pour l'incident reel (degenerescence
# de langue, reponse dangereuse) qui justifie ces deux constantes,
# critique d'autant plus ici que ce cas d'usage produit le diagnostic
# final presente a l'infirmier.
REPETITION_PENALTY_DEFAUT = 1.2
TEMPERATURE_DEFAUT = 0.0

NOMBRE_TOKENS_GENERES_DIAGNOSTIC = 512


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ResultatDiagnostic:
    """
    Sortie de `ObtenirDiagnosticUseCase.executer()`. `diagnostic` est
    `None` quand le modele n'a pas respecte le format cible :
    `texte_brut`/`format_respecte` restent disponibles pour l'appelant
    (jamais un echec silencieux, F6 exige la tracabilite meme d'une
    sortie mal formee).
    """

    diagnostic: DiagnosticClinique | None
    texte_brut: str
    format_respecte: bool


@dataclass(slots=True)
class ObtenirDiagnosticUseCase:
    """Orchestre l'appel diagnostic : historique complet -> classification ESI structuree."""

    moteur: MoteurInference
    journal: JournalAudit
    version_modele: str = "mombasstic/chsa-triage-dpo-lora"
    horloge: Callable[[], str] = _horodatage_utc_iso

    def executer(
        self, conversation_id: str, historique: Sequence[Message]
    ) -> ResultatDiagnostic:
        """
        `historique` doit deja contenir au moins un tour (l'API rejette
        une conversation vide avant d'appeler ce cas d'usage : demander
        un diagnostic sans aucun echange n'a pas de sens clinique).
        Consigne systematiquement un `EntreeAudit` (F6), que le format
        cible ait ete respecte ou non.
        """
        messages_pour_modele = [
            {"role": m.role, "content": m.contenu} for m in historique
        ] + [{"role": "user", "content": PROMPT_DIAGNOSTIC}]

        reponse = self.moteur.generer(
            messages_pour_modele,
            {
                "n_predict": NOMBRE_TOKENS_GENERES_DIAGNOSTIC,
                "repetition_penalty": REPETITION_PENALTY_DEFAUT,
                "temperature": TEMPERATURE_DEFAUT,
            },
        )

        diagnostic = parser_diagnostic_strict(reponse.texte)

        self.journal.consigner(
            EntreeAudit(
                horodatage=self.horloge(),
                type_evenement="diagnostic",
                conversation_id=conversation_id,
                entree=json.dumps(
                    [
                        {"role": m.role, "contenu": m.contenu}
                        for m in historique
                    ],
                    ensure_ascii=False,
                ),
                sortie=reponse.texte,
                version_modele=self.version_modele,
                metadonnees={"format_respecte": diagnostic is not None},
            )
        )

        return ResultatDiagnostic(
            diagnostic=diagnostic,
            texte_brut=reponse.texte,
            format_respecte=diagnostic is not None,
        )
