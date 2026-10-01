"""
Cas d'usage : deuxieme tentative, a la demande explicite de l'infirmier
(bouton dedie, jamais automatique), de convertir en JSON structure un
diagnostic deja genere mais dont le format n'a pas ete respecte
(`ResultatDiagnostic.format_respecte == False`, cf.
E4_01_uc_obtenir_diagnostic.py) avant affichage pret pour le SIH.
Reutilise le meme prompt de reformulation que la preparation des
donnees DPO (`PROMPT_REFORMULATION_CHOSEN`,
E3_00_uc_reformuler_preference_dpo.py) et le meme parseur que "obtenir
diagnostic" (`parser_diagnostic_strict`) : aucune nouvelle logique de
format. Ce prompt n'a jamais ete confirme fiable sur un run GPU reel
(cf. docstring de E3_00) : un second echec est attendu et gere, pas
une garantie de succes.

Bug reel trouve en deploiement (01/10/2026, meme incident que
E4_01_uc_obtenir_diagnostic.py) : sans `repetition_penalty` ni
contrainte de format, un essai reel a produit une longue digression
hors sujet (plusieurs paragraphes sur la fievre chez l'enfant, tronques
en fin de budget de tokens) au lieu d'une reformulation JSON. Reutilise
telles quelles `REPETITION_PENALTY_DEFAUT` et `PATRON_DIAGNOSTIC_REGEX`
de E4_01 (garde-fou NF4 + decodage contraint vLLM, meme forme cible,
meme parseur) plutot que de dupliquer ces constantes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from chsa_triage.application.use_cases.E3_00_uc_reformuler_preference_dpo import (
    MIN_TOKENS_GENERES_REFORMULATION,
    NOMBRE_TOKENS_GENERES_REFORMULATION,
    PROMPT_REFORMULATION_CHOSEN,
    TEMPERATURE_REFORMULATION,
)
from chsa_triage.application.use_cases.E4_01_uc_obtenir_diagnostic import (
    PATRON_DIAGNOSTIC_REGEX,
    REPETITION_PENALTY_DEFAUT,
    ResultatDiagnostic,
)
from chsa_triage.application.validation_diagnostic import (
    parser_diagnostic_strict,
)
from chsa_triage.domain.model.entree_audit import EntreeAudit
from chsa_triage.domain.ports.journal_audit import JournalAudit
from chsa_triage.domain.ports.moteur_inference import MoteurInference


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ReformulerDiagnosticJsonUseCase:
    """Orchestre la reformulation a la demande d'un diagnostic mal forme."""

    moteur: MoteurInference
    journal: JournalAudit
    version_modele: str = "mombasstic/chsa-triage-dpo-lora"
    horloge: Callable[[], str] = _horodatage_utc_iso

    def executer(
        self, conversation_id: str, texte_brut: str
    ) -> ResultatDiagnostic:
        """
        `texte_brut` est la reponse deja generee par "obtenir diagnostic"
        (F2/F3/F4), non conforme au format cible. Consigne
        systematiquement un `EntreeAudit` ("diagnostic_reformule"),
        que cette seconde tentative reussisse ou non : F6 exige la
        tracabilite de chaque appel au moteur d'inference.
        """
        contenu_utilisateur = f"{PROMPT_REFORMULATION_CHOSEN}\n\nReponse a reformuler :\n{texte_brut}"
        messages = [{"role": "user", "content": contenu_utilisateur}]
        parametres_generation = {
            "n_predict": NOMBRE_TOKENS_GENERES_REFORMULATION,
            "temperature": TEMPERATURE_REFORMULATION,
            "min_new_tokens": MIN_TOKENS_GENERES_REFORMULATION,
            "repetition_penalty": REPETITION_PENALTY_DEFAUT,
            "structured_outputs": {"regex": PATRON_DIAGNOSTIC_REGEX},
        }
        reponse = self.moteur.generer(messages, parametres_generation)

        diagnostic = parser_diagnostic_strict(reponse.texte)

        self.journal.consigner(
            EntreeAudit(
                horodatage=self.horloge(),
                type_evenement="diagnostic_reformule",
                conversation_id=conversation_id,
                entree=texte_brut,
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
