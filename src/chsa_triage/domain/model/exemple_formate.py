"""
Entite ExempleFormate : rendu ChatML complet d'un ExemplePivot, produit
par un FormateurConversation (cf. domain.ports.formateur_conversation)
et consomme par EntraineurSupervise.entrainer().

Dataclass pure, aucune dependance externe (pas de transformers ici :
l'appel a `apply_chat_template` est un souci d'infrastructure), meme
regle que le reste de `domain.model`.

`tours` porte les bornes (caracteres, dans `texte`) de chaque tour du
rendu ChatML : necessaire pour que `TrlSftEntraineurAdapter` puisse
masquer la perte aux tours assistant uniquement
(`assistant_only_loss=True`, cf. backlog
`m14-ocr-assistant-only-loss-estructurado`). Vide par defaut
(`()`) pour rester compatible avec tout `ExempleFormate` construit
avant cette entite (`texte` seul, perte pleine sequence) : un `tours`
vide signifie "bornes inconnues", jamais "aucun tour".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LimiteTour:
    """
    Bornes (caracteres, demi-ouvert [debut, fin[) d'un seul tour ChatML
    dans `ExempleFormate.texte`. `role` reprend `Message.role`
    ("system" | "user" | "assistant").

    Construites par decoupage de prefixes successifs du rendu ChatML
    (cf. `ChatMLFormateurAdapter.formater`) plutot que par un second
    appel a `apply_chat_template` sur un sous-ensemble contenant un
    tour assistant : ce dernier strippe silencieusement les blocs
    `<think>` d'un tour assistant non-final (bug reel trouve et corrige
    cote DPO, cf. `ChatMLFormateurAdapter._rendre_tour_assistant_seul`).
    Les tours `prompt` (system/user, jamais assistant) sont toujours
    surs a decouper par prefixes `apply_chat_template` successifs ; le
    tour `completion` final est sur par construction (toujours le
    dernier tour du rendu complet, jamais strippe par ce bug).
    """

    role: str
    debut: int
    fin: int


@dataclass(frozen=True, slots=True)
class ExempleFormate:
    """Rendu ChatML d'un ExemplePivot, pret a etre tokenize pour l'entrainement SFT."""

    identifiant: str  # repris de ExemplePivot.identifiant, jamais regenere
    texte: str  # rendu ChatML complet (system+user+assistant)
    tours: tuple[LimiteTour, ...] = ()  # bornes par tour ; () si inconnues

    def textes_assistant(self) -> tuple[str, ...]:
        """Sous-chaines de `texte` correspondant aux tours
        `role == "assistant"`, dans l'ordre. Utilise par
        `TrlSftEntraineurAdapter` pour construire les `labels` masques
        quand `assistant_only_loss=True`."""
        return tuple(
            self.texte[tour.debut : tour.fin]
            for tour in self.tours
            if tour.role == "assistant"
        )
