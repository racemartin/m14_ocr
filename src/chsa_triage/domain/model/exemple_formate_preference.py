"""
Entite ExempleFormatePreference : rendu (ChatML ou equivalent) d'un
ExemplePivot DPO, triplet prompt/chosen/rejected distinct, jamais
concatene en un seul texte (contrairement a `ExempleFormate`, SFT).
Necessaire parce que la perte DPO compare pi_theta(y_w|x) a
pi_theta(y_l|x) (et pareil pour pi_ref) : trois quantites distinctes
par exemple, cf. `domain.ports.entraineur_preference` et
docs/04_etape3_dpo/00_introduction_concepts.md §4.3.

Dataclass pure, aucune dependance externe, meme famille que
`domain.model.exemple_formate`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExempleFormatePreference:
    """Rendu texte d'un ExemplePivot DPO, triplet prompt/chosen/rejected distinct."""

    identifiant    : str   # repris de ExemplePivot.identifiant, jamais regenere
    texte_prompt    : str
    texte_chosen     : str
    texte_rejected    : str
