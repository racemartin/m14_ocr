"""
Entite ChosenReformule : chosen reformule vers <think>+JSON pour un
ExemplePivot DPO donne (decision tranchee en
docs/04_etape3_dpo/00_introduction_concepts.md §3.2 : seul `chosen` est
reformule, jamais `rejected`). Produite par
`ReformulerPreferenceDpoUseCase`, persistee dans un fichier derive
separe (`data/processed/dataset_dpo_chosen_reformule.jsonl`), jamais
une mutation du pivot original (meme principe que le pivot/anonymise
deja en place, cf. AGENTS.md).

Dataclass pure, aucune dependance externe.
"""

from __future__ import annotations

from dataclasses import dataclass

from chsa_triage.domain.model.exemple_pivot import Message


@dataclass(frozen=True, slots=True)
class ChosenReformule:
    """Chosen reformule vers <think>+JSON pour un ExemplePivot DPO donne."""

    identifiant       : str               # repris de ExemplePivot.identifiant, jamais regenere
    chosen_reformule    : tuple[Message, ...]  # nouveau tour assistant, remplace chosen a l'usage
    horodatage            : str
