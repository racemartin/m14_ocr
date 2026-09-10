"""
Port de persistance des decisions humaines sur les candidats de PII
residuelle marques `VERDICT_REVISION_HUMAINE` par
`controler_qualite_anonymisation.py` (ni le regex ni la seconde
opinion spaCy ne tranchent).

Avant ce port, `CandidatPiiResiduelle.verdict` etait uniquement un
calcul automatique, jamais une decision de personne -- ce port ferme
cet ecart pour l'exigence NF2 du cahier des charges (anonymisation
"validee manuellement"). Une decision est identifiee par une cle
stable (cf. `domain.model.CleCandidatRevision`) plutot que par une
position dans une liste, pour rester valide meme si le meme candidat
reapparait dans une execution ulterieure.
"""

from __future__ import annotations

from typing import Protocol

from chsa_triage.domain.model import CleCandidatRevision, DecisionRevisionHumaine


class RegistreDecisionsRevisionHumaine(Protocol):
    """Port de lecture/ecriture des decisions humaines, indexees par cle stable de candidat."""

    def toutes(self) -> list[DecisionRevisionHumaine]:
        """Retourne toutes les decisions enregistrees."""
        ...

    def cles_decidees(self) -> set[CleCandidatRevision]:
        """Retourne l'ensemble des cles de candidats ayant deja une decision (sans reconstruire chaque decision)."""
        ...

    def trouver(self, cle: CleCandidatRevision) -> DecisionRevisionHumaine | None:
        """Retourne la decision associee a `cle`, ou None si aucune."""
        ...

    def enregistrer(self, decision: DecisionRevisionHumaine) -> None:
        """Persiste `decision` immediatement (remplace toute decision anterieure de meme cle)."""
        ...
