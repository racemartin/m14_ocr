"""
Entites du controle qualite d'anonymisation qui ne peuvent PAS etre
calculees automatiquement : la decision d'une personne sur un candidat
de PII residuelle que ni le regex ni la seconde opinion spaCy ne
tranchent (`VERDICT_REVISION_HUMAINE`, cf.
`application.use_cases.E1_04_02_controler_qualite_anonymisation`).

Aucune dependance externe ; comme le reste de `domain.model`, ces
entites ne savent rien du format de fichier utilise pour les
persister (cf. `infrastructure.adapters.jsonl_decisions_revision_humaine`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# Les 3 listes de candidats de PII residuelle produites par
# `ControleQualiteAnonymisation` (cf. E1_04_02_controler_qualite_anonymisation) ; un candidat n'existe
# que dans UNE de ces trois listes, jamais plusieurs a la fois.
SOURCE_CANDIDATS_PII = "candidats_pii"
SOURCE_CANDIDATS_FAUX_POSITIFS = "candidats_faux_positifs"
SOURCE_CANDIDATS_PII_SANS_ENTITE = "candidats_pii_sans_entite"

# Decision humaine sur un candidat : "accepte" = confirme que ce n'est
# PAS une PII reelle (faux positif humain, pas seulement regex/spaCy) ;
# "rejete" = confirme que c'EST une fuite / qu'il faut agir dessus.
DECISION_ACCEPTE = "accepte"
DECISION_REJETE = "rejete"


class CleCandidatRevision(NamedTuple):
    """Cle stable d'un candidat de PII residuelle, independante de sa
    position dans la liste. `debut`/`fin` desambiguisent plusieurs
    matches du meme `type_motif` dans le meme `champ` (vu en pratique :
    jusqu'a 17 matches dans un seul champ). `type_motif` vaut "" pour
    un faux positif (fragment brut, pas de regex typee)."""

    source_liste: str
    identifiant: str
    champ: str
    type_motif: str
    debut: int
    fin: int


@dataclass(frozen=True, slots=True)
class DecisionRevisionHumaine:
    """Decision d'une personne sur un candidat de PII residuelle identifie par sa cle stable."""

    cle: CleCandidatRevision
    passage: (
        str  # informationnel (affichage/audit) ; ne fait PAS partie de la cle
    )
    decision: str  # DECISION_ACCEPTE | DECISION_REJETE
    horodatage: str
    note: str = ""
