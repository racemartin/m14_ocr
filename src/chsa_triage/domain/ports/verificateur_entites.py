"""
Port generique de "seconde opinion" sur un passage de texte suspecte
de contenir une entite nommee (personne, organisation, lieu).

Utilise par le controle qualite d'anonymisation (cf.
`E1_04_02_controler_qualite_anonymisation.py`) pour trancher les candidats de
PII residuelle trouves par une heuristique regex sur le texte deja
anonymise : le regex seul confond frequemment un terme medical
capitalise ("Polycystic", "Mitochondrial Myopathy") avec un nom
propre. Ce port ne sait rien de spaCy ; il expose seulement un
verdict a 3 etats.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class VerdictEntiteNommee(str, Enum):
    """Verdict de la seconde opinion sur un passage [debut:fin] d'un texte."""

    # Une entite PERSON/ORGANIZATION/LOCATION (ou equivalent) chevauche
    # le passage : le candidat est probablement une vraie entite nommee.
    ENTITE_PERTINENTE = "entite_pertinente"
    # Aucune entite nommee ne chevauche le passage : le regex a
    # vraisemblablement confondu un terme non-PII (souvent medical) avec
    # un nom propre.
    AUCUNE_ENTITE = "aucune_entite"
    # Une entite chevauche le passage mais d'un type non pertinent pour
    # la PII (ex. DATE, PRODUCT, MISC) : le verdict n'est pas tranche.
    ENTITE_NON_PERTINENTE = "entite_non_pertinente"


class VerificateurEntitesNommees(Protocol):
    """Port generique de verification d'entite nommee sur un passage de texte."""

    def verifier(self, texte: str, langue: str, debut: int, fin: int) -> VerdictEntiteNommee:
        """Indique si le passage [debut:fin] de `texte` chevauche une entite nommee."""
        ...
