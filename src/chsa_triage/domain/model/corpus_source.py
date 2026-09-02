"""
Entite CorpusSource : decrit une source de donnees brute avant
conversion vers le schema pivot (ExemplePivot).

Aucune dependance externe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chsa_triage.domain.model.enums import Langue


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """
    Represente l'un des quatre corpus medicaux d'origine de la mission
    (MediQAl, FrenchMedMCQA, MedQuAD, UltraMedical-Preference).

    Cette entite ne connait rien du format de fichier ni de la
    bibliotheque utilisee pour le charger : elle decrit seulement
    l'identite et les caracteristiques du corpus.
    """

    nom                : str             # Identifiant court, ex. "MediQAl"
    identifiant_hub     : str             # Chemin Hugging Face Hub, ex. "ANR-MALADES/MediQAl"
    langue              : Langue          # Langue dominante du corpus
    licence             : str             # Licence declaree (auditabilite)
    url_documentation   : str = ""        # Lien vers la fiche/carte du dataset
    notes               : dict = field(default_factory=dict)  # Metadonnees libres
