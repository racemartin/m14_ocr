"""
Validation STRICTE de la sortie "obtenir diagnostic" (F2/F3/F4 : bloc
`<think>...</think>` suivi d'un JSON strict `{niveau, categorie,
ressources_estimees}`). Fonction pure, aucun port. Reutilise
`CLES_REFORMULATION_ATTENDUES` de `validation_reformulation_dpo.py`
mais reste distincte : ici l'appelant (l'API live) a besoin des champs
individuellement (ex. badge ESI colore), pas du texte brut validé.
"""

from __future__ import annotations

import json
import re

from chsa_triage.application.validation_reformulation_dpo import (
    CLES_REFORMULATION_ATTENDUES,
)
from chsa_triage.domain.model.diagnostic_clinique import DiagnosticClinique

_MOTIF_BLOC_THINK = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def parser_diagnostic_strict(texte: str) -> DiagnosticClinique | None:
    """
    Extrait le bloc `<think>...</think>` puis parse le reste comme JSON
    strict avec exactement les cles `niveau`/`categorie`/`ressources_estimees`.
    Retourne `None` (jamais une exception) des que le format n'est pas
    respecte : bloc `<think>` absent/vide, JSON invalide, cle
    manquante/en trop, ou `niveau` non entier. L'appelant (use case
    "obtenir diagnostic") doit consigner le texte brut au journal
    d'audit MEME quand cette fonction retourne `None` : F6 exige la
    tracabilite de l'appel, pas seulement des reponses bien formees.
    """
    correspondance = _MOTIF_BLOC_THINK.match(texte.strip()) if texte else None
    if correspondance is None:
        return None

    raisonnement = correspondance.group(1).strip()
    if not raisonnement:
        return None

    reste = correspondance.group(2).strip()
    try:
        objet = json.loads(reste)
    except json.JSONDecodeError:
        return None

    if not isinstance(objet, dict):
        return None
    if set(objet.keys()) != CLES_REFORMULATION_ATTENDUES:
        return None

    niveau = objet["niveau"]
    if not isinstance(niveau, int) or isinstance(niveau, bool):
        return None

    categorie = objet["categorie"]
    ressources_estimees = objet["ressources_estimees"]
    if not isinstance(categorie, str) or not isinstance(
        ressources_estimees, str
    ):
        return None

    return DiagnosticClinique(
        raisonnement=raisonnement,
        niveau=niveau,
        categorie=categorie,
        ressources_estimees=ressources_estimees,
    )
