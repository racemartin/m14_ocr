"""
Validation STRICTE du format de sortie attendu apres reformulation d'un
`chosen` DPO (cahier des charges F3-F4 : `<think>...</think>` suivi
d'un JSON strict `{niveau, categorie, ressources_estimees}`). Fonction
pure, aucun port ; tout echec de format retourne `None`, jamais une
exception, pour ecarter un exemple degenere sans faire echouer le lot.

Plus stricte que `metriques_evaluation_baseline.py::extraire_json`
(qui tolere un JSON noye dans du texte libre, pour evaluer une
generation zero-shot) : ici la sortie sera persistee comme donnee
d'entrainement DPO, elle doit deja respecter le format au mot pres.
"""

from __future__ import annotations

import json
import re

from chsa_triage.domain.model.exemple_pivot import Message

# Cles EXACTES attendues dans le JSON cible (F3-F4 du cahier des
# charges) : ni cle manquante, ni cle en trop.
CLES_REFORMULATION_ATTENDUES = frozenset(
    {"niveau", "categorie", "ressources_estimees"}
)

_MOTIF_BLOC_THINK = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def parser_reformulation_stricte(texte: str) -> tuple[Message, ...] | None:
    """
    Extrait le bloc `<think>...</think>` puis parse le reste comme JSON
    strict avec exactement les cles `niveau`/`categorie`/`ressources_estimees`
    (cahier des charges F3-F4). Retourne `None` (jamais une exception)
    des que le format n'est pas respecte : bloc `<think>` absent/vide,
    JSON invalide, cle manquante ou en trop. Sinon, retourne le nouveau
    tour assistant reformule, `texte` inchange (le format valide EST
    directement le contenu cible), pret a remplacer `chosen` dans
    `ChosenReformule.chosen_reformule`.
    """
    correspondance = _MOTIF_BLOC_THINK.match(texte.strip()) if texte else None
    if correspondance is None:
        return None

    contenu_think = correspondance.group(1).strip()
    if not contenu_think:
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

    return (Message(role="assistant", contenu=texte.strip()),)
