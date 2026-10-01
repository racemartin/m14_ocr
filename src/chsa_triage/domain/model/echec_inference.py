"""
Exception de domaine levee quand `MoteurInference.generer()` echoue
(timeout, 500 serveur, reponse malformee cote vLLM/llama.cpp), dans le
contexte requete-unique de l'API (Etape 4). Par convention, elle n'est
levee qu'APRES que l'appelant a deja consigne un `EntreeAudit`
(`type_evenement="echec_inference"`) : F6 exige la tracabilite meme des
echecs, jamais un echec d'inference silencieusement perdu.
"""

from __future__ import annotations


class EchecInferenceError(Exception):
    """L'appel au moteur d'inference a echoue ; deja consigne au journal d'audit."""
