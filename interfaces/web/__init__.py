"""
Paquet du frontend Streamlit de TEST de l'entretien clinique (Etape 4),
contre l'API FastAPI reelle (`interfaces/api/`) : `app_test_inference.py`
(Streamlit) + `logica_test_inference.py` (logique pure, testable sans
Streamlit ni reseau, meme discipline que
`monitoring/logica_suivi_entrainement.py`).

Architecture a 2 pieces (decision produit du 24/09/2026, cf. AGENTS.md) :
ce frontend s'execute EN LOCAL sur la machine de l'operateur humain,
jamais comme Space HF separe (creer un Space Docker rien que pour
Streamlit exigerait un abonnement HF PRO ou du hardware payant, pour
un frontend qui ne fait qu'appeler l'API en HTTP). Il cible le Space
HF Docker/GPU qui sert l'API+vLLM, cf. `interfaces/web/README_space.md`
et README.md §4.5 pour la commande d'execution reelle.
"""

from __future__ import annotations
