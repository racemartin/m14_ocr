"""
Paquet du frontend Streamlit de TEST de l'entretien clinique (Etape 4),
contre l'API FastAPI reelle (`interfaces/api/`) : `app_test_inference.py`
(Streamlit) + `logica_test_inference.py` (logique pure, testable sans
Streamlit ni reseau, meme discipline que
`monitoring/logica_suivi_entrainement.py`).

Deploiement cible (jamais realise, decision produit actee) : un Space HF
Streamlit/CPU separe du Space HF Docker/GPU qui sert l'API+vLLM, cf.
`interfaces/web/README_space.md` pour son frontmatter et
`interfaces/web/requirements.txt` pour ses dependances.
"""

from __future__ import annotations
