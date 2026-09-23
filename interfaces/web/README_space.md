---
title: CHSA Triage Test Chat
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: interfaces/web/app_test_inference.py
pinned: false
---

Space Streamlit/CPU, leger et bon marche, laisse allume en permanence
(a la difference du Space Docker/GPU compagnon, `README_space_gpu.md`
a la racine du depot, couteux et allume seulement pendant les tests).
Interface de test de l'entretien clinique CHSA Triage : sonde `GET
/sante` de l'API en boucle et affiche automatiquement l'interface de
chat des que le Space GPU devient disponible, sans synchronisation
manuelle des deux demarrages.

Ne contient aucun code d'entrainement ni d'inference lui-meme : parle
uniquement HTTP a l'API FastAPI reelle (`interfaces/api/`).

Secrets du Space a definir avant publication :
- `CHSA_API_URL_BASE` : URL du Space Docker/GPU compagnon.
- `CHSA_API_CLE` : meme cle que `CHSA_CLE_API_DEMO` cote API.

Publication (jamais realisee dans cette tache, effet externe reel,
action reservee a l'operateur humain) : uploader `interfaces/web/app_test_inference.py`,
`interfaces/web/logica_test_inference.py`, ce fichier (renomme
`README.md` a la racine du Space) et `interfaces/web/requirements.txt`
(renomme `requirements.txt` a la racine du Space) vers le depot HF
Space cible, meme patron que `monitoring/README_space.md`.
