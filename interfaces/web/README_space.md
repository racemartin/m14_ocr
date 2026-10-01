# `interfaces/web/` : pas de Space HF dédié

Ce dossier n'a pas de Space HF Streamlit/CPU dédié : créer un Space
**Docker** rien que pour Streamlit exigerait soit un abonnement HF PRO
(sur le hardware gratuit `cpu-basic`), soit du hardware payant, alors
que `app_test_inference.py` ne fait qu'appeler l'API FastAPI
(`interfaces/api/`) en HTTP, sans aucun calcul GPU.

Depuis le 01/10/2026, ce frontend est servi directement par l'unique
Space Docker/GPU existant (API+vLLM, cf. `deploy/space_gpu_api_vllm/`),
seul process branché sur son unique port public (`demarrer.sh`) : pas
de deuxième Space, pas de nouvel abonnement. Il reste aussi possible de
le lancer en local, pointé vers l'API de ce même Space (avant le
01/10/2026, c'était la seule façon de l'utiliser).

Instructions d'exécution réelles : voir `README_AVEC_DETAILS.md` §4.5
à la racine du dépôt.
