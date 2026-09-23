# `interfaces/web/` : pas de Space HF dédié

Ce dossier n'a plus de Space HF Streamlit/CPU compagnon (décision
produit du 24/09/2026) : créer un Space **Docker** rien que pour
Streamlit exige soit un abonnement HF PRO (sur le hardware gratuit
`cpu-basic`), soit du hardware payant, alors que
`app_test_inference.py` ne fait qu'appeler l'API FastAPI
(`interfaces/api/`) en HTTP, sans aucun calcul GPU. Architecture
retenue, à 2 pièces : un unique Space Docker/GPU (API+vLLM, cf.
`deploy/space_gpu_api_vllm/`) et ce frontend Streamlit exécuté en
LOCAL sur la machine de l'opérateur humain, jamais déployé comme
Space séparé.

Instructions d'exécution réelles : voir `README.md` §4.5 à la racine
du dépôt.
