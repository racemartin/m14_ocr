---
title: CHSA Triage SFT Monitor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: monitoring/app_suivi_entrainement.py
pinned: false
---

Dashboard de suivi EN VIVO d'un run SFT-LoRA (perte train/validation,
verdict de convergence) pour le POC CHSA Triage. Lit en continu le
dataset de metriques publie par `training/E2_04_sft_train.py
--suivi-hf-repo <repo>` (`suivi.backend: hf_dataset`) ; ne contient
aucune donnee ni aucun code d'entrainement lui-meme.
