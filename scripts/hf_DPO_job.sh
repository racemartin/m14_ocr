#!/bin/bash
hf jobs uv run \
    --flavor l4x1 \
    --timeout 2h \
    --with "chsa-triage[remote] @ git+https://github.com/racemartin/m14_ocr.git@main" \
    --secrets HF_TOKEN \
    -v hf://datasets/mombasstic/chsa-triage-dpo-train-data:/mnt/train-data \
    https://raw.githubusercontent.com/racemartin/m14_ocr/main/training/E3_03_dpo_train.py \
    --recette recipes/dpo_qwen3_lora.yaml \
    --dataset /mnt/train-data/dataset_chsa_triage_dpo_anonymise_100.jsonl \
    --suivi-hf-repo mombasstic/chsa-triage-dpo-metrics \
    --checkpoint-hf-repo mombasstic/chsa-triage-dpo-lora \
    --skip-reformulation