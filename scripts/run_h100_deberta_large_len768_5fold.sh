#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python -u -m aes2.train_transformer \
  --model-name models/hf/microsoft_deberta-v3-large \
  --run-name deberta-v3-large-len768-5fold \
  --local-files-only \
  --fold -1 \
  --max-length 768 \
  --batch-size 8 \
  --grad-accum 2 \
  --epochs 4 \
  --lr 8e-6 \
  --fp16
