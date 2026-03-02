#!/usr/bin/env bash
set -euo pipefail

DATASET_CSV="${1:-outputs/latest/dataset_labeled.csv}"
OUTPUT_DIR="${2:-outputs/gpu}"

if [[ -f ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

PYTHONPATH=src "$PY" -m gnss_risk.gpu_train \
  --dataset-csv "$DATASET_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --allow-cpu-fallback
