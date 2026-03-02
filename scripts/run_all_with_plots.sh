#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/storm_may2024.json}"
OUTPUT_ROOT="${2:-outputs}"
RUN_GPU="${RUN_GPU:-1}"

if [[ -f ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

export PYTHONPATH=src

"$PY" scripts/run_pipeline.py --config "$CONFIG_PATH" --generate-synthetic --output-root "$OUTPUT_ROOT"

RUN_DIR="$OUTPUT_ROOT/latest"
GPU_PRED_ARG=()

if [[ "$RUN_GPU" == "1" ]]; then
  if "$PY" -c "import xgboost" >/dev/null 2>&1; then
    "$PY" -m gnss_risk.gpu_train \
      --dataset-csv "$RUN_DIR/dataset_labeled.csv" \
      --output-dir "$RUN_DIR/gpu" \
      --allow-cpu-fallback

    if [[ -f "$RUN_DIR/gpu/predictions_gpu_xgboost.csv" ]]; then
      GPU_PRED_ARG=(--gpu-predictions "$RUN_DIR/gpu/predictions_gpu_xgboost.csv")
    fi
  else
    echo "xgboost not installed; skipping GPU model."
  fi
fi

"$PY" scripts/plot_metrics_and_comparison.py --run-dir "$RUN_DIR" "${GPU_PRED_ARG[@]}"

echo "All artifacts ready in: $RUN_DIR"
