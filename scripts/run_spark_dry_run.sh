#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <processed_csv_path> <output_dir> [config_path]"
  exit 1
fi

PROCESSED_CSV="$1"
OUTPUT_DIR="$2"
CONFIG_PATH="${3:-}"

if [[ -n "${CONFIG_PATH}" ]]; then
  PYTHONPATH=src python3 -m gnss_risk.spark_job --dry-run --processed-csv "${PROCESSED_CSV}" --output-dir "${OUTPUT_DIR}" --config "${CONFIG_PATH}"
else
  PYTHONPATH=src python3 -m gnss_risk.spark_job --dry-run --processed-csv "${PROCESSED_CSV}" --output-dir "${OUTPUT_DIR}"
fi
