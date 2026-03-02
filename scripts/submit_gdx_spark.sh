#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <project_root_uri> <config_uri> <output_uri> [processed_csv_uri]"
  echo "Example:"
  echo "  $0 gs://my-bucket/gnss-scintillation-risk-ml gs://my-bucket/gnss-scintillation-risk-ml/configs/storm_may2024.json gs://my-bucket/gnss-scintillation-risk-ml/outputs/spark gs://my-bucket/gnss-scintillation-risk-ml/outputs/latest/dataset_labeled.csv"
  exit 1
fi

PROJECT_ROOT_URI="$1"
CONFIG_URI="$2"
OUTPUT_URI="$3"
PROCESSED_CSV_URI="${4:-${PROJECT_ROOT_URI}/outputs/latest/dataset_labeled.csv}"

# Adjust master/deploy settings to your GDX Spark environment.
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  "${PROJECT_ROOT_URI}/src/gnss_risk/spark_job.py" \
  --config "${CONFIG_URI}" \
  --processed-csv "${PROCESSED_CSV_URI}" \
  --output-dir "${OUTPUT_URI}"
