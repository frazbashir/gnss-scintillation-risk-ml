# GNSS Scintillation and Positioning-Risk ML Framework

Repository for a bounded early-warning ML framework focused on the May 2024 geomagnetic storm (primary window: 2024-05-10 to 2024-05-12, with pre/post baseline context).

## Goal

Predict GNSS integrity degradation using binary classification:

- `VPL_exceed = 1` when `VPL > threshold`
- `VPL_exceed = 0` otherwise

Primary target: operational risk classification.
Secondary targets (future): VPL regression and position error regression.

## Data Sources in Scope

- EarthScope GNSS receiver observations
- CDDIS ephemeris inputs (for position/quality context)
- WAAS VPL time series
- ARTEMIS solar wind drivers
- SuperMAG geomagnetic activity

## Repository Layout

- `src/gnss_risk/ingest`: source adapters and synthetic data generator
- `src/gnss_risk/preprocess`: time alignment, lag and rolling feature creation
- `src/gnss_risk/labels`: VPL threshold labeling
- `src/gnss_risk/models`: baseline logistic regression and random-forest-lite model
- `src/gnss_risk/eval`: ROC-AUC, PR-AUC, F1, confusion matrix, lead-time evaluation
- `src/gnss_risk/report`: markdown/json report generation
- `src/gnss_risk/pipeline.py`: offline/local end-to-end pipeline
- `src/gnss_risk/spark_job.py`: Spark execution entrypoint for cluster runs
- `scripts/submit_gdx_spark.sh`: template submit command for GDX Spark

## Quick Start (Offline, No External Dependencies)

```bash
cd gnss-scintillation-risk-ml
python3 scripts/run_pipeline.py --config configs/storm_may2024.json --generate-synthetic
```

This will:

1. Generate synthetic source files matching the target schema.
2. Build aligned, lagged, and rolled features.
3. Train baseline and main models.
4. Evaluate nowcast and lead times (10/30/60 min).
5. Write outputs under `outputs/<run_id>/`.

## Full Environment (Online)

When internet/package access is available:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Spark Path (GDX)

Use the provided submit template:

```bash
bash scripts/submit_gdx_spark.sh \
  gs://<bucket>/gnss-scintillation-risk-ml \
  gs://<bucket>/gnss-scintillation-risk-ml/configs/storm_may2024.json \
  gs://<bucket>/gnss-scintillation-risk-ml/outputs/spark
```

## Output Artifacts

Each run writes:

- `metrics_nowcast.json`
- `metrics_lead_10.json`, `metrics_lead_30.json`, `metrics_lead_60.json`
- `predictions_nowcast.csv`
- `summary.md`
- `model_logistic.json`
- `model_forest_lite.json`

## GPU Training (Single Node)

If your GDX VM has an NVIDIA GPU and CUDA driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip xgboost
```

Generate the dataset and run GPU training:

```bash
PYTHONPATH=src python3 scripts/run_pipeline.py --config configs/storm_may2024.json --generate-synthetic
bash scripts/run_gpu_training.sh outputs/latest/dataset_labeled.csv outputs/gpu
```

Expected outputs:

- `outputs/gpu/metrics_gpu_xgboost.json`
- `outputs/gpu/predictions_gpu_xgboost.csv`

## Scope Protection

- No deep learning in this phase.
- No multi-storm generalization in this phase.
- No dashboard build in this phase.
- Minimal reliable pipeline first.
