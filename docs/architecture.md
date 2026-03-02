# Architecture

## Modules

- `ingest`: fetch or synthesize source streams.
- `preprocess`: convert to uniform cadence and build lag/rolling features.
- `labels`: create `VPL_exceed` labels from configured threshold logic.
- `models`: train baseline logistic and main ensemble model.
- `eval`: evaluate with ROC-AUC, PR-AUC, confusion matrix, F1.
- `report`: summarize outputs into markdown and JSON files.
- `spark_job`: distributed batch path for GDX Spark.

## Decision Variables

Primary label:
- `VPL_exceed(t) = 1 if VPL(t) > threshold else 0`

Lead-time labels:
- `VPL_exceed(t + Delta)` for `Delta in {10, 30, 60}` minutes.
