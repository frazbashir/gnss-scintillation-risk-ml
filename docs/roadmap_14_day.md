# 14-Day Delivery Plan

## Day 1-2
- Confirm data access credentials and endpoint health for EarthScope, CDDIS, WAAS, ARTEMIS, SuperMAG.
- Freeze schema and timestamp conventions in UTC.
- Validate storm window and quiet reference window definitions.

## Day 3-4
- Implement source-specific ingestion scripts and metadata checks.
- Execute first full raw data pull for May 9-13, 2024.
- Generate ingestion QA report (coverage, gaps, duplicates).

## Day 5-6
- Build unified timeline at 5-minute cadence.
- Add missing-data handling and source lag alignment assumptions.
- Produce first processed dataset with lag and rolling features.

## Day 7
- Finalize VPL threshold policy (fixed operational threshold or approved interim percentile).
- Freeze label generation code and version the labeled dataset.

## Day 8-9
- Train baseline logistic regression.
- Train main random-forest/boosting model.
- Tune minimal hyperparameters under time-series split constraints.

## Day 10
- Run lead-time experiments for 10, 30, 60 minutes.
- Compare performance decay vs lead time.

## Day 11
- Produce required figures:
  - VPL and predicted risk probability time series
  - ROC and PR curves
  - Optional station panel/map summary

## Day 12
- Package reproducible run scripts and Spark submit job.
- Run one full end-to-end batch in GDX Spark environment.

## Day 13
- Prepare results brief with methods, assumptions, and limitations.
- Add ablation summary (baseline vs main model and key features).

## Day 14
- Final QA and handoff:
  - code, config, data dictionary, runbook, and metrics snapshot
  - risks, next iteration items, and required approvals
