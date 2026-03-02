from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from gnss_risk.config import load_config
from gnss_risk.eval.metrics import evaluate_classification
from gnss_risk.ingest.downloaders import attempt_download_sources
from gnss_risk.ingest.synthetic import generate_synthetic_sources
from gnss_risk.labels.vpl import apply_vpl_labels, determine_vpl_threshold
from gnss_risk.models.forest_lite import RandomForestLite
from gnss_risk.models.logistic import LogisticRegressionGD
from gnss_risk.preprocess.alignment import align_and_engineer, write_records_csv
from gnss_risk.report.generate import write_json, write_predictions_csv, write_summary_markdown

REQUIRED_RAW_FILES = [
    "artemis_solar_wind.csv",
    "supermag.csv",
    "waas_vpl.csv",
    "earthscope_receiver.csv",
    "cddis_ephemeris.csv",
]


def _has_minimum_raw_data(raw_dir: Path) -> bool:
    return all((raw_dir / f).exists() for f in REQUIRED_RAW_FILES)


def _select_feature_sets(records: List[Dict], config: Dict) -> Tuple[List[str], List[str]]:
    first = records[0]

    numeric_keys = [
        key
        for key, value in first.items()
        if isinstance(value, (int, float))
    ]

    def keep_feature(name: str) -> bool:
        blocked = (
            name == "label"
            or name == "position_error_m"
            or name == "timestamp"
            or name.startswith("vpl")
            or name.startswith("d_vpl")
            or "_vpl_" in name
        )
        return not blocked

    candidates = [k for k in numeric_keys if keep_feature(k)]

    baseline_order = [
        "bz",
        "vsw",
        "pdyn",
        "sml",
        "d_bz_dt",
        "d_sml_dt",
        "bz_lag_10m",
        "vsw_lag_30m",
        "sml_lag_30m",
    ]
    baseline = [k for k in baseline_order if k in candidates]

    if len(baseline) < 4:
        baseline = candidates[: min(6, len(candidates))]

    main_base = [k for k in config["features"]["base"] if k in candidates]

    derived = [
        k
        for k in candidates
        if (
            k.startswith("d_")
            or "_lag_" in k
            or "_roll_mean_" in k
            or "_roll_std_" in k
        )
    ]

    main_features = sorted(set(main_base + derived))
    if len(main_features) < len(baseline):
        main_features = sorted(set(candidates))

    return baseline, main_features


def _build_supervised_rows(records: List[Dict], lead_steps: int) -> List[Dict]:
    if lead_steps < 0:
        raise ValueError("lead_steps must be non-negative")

    rows: List[Dict] = []
    usable = len(records) - lead_steps
    for i in range(max(0, usable)):
        src = records[i]
        target = records[i + lead_steps]
        rows.append(
            {
                "timestamp": src["timestamp"],
                "target_timestamp": target["timestamp"],
                "target_label": int(target["label"]),
                "target_vpl": float(target["vpl"]),
                "features": src,
            }
        )
    return rows


def _extract_matrix(rows: List[Dict], feature_names: List[str]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for row in rows:
        feature_row = row["features"]
        matrix.append([float(feature_row[name]) for name in feature_names])
    return matrix


def _extract_target(rows: List[Dict]) -> List[int]:
    return [int(row["target_label"]) for row in rows]


def _split_time_ordered(rows: List[Dict], train_fraction: float) -> Tuple[List[Dict], List[Dict]]:
    if len(rows) < 8:
        raise ValueError("Not enough rows to split train/test")

    split_idx = int(len(rows) * train_fraction)
    split_idx = max(4, min(len(rows) - 4, split_idx))
    return rows[:split_idx], rows[split_idx:]


def _train_models(
    train_rows: List[Dict],
    test_rows: List[Dict],
    baseline_features: List[str],
    main_features: List[str],
    config: Dict,
) -> Dict:
    logistic_cfg = config["training"]["logistic"]
    forest_cfg = config["training"]["forest_lite"]
    seed = int(config["training"]["random_seed"])

    x_train_baseline = _extract_matrix(train_rows, baseline_features)
    x_test_baseline = _extract_matrix(test_rows, baseline_features)

    x_train_main = _extract_matrix(train_rows, main_features)
    x_test_main = _extract_matrix(test_rows, main_features)

    y_train = _extract_target(train_rows)
    y_test = _extract_target(test_rows)

    logistic = LogisticRegressionGD(
        learning_rate=float(logistic_cfg["learning_rate"]),
        epochs=int(logistic_cfg["epochs"]),
        l2=float(logistic_cfg["l2"]),
    )
    logistic.fit(x_train_baseline, y_train)
    prob_logistic = logistic.predict_proba(x_test_baseline)

    forest = RandomForestLite(
        n_estimators=int(forest_cfg["n_estimators"]),
        max_features=int(forest_cfg["max_features"]),
        max_threshold_candidates=int(forest_cfg["max_threshold_candidates"]),
        random_seed=seed,
    )
    forest.fit(x_train_main, y_train)
    prob_forest = forest.predict_proba(x_test_main)

    threshold = float(config["evaluation"]["decision_threshold"])

    metrics_logistic = evaluate_classification(y_test, prob_logistic, threshold=threshold)
    metrics_forest = evaluate_classification(y_test, prob_forest, threshold=threshold)

    return {
        "logistic": logistic,
        "forest_lite": forest,
        "y_test": y_test,
        "prob_logistic": prob_logistic,
        "prob_forest": prob_forest,
        "metrics_logistic": metrics_logistic,
        "metrics_forest": metrics_forest,
    }


def run_pipeline(
    config_path: str,
    output_root: str = "outputs",
    generate_synthetic: bool = False,
    download_real: bool = False,
    raw_dir: str | None = None,
) -> Path:
    config = load_config(config_path)

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = output_root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_raw_dir = Path(raw_dir) if raw_dir else run_dir / "raw"
    selected_raw_dir.mkdir(parents=True, exist_ok=True)

    if download_real:
        download_manifest = attempt_download_sources(selected_raw_dir)
        write_json(run_dir / "download_manifest.json", download_manifest)

    if generate_synthetic:
        generate_synthetic_sources(config, selected_raw_dir)

    if not _has_minimum_raw_data(selected_raw_dir):
        raise FileNotFoundError(
            "Raw data is incomplete. Provide raw files or run with --generate-synthetic."
        )

    records = align_and_engineer(selected_raw_dir, config)
    threshold = determine_vpl_threshold(records, config)
    labeled = apply_vpl_labels(records, threshold=threshold)

    write_records_csv(labeled, run_dir / "dataset_labeled.csv")

    baseline_features, main_features = _select_feature_sets(labeled, config)
    write_json(
        run_dir / "feature_manifest.json",
        {
            "baseline_features": baseline_features,
            "main_features": main_features,
        },
    )

    lead_minutes = [0] + [int(v) for v in config["evaluation"]["lead_minutes"]]
    cadence = int(config["cadence_minutes"])
    train_fraction = float(config["training"]["train_fraction"])

    nowcast_summary: Dict[str, Dict] = {}
    lead_summary: Dict[int, Dict[str, Dict]] = {}

    for lead_min in lead_minutes:
        lead_steps = lead_min // cadence
        supervised = _build_supervised_rows(labeled, lead_steps=lead_steps)
        train_rows, test_rows = _split_time_ordered(supervised, train_fraction=train_fraction)

        result = _train_models(
            train_rows=train_rows,
            test_rows=test_rows,
            baseline_features=baseline_features,
            main_features=main_features,
            config=config,
        )

        metrics_payload = {
            "lead_minutes": lead_min,
            "logistic": result["metrics_logistic"],
            "forest_lite": result["metrics_forest"],
        }

        if lead_min == 0:
            nowcast_summary = {
                "logistic": result["metrics_logistic"],
                "forest_lite": result["metrics_forest"],
            }
            write_json(run_dir / "metrics_nowcast.json", metrics_payload)

            write_json(
                run_dir / "model_logistic.json",
                {
                    "feature_names": baseline_features,
                    "model": result["logistic"].to_dict(),
                },
            )
            write_json(
                run_dir / "model_forest_lite.json",
                {
                    "feature_names": main_features,
                    "model": result["forest_lite"].to_dict(),
                },
            )

            preds: List[Dict] = []
            for row, y_true, p_logit, p_forest in zip(
                test_rows,
                result["y_test"],
                result["prob_logistic"],
                result["prob_forest"],
            ):
                preds.append(
                    {
                        "timestamp": row["timestamp"],
                        "target_timestamp": row["target_timestamp"],
                        "target_vpl": row["target_vpl"],
                        "target_label": y_true,
                        "prob_logistic": round(p_logit, 6),
                        "prob_forest_lite": round(p_forest, 6),
                    }
                )
            write_predictions_csv(run_dir / "predictions_nowcast.csv", preds)
        else:
            lead_summary[lead_min] = {
                "logistic": result["metrics_logistic"],
                "forest_lite": result["metrics_forest"],
            }
            write_json(run_dir / f"metrics_lead_{lead_min}.json", metrics_payload)

    write_summary_markdown(
        run_dir / "summary.md",
        project_name=config["project_name"],
        threshold=threshold,
        nowcast=nowcast_summary,
        lead_results=lead_summary,
        baseline_features=baseline_features,
        main_features=main_features,
    )

    write_json(
        run_dir / "run_manifest.json",
        {
            "config": str(config_path),
            "raw_dir": str(selected_raw_dir),
            "run_dir": str(run_dir),
            "threshold": threshold,
        },
    )

    latest_link = output_root_path / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
    except OSError:
        # Symlinks may be unavailable in some environments; keep pipeline success.
        pass

    return run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GNSS risk ML pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--output-root", default="outputs", help="Root output directory")
    parser.add_argument("--raw-dir", default=None, help="Existing raw data directory")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic source data")
    parser.add_argument("--download-real", action="store_true", help="Attempt source downloads")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_dir = run_pipeline(
        config_path=args.config,
        output_root=args.output_root,
        generate_synthetic=args.generate_synthetic,
        download_real=args.download_real,
        raw_dir=args.raw_dir,
    )
    print(f"Pipeline completed. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
