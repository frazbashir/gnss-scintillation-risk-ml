from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from gnss_risk.eval.metrics import evaluate_classification


EXCLUDED_COLS = {"timestamp", "label", "position_error_m"}


def _read_dataset(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return reader.fieldnames, rows


def _numeric_feature_candidates(fieldnames: Sequence[str], rows: List[Dict[str, str]]) -> List[str]:
    candidates: List[str] = []
    for col in fieldnames:
        if col in EXCLUDED_COLS:
            continue
        if col.startswith("vpl") or col.startswith("d_vpl") or "_vpl_" in col:
            continue

        ok = True
        for row in rows[:200]:
            val = row.get(col, "")
            try:
                float(val)
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            candidates.append(col)
    return candidates


def _to_matrix(rows: List[Dict[str, str]], feature_cols: Sequence[str]) -> Tuple[List[List[float]], List[int], List[str]]:
    x: List[List[float]] = []
    y: List[int] = []
    ts: List[str] = []

    for row in rows:
        try:
            label = int(float(row["label"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("Dataset must contain numeric 'label' column") from exc

        x.append([float(row[c]) for c in feature_cols])
        y.append(label)
        ts.append(row.get("timestamp", ""))

    return x, y, ts


def _split_time_ordered(x: List[List[float]], y: List[int], ts: List[str], train_fraction: float):
    if len(x) < 20:
        raise ValueError("Not enough rows to train/test split")
    split = int(len(x) * train_fraction)
    split = max(10, min(len(x) - 10, split))

    return (
        x[:split],
        y[:split],
        ts[:split],
        x[split:],
        y[split:],
        ts[split:],
    )


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_predictions(path: Path, timestamps: Sequence[str], y_true: Sequence[int], y_score: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "target_label", "probability", "predicted_label"],
        )
        writer.writeheader()
        for t, y, p in zip(timestamps, y_true, y_score):
            writer.writerow(
                {
                    "timestamp": t,
                    "target_label": y,
                    "probability": round(float(p), 6),
                    "predicted_label": 1 if p >= 0.5 else 0,
                }
            )


def run_gpu_training(
    dataset_csv: str,
    output_dir: str,
    train_fraction: float,
    allow_cpu_fallback: bool,
) -> Dict:
    dataset_path = Path(dataset_csv)
    out_dir = Path(output_dir)

    fieldnames, rows = _read_dataset(dataset_path)
    feature_cols = _numeric_feature_candidates(fieldnames, rows)
    if not feature_cols:
        raise ValueError("No numeric feature columns found")

    x, y, ts = _to_matrix(rows, feature_cols)
    x_train, y_train, _, x_test, y_test, ts_test = _split_time_ordered(x, y, ts, train_fraction)

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost") from exc

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
    }

    model = XGBClassifier(**params)
    backend = "cuda"

    try:
        model.fit(x_train, y_train)
    except Exception as exc:
        if not allow_cpu_fallback:
            raise RuntimeError(f"GPU training failed: {exc}") from exc

        params["device"] = "cpu"
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        backend = "cpu-fallback"

    proba = model.predict_proba(x_test)
    y_score = [float(row[1]) for row in proba]

    metrics = evaluate_classification(y_test, y_score, threshold=0.5)

    payload = {
        "backend": backend,
        "samples_train": len(x_train),
        "samples_test": len(x_test),
        "feature_count": len(feature_cols),
        "metrics": metrics,
        "features": feature_cols,
    }

    _write_json(out_dir / "metrics_gpu_xgboost.json", payload)
    _write_predictions(out_dir / "predictions_gpu_xgboost.csv", ts_test, y_test, y_score)

    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU training (XGBoost CUDA) for GNSS risk dataset")
    parser.add_argument("--dataset-csv", default="outputs/latest/dataset_labeled.csv")
    parser.add_argument("--output-dir", default="outputs/gpu")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = run_gpu_training(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
