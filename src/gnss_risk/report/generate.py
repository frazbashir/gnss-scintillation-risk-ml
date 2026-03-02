from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def write_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_predictions_csv(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(model_name: str, metrics: Dict) -> str:
    conf = metrics["confusion_matrix"]
    return (
        f"| {model_name} | {metrics['samples']} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
        f"{metrics['f1']:.4f} | TP={conf['tp']}, FP={conf['fp']}, TN={conf['tn']}, FN={conf['fn']} |"
    )


def write_summary_markdown(
    path: str | Path,
    project_name: str,
    threshold: float,
    nowcast: Dict,
    lead_results: Dict[int, Dict],
    baseline_features: List[str],
    main_features: List[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# {project_name}")
    lines.append("")
    lines.append("## Labeling")
    lines.append("")
    lines.append(f"- VPL exceed threshold: **{threshold:.2f} m**")
    lines.append("")
    lines.append("## Nowcast Metrics (Delta = 0 min)")
    lines.append("")
    lines.append("| Model | Samples | ROC-AUC | PR-AUC | F1 | Confusion Matrix |")
    lines.append("|---|---:|---:|---:|---:|---|")
    lines.append(_metric_row("Logistic (baseline)", nowcast["logistic"]))
    lines.append(_metric_row("ForestLite (main)", nowcast["forest_lite"]))
    lines.append("")
    lines.append("## Lead-Time Metrics")
    lines.append("")

    for lead_min in sorted(lead_results):
        lines.append(f"### Delta = {lead_min} min")
        lines.append("")
        lines.append("| Model | Samples | ROC-AUC | PR-AUC | F1 | Confusion Matrix |")
        lines.append("|---|---:|---:|---:|---:|---|")
        lines.append(_metric_row("Logistic (baseline)", lead_results[lead_min]["logistic"]))
        lines.append(_metric_row("ForestLite (main)", lead_results[lead_min]["forest_lite"]))
        lines.append("")

    lines.append("## Features")
    lines.append("")
    lines.append("### Baseline")
    lines.append("")
    lines.append(", ".join(baseline_features))
    lines.append("")
    lines.append("### Main")
    lines.append("")
    lines.append(", ".join(main_features))
    lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
