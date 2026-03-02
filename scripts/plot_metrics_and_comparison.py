#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from gnss_risk.eval.metrics import confusion_counts, evaluate_classification, pr_curve, roc_curve

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install with: pip install matplotlib"
    ) from exc


def _parse_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _collect_lead_metrics(run_dir: Path) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}

    nowcast = run_dir / "metrics_nowcast.json"
    if nowcast.exists():
        out[0] = _read_json(nowcast)

    for lead in (10, 30, 60):
        p = run_dir / f"metrics_lead_{lead}.json"
        if p.exists():
            out[lead] = _read_json(p)

    if not out:
        raise FileNotFoundError(f"No metrics files found in {run_dir}")

    return out


def _plot_lead_time_metrics(lead_metrics: Dict[int, Dict], output_dir: Path) -> Path:
    leads = sorted(lead_metrics.keys())

    logistic_roc = [float(lead_metrics[l]["logistic"]["roc_auc"]) for l in leads]
    forest_roc = [float(lead_metrics[l]["forest_lite"]["roc_auc"]) for l in leads]

    logistic_pr = [float(lead_metrics[l]["logistic"]["pr_auc"]) for l in leads]
    forest_pr = [float(lead_metrics[l]["forest_lite"]["pr_auc"]) for l in leads]

    logistic_f1 = [float(lead_metrics[l]["logistic"]["f1"]) for l in leads]
    forest_f1 = [float(lead_metrics[l]["forest_lite"]["f1"]) for l in leads]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(leads, logistic_roc, marker="o", label="Logistic")
    axes[0].plot(leads, forest_roc, marker="o", label="ForestLite")
    axes[0].set_title("ROC-AUC vs Lead Time")
    axes[0].set_xlabel("Lead time (min)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.3)

    axes[1].plot(leads, logistic_pr, marker="o", label="Logistic")
    axes[1].plot(leads, forest_pr, marker="o", label="ForestLite")
    axes[1].set_title("PR-AUC vs Lead Time")
    axes[1].set_xlabel("Lead time (min)")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.3)

    axes[2].plot(leads, logistic_f1, marker="o", label="Logistic")
    axes[2].plot(leads, forest_f1, marker="o", label="ForestLite")
    axes[2].set_title("F1 vs Lead Time")
    axes[2].set_xlabel("Lead time (min)")
    axes[2].set_ylabel("F1")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(alpha=0.3)

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    out_path = output_dir / "lead_time_metrics.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _plot_roc_pr_curves(
    y_true: List[int],
    model_scores: Dict[str, List[float]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for name, scores in model_scores.items():
        pts = roc_curve(y_true, scores)
        auc = evaluate_classification(y_true, scores)["roc_auc"]
        ax_roc.plot([p[0] for p in pts], [p[1] for p in pts], label=f"{name} (AUC={auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title("ROC Curve (Nowcast)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc="lower right")

    roc_path = output_dir / "nowcast_roc_curve.png"
    fig_roc.savefig(roc_path, dpi=170)
    plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for name, scores in model_scores.items():
        pts = pr_curve(y_true, scores)
        auc = evaluate_classification(y_true, scores)["pr_auc"]
        ax_pr.plot([p[0] for p in pts], [p[1] for p in pts], label=f"{name} (AUC={auc:.3f})")

    baseline = (sum(y_true) / len(y_true)) if y_true else 0.0
    ax_pr.hlines(baseline, 0.0, 1.0, linestyles="dashed", colors="k", alpha=0.4)
    ax_pr.set_title("Precision-Recall Curve (Nowcast)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.02)
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc="lower left")

    pr_path = output_dir / "nowcast_pr_curve.png"
    fig_pr.savefig(pr_path, dpi=170)
    plt.close(fig_pr)

    return roc_path, pr_path


def _draw_confusion(ax, conf: Dict[str, int], title: str) -> None:
    mat = [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center", color="black")


def _plot_confusion_matrices(
    y_true: List[int],
    model_scores: Dict[str, List[float]],
    output_dir: Path,
) -> Path:
    cols = len(model_scores)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), constrained_layout=True)
    if cols == 1:
        axes = [axes]

    for ax, (name, scores) in zip(axes, model_scores.items()):
        conf = confusion_counts(y_true, scores, threshold=0.5)
        _draw_confusion(ax, conf, f"{name} Confusion")

    out_path = output_dir / "nowcast_confusion_matrices.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _plot_original_vs_predicted(
    timestamps: List[datetime],
    vpl: List[float],
    label: List[int],
    model_scores: Dict[str, List[float]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    fig1, ax1 = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(timestamps, vpl, color="#1f2937", linewidth=1.5, label="Original VPL")
    ax1.set_ylabel("VPL (m)")
    ax1.set_xlabel("Timestamp (UTC)")
    ax1.set_title("Original VPL vs Predicted Risk Probability")

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for i, (name, scores) in enumerate(model_scores.items()):
        ax2.plot(
            timestamps,
            scores,
            linewidth=1.1,
            alpha=0.9,
            color=palette[i % len(palette)],
            label=f"{name} risk prob",
        )

    ax2.set_ylabel("Predicted risk probability")
    ax2.set_ylim(0.0, 1.02)
    ax1.grid(alpha=0.25)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig1.autofmt_xdate()

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    out1 = output_dir / "comparison_vpl_vs_predicted_probability.png"
    fig1.savefig(out1, dpi=170)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.step(timestamps, label, where="post", linewidth=2, color="#111827", label="Original label")

    for i, (name, scores) in enumerate(model_scores.items()):
        pred = [1 if s >= 0.5 else 0 for s in scores]
        ax.step(
            timestamps,
            pred,
            where="post",
            linewidth=1.2,
            alpha=0.85,
            color=palette[i % len(palette)],
            label=f"{name} predicted label",
        )

    ax.set_title("Original vs Predicted Label Comparison")
    ax.set_ylabel("Label")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Timestamp (UTC)")
    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig2.autofmt_xdate()
    ax.legend(loc="upper left")

    out2 = output_dir / "comparison_original_vs_predicted_labels.png"
    fig2.savefig(out2, dpi=170)
    plt.close(fig2)

    return out1, out2


def _write_metric_table(lead_metrics: Dict[int, Dict], output_dir: Path) -> Path:
    path = output_dir / "metrics_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lead_minutes", "model", "roc_auc", "pr_auc", "f1", "accuracy", "tp", "fp", "tn", "fn"],
        )
        writer.writeheader()
        for lead in sorted(lead_metrics):
            for model_key in ("logistic", "forest_lite"):
                m = lead_metrics[lead][model_key]
                c = m["confusion_matrix"]
                writer.writerow(
                    {
                        "lead_minutes": lead,
                        "model": model_key,
                        "roc_auc": m["roc_auc"],
                        "pr_auc": m["pr_auc"],
                        "f1": m["f1"],
                        "accuracy": m["accuracy"],
                        "tp": c["tp"],
                        "fp": c["fp"],
                        "tn": c["tn"],
                        "fn": c["fn"],
                    }
                )
    return path


def _load_gpu_predictions(path: Path) -> Tuple[List[str], List[int], List[float]]:
    rows = _read_csv(path)
    ts = [r["timestamp"] for r in rows]
    y = [int(float(r["target_label"])) for r in rows]
    p = [float(r["probability"]) for r in rows]
    return ts, y, p


def generate_plots(run_dir: Path, output_dir: Path, gpu_predictions: Path | None = None) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    lead_metrics = _collect_lead_metrics(run_dir)
    predictions = _read_csv(run_dir / "predictions_nowcast.csv")

    timestamps = [_parse_ts(r["timestamp"]) for r in predictions]
    y_true = [int(float(r["target_label"])) for r in predictions]
    vpl = [float(r["target_vpl"]) for r in predictions]

    model_scores: Dict[str, List[float]] = {
        "Logistic": [float(r["prob_logistic"]) for r in predictions],
        "ForestLite": [float(r["prob_forest_lite"]) for r in predictions],
    }

    if gpu_predictions is not None and gpu_predictions.exists():
        ts_gpu, y_gpu, p_gpu = _load_gpu_predictions(gpu_predictions)
        gpu_map = {t: (y, p) for t, y, p in zip(ts_gpu, y_gpu, p_gpu)}

        aligned_scores: List[float] = []
        aligned_ok = True
        for ts, y in zip([r["timestamp"] for r in predictions], y_true):
            rec = gpu_map.get(ts)
            if rec is None:
                aligned_ok = False
                break
            y_g, p_g = rec
            if y_g != y:
                aligned_ok = False
                break
            aligned_scores.append(p_g)

        if aligned_ok:
            model_scores["XGBoost-GPU"] = aligned_scores

    artifacts: Dict[str, str] = {}

    artifacts["lead_time_metrics"] = str(_plot_lead_time_metrics(lead_metrics, output_dir))
    roc_path, pr_path = _plot_roc_pr_curves(y_true, model_scores, output_dir)
    artifacts["roc_curve"] = str(roc_path)
    artifacts["pr_curve"] = str(pr_path)
    artifacts["confusion_matrices"] = str(_plot_confusion_matrices(y_true, model_scores, output_dir))

    comp1, comp2 = _plot_original_vs_predicted(timestamps, vpl, y_true, model_scores, output_dir)
    artifacts["comparison_vpl_prob"] = str(comp1)
    artifacts["comparison_labels"] = str(comp2)

    artifacts["metrics_summary_csv"] = str(_write_metric_table(lead_metrics, output_dir))

    manifest_path = output_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    artifacts["manifest"] = str(manifest_path)
    return artifacts


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot GNSS risk metrics and original-vs-predicted comparisons")
    parser.add_argument("--run-dir", default="outputs/latest", help="Pipeline run directory")
    parser.add_argument("--output-dir", default=None, help="Plot output directory (default: <run-dir>/plots)")
    parser.add_argument(
        "--gpu-predictions",
        default=None,
        help="Optional GPU predictions CSV path (timestamp,target_label,probability,predicted_label)",
    )
    return parser


def main() -> None:
    args = _arg_parser().parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    gpu_predictions = Path(args.gpu_predictions) if args.gpu_predictions else None

    artifacts = generate_plots(run_dir=run_dir, output_dir=output_dir, gpu_predictions=gpu_predictions)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
