from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def _trapezoid_auc(points: Sequence[Tuple[float, float]]) -> float:
    area = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        area += (x1 - x0) * (y0 + y1) * 0.5
    return area


def roc_curve(y_true: List[int], y_score: List[float]) -> List[Tuple[float, float]]:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    positives = sum(y_true)
    negatives = len(y_true) - positives

    if positives == 0 or negatives == 0:
        return [(0.0, 0.0), (1.0, 1.0)]

    tp = 0
    fp = 0
    points: List[Tuple[float, float]] = [(0.0, 0.0)]

    for _, target in pairs:
        if target == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / positives
        fpr = fp / negatives
        points.append((fpr, tpr))

    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))

    return points


def pr_curve(y_true: List[int], y_score: List[float]) -> List[Tuple[float, float]]:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    positives = sum(y_true)

    if positives == 0:
        return [(0.0, 1.0), (1.0, 0.0)]

    tp = 0
    fp = 0
    points: List[Tuple[float, float]] = [(0.0, 1.0)]

    for _, target in pairs:
        if target == 1:
            tp += 1
        else:
            fp += 1

        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        points.append((recall, precision))

    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))

    return points


def confusion_counts(y_true: List[int], y_score: List[float], threshold: float) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_score):
        pred = 1 if p >= threshold else 0
        if t == 1 and pred == 1:
            tp += 1
        elif t == 0 and pred == 1:
            fp += 1
        elif t == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def f1_from_confusion(conf: Dict[str, int]) -> float:
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_classification(y_true: List[int], y_score: List[float], threshold: float = 0.5) -> Dict:
    roc_pts = roc_curve(y_true, y_score)
    pr_pts = pr_curve(y_true, y_score)

    conf = confusion_counts(y_true, y_score, threshold=threshold)
    f1 = f1_from_confusion(conf)

    total = len(y_true)
    accuracy = (conf["tp"] + conf["tn"]) / total if total > 0 else 0.0

    return {
        "samples": total,
        "positives": sum(y_true),
        "negative": total - sum(y_true),
        "roc_auc": _trapezoid_auc(roc_pts),
        "pr_auc": _trapezoid_auc(pr_pts),
        "f1": f1,
        "accuracy": accuracy,
        "decision_threshold": threshold,
        "confusion_matrix": conf,
    }
