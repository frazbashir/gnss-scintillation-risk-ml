from __future__ import annotations

import math
from typing import Dict, List, Tuple

from gnss_risk.time_utils import parse_utc


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        raise ValueError("Percentile requested on empty value set")

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    p = max(0.0, min(100.0, percentile)) / 100.0
    idx = p * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]

    fraction = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * fraction


def determine_vpl_threshold(records: List[Dict], config: Dict) -> float:
    labeling = config["labeling"]
    mode = labeling.get("mode", "fixed").lower()

    if mode == "fixed":
        return float(labeling["vpl_threshold_m"])

    if mode == "percentile":
        quiet_end = parse_utc(labeling["quiet_reference_end_utc"])
        quiet_vals = [
            float(r["vpl"])
            for r in records
            if parse_utc(str(r["timestamp"])) <= quiet_end
        ]
        if not quiet_vals:
            raise ValueError("No quiet-window VPL values available for percentile threshold")
        return _percentile(quiet_vals, float(labeling.get("quiet_percentile", 95.0)))

    raise ValueError(f"Unsupported labeling mode: {mode}")


def apply_vpl_labels(records: List[Dict], threshold: float) -> List[Dict]:
    labeled: List[Dict] = []
    for row in records:
        row_copy = dict(row)
        vpl = float(row_copy["vpl"])
        row_copy["label"] = 1 if vpl > threshold else 0
        labeled.append(row_copy)
    return labeled
