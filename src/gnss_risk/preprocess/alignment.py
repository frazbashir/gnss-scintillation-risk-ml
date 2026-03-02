from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

from gnss_risk.time_utils import build_timeline, parse_utc, to_utc_string


NUMERIC_DEFAULTS = {
    "bz": -2.0,
    "by": 0.0,
    "vsw": 380.0,
    "nsw": 5.0,
    "pdyn": 2.0,
    "sml": -100.0,
    "smu": 80.0,
    "vpl": 20.0,
    "sat_count": 20.0,
    "residual_rms": 0.6,
    "position_error_m": 2.0,
    "ephemeris_quality": 0.95,
}


def _parse_numeric_row(row: Dict[str, str], numeric_fields: Iterable[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for field in numeric_fields:
        raw = row.get(field, "")
        try:
            parsed[field] = float(raw)
        except (TypeError, ValueError):
            parsed[field] = NUMERIC_DEFAULTS.get(field, 0.0)
    return parsed


def _load_timeseries(path: Path, numeric_fields: Iterable[str]) -> Dict:
    data: Dict = {}
    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = parse_utc(row["timestamp"])
            data[ts] = _parse_numeric_row(row, numeric_fields)
    return data


def _load_receiver(path: Path, station_id: str) -> Dict:
    data: Dict = {}
    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("station_id") != station_id:
                continue
            ts = parse_utc(row["timestamp"])
            data[ts] = _parse_numeric_row(row, ["sat_count", "residual_rms", "position_error_m"])
    return data


def load_sources(raw_dir: str | Path, station_id: str) -> Dict[str, Dict]:
    raw_dir = Path(raw_dir)
    return {
        "artemis": _load_timeseries(raw_dir / "artemis_solar_wind.csv", ["bz", "by", "vsw", "nsw", "pdyn"]),
        "supermag": _load_timeseries(raw_dir / "supermag.csv", ["sml", "smu"]),
        "waas": _load_timeseries(raw_dir / "waas_vpl.csv", ["vpl"]),
        "earthscope": _load_receiver(raw_dir / "earthscope_receiver.csv", station_id),
        "cddis": _load_timeseries(raw_dir / "cddis_ephemeris.csv", ["ephemeris_quality"]),
    }


def _fill_from_source(current: Dict[str, float], source_row: Dict[str, float] | None, last_row: Dict[str, float]) -> None:
    if source_row is not None:
        last_row.update(source_row)

    for key, default in NUMERIC_DEFAULTS.items():
        if key in last_row:
            current[key] = last_row[key]
        elif key in current:
            continue
        else:
            current[key] = default


def align_and_engineer(raw_dir: str | Path, config: Dict) -> List[Dict[str, float | str]]:
    station_id = config.get("station_id", "P123")
    sources = load_sources(raw_dir, station_id=station_id)

    storm_window = config["storm_window"]
    start = parse_utc(storm_window["start_utc"])
    end = parse_utc(storm_window["end_utc"])
    cadence = int(config["cadence_minutes"])
    timeline = build_timeline(start, end, cadence)

    last_values = {
        "artemis": {},
        "supermag": {},
        "waas": {},
        "earthscope": {},
        "cddis": {},
    }

    records: List[Dict[str, float | str]] = []
    for ts in timeline:
        record: Dict[str, float | str] = {"timestamp": to_utc_string(ts)}

        _fill_from_source(record, sources["artemis"].get(ts), last_values["artemis"])
        _fill_from_source(record, sources["supermag"].get(ts), last_values["supermag"])
        _fill_from_source(record, sources["waas"].get(ts), last_values["waas"])
        _fill_from_source(record, sources["earthscope"].get(ts), last_values["earthscope"])
        _fill_from_source(record, sources["cddis"].get(ts), last_values["cddis"])

        records.append(record)

    derivative_fields = ["bz", "vsw", "sml", "vpl"]
    for idx, record in enumerate(records):
        if idx == 0:
            for field in derivative_fields:
                record[f"d_{field}_dt"] = 0.0
            continue

        prev = records[idx - 1]
        for field in derivative_fields:
            current_v = float(record[field])
            prev_v = float(prev[field])
            record[f"d_{field}_dt"] = (current_v - prev_v) / float(cadence)

    lag_fields = ["bz", "by", "vsw", "nsw", "pdyn", "sml", "smu", "sat_count", "residual_rms", "ephemeris_quality"]
    lag_minutes = [int(v) for v in config["features"]["lag_minutes"]]
    for idx, record in enumerate(records):
        for lag_min in lag_minutes:
            lag_steps = max(1, lag_min // cadence)
            lag_idx = max(0, idx - lag_steps)
            ref = records[lag_idx]
            for field in lag_fields:
                record[f"{field}_lag_{lag_min}m"] = float(ref[field])

    rolling_fields = ["bz", "vsw", "pdyn", "sml", "vpl"]
    rolling_minutes = [int(v) for v in config["features"]["rolling_windows_minutes"]]
    for idx, record in enumerate(records):
        for window_min in rolling_minutes:
            window_steps = max(1, window_min // cadence)
            start_idx = max(0, idx - window_steps + 1)
            window = records[start_idx : idx + 1]
            for field in rolling_fields:
                values = [float(r[field]) for r in window]
                record[f"{field}_roll_mean_{window_min}m"] = mean(values)
                record[f"{field}_roll_std_{window_min}m"] = pstdev(values) if len(values) > 1 else 0.0

    return records


def write_records_csv(records: List[Dict[str, float | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No records to write")

    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
