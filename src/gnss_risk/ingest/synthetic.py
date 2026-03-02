from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

from gnss_risk.time_utils import build_timeline, parse_utc, to_utc_string


def _storm_level(ts, peak_start, peak_end) -> float:
    lead_ramp_hours = 12
    decay_hours = 12

    if ts < peak_start:
        dt_hours = (peak_start - ts).total_seconds() / 3600.0
        if dt_hours <= lead_ramp_hours:
            return max(0.0, 1.0 - dt_hours / lead_ramp_hours)
        return 0.0

    if peak_start <= ts <= peak_end:
        span = max(1.0, (peak_end - peak_start).total_seconds())
        x = (ts - peak_start).total_seconds() / span
        return 1.0 + 0.2 * math.sin(2.0 * math.pi * x)

    dt_hours = (ts - peak_end).total_seconds() / 3600.0
    if dt_hours <= decay_hours:
        return max(0.0, 1.0 - dt_hours / decay_hours)
    return 0.0


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_synthetic_sources(config: Dict, output_dir: str | Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_seed = int(config["training"]["random_seed"])
    rng = random.Random(random_seed)

    storm_window = config["storm_window"]
    start = parse_utc(storm_window["start_utc"])
    end = parse_utc(storm_window["end_utc"])
    peak_start = parse_utc(storm_window["peak_start_utc"])
    peak_end = parse_utc(storm_window["peak_end_utc"])
    cadence = int(config["cadence_minutes"])

    timeline = build_timeline(start, end, cadence)

    solar_rows: List[Dict[str, object]] = []
    mag_rows: List[Dict[str, object]] = []
    waas_rows: List[Dict[str, object]] = []
    receiver_rows: List[Dict[str, object]] = []
    eph_rows: List[Dict[str, object]] = []

    station_id = config.get("station_id", "P123")

    for i, ts in enumerate(timeline):
        phase = i / 9.0
        noise = rng.gauss(0.0, 1.0)
        storm = _storm_level(ts, peak_start, peak_end)

        bz = -3.0 + 2.2 * math.sin(phase) - 18.0 * storm + 1.5 * noise
        by = 1.0 + 1.8 * math.sin(phase / 2.2) + 2.8 * storm + 0.8 * noise
        vsw = 385.0 + 35.0 * math.sin(phase / 3.0) + 250.0 * storm + 9.0 * noise
        nsw = 4.5 + 1.1 * math.sin(phase / 4.5) + 8.5 * storm + 0.6 * noise
        pdyn = max(0.5, (nsw / 10.0) * ((vsw / 400.0) ** 2) * 12.0)

        sml = -90.0 - 420.0 * storm + 35.0 * math.sin(phase * 0.8) + 20.0 * noise
        smu = 70.0 + 150.0 * storm + 18.0 * math.sin(phase * 0.6) + 10.0 * noise

        sat_count = max(6, int(round(21.0 - 4.8 * storm + 1.3 * noise)))
        residual_rms = max(0.2, 0.55 + 0.95 * storm + abs(noise) * 0.28)
        ephemeris_quality = max(0.50, min(1.00, 0.98 - 0.08 * storm + 0.012 * noise))

        vpl = max(
            6.0,
            18.0
            + max(0.0, -bz) * 0.95
            + pdyn * 1.35
            + abs(sml) / 175.0
            + residual_rms * 7.5
            + 18.0 * storm
            + 2.2 * noise,
        )
        position_error = max(0.3, 1.2 + 0.12 * vpl + 0.4 * noise)

        ts_str = to_utc_string(ts)

        solar_rows.append(
            {
                "timestamp": ts_str,
                "bz": round(bz, 4),
                "by": round(by, 4),
                "vsw": round(vsw, 4),
                "nsw": round(nsw, 4),
                "pdyn": round(pdyn, 4),
            }
        )
        mag_rows.append(
            {
                "timestamp": ts_str,
                "sml": round(sml, 4),
                "smu": round(smu, 4),
            }
        )
        waas_rows.append({"timestamp": ts_str, "vpl": round(vpl, 4)})
        receiver_rows.append(
            {
                "timestamp": ts_str,
                "station_id": station_id,
                "sat_count": sat_count,
                "residual_rms": round(residual_rms, 4),
                "position_error_m": round(position_error, 4),
            }
        )
        eph_rows.append(
            {
                "timestamp": ts_str,
                "ephemeris_quality": round(ephemeris_quality, 4),
            }
        )

    paths = {
        "artemis": output_dir / "artemis_solar_wind.csv",
        "supermag": output_dir / "supermag.csv",
        "waas": output_dir / "waas_vpl.csv",
        "earthscope": output_dir / "earthscope_receiver.csv",
        "cddis": output_dir / "cddis_ephemeris.csv",
    }

    _write_csv(paths["artemis"], ["timestamp", "bz", "by", "vsw", "nsw", "pdyn"], solar_rows)
    _write_csv(paths["supermag"], ["timestamp", "sml", "smu"], mag_rows)
    _write_csv(paths["waas"], ["timestamp", "vpl"], waas_rows)
    _write_csv(
        paths["earthscope"],
        ["timestamp", "station_id", "sat_count", "residual_rms", "position_error_m"],
        receiver_rows,
    )
    _write_csv(paths["cddis"], ["timestamp", "ephemeris_quality"], eph_rows)

    return paths
