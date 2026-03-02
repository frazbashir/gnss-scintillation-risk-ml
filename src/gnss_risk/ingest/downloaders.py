from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    target_filename: str


DEFAULT_SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec("earthscope", "https://example.org/earthscope/rinex_may2024.csv", "earthscope_receiver.csv"),
    SourceSpec("cddis", "https://example.org/cddis/ephemeris_may2024.csv", "cddis_ephemeris.csv"),
    SourceSpec("waas", "https://example.org/waas/vpl_may2024.csv", "waas_vpl.csv"),
    SourceSpec("artemis", "https://example.org/artemis/solarwind_may2024.csv", "artemis_solar_wind.csv"),
    SourceSpec("supermag", "https://example.org/supermag/indices_may2024.csv", "supermag.csv"),
]


def attempt_download_sources(
    output_dir: str | Path,
    timeout_seconds: int = 30,
    source_specs: List[SourceSpec] | None = None,
) -> Dict[str, Dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = source_specs or DEFAULT_SOURCE_SPECS
    result: Dict[str, Dict[str, str]] = {}

    for spec in specs:
        target = output_dir / spec.target_filename
        try:
            urllib.request.urlretrieve(spec.url, target)
            result[spec.name] = {
                "status": "downloaded",
                "path": str(target),
                "url": spec.url,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result[spec.name] = {
                "status": "failed",
                "path": str(target),
                "url": spec.url,
                "error": str(exc),
            }

    manifest = output_dir / "download_manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
