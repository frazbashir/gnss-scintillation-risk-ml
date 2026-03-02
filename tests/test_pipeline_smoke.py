from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gnss_risk.pipeline import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_runs_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_src = Path(__file__).resolve().parents[1] / "configs" / "storm_may2024.json"

            config = json.loads(config_src.read_text(encoding="utf-8"))
            config["storm_window"]["start_utc"] = "2024-05-09T00:00:00Z"
            config["storm_window"]["end_utc"] = "2024-05-10T12:00:00Z"
            config["storm_window"]["peak_start_utc"] = "2024-05-09T12:00:00Z"
            config["storm_window"]["peak_end_utc"] = "2024-05-10T03:00:00Z"
            config["evaluation"]["lead_minutes"] = [10, 30]

            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            run_dir = run_pipeline(
                config_path=str(config_path),
                output_root=str(tmp_path / "outputs"),
                generate_synthetic=True,
                download_real=False,
                raw_dir=None,
            )

            self.assertTrue((run_dir / "metrics_nowcast.json").exists())
            self.assertTrue((run_dir / "metrics_lead_10.json").exists())
            self.assertTrue((run_dir / "metrics_lead_30.json").exists())
            self.assertTrue((run_dir / "predictions_nowcast.csv").exists())
            self.assertTrue((run_dir / "summary.md").exists())

            metrics = json.loads((run_dir / "metrics_nowcast.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(metrics["forest_lite"]["roc_auc"], 0.0)
            self.assertLessEqual(metrics["forest_lite"]["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
