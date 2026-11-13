import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from detector_evaluation import (LEAK_STATUS, choose_thresholds, daily_medians, delay_study, main as evaluate_main,
                                 make_scenarios, rolling_metrics, score, summarize_delay, sweep)
from pressure_pipeline import (Database, clean_readings, daily_trend_metrics, demo_readings, hourly_window,
                               load_thresholds, main as pipeline_main, trend_indicators)


def metrics_row(label, drop, slope, step=0.1, ratio=0.1):
    return {"label": label, "drop_psi": drop, "slope_psi_per_30_days": slope,
            "trend_r2": 0.9, "max_daily_step_psi": step, "step_ratio": ratio}


class EvaluationTests(unittest.TestCase):
    def test_refactor_matches_pipeline_aggregation(self):
        clean, _, _ = clean_readings(demo_readings(90))
        group = clean[clean.sensor_id == "synthetic_decline"]
        series, _ = hourly_window(group, 90)
        expected = trend_indicators(series)
        metrics = daily_trend_metrics(daily_medians(series.to_numpy()))
        self.assertEqual(expected["status"], LEAK_STATUS)
        for key, value in metrics.items():
            self.assertAlmostEqual(expected[key], value)
        with self.assertRaises(ValueError):
            daily_medians(np.ones(25))
        with self.assertRaises(ValueError):
            daily_trend_metrics(np.ones(10))

    def test_score_counts_flags_by_class(self):
        table = pd.DataFrame([metrics_row("leak", 3, -1), metrics_row("leak", 0.2, 0),
                              metrics_row("stable", 0, 0), metrics_row("valve_closure", 3, -1)])
        result = score(table, 1.0, 0.1)
        self.assertEqual(result["leak_recall"], 0.5)
        self.assertEqual(result["false_alarm_rate"], 0.5)
        self.assertEqual(result["false_alarm_rate_by_class"], {"valve_closure": 1.0, "stable": 0.0})
        self.assertEqual(result["confusion"]["leak"][LEAK_STATUS], 1)
        json.dumps(result, allow_nan=False)

    def test_choose_respects_budget_and_falls_back_to_balance(self):
        df = pd.DataFrame({"min_drop_psi": [1, 2, 3], "min_decline_psi_per_30_days": [0.1, 0.1, 0.1],
                           "leak_recall": [1.0, 0.8, 0.1], "false_alarm_rate": [0.3, 0.04, 0.0],
                           "precision": [0.5, 0.9, 1.0]})
        chosen = choose_thresholds(df, 0.05)
        self.assertEqual(chosen["min_drop_psi"], 2)
        self.assertTrue(chosen["constraint_met"])
        unmet = df.assign(false_alarm_rate=[0.3, 0.2, 0.09])
        chosen = choose_thresholds(unmet, 0.05)
        self.assertFalse(chosen["constraint_met"])
        self.assertEqual(chosen["min_drop_psi"], 1)  # 1.0-0.3 beats 0.8-0.2 and 0.1-0.09

    def test_sweep_separates_easy_synthetic_classes(self):
        table = make_scenarios(5, 30, 0.05, (6, 8), np.random.default_rng(1))
        grid = sweep(table, drops=[1.0, 2.0], slopes=[0.1])
        self.assertEqual(len(grid), 2)
        stable = score(table[table.label == "stable"], 2.0, 0.1)
        self.assertEqual(stable["false_alarm_rate"], 0.0)

    def test_delay_measured_from_onset(self):
        rng = np.random.default_rng(3)
        leak_runs, stable_runs = delay_study(rng, 3, 120, 30, 40, 0.05, (8, 10), step_days=1)
        self.assertEqual(leak_runs[0][0][0], 30)
        summary = summarize_delay(leak_runs, stable_runs, 40, 1.0, 0.1)
        self.assertEqual(summary["detected_fraction"], 1.0)
        self.assertGreater(summary["median_delay_days"], 0)
        with self.assertRaises(ValueError):
            delay_study(rng, 1, 100, 30, 10, 0.05, (8, 10))
        with self.assertRaises(ValueError):
            rolling_metrics(np.ones(20), 30)

    def test_cli_outputs_feed_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evaluation"
            evaluate_main(["--n-per-class", "8", "--days", "30", "--delay-series", "2",
                           "--delay-total-days", "60", "--step-days", "5", "--out", str(out)])
            for name in ["evaluation_report.json", "threshold_sweep.csv", "recommended_thresholds.json",
                         "threshold_tradeoff.png"]:
                self.assertTrue((out / name).exists(), name)
            report = json.loads((out / "evaluation_report.json").read_text())
            self.assertEqual(report["data_source"], "synthetic_only")
            drop, slope, origin = load_thresholds(out / "recommended_thresholds.json")
            self.assertEqual(origin, "synthetic_only")

            db = str(Path(tmp) / "demo.sqlite")
            run = str(Path(tmp) / "run")
            thresholds = str(out / "recommended_thresholds.json")
            self.assertEqual(pipeline_main(["demo", "--db", db, "--out", run, "--thresholds", thresholds]), 0)
            config = json.loads((Path(run) / "report.json").read_text())["configuration"]
            self.assertEqual((config["min_drop_psi"], config["min_decline_psi_per_30_days"]), (drop, slope))
            self.assertEqual(config["threshold_source"]["min_drop"], "file")
            pipeline_main(["analyze", "--db", db, "--out", run, "--thresholds", thresholds, "--min-drop", "2.5"])
            config = json.loads((Path(run) / "report.json").read_text())["configuration"]
            self.assertEqual(config["min_drop_psi"], 2.5)
            self.assertEqual(config["threshold_source"]["min_drop"], "cli")
            self.assertEqual(config["threshold_source"]["min_slope"], "file")

    def test_bad_threshold_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            for payload in [{}, {"min_drop_psi": -1, "min_decline_psi_per_30_days": 0.1},
                            {"min_drop_psi": "x", "min_decline_psi_per_30_days": 0.1}]:
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_thresholds(path)


if __name__ == "__main__":
    unittest.main()
