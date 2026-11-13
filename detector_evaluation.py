"""LLB-13: benchmark and calibrate the deterministic pressure-trend detector.

The LLB-12 rules in pressure_pipeline.py shipped with demonstrative thresholds.
This script measures how those rules behave on LABELED scenarios and chooses
thresholds under an explicit false-alarm budget:

1. Benchmark: leak / valve_closure / demand_swing / stable windows from
   leak_toolkit_working.synth_series -> confusion table, leak recall,
   false-alarm rate (overall and by class), precision.
2. Calibration: sweep (min_drop, min_slope) on a CALIBRATION split, keep the
   pair with the highest leak recall whose false-alarm rate fits the budget,
   then re-score it on a separate TEST split so the reported numbers are not
   tuned on the data they describe.
3. Detection delay: slide the analysis window one day at a time over long
   series with a known leak onset -> days from onset to first flag, plus
   how often stable series or pre-onset windows raise a flag.

Every scenario is synthetic. The output measures the detector against our own
assumptions (noise, leak growth, valve and demand behavior), not field
accuracy; recalibrate when sponsor-labelled events become available.

Usage:
    python detector_evaluation.py
    python detector_evaluation.py --max-false-alarm 0.02 --n-per-class 200
    python pressure_pipeline.py analyze --thresholds results/evaluation/recommended_thresholds.json
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leak_toolkit_working import CLASSES, synth_series
from pressure_pipeline import DEFAULT_MIN_DROP, DEFAULT_MIN_SLOPE, daily_trend_metrics, trend_status

LEAK_STATUS = "possible_leak_indicator"
STATUSES = ["possible_leak_indicator", "abrupt_change_review", "no_sustained_decline_indicator"]
DEFAULT_DROPS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_SLOPES = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def daily_medians(hourly):
    """Same aggregation as pressure_pipeline.trend_indicators (whole UTC days)."""
    y = np.asarray(hourly, dtype=float)
    if len(y) == 0 or len(y) % 24 or not np.isfinite(y).all():
        raise ValueError("Need complete finite hourly readings covering whole days")
    return np.median(y.reshape(-1, 24), axis=1)


def make_scenarios(n_per_class, days, sigma, drop_range, rng):
    """Labeled windows -> threshold-independent metrics, one row per window."""
    if n_per_class < 1:
        raise ValueError("Need at least one scenario per class")
    rows = []
    for kind in CLASSES:
        for _ in range(n_per_class):
            series = synth_series(kind, rng, days, sigma, drop_range)
            rows.append({"label": kind, **daily_trend_metrics(daily_medians(series))})
    return pd.DataFrame(rows)


def statuses(table, min_drop, min_slope):
    return pd.Series([trend_status(r, min_drop, min_slope) for r in table.to_dict("records")],
                     index=table.index, dtype=object)


def _ratio(num, den):
    return float(num / den) if den else None


def score(table, min_drop, min_slope):
    """Leak screening quality: 'flagged' means status == possible_leak_indicator."""
    status = statuses(table, min_drop, min_slope)
    flagged = status.eq(LEAK_STATUS)
    leak = table.label.eq("leak")
    tp, fp = int((flagged & leak).sum()), int((flagged & ~leak).sum())
    confusion = (pd.crosstab(table.label, status)
                 .reindex(index=[c for c in CLASSES if table.label.eq(c).any()], columns=STATUSES, fill_value=0))
    return {
        "min_drop_psi": float(min_drop),
        "min_decline_psi_per_30_days": float(min_slope),
        "leak_recall": _ratio(tp, int(leak.sum())),
        "false_alarm_rate": _ratio(fp, int((~leak).sum())),
        "precision": _ratio(tp, tp + fp),
        "false_alarm_rate_by_class": {c: float(flagged[table.label.eq(c)].mean())
                                      for c in CLASSES if c != "leak" and table.label.eq(c).any()},
        "confusion": {label: {s: int(v) for s, v in row.items()} for label, row in confusion.iterrows()},
    }


def sweep(table, drops=DEFAULT_DROPS, slopes=DEFAULT_SLOPES):
    keys = ["min_drop_psi", "min_decline_psi_per_30_days", "leak_recall", "false_alarm_rate", "precision"]
    rows = []
    for d in drops:
        for s in slopes:
            result = score(table, d, s)
            rows.append({k: result[k] for k in keys})
    return pd.DataFrame(rows)


def choose_thresholds(sweep_df, max_false_alarm):
    """Highest recall within the false-alarm budget; ties -> fewer alarms, then
    larger (more conservative) thresholds.

    If no pair fits the budget, fall back to the best balance (max recall minus
    false-alarm rate, Youden's J) instead of a near-zero-recall corner, and
    report constraint_met=False so the gap stays visible.
    """
    if not 0 <= max_false_alarm <= 1:
        raise ValueError("max_false_alarm must be in [0, 1]")
    if sweep_df.empty:
        raise ValueError("Empty sweep")
    df = sweep_df.assign(balance=sweep_df.leak_recall - sweep_df.false_alarm_rate)
    within = df[df.false_alarm_rate <= max_false_alarm]
    met = not within.empty
    primary = "leak_recall" if met else "balance"
    best = (within if met else df).sort_values(
        [primary, "false_alarm_rate", "min_drop_psi", "min_decline_psi_per_30_days"],
        ascending=[False, True, False, False]).iloc[0]
    return {"min_drop_psi": float(best.min_drop_psi),
            "min_decline_psi_per_30_days": float(best.min_decline_psi_per_30_days),
            "constraint_met": bool(met),
            "selection": "max_recall_within_budget" if met else "budget_unmet_max_recall_minus_false_alarm"}


# ----------------------------------------------------------------------
# Detection delay on rolling windows
# ----------------------------------------------------------------------
def onset_leak_series(rng, total_days, onset_day, magnitude, sigma):
    """Stable background plus a compounding leak that starts at onset_day and
    reaches `magnitude` psi after a random 60-150 day ramp."""
    y = synth_series("stable", rng, total_days, sigma, (1.0, 1.0))
    t = np.arange(total_days * 24) / 24.0
    growth, ramp_days = rng.uniform(1.0, 4.0), rng.uniform(60, 150)
    xe = np.clip((t - onset_day) / ramp_days, 0, 1)
    return y - magnitude * (np.exp(growth * xe) - 1) / (np.exp(growth) - 1)


def rolling_metrics(daily, window_days, step_days=1):
    """[(window_end_day_exclusive, metrics)] for each complete window."""
    if window_days < 14 or step_days < 1 or len(daily) < window_days:
        raise ValueError("Need window_days >= 14, step_days >= 1 and enough data")
    return [(e, daily_trend_metrics(daily[e - window_days:e]))
            for e in range(window_days, len(daily) + 1, step_days)]


def delay_study(rng, n_series, total_days, window_days, onset_day, sigma, drop_range, step_days=1):
    if not window_days <= onset_day < total_days:
        raise ValueError("Need window_days <= onset_day < total_days")
    leak_runs, stable_runs = [], []
    for _ in range(n_series):
        magnitude = rng.uniform(*drop_range)
        leak = onset_leak_series(rng, total_days, onset_day, magnitude, sigma)
        stable = synth_series("stable", rng, total_days, sigma, (1.0, 1.0))
        leak_runs.append(rolling_metrics(daily_medians(leak), window_days, step_days))
        stable_runs.append(rolling_metrics(daily_medians(stable), window_days, step_days))
    return leak_runs, stable_runs


def summarize_delay(leak_runs, stable_runs, onset_day, min_drop, min_slope):
    """A window ending at day e is first available on day e; delay = e - onset."""
    delays, pre_onset = [], 0
    for run in leak_runs:
        flags = [e for e, m in run if trend_status(m, min_drop, min_slope) == LEAK_STATUS]
        pre_onset += any(e <= onset_day for e in flags)
        after = [e - onset_day for e in flags if e > onset_day]
        if after:
            delays.append(after[0])
    stable_flagged = sum(any(trend_status(m, min_drop, min_slope) == LEAK_STATUS for _, m in run)
                         for run in stable_runs)
    return {
        "leak_series": len(leak_runs),
        "detected_fraction": _ratio(len(delays), len(leak_runs)),
        "median_delay_days": float(np.median(delays)) if delays else None,
        "p90_delay_days": float(np.percentile(delays, 90)) if delays else None,
        "pre_onset_flag_fraction": _ratio(pre_onset, len(leak_runs)),
        "stable_series": len(stable_runs),
        "stable_series_ever_flagged_fraction": _ratio(stable_flagged, len(stable_runs)),
    }


def plot_tradeoff(sweep_df, default, chosen, max_false_alarm, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sweep_df.false_alarm_rate, sweep_df.leak_recall, s=18, color="#9aa5b1", label="sweep (calibration split)")
    ax.axvline(max_false_alarm, color="#c0392b", ls="--", lw=1, label=f"false-alarm budget {max_false_alarm:.0%}")
    for point, color, name in [(default, "#e67e22", "current defaults"), (chosen, "#2471a3", "recommended")]:
        ax.scatter(point["false_alarm_rate"], point["leak_recall"], s=90, color=color, edgecolor="black", zorder=3,
                   label=f"{name} ({point['min_drop_psi']:g} psi, {point['min_decline_psi_per_30_days']:g} psi/30d)")
    ax.set_xlabel("False-alarm rate (non-leak windows flagged)")
    ax.set_ylabel("Leak recall")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Trend detector threshold trade-off (synthetic, test split for markers)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=100, help="windows per class in EACH split")
    ap.add_argument("--days", type=int, default=90, help="analysis window, days (pipeline default 90)")
    ap.add_argument("--sigma", type=float, default=0.25, help="sensor noise, psi")
    ap.add_argument("--drop-min", type=float, default=2.0)
    ap.add_argument("--drop-max", type=float, default=8.0)
    ap.add_argument("--max-false-alarm", type=float, default=0.05)
    ap.add_argument("--delay-series", type=int, default=30)
    ap.add_argument("--delay-total-days", type=int, default=270)
    ap.add_argument("--delay-onset-day", type=int, default=None, help="default: --days")
    ap.add_argument("--step-days", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/evaluation")
    a = ap.parse_args(argv)
    onset = a.delay_onset_day if a.delay_onset_day is not None else a.days
    if (a.n_per_class < 1 or a.days < 14 or a.sigma <= 0 or a.drop_min <= 0 or a.drop_max < a.drop_min
            or not 0 <= a.max_false_alarm <= 1 or a.delay_series < 1 or a.step_days < 1
            or not a.days <= onset < a.delay_total_days):
        ap.error("Invalid evaluation settings")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    drop_range = (a.drop_min, a.drop_max)

    calibration = make_scenarios(a.n_per_class, a.days, a.sigma, drop_range, rng)
    test = make_scenarios(a.n_per_class, a.days, a.sigma, drop_range, rng)
    grid = sweep(calibration)
    grid.to_csv(os.path.join(a.out, "threshold_sweep.csv"), index=False)
    chosen = choose_thresholds(grid, a.max_false_alarm)
    default_test = score(test, DEFAULT_MIN_DROP, DEFAULT_MIN_SLOPE)
    chosen_test = score(test, chosen["min_drop_psi"], chosen["min_decline_psi_per_30_days"])

    leak_runs, stable_runs = delay_study(rng, a.delay_series, a.delay_total_days, a.days, onset,
                                         a.sigma, drop_range, a.step_days)
    report = {
        "data_source": "synthetic_only", "seed": a.seed, "algorithm": "daily-median-trend-v1",
        "configuration": {"window_days": a.days, "sigma_psi": a.sigma, "drop_range_psi": list(drop_range),
                          "windows_per_class_per_split": a.n_per_class, "max_false_alarm": a.max_false_alarm,
                          "delay_series": a.delay_series, "delay_total_days": a.delay_total_days,
                          "delay_onset_day": onset, "step_days": a.step_days},
        "default_thresholds_test": default_test,
        "recommended_thresholds": chosen,
        "recommended_thresholds_test": chosen_test,
        "detection_delay": {
            "default": summarize_delay(leak_runs, stable_runs, onset, DEFAULT_MIN_DROP, DEFAULT_MIN_SLOPE),
            "recommended": summarize_delay(leak_runs, stable_runs, onset, chosen["min_drop_psi"],
                                           chosen["min_decline_psi_per_30_days"]),
        },
        "limitation": ("Synthetic scenarios with assumed noise, leak growth, valve and demand behavior. "
                       "Rates are not field accuracy or calibrated probabilities; recalibrate on sponsor-labelled events."),
    }
    with open(os.path.join(a.out, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    with open(os.path.join(a.out, "recommended_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({**{k: chosen[k] for k in ("min_drop_psi", "min_decline_psi_per_30_days")},
                   "constraint_met": chosen["constraint_met"], "selection": chosen["selection"],
                   "max_false_alarm": a.max_false_alarm,
                   "window_days": a.days, "seed": a.seed, "source": "detector_evaluation.py",
                   "data_source": "synthetic_only"}, f, indent=2)
    plot_tradeoff(grid, default_test, chosen_test, a.max_false_alarm, os.path.join(a.out, "threshold_tradeoff.png"))

    def line(name, r, d):
        return (f"  {name:<12} drop>={r['min_drop_psi']:g} psi, slope<=-{r['min_decline_psi_per_30_days']:g} psi/30d | "
                f"recall {r['leak_recall']:.2f}, false alarms {r['false_alarm_rate']:.2f} | "
                f"rolling: detected {d['detected_fraction']:.2f}, median delay {d['median_delay_days']} d, "
                f"stable ever flagged {d['stable_series_ever_flagged_fraction']:.2f}")
    print("Detector evaluation on held-out synthetic test split (synthetic only):")
    print(line("defaults", default_test, report["detection_delay"]["default"]))
    print(line("recommended", chosen_test, report["detection_delay"]["recommended"]))
    if not chosen["constraint_met"]:
        print(f"  NOTE: no swept pair met the {a.max_false_alarm:.0%} false-alarm budget; best recall-minus-false-alarm "
              "pair chosen. Trend thresholds alone cannot separate these confounders (see confusion by class).")
    print(f"files saved in {a.out}")
    return report


if __name__ == "__main__":
    main()
