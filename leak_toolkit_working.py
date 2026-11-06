"""Completed research toolkit; see README.md for the LLB-10..12 pipeline.
Synthetic classifier scores are not field-validated leak probabilities.
"""
import argparse
import datetime as dt
import os
import re
import sqlite3
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pressure_pipeline import Database, clean_readings, hourly_window

PSI_PER_M = 1.4223
DEFAULT_INP = os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net3.inp")
DEFAULT_SENSOR_NODES = ["101", "10", "40", "121"]
CLASSES = ["leak", "valve_closure", "demand_swing", "stable"]
FEATURES = ["total_change", "max_step", "step_ratio", "accel", "c2", "ac7", "range_change"]


# ======================================================================
# STAGE 1a -- Concept 1: sensor-gap map  (DONE)
# ======================================================================
def run_one(inp_path, leak_node, area_m2, hours=24):
    """One hydraulic run -> last-day mean pressure (psi) at ALL junctions."""
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.time.duration = hours * 3600
    if area_m2 > 0:
        wn.get_node(leak_node).add_leak(
            wn, area=area_m2, discharge_coeff=0.75,
            start_time=0, end_time=wn.options.time.duration)
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    p = res.node["pressure"][wn.junction_name_list].iloc[-24:].mean() * PSI_PER_M
    return p


def build_response_cube(inp_path, leak_nodes, months, area0_cm2, growth):
    """For each leak node: DataFrame [month x junction] of pressure change (psi)
    versus the no-leak baseline, with the leak area growing every month."""
    baseline = run_one(inp_path, leak_nodes[0], 0.0)
    cube = {}
    for i, node in enumerate(leak_nodes, 1):
        rows = []
        for m in range(months):
            area_cm2 = area0_cm2 * growth ** m
            rows.append(run_one(inp_path, node, area_cm2 * 1e-4) - baseline)
        cube[node] = pd.DataFrame(rows, index=range(months))
        print(f"  leak node {node} done ({i}/{len(leak_nodes)})")
    return cube


def first_detect_matrix(cube, threshold_psi):
    """leak x junction table: first month |delta| >= threshold (inf = never)."""
    out = {}
    for leak, df in cube.items():
        hit = df.abs().ge(threshold_psi)
        first = hit.idxmax().astype(float).where(hit.any(), np.inf)
        out[leak] = first
    return pd.DataFrame(out).T


def coverage_table(first_df, sensors):
    best = first_df[sensors].min(axis=1)
    return pd.DataFrame({
        "leak_node": first_df.index,
        "best_case_detect_month": best.replace(np.inf, np.nan).values,
        "blind_spot": np.isinf(best.values),
    })


def plot_coverage_map(inp_path, cov_before, cov_after, new_sites, out_path):
    wn = wntr.network.WaterNetworkModel(inp_path)
    xy = {n: wn.get_node(n).coordinates for n in wn.junction_name_list}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(*zip(*xy.values()), color="lightgray", s=12, zorder=1)
    for _, r in cov_after.iterrows():
        x, y = xy[r["leak_node"]]
        was_blind = bool(cov_before.set_index("leak_node").loc[r["leak_node"], "blind_spot"])
        if r["blind_spot"]:
            ax.scatter(x, y, color="red", marker="x", s=120, zorder=3)
        else:
            ax.scatter(x, y, color="green" if not was_blind else "orange", s=90, zorder=3)
    for n in new_sites:
        ax.scatter(*xy[n], marker="s", s=140, facecolors="none", edgecolors="blue", linewidths=2, zorder=4)
    ax.scatter([], [], color="red", marker="x", label="still blind")
    ax.scatter([], [], color="green", label="seen by current sensors")
    ax.scatter([], [], color="orange", label="rescued by new meters")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors="blue", label="new meter site")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Coverage before/after temporary meters (Net3, simulated)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ======================================================================
# STAGE 2a -- synthetic data generator  (DONE)
# ======================================================================
def synth_series(kind, rng, days, sigma, drop_range, wntr_shape=None):
    """Hourly pressure series (psi). Same size range for every class on
    purpose, so total drop alone cannot separate them. Harder-than-textbook
    on purpose: slow supply wander (AR(1) noise) is added to every class, a
    leak may start late, and a valve closure may be a gradual ramp."""
    n = days * 24
    t = np.arange(n) / 24.0
    x = t / days
    y = rng.uniform(55, 68) - rng.uniform(1.0, 3.0) * np.sin(2 * np.pi * (t % 1.0 - 0.25))
    mag = rng.uniform(*drop_range)
    if kind == "leak":
        onset = rng.uniform(0.0, 0.4)                    # leak may start mid-window
        xe = np.clip((x - onset) / (1 - onset), 0, 1)
        if wntr_shape is not None and rng.random() < 0.5:
            shape = np.interp(xe * (len(wntr_shape) - 1), np.arange(len(wntr_shape)), wntr_shape)
        else:
            g = rng.uniform(1.0, 4.0)                    # compounding growth
            shape = (np.exp(g * xe) - 1) / (np.exp(g) - 1)
        y = y - mag * shape
    elif kind == "valve_closure":
        t0 = rng.uniform(0.2, 0.8) * days
        ramp = rng.uniform(0.1, 6.0)                     # days to settle (abrupt..gradual)
        frac = np.clip((t - t0) / ramp, 0, 1)
        y = y - mag * rng.choice([1, -1], p=[0.7, 0.3]) * frac
    elif kind == "demand_swing":                       # weekly repeat + trend
        amp = mag * rng.uniform(0.3, 0.6)
        y = y - amp * (1 + np.sin(2 * np.pi * t / 7.0 + rng.uniform(0, 2 * np.pi)))
        y = y - rng.normal(0, 0.3 * mag) * x             # small unrelated demand trend
    elif kind != "stable":
        raise ValueError("Unknown synthetic class")
    wander = np.zeros(n)                                 # slow supply-pressure wander
    e = rng.normal(0, 0.06, n)
    for i in range(1, n):
        wander[i] = 0.995 * wander[i - 1] + e[i]
    return y + wander + rng.normal(0, sigma, n)


# ----------------------------------------------------------------------
# Concept 1: choose new meter sites
# ----------------------------------------------------------------------
def greedy_placement(first_df, train_leaks, existing, candidates, k, months):
    """Greedy blind-spot rescue using TRAIN rows only; ties minimize delay."""
    if not train_leaks or months < 1 or k < 0:
        raise ValueError("Need training leaks, positive months, nonnegative budget")
    train = first_df.loc[train_leaks]
    best = train[existing].min(axis=1).to_numpy() if existing else np.full(len(train), np.inf)
    selected = []
    for _ in range(k):
        scores = []
        for c in sorted(set(candidates) - set(existing) - set(selected)):
            improved = np.minimum(best, train[c].to_numpy())
            rescued = int((~np.isfinite(best) & np.isfinite(improved)).sum())
            saved = float((np.minimum(best, months) - np.minimum(improved, months)).sum())
            scores.append((rescued, saved, c, improved))
        if not scores:
            break
        winner = max(scores, key=lambda row: (row[0], row[1]))
        if winner[0] == 0 and winner[1] <= 0:
            break
        selected.append(winner[2])
        best = winner[3]
    return selected


def leak_shape_from_cube(cube, first_df, sensors):
    """Return normalized absolute hydraulic change for first detected leak."""
    for leak in first_df.index:
        times = first_df.loc[leak, sensors]
        if np.isfinite(times).any():
            sensor = times.idxmin()
            curve = cube[leak][sensor].abs().to_numpy()
            if np.isfinite(curve).all() and curve[-1] > 0:
                return curve / curve[-1], leak, sensor
    return None, None, None


def features(y, days):
    """Seven reproducible shape features on complete hourly UTC windows."""
    y = np.asarray(y, dtype=float)
    if days < 14 or len(y) != days * 24 or not np.isfinite(y).all():
        raise ValueError("Need >=14 days of complete finite hourly readings")
    dm = y.reshape(days, 24).mean(axis=1)
    change = dm[-7:].mean() - dm[:7].mean()
    step = max(abs(dm[i:i+5].mean() - dm[i-5:i].mean()) for i in range(5, days-4))
    mid = days // 2
    accel = (dm[-1] - dm[mid]) - (dm[mid] - dm[0])
    x = np.linspace(0, 1, days)
    c2 = np.polyfit(x, dm, 2)[0]
    detrended = dm - np.polyval(np.polyfit(x, dm, 1), x)
    ac7 = 0.0
    if np.std(detrended[:-7]) > 1e-8 and np.std(detrended[7:]) > 1e-8:
        ac7 = np.corrcoef(detrended[:-7], detrended[7:])[0, 1]
    range_change = np.ptp(dm[mid:]) - np.ptp(dm[:mid])
    return [float(v) for v in [change, step, step / (np.ptp(dm)+1e-6), accel, c2, ac7, range_change]]


def train_and_evaluate(args, wntr_shape, out_dir):
    """Seeded synthetic split; threshold tuning uses training data only."""
    if args.n_per_class < 10 or args.days < 14:
        raise ValueError("Need >=10 examples per class and >=14 days")
    rng = np.random.default_rng(args.seed)
    X, y = [], []
    for kind in CLASSES:
        for _ in range(args.n_per_class):
            series = synth_series(kind, rng, args.days, args.sigma,
                                  (args.drop_min, args.drop_max), wntr_shape)
            X.append(features(series, args.days)); y.append(kind)
    X = np.asarray(X); y = np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)
    model = RandomForestClassifier(n_estimators=150, random_state=args.seed, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    drops = -x_train[:, 0]
    thresholds = np.r_[-np.inf, np.unique(drops), np.inf]
    threshold = max(thresholds, key=lambda t: np.mean((drops >= t) == (y_train == "leak")))
    baseline = -x_test[:, 0] >= threshold
    metrics = {
        "data_source": "synthetic_only", "seed": args.seed,
        "train_count": len(y_train), "test_count": len(y_test),
        "class_order": CLASSES, "feature_order": FEATURES,
        "multiclass_accuracy": float(np.mean(pred == y_test)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
        "classifier_leak_accuracy": float(np.mean((pred == "leak") == (y_test == "leak"))),
        "baseline_leak_accuracy": float(np.mean(baseline == (y_test == "leak"))),
        "baseline_threshold_psi": float(threshold) if np.isfinite(threshold) else str(threshold),
        "limitation": "Random holdout of synthetic examples; no sponsor-data validation or calibrated confidence.",
    }
    metrics["beats_baseline"] = metrics["classifier_leak_accuracy"] > metrics["baseline_leak_accuracy"]
    with open(os.path.join(out_dir, "classifier_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2))
    return model


def classify_database(model, db, args):
    """Classify quality-gated sensor windows; persist evidence and skipped sensors."""
    raw = db.read(args.table, args.time_col, args.sensor_col, args.value_col)
    clean, rejected, audit = clean_readings(raw)
    rejected.to_csv(os.path.join(args.out, "classifier_rejected.csv"), index=False)
    rows = []
    end = clean.ts.max().floor("h") + pd.Timedelta(hours=1) if not clean.empty else None
    for sensor, group in clean.groupby("sensor_id"):
        window, quality = hourly_window(group, args.days, end=end)
        row = {"sensor_id": str(sensor), "analyzed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "coverage": quality["coverage"], "predicted_class": "insufficient_data"}
        if quality["coverage"] >= 0.95 and window.notna().all():
            probs = model.predict_proba([features(window.to_numpy(), args.days)])[0]
            row["predicted_class"] = str(model.classes_[np.argmax(probs)])
            row.update({"score_"+str(c): float(p) for c,p in zip(model.classes_, probs)})
        rows.append(row)
    # Stable schema even when an entire batch is insufficient.
    result = pd.DataFrame(rows).reindex(columns=["sensor_id", "analyzed_at", "coverage", "predicted_class"] + ["score_"+c for c in CLASSES])
    result.to_csv(os.path.join(args.out, "classifier_results.csv"), index=False)
    with open(os.path.join(args.out, "classifier_quality.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    db.write(result, "classifier_analysis_results")
    print(result.to_string(index=False))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default=DEFAULT_INP)
    ap.add_argument("--sensor-nodes", nargs="+", default=DEFAULT_SENSOR_NODES)
    ap.add_argument("--leak-count", type=int, default=12)
    ap.add_argument("--k-new", type=int, default=3)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--area0", type=float, default=2.0, help="starting leak area, cm^2")
    ap.add_argument("--growth", type=float, default=1.30, help="monthly area growth factor")
    ap.add_argument("--sigma", type=float, default=0.25, help="sensor noise, psi")
    ap.add_argument("--threshold", type=float, default=None, help="psi; default 3*sigma")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--drop-min", type=float, default=2.0)
    ap.add_argument("--drop-max", type=float, default=8.0)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--db", default=None)
    ap.add_argument("--table", default="pressure_readings")
    ap.add_argument("--time-col", default="ts")
    ap.add_argument("--sensor-col", default="sensor_id")
    ap.add_argument("--value-col", default="pressure_psi")
    ap.add_argument("--out", default="results/toolkit")
    a = ap.parse_args()
    if a.days < 14 or a.months < 1 or a.leak_count < 2 or a.k_new < 0 or a.sigma <= 0 or a.area0 <= 0 or a.growth <= 0 or a.drop_min <= 0 or a.drop_max < a.drop_min or (a.threshold is not None and a.threshold <= 0):
        ap.error("Invalid simulation/window/threshold settings")
    thr = a.threshold if a.threshold is not None else 3 * a.sigma
    os.makedirs(a.out, exist_ok=True)
    wntr_shape = None

    if not a.skip_coverage:
        print(f"=== Stage 1: Sensor-gap map (threshold {thr:.2f} psi, area0 {a.area0} cm^2, "
              f"growth {a.growth}x/month) ===")
        wn = wntr.network.WaterNetworkModel(a.inp)
        rng = np.random.default_rng(a.seed)
        pool = [n for n in wn.junction_name_list if n not in a.sensor_nodes]
        leaks = [str(n) for n in rng.choice(pool, size=min(a.leak_count, len(pool)), replace=False)]
        cube = build_response_cube(a.inp, leaks, a.months, a.area0, a.growth)
        first = first_detect_matrix(cube, thr)
        before = coverage_table(first, a.sensor_nodes)
        print("\nCoverage with CURRENT sensors:")
        print(before.to_string(index=False))
        print(f"\nblind spots now: {int(before.blind_spot.sum())}/{len(before)}")
        before.to_csv(os.path.join(a.out, "coverage_before.csv"), index=False)

        train, test = leaks[: len(leaks) // 2], leaks[len(leaks) // 2:]
        new_sites = greedy_placement(first, train, a.sensor_nodes, list(first.columns), a.k_new, a.months)
        after = coverage_table(first, a.sensor_nodes + new_sites)
        after.to_csv(os.path.join(a.out, "coverage_after.csv"), index=False)
        coverage_metrics = {"new_sites": new_sites, "train_leaks": train, "held_out_leaks": test}
        for name, nodes in [("train", train), ("held_out", test)]:
            b = float(before.set_index("leak_node").loc[nodes, "blind_spot"].mean())
            f = float(after.set_index("leak_node").loc[nodes, "blind_spot"].mean())
            coverage_metrics[name] = {"before_blind_fraction": b, "after_blind_fraction": f}
        with open(os.path.join(a.out, "coverage_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(coverage_metrics, f, indent=2)
        print(json.dumps(coverage_metrics, indent=2))
        plot_coverage_map(a.inp, before, after, new_sites, os.path.join(a.out, "coverage_map.png"))
        # Keep held-out hydraulic leak locations out of classifier training-shape selection.
        wntr_shape, _, _ = leak_shape_from_cube({n: cube[n] for n in train}, first.loc[train], a.sensor_nodes + new_sites)
    model = train_and_evaluate(a, wntr_shape, a.out)
    if a.db:
        with Database(a.db) as db:
            classify_database(model, db, a)
    print(f"\nfiles saved in {a.out}")


if __name__ == "__main__":
    main()
