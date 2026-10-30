"""
leak_toolkit_half.py   (HALF-FINISHED VERSION -- finish the TODOs)
LL Blue Engineering Capstone -- merges Concept 1 (Sensor-Gap-Guided Placement)
and Concept 3 (Decline-Shape Classifier), plus a database link.
Companion to leak_toolkit_full.py (the finished reference).

WHAT IS ALREADY DONE (runs today)
---------------------------------
 * Stage 1a: hydraulic simulation on EPANET Net3 (WNTR), realistic leak size
             and growth, noise-aware detection threshold, blind-spot table + map
 * Stage 2a: synthetic pressure-series generator for the three classes
             (leak / valve closure / demand swing)

WHAT YOU FINISH (search for "TODO")
-----------------------------------
 TODO 1  greedy_placement()      choose new meter sites, test on HELD-OUT leaks
 TODO 2  leak_shape_from_cube()  hand the hydraulic leak curve to Stage 2
 TODO 3  features()              shape features for the classifier
 TODO 4  train_and_evaluate()    classifier vs simple-threshold baseline
 TODO 5  Database.read/write     connect your own database (SQLite or SQLAlchemy URL)
 TODO 6  classify_database()     classify each sensor in the DB, save results

Unfinished stages are skipped with a message, so the script always runs.

ASSUMPTIONS -- NOT VERIFIED (all are command-line options)
----------------------------------------------------------
 * The 0.4 psi/month figure from the Cycle 1 notes was described as
   noise-level, so it is NOT the leak signal here. The leak starts at --area0
   cm^2 and grows --growth x per month; the hydraulic model decides the psi.
 * Sensor noise sigma = 0.25 psi; detection threshold = 3 sigma = 0.75 psi.
 * Clamp-on meters measure flow, this model measures pressure (open issue).
 * Simulation on a public model only: it tests the method, not a real utility.

USAGE
-----
 pip install wntr scikit-learn pandas matplotlib numpy
 python leak_toolkit_half.py
"""
import argparse
import datetime as dt
import os
import re
import sqlite3
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PSI_PER_M = 1.4223
DEFAULT_INP = os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net3.inp")
DEFAULT_SENSOR_NODES = ["101", "10", "40", "121"]
CLASSES = ["leak", "valve_closure", "demand_swing"]
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
    else:                                                # demand swing: weekly repeat + trend
        amp = mag * rng.uniform(0.3, 0.6)
        y = y - amp * (1 + np.sin(2 * np.pi * t / 7.0 + rng.uniform(0, 2 * np.pi)))
        y = y - rng.normal(0, 0.3 * mag) * x             # small unrelated demand trend
    wander = np.zeros(n)                                 # slow supply-pressure wander
    e = rng.normal(0, 0.06, n)
    for i in range(1, n):
        wander[i] = 0.995 * wander[i - 1] + e[i]
    return y + wander + rng.normal(0, sigma, n)


# ----------------------------------------------------------------------
# TODO 1 -- Concept 1: choose new meter sites
# ----------------------------------------------------------------------
def greedy_placement(first_df, train_leaks, existing, candidates, k, months):
    """TODO 1.
    first_df : DataFrame [leak node x junction] = first month that junction sees
               the leak (np.inf = never).
    Goal     : return a list of k new junction names.
    Hint     : repeat k times -- for every candidate c, compute
               new = np.minimum(current_best_month, first_df.loc[train_leaks, c])
               and count how many currently-blind (inf) train leaks become
               finite. Add the best c (tie-break: total months saved).
    Success test (slide 8): after choosing on TRAIN leaks, the blind fraction
               must also drop on the HELD-OUT leaks -- otherwise Concept 1 failed.
    """
    raise NotImplementedError("TODO 1")


def leak_shape_from_cube(cube, first_df, sensors):
    """TODO 2. Return (curve, leak_node, sensor) where curve is the month-by-month
    |pressure change| at the best-seeing sensor for the first DETECTED leak,
    divided by its last value (so it runs 0..1). Return (None, None, None) if no
    leak is detected. Stage 2 uses this as the 'leak' shape, which is what
    actually joins Concept 1 and Concept 3."""
    raise NotImplementedError("TODO 2")


# ======================================================================
# STAGE 2b -- Concept 3: classifier   (synthetic generator is above)
# ======================================================================
def features(y, days):
    """TODO 3. y = hourly pressure (days*24 values). Return a list of numbers
    that describe the SHAPE of the decline. Ideas (daily means dm = y.reshape(days,24).mean(1)):
      total_change : mean(last 7 days) - mean(first 7 days)
      max_step     : biggest |mean(next 5 days) - mean(previous 5 days)|  (valve = big step)
      step_ratio   : max_step / (np.ptp(dm) + 1e-6)
      accel        : drop in 2nd half minus drop in 1st half  (leak compounds)
      c2           : np.polyfit(np.linspace(0,1,days), dm, 2)[0]
      ac7          : lag-7 autocorrelation of the detrended dm (weekly demand cycle)
    Keep the list order identical to FEATURES."""
    raise NotImplementedError("TODO 3")


def train_and_evaluate(args, wntr_shape, out_dir):
    """TODO 4.
    1. Build X, y: loop over CLASSES, call synth_series(...) then features(...)
       args.n_per_class times each.
    2. train_test_split (stratified, 30% test), fit RandomForestClassifier.
    3. Baseline: 'leak' if total pressure fall >= T (tune T on train only).
    4. Print 3-class accuracy, confusion matrix, and leak-vs-other accuracy for
       classifier AND baseline.
    5. Slide-8 criterion: classifier must beat the threshold, otherwise report
       it as a failed (but valid) result.
    Return the fitted model."""
    raise NotImplementedError("TODO 4")


# ======================================================================
# STAGE 3 -- Database link
# ======================================================================
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name):
    """Table/column names cannot be passed as SQL parameters, so whitelist them."""
    if not _IDENT.match(name):
        raise ValueError(f"unsafe table/column name: {name!r}")
    return name


class Database:
    """target = SQLite file path, or a SQLAlchemy URL such as postgresql://user:pw@host/db"""

    def __init__(self, target):
        if "://" in target:
            from sqlalchemy import create_engine      # pip install sqlalchemy + a driver
            self.conn = create_engine(target)
        else:
            self.conn = sqlite3.connect(target)

    def read(self, table, time_col, sensor_col, value_col):
        """TODO 5a. SELECT the three columns (use _ident on every name), alias them
        ts / sensor_id / pressure_psi, convert ts with pd.to_datetime, return the
        DataFrame. Tool: pd.read_sql_query(sql, self.conn)."""
        raise NotImplementedError("TODO 5a")

    def write(self, df, table):
        """TODO 5b. df.to_sql(_ident(table), self.conn, if_exists='append', index=False)"""
        raise NotImplementedError("TODO 5b")


def classify_database(model, db, args):
    """TODO 6. df = db.read(...). For each sensor_id: sort by ts, resample to 1 hour
    (.resample('1h').mean().interpolate()), skip with a message if fewer than
    args.days*24 values, take the last args.days*24 values, run features(),
    model.predict_proba(), and collect sensor_id / predicted class / probabilities.
    Print the table and db.write(result_df, 'analysis_results')."""
    raise NotImplementedError("TODO 6")


# ======================================================================
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
    ap.add_argument("--out", default="/mnt/user-data/outputs/leak_toolkit_half_out")
    a = ap.parse_args()
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

        try:
            train, test = leaks[: len(leaks) // 2], leaks[len(leaks) // 2:]
            new_sites = greedy_placement(first, train, a.sensor_nodes, list(first.columns), a.k_new, a.months)
            after = coverage_table(first, a.sensor_nodes + new_sites)
            for name, nodes in [("train", train), ("HELD-OUT", test)]:
                b = before.set_index("leak_node").loc[nodes, "blind_spot"].mean()
                f = after.set_index("leak_node").loc[nodes, "blind_spot"].mean()
                print(f"blind fraction on {name:8s} leaks: {b:.2f} -> {f:.2f}")
            plot_coverage_map(a.inp, before, after, new_sites, os.path.join(a.out, "coverage_map.png"))
            wntr_shape, _, _ = leak_shape_from_cube(cube, first, a.sensor_nodes + new_sites)
        except NotImplementedError as e:
            print(f"\n[skipped] {e} not finished yet (placement / leak-shape hand-off)")

    model = None
    try:
        model = train_and_evaluate(a, wntr_shape, a.out)
    except NotImplementedError as e:
        print(f"\n[skipped] Stage 2 classifier: {e} not finished yet "
              "(generator is ready: synth_series)")

    if a.db:
        try:
            if model is None:
                raise NotImplementedError("classifier (TODO 3/4)")
            classify_database(model, Database(a.db), a)
        except NotImplementedError as e:
            print(f"\n[skipped] Stage 3 database: {e} not finished yet")
    print(f"\nfiles saved in {a.out}")


if __name__ == "__main__":
    main()
