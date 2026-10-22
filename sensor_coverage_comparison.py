"""
sensor_coverage_comparison.py
LL Blue Engineering Capstone — Gate 2 evidence: sensor coverage / observability map

Purpose
-------
Challenge A (information scarcity) asks: with a limited sensor budget, where
should temporary measurement be placed? Before proposing a placement
*algorithm*, the team needs honest evidence of the current gap: with the
CURRENT sensor set, which parts of the network can already be seen, and
which cannot be seen at all within a realistic detection window?

This script reuses the Cycle 1 leak-growth model (leak_growth_sandbox.py)
and runs it for a leak at each of several network locations, all against
the SAME fixed sensor set. For each leak location it reports the first
month any sensor clears the detection threshold -- or "never" within the
simulated window. A location is a documented BLIND SPOT if every sensor
reports "never". That is a legitimate negative result, not a failure of
the script.

This is intended as evidence for Gate 2, not a finished concept: it shows
WHERE the current instrumentation is blind, as a first input into deciding
where a small number of new temporary meters would do the most good.

Usage
-----
    python sensor_coverage_comparison.py
    python sensor_coverage_comparison.py --leak-nodes 103 197 211 219 50 --sensor-nodes 101 10 40 121
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wntr

from leak_growth_sandbox import DEFAULT_INP, DEFAULT_SENSOR_NODES, run_monthly_trend

# Spread across the Net3 footprint (see node coordinates in the .inp file):
# near-sensor, mid-network, and far-corner locations, to deliberately probe
# whether distance/topology from the existing sensors predicts blindness.
DEFAULT_LEAK_NODES = ["103", "20", "157", "197", "50", "131", "211", "219", "231", "15"]


def build_coverage_table(
    inp_path, leak_nodes, sensor_nodes, area0_cm2, growth_factor, months, threshold_psi
):
    rows = []
    for leak_node in leak_nodes:
        _, first_detectable, _ = run_monthly_trend(
            inp_path, leak_node, sensor_nodes,
            area0_cm2=area0_cm2, growth_factor=growth_factor,
            months=months, threshold_psi=threshold_psi,
        )
        best_month = min(
            (m for m in first_detectable.values() if m is not None), default=None
        )
        row = {"leak_node": leak_node}
        for n in sensor_nodes:
            row[f"detect_month_{n}"] = first_detectable[n]
        row["best_case_detect_month"] = best_month
        row["blind_spot"] = best_month is None
        rows.append(row)
    return pd.DataFrame(rows)


def plot_coverage_map(inp_path, df, out_path):
    wn = wntr.network.WaterNetworkModel(inp_path)
    coords = {n: wn.get_node(n).coordinates for n in wn.junction_name_list}

    fig, ax = plt.subplots(figsize=(8, 7))
    # faint背景: all junctions
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    ax.scatter(xs, ys, color="lightgray", s=15, zorder=1, label="network junctions")

    for _, r in df.iterrows():
        x, y = coords[r["leak_node"]]
        if r["blind_spot"]:
            ax.scatter(x, y, color="red", s=140, marker="x", zorder=3)
        else:
            ax.scatter(x, y, color="green", s=100, zorder=3)
            ax.annotate(f"{int(r['best_case_detect_month'])}mo", (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.scatter([], [], color="red", marker="x", s=140, label="blind spot (never detected)")
    ax.scatter([], [], color="green", s=100, label="detected (month shown)")
    ax.set_title("Leak-location coverage under current sensor set (WNTR / Net3)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", default=DEFAULT_INP)
    ap.add_argument("--leak-nodes", nargs="+", default=DEFAULT_LEAK_NODES)
    ap.add_argument("--sensor-nodes", nargs="+", default=DEFAULT_SENSOR_NODES)
    ap.add_argument("--area0", type=float, default=1.0)
    ap.add_argument("--growth", type=float, default=1.25)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/sensor_coverage")
    args = ap.parse_args()

    df = build_coverage_table(
        args.inp, args.leak_nodes, args.sensor_nodes,
        args.area0, args.growth, args.months, args.threshold,
    )

    print(df.to_string(index=False))
    n_blind = df["blind_spot"].sum()
    print(f"\n{n_blind} of {len(df)} tested leak locations are BLIND SPOTS "
          f"(never cross {args.threshold} psi at any current sensor, "
          f"within {args.months} months, at the modeled growth rate).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(f"{args.out}_table.csv", index=False)
    plot_coverage_map(args.inp, df, f"{args.out}_map.png")
    print(f"\nSaved: {args.out}_table.csv, {args.out}_map.png")


if __name__ == "__main__":
    main()
