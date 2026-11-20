"""Validation check for the Concept-1 greedy placement in leak_toolkit_working.py.

Why: with the default settings the greedy search picks junction 601 as the new
meter site, and it "rescues" every blind spot. Junction 601 has zero demand and
its simulated pressure change moves in ~1.57 psi steps that do not scale with
leak size. This script re-runs stage 1 with the same seed and reports
(a) the response at 601 for each test leak, (b) a tiny-leak control, and
(c) the placement result when 601 (and its neighbour 61) are excluded.

Usage:  python check_placement_artifact.py [--out results/placement_check]
The cause of the staircase response is NOT established here; this only shows
that the placement result depends on it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import wntr

from leak_toolkit_working import (DEFAULT_INP, DEFAULT_SENSOR_NODES, build_response_cube, coverage_table,
                                  first_detect_matrix, greedy_placement, run_one)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak-count", type=int, default=12)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--area0", type=float, default=2.0)
    ap.add_argument("--growth", type=float, default=1.30)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--suspect", nargs="+", default=["601"], help="junctions to test")
    ap.add_argument("--out", default="results/placement_check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    wn = wntr.network.WaterNetworkModel(DEFAULT_INP)
    rng = np.random.default_rng(a.seed)
    pool = [n for n in wn.junction_name_list if n not in DEFAULT_SENSOR_NODES]
    leaks = [str(n) for n in rng.choice(pool, size=a.leak_count, replace=False)]
    cube = build_response_cube(DEFAULT_INP, leaks, a.months, a.area0, a.growth)
    first = first_detect_matrix(cube, a.threshold)
    train, test = leaks[: len(leaks) // 2], leaks[len(leaks) // 2:]

    suspect = a.suspect[0]
    response = pd.concat({k: v[suspect] for k, v in cube.items()}, axis=1).T.round(2)
    response.to_csv(os.path.join(a.out, f"response_at_{suspect}.csv"))

    base = run_one(DEFAULT_INP, leaks[0], 0.0)
    control = {}
    for area in [1e-8, 1e-6, 1e-5, 5e-5]:
        d = run_one(DEFAULT_INP, leaks[0], area) - base
        control[f"area_{area:g}_m2"] = {f"delta_{suspect}": round(float(d[suspect]), 3),
                                        "max_abs_delta_other_nodes": round(float(d.drop(a.suspect).abs().max()), 3)}

    outcomes = {}
    for label, excl in [("default", []), ("exclude_suspect", a.suspect),
                        ("exclude_suspect_and_neighbour_61", a.suspect + ["61"])]:
        cands = [c for c in first.columns if c not in excl]
        sites = greedy_placement(first, train, DEFAULT_SENSOR_NODES, cands, 3, a.months)
        after = coverage_table(first, DEFAULT_SENSOR_NODES + sites).set_index("leak_node")
        outcomes[label] = {"excluded": excl, "new_sites": sites,
                           "train_blind_fraction": float(after.loc[train, "blind_spot"].mean()),
                           "held_out_blind_fraction": float(after.loc[test, "blind_spot"].mean())}
    result = {"leaks": leaks, "train": train, "held_out": test, "tiny_leak_control": control, "placement": outcomes}
    with open(os.path.join(a.out, "placement_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(response.to_string())
    print(json.dumps({"tiny_leak_control": control, "placement": outcomes}, indent=2))


if __name__ == "__main__":
    main()
