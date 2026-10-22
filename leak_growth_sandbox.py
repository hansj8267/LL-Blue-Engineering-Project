"""
leak_growth_sandbox.py
LL Blue Engineering Capstone — Gate 2 prep sandbox

Purpose
-------
Stand up the EPANET/WNTR toolchain (per sponsor Day-1 guidance) and give the
team a reusable, scriptable way to test the two hypotheses from Cycle 1
research before committing to a Gate 2 concept:

  1. "Gradual decline" hypothesis — a leak shows up as a slow, compounding
     pressure drop over months, not a sudden step change.
  2. "Bulging" hypothesis — a leak orifice may enlarge over time as a
     function of pressure, which would make the pressure-drop trend
     *accelerate* rather than stay linear.

What this script does
----------------------
- Loads a public EPANET network model (WNTR ships Net1/Net2/Net3/Net6 as
  example networks; Net3 is used by default — 92 junctions, 2 tanks,
  2 pumps, a realistic looped topology).
- Runs a no-leak baseline.
- Runs a month-by-month scenario where a leak at a chosen junction grows
  according to a simple exponential "bulging" model:
        area(month) = area0 * growth_factor ** month
  This is a deliberately simple placeholder for the real bulging mechanism
  (which depends on pipe material and local pressure) — it is meant to let
  us see *whether the resulting pressure trend looks like what we think we
  see in practice*, not to be a validated physical model yet.
- Records end-of-month average pressure at a small set of "sensor" nodes
  (stand-ins for the handful of points a real utility would actually have
  instrumented) and compares each to the no-leak baseline.
- Flags the first month at which the drop at each sensor node would clear a
  configurable detection threshold (default 0.5 psi, roughly the kind of
  noise floor a real pressure transmitter/SCADA tag might have) — this is
  an observability check: does this sensor ever "see" this leak at all,
  and if so, how long does it take?
- Saves a CSV of the full trend and two plots (absolute pressure, and
  delta-from-baseline) to the outputs folder.

Known simplifications (be upfront about these in the Gate 2 writeup)
----------------------------------------------------------------------
- Each month is re-simulated independently from the same initial tank
  levels/demand pattern; tank state is not carried over month-to-month.
  This is fine for seeing the *shape* of the trend but is not a continuous
  12-month hydraulic simulation.
- The exponential growth model for leak area is a hypothesis stand-in, not
  a material-science-derived model. Swapping in a real bulging model
  (pipe-material-dependent) is the obvious next step.
- Uses WNTR's built-in pressure-dependent-demand (PDD) solver
  (WNTRSimulator), not the official EPANET binary (EpanetSimulator),
  because leak controls in WNTR currently only affect results under PDD.

Usage
-----
    python leak_growth_sandbox.py
    python leak_growth_sandbox.py --leak-node 101 --area0 0.3 --growth 1.15 --months 12
    python leak_growth_sandbox.py --compare-nodes 101 60 139 201   # try other leak locations
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wntr

PSI_PER_M = 1.4223  # psi per meter of head, for converting WNTR's SI pressure output

# Net3 ships with WNTR itself, so this works with no external download.
DEFAULT_INP = os.path.join(
    os.path.dirname(wntr.__file__), "library", "networks", "Net3.inp"
)

DEFAULT_SENSOR_NODES = ["101", "10", "40", "121"]  # mix of near/far junctions


def run_scenario(inp_path, leak_node, area_m2, sensor_nodes, discharge_coeff=0.75):
    """Run one hydraulic simulation and return last-day average pressure (psi)
    at the requested sensor nodes. area_m2 = 0 runs the no-leak baseline."""
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.hydraulic.demand_model = "PDD"  # required for WNTR leak controls to take effect

    if area_m2 > 0:
        wn.get_node(leak_node).add_leak(
            wn,
            area=area_m2,
            discharge_coeff=discharge_coeff,
            start_time=0,
            end_time=wn.options.time.duration,
        )

    results = wntr.sim.WNTRSimulator(wn).run_sim()
    # average over the final simulated day, to smooth out diurnal demand swings
    last_day = results.node["pressure"][sensor_nodes].iloc[-24:].mean() * PSI_PER_M
    return last_day


def run_monthly_trend(
    inp_path,
    leak_node,
    sensor_nodes,
    area0_cm2=0.3,
    growth_factor=1.15,
    months=12,
    threshold_psi=0.5,
):
    """Simulate a leak whose area grows month over month, and compare each
    month's pressure at the sensor nodes against a no-leak baseline.

    Returns
    -------
    df : DataFrame, one row per month, with pressure and delta-vs-baseline
         for every sensor node.
    first_detectable : dict mapping sensor node -> first month its |delta|
         clears threshold_psi, or None if it never does within the window.
    """
    baseline = run_scenario(inp_path, leak_node, 0.0, sensor_nodes)

    rows = []
    for m in range(months):
        area_cm2 = area0_cm2 * (growth_factor**m)
        pres = run_scenario(inp_path, leak_node, area_cm2 * 1e-4, sensor_nodes)
        row = {"month": m, "leak_area_cm2": area_cm2}
        for n in sensor_nodes:
            row[f"{n}_psi"] = pres[n]
            row[f"{n}_delta_psi"] = pres[n] - baseline[n]
        rows.append(row)

    df = pd.DataFrame(rows)

    first_detectable = {}
    for n in sensor_nodes:
        hit = df[df[f"{n}_delta_psi"].abs() >= threshold_psi]
        first_detectable[n] = int(hit["month"].iloc[0]) if len(hit) else None

    return df, first_detectable, baseline


def plot_trend(df, sensor_nodes, out_prefix, threshold_psi):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in sensor_nodes:
        ax.plot(df["month"], df[f"{n}_delta_psi"], marker="o", label=f"node {n}")
    ax.axhline(-threshold_psi, color="gray", linestyle="--", linewidth=1,
                label=f"±{threshold_psi} psi detection floor")
    ax.axhline(threshold_psi, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Month since leak onset")
    ax.set_ylabel("Pressure delta vs. no-leak baseline (psi)")
    ax.set_title("Simulated leak-growth pressure trend (WNTR / Net3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_delta_trend.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", default=DEFAULT_INP, help="Path to an EPANET .inp file")
    ap.add_argument("--leak-node", default="101", help="Junction name to leak at")
    ap.add_argument(
        "--sensor-nodes", nargs="+", default=DEFAULT_SENSOR_NODES,
        help="Junction names to treat as instrumented 'sensors'",
    )
    ap.add_argument("--area0", type=float, default=0.3, help="Starting leak area, cm^2")
    ap.add_argument("--growth", type=float, default=1.15, help="Month-over-month area growth factor")
    ap.add_argument("--months", type=int, default=12, help="Number of months to simulate")
    ap.add_argument("--threshold", type=float, default=0.5, help="Detection threshold, psi")
    ap.add_argument("--out", default="results/leak_growth", help="Output file prefix")
    args = ap.parse_args()

    df, first_detectable, baseline = run_monthly_trend(
        args.inp,
        args.leak_node,
        args.sensor_nodes,
        area0_cm2=args.area0,
        growth_factor=args.growth,
        months=args.months,
        threshold_psi=args.threshold,
    )

    print(f"\nLeak node: {args.leak_node}")
    print("Baseline pressure (psi):")
    print(baseline.round(3).to_string())

    print(f"\nMonth-by-month trend (leak area starts at {args.area0} cm^2, "
          f"grows {args.growth}x/month):")
    print(df.round(4).to_string(index=False))

    print(f"\nFirst month each sensor clears the ±{args.threshold} psi detection floor:")
    for node, month in first_detectable.items():
        label = f"month {month}" if month is not None else f"never (within {args.months} months)"
        print(f"  node {node}: {label}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(f"{args.out}_trend.csv", index=False)
    plot_trend(df, args.sensor_nodes, args.out, args.threshold)
    print(f"\nSaved: {args.out}_trend.csv, {args.out}_delta_trend.png")


if __name__ == "__main__":
    main()
