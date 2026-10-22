"""
demo.py - quick demo of Concept 1 (Sensor-Gap-Guided Placement)

Runs in about a minute. It compares two simulated leaks on EPANET Net3:
  - one at junction 103 (near a sensor)
  - one at junction 20 (far from every sensor)
at the end of month 12, when the leak is largest, and prints how much the
pressure changed at the 4 assumed sensors. A change below the threshold means
the leak is invisible to those sensors.

For the full 12-month, 10-location analysis, run sensor_coverage_comparison.py
(takes a few minutes).

Usage:   python demo.py
Needs:   leak_growth_sandbox.py in the same folder, and pip install -r requirements.txt

Caveat: a simulation on a public model. Sensor sites, leak size and the
threshold are our own choices. It shows the method can expose gaps, not that a
real utility has them.
"""
import argparse

from leak_growth_sandbox import DEFAULT_INP, DEFAULT_SENSOR_NODES, run_scenario


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leak-nodes", nargs="+", default=["103", "20"])
    ap.add_argument("--area0", type=float, default=1.0, help="starting leak area, cm^2")
    ap.add_argument("--growth", type=float, default=1.25, help="monthly growth factor")
    ap.add_argument("--month", type=int, default=11, help="month to check (0 = first)")
    ap.add_argument("--threshold", type=float, default=0.5, help="psi")
    args = ap.parse_args()

    sensors = DEFAULT_SENSOR_NODES
    area_cm2 = args.area0 * args.growth ** args.month
    print(f"Net3, sensors at {sensors}, leak area {area_cm2:.1f} cm2 (month {args.month + 1}), "
          f"threshold {args.threshold} psi\n")

    base = run_scenario(DEFAULT_INP, args.leak_nodes[0], 0.0, sensors)
    for leak in args.leak_nodes:
        leaked = run_scenario(DEFAULT_INP, leak, area_cm2 * 1e-4, sensors)
        delta = (leaked - base).abs()
        seen = [s for s in sensors if delta[s] >= args.threshold]
        print(f"Leak at junction {leak}:")
        for s in sensors:
            print(f"   sensor {s:>4}: pressure change {delta[s]:.2f} psi")
        print("   -> " + (f"DETECTED by sensor(s) {seen}" if seen else "BLIND SPOT: no sensor sees it") + "\n")


if __name__ == "__main__":
    main()
