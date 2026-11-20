# Team 20 – Cycle 3 code: how to run

Everything below was run from a fresh virtual environment. All data is SYNTHETIC
(WNTR Net3 hydraulics + our own generated pressure series). No sponsor data is used.

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt

| Step | Command | Time (our run) | What it shows |
|---|---|---|---|
| 1 Pipeline demo | `python pressure_pipeline.py demo --db demo.sqlite --out results/pipeline` | <1 s | CSV/DB -> quality gates -> trend indicators -> report.json (LLM off) |
| 2 Detector benchmark | `python detector_evaluation.py --out results/evaluation` | ~5 s | recall / false alarms on held-out synthetic windows, detection delay |
| 3 Integrated toolkit (Concept 1+3) | `python leak_toolkit_working.py --out results/toolkit` | ~50 s | sensor-gap map, greedy meter placement, decline-shape classifier |
| 4 Cycle 2 coverage table | `python sensor_coverage_comparison.py --out results/sensor_coverage` | several minutes | 9 of 10 test leak locations blind to the 4 current sensors |
| 5 Placement check | `python check_placement_artifact.py` | ~1 min | shows the step-3 placement depends on junction 601 (see below) |

Classify a database table with the trained classifier (window must equal `--days`):

    python leak_toolkit_working.py --skip-coverage --days 90 --n-per-class 100 --db demo.sqlite --out results/dbtest

Known limitations (found while running, not fixed in the code):
1. Step 3 picks junction 601 as the new meter. Its simulated response is a ~1.57 psi staircase that does not
   scale with leak size; excluding 601 and 61 leaves 2 of 6 held-out leaks blind. Cause not established.
2. With 601 included, half of the classifier's synthetic leaks reuse that staircase as their "hydraulic shape".
3. No swept threshold pair meets the 5% false-alarm budget (best: 10.3% false alarms at 0.19 recall).
4. On the pipeline's own `synthetic_decline` demo series the rule detector says "possible_leak_indicator"
   while the classifier says "demand_swing" (0.58). The two parts are not yet reconciled.
5. LLM summary (`--llm openai`) was not run. Synthetic accuracy is not field accuracy.
