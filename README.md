# LL Blue Engineering — prototype through LLB-12

## Cycle 1 (Gate 1 — Weeks 1–4)

Key finding: leak-driven pressure decline is gradual and compounding over months, not a sudden drop. Simulation testing (WNTR/Net3) showed 9 of 10 tested leak locations were undetectable with the current sensor set.

Full writeup: [Cycle 1_Research_Report.md](Cycle%201_Research_Report.md)

## Cycle 2 (Gate 2 - Weeks 5–7)

Key finding: The best candidate concept is actually a combination of two concepts: (A) sensor-gap-guided placement and (C) LLM-assisted trend detection. The sensor gap concept is supported by the Cycle 1 blind-spot finding, while the LLM-assisted trend detection is supported by the Cycle 2 pressure-trend analysis.

Full writeup: [Cycle2_concept.md](Cycle2_concept.md)

## Cycle 3 Update (Gate 3 - Weeks 8–13)

Currently developing the diffearent AGILE schedule steps outlines by LL Blue Engineering.

## Quick start (Python 3.12)

Run these commands. 
On Windows, `py -3.12` can replace `python` when creating the environment.

```powershell
python -m venv .venv
# Windows PowerShell, without needing to activate:
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe pressure_pipeline.py demo --db demo.sqlite --days 90 --llm off
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This demo creates 180 days-worth of data for three sensors in SQLite. It then analyzes the last 90 days.

Outputs are placed into `results/pipeline/`:

- `report.json`: Contains the run ID, timestamp, source, algorithm version, thresholds, quality counts, per-sensor evidence, and LLM status/text.
- `clean_readings.csv`: Contains the validated UTC readings, with duplicate timestamp/sensor pairs averaged.
- `rejected_readings.csv`: Contains the invalid source records and reasons.
- Database tables `pressure_analysis_results` and `pressure_analysis_runs`: Contains append-only history with run IDs. Repeated analysis adds a run; it does not overwrite measurements.

## LLB-10: configuration and connectivity

Copy `.env.example` to `.env`:

```dotenv
DATABASE_URL=demo.sqlite
LLM_MODE=openai
OPENAI_API_KEY=your-real-key
OPENAI_MODEL=your-available-responses-model
```

For MySQL, use `mysql+pymysql://USER:PASSWORD@HOST:3306/DATABASE`. The database must already exist.

```powershell
.venv\Scripts\python.exe pressure_pipeline.py check --llm openai
.venv\Scripts\python.exe pressure_pipeline.py analyze --llm openai
```

`check` executes `SELECT 1` and, if requested, a real small LLM request. `--llm openai` is explicit opt-in to send aggregate sensor identifiers, quality metrics, and trend evidence to OpenAI. 
Raw hourly readings, database URLs, and credentials are not included. LLM output is stored as a narrative; it cannot alter deterministic statuses. Requests use a timeout, bounded SDK retries, and `store=False`. Missing configuration, API failures, empty/incomplete responses are marked as errors and produce a nonzero exit code. 
Numerical evidence is retained on LLM failure. Disabled mode is never represented as a successful LLM call.

The provider adapter uses the [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create). No live request was made during this update because no project API credentials were supplied.

## LLB-11: historical database integration

Canonical source schema (SQLite, or created automatically by CSV/demo ingestion):

```sql
CREATE TABLE pressure_readings (
    ts TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    pressure_psi REAL NOT NULL
);
```

For database rows, use ISO-8601 timestamps with timezone offsets and numeric **psi** values.

## LLB-12: pressure trend and LLM prototype

The detector calculates daily median pressure, first-week versus last-week decline, linear slope in psi per 30 days, fit R², largest daily step, and step/range ratio.

Default rules:

1. A daily step of at least 1 psi accounting for at least 60% of the window range triggers `abrupt_change_review`.
2. Otherwise, a first-to-last-week drop of at least 1 psi and slope <= -0.1 psi per 30 days triggers `possible_leak_indicator`.
3. Otherwise, return `no_sustained_decline_indicator`.

Thresholds (`--min-drop`, `--min-slope`) are demonstrative, not calibrated against sponsor labels. The LLM explains evidence and suggests follow-up checks for operating conditions, demand, flow, and sensor drift. It does not confirm or locate a leak. No decline does not rule out a leak outside sensor coverage.

Architecture:

| Component | Responsibility |
|---|---|
| `.env` + CLI | DB/provider configuration and configurable analysis assumptions |
| `Database` | SQLite/SQLAlchemy connection, identifier validation, read/write lifecycle |
| `clean_readings` / `hourly_window` | Validation, UTC normalization, aggregation, bounded gap filling, quality gates |
| `analyze` / `trend_indicators` | Deterministic screening with auditable numerical evidence |
| `llm_summary` | Optional explanation of aggregate evidence; explicit success/error/disabled status |
| JSON + result tables | Reviewable output and append-only run history |

## Completed research toolkit

```powershell
.venv\Scripts\python.exe leak_toolkit_working.py --skip-coverage --days 90 --n-per-class 30 --db demo.sqlite
.venv\Scripts\python.exe leak_toolkit_working.py --leak-count 4 --months 2 --k-new 1 --days 30 --n-per-class 10
```

## Verification and known limits

Run `python -m unittest discover -s tests -v`. Tests cover known signals, invalid records, duplicate/timezone handling, short/long gaps, stale sensors, short windows, SQL identifier rejection, SQLite and SQLAlchemy connection paths, LLM mock success/failure, persistent run history, all-invalid sensors, finite features, and train-only placement.

`tested_versions.txt` records actual direct dependency versions. 

`requirements.txt` uses bounded ranges rather than claiming an untested universal lockfile. MySQL adapter is provided but live MySQL was not available for validation. LLM behavior was mock-tested; credentials, model access, and a live run remain acceptance items.

Reads currently load the source table into memory. Suitable for the capstone prototype; large deployments need indexed time-range queries, pagination, migrations, calibrated alarm thresholds, real labeled evaluation, and operational monitoring. No web server, scheduled ingestion, or production deployment is included in this milestone.
