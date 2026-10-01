## Cycle 1 Update (Gate 1 — Weeks 1–4)

**Status:** Toolchain established; initial hypothesis testing in progress.

### Summary
This cycle focused on building the team's understanding of water-loss practice and getting the research toolchain running. The EPANET/WNTR simulation environment is now set up and operational, and the team has started testing early hypotheses from the literature review against simulated data.

### Toolchain
- EPANET hydraulic engine + WNTR (Python) installed and verified against a sample public network model.
- Able to load an EPANET network file, run a WNTR simulation, and inject a student-defined leak scenario for comparison against baseline (no-leak) conditions.
- LeakDB and BattLeDIM benchmark structures reviewed; not yet integrated as the primary test set (planned for Cycle 2 / Gate 2 prep).

### Research findings this cycle
- Literature review indicates leak-driven pressure decline is typically **gradual and compounding over months**, not an abrupt step change (e.g., ~0.4 PSI/month compounding decrease from a 62 PSI baseline), which has implications for how a detector should be designed (trend-based, not threshold-based).
- Identified a candidate feedback mechanism ("bulging") where leak orifice size may grow over time as a function of pipe material and local pressure, potentially compounding leak rate — flagged as a hypothesis to validate in WNTR, not yet simulation-confirmed.
- Reviewed NAMUR NE 107 / HART diagnostic status signals (Failure, Out of Specification, Maintenance Required, Function Check) as a possible explanation for "meters that lie" under Challenge B — diagnostic data may already exist at the device level but be discarded before reaching SCADA.
- AWWA M36 water audit methodology reviewed as the industry standard for NRW quantification, including its data-validity scoring approach — used as a model for how this project should report its own confidence levels.

### Open items heading into Cycle 2 / Gate 2
- Validate the gradual-decline and bulging hypotheses against a WNTR-generated leak scenario.
- Down-select between Challenge A (sensor placement under scarcity) and Challenge B (measurement credibility), or scope a combined approach.
- Begin drafting at least three materially different concepts per the Gate 2 rubric.
