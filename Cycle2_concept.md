# Gate 2 — Concept Selection
## L.L. Blue Engineering Capstone Research Challenge | Weeks 5–7

**Status: Cycle 2, Week 5 — just started. Concepts below are initial candidates and will be updated as research continues.**

---

## 1. Recap: What Gate 1 Evidence Tells Us

Our Cycle 1/Gate 1 research produced one concrete, simulation-backed result: running a growing leak at 10 locations across a public benchmark network (WNTR/Net3), against our current 4-sensor set, **9 of 10 locations never crossed the detection threshold within 12 months.** Only the leak closest to an existing sensor was detectable, and only after 7 months.

This result is our starting point for Gate 2. It confirms the Gate 1 hypothesis that observability is a *structural* problem (sensor placement/count), not just an algorithm problem.

---

## 2. Candidate Concepts (Draft — one So Far)

*These are the concepts we're evaluating as of the start of Cycle 2. This list, the evaluation criteria below, and the leaning in Section 4 are all expected to change as we get further into the gate.*

### **Concept 1 — Sensor-Gap-Guided Placement (Challenge A)**
Use the blind-spot map itself as the product: for any candidate set of new temporary meter locations (within a fixed rental budget), predict how much of the network's currently-blind area becomes observable. Output is a ranked list of "where to put the next N meters" with an honest coverage-improvement estimate attached.

## Cycle 2 Update (Gate 2: Concept Selection, Weeks 5–7), In Progress

**Status:** Cycle 2 has just started. The concepts below are initial candidates and **will be updated** as we get further into the gate.

### Candidate concepts
1. **Sensor-Gap-Guided Placement (Challenge A):** Map where the current sensors cannot see a leak, then rank candidate locations for a small number of temporary clamp-on flow meters by how much of that blind area each one would cover.
2. **Diagnostic-Flag Recovery (Challenge B):** Read HART / NAMUR NE 107 device status flags that SCADA may be discarding, and attach a trust indicator to each existing reading.
3. **Decline-Shape Classifier (Challenge A/B):** Use the slow, compounding shape of leak-driven pressure decline to tell leaks apart from valve closures, network changes, and normal demand swings, using only existing sensors.


### Starting point from Cycle 1
Using WNTR/EPANET on the public Net3 network, we modeled a leak whose area grows each month and checked it against a fixed set of 4 sensors. With a 0.5 psi detection threshold and 12 months of simulated growth, **9 of 10 tested leak locations were never detected**.

**Caveats:** This result is specific to Net3 and to assumptions we chose ourselves (sensor locations, leak growth rate, detection threshold). It is not based on sponsor data and has not been validated on a real network.

### Open questions
- How many clamp-on meters can we rent, and for how long? (sponsor)
- Does the sponsor's SCADA integration discard HART / NE 107 diagnostic flags? (sponsor)
- Can a leak's pressure decline be reliably separated from valve closures and demand changes?

### Next steps
- Extend the Gate 1 simulation to test candidate sensor additions.
- Compare the three concepts against agreed criteria before any down-select.
- Update this section and the full writeup as answers come in.

Full writeup: [Cycle2_concept.md](Cycle2_concept.md)
