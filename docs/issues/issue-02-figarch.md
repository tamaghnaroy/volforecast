# Issue 02 — Implement FIGARCH forecaster for long-memory volatility

## Why
Volatility persistence often exceeds GARCH's geometric decay. FIGARCH provides fractional integration and is a canonical long-memory cvol model.

## Gap evidence
- FIGARCH appears in the knowledge graph but has no runnable forecaster implementation.

## Proposal
Add `FIGARCHForecaster` under `volforecast/models/garch.py` or a dedicated module.

## Implementation notes
- Prefer `arch` package support if available for robustness.
- If custom recursion is used, expose truncation controls and numerical safeguards.
- Ensure explicit stationarity and parameter constraint checks.

## Acceptance criteria
- [ ] FIGARCH class implementing BaseForecaster API.
- [ ] Unit tests covering parameter constraints and forecast positivity.
- [ ] Benchmark integration and documentation update.

## References
- Baillie, Bollerslev, Mikkelsen (1996), Journal of Econometrics.
