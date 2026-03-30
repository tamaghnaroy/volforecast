# Issue 03 — Implement HEAVY model with realized measures

## Why
HEAVY is a distinct realized-measure-driven volatility framework and a key competitor to Realized GARCH.

## Gap evidence
- HEAVY is listed in the knowledge graph relationships but absent as executable model code.

## Proposal
Add `HEAVYForecaster` using daily returns + realized measure (`RV` by default), with optional alternative realized inputs.

## Implementation notes
- Implement return-variance and realized-measure recursions per Shephard & Sheppard (2010).
- Include robust handling when realized inputs are missing/irregular.
- Keep target declaration as `CONDITIONAL_VARIANCE`.

## Acceptance criteria
- [ ] HEAVY forecaster with fit/predict/update.
- [ ] Tests validating improved nowcast behavior vs plain GARCH on synthetic realized-data DGP.
- [ ] Docs + benchmark recipe added.

## References
- Shephard & Sheppard (2010), Journal of Financial Econometrics.
