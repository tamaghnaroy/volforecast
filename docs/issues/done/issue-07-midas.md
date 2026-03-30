# Issue 07 — Implement MIDAS volatility models (GARCH-MIDAS / HAR-MIDAS)

## Why
MIDAS bridges short-run high-frequency volatility with long-run macro/low-frequency components, representing a different decomposition class.

## Gap evidence
- No MIDAS-family model exists despite strong relevance for medium/long horizon volatility forecasting.

## Proposal
Implement:
1. `GARCHMIDASForecaster`
2. `HARMIDASForecaster` (optional follow-up)

## Implementation notes
- Beta or exponential Almon lag weighting for low-frequency covariates.
- Separate long-run and short-run volatility components.
- Support optional macro covariate matrices.

## Acceptance criteria
- [ ] At least one MIDAS model with robust parameter constraints.
- [ ] Unit/integration tests with synthetic low-frequency drivers.
- [ ] Example notebook/script for horizon-specific forecasts.

## References
- Engle, Ghysels, Sohn (2013).
