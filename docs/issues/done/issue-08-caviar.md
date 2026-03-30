# Issue 08 — Add quantile-native volatility/risk methods (CAViaR)

## Why
Variance forecasts are not always sufficient for tail-risk use-cases. Quantile-dynamics models provide fundamentally different forecasting targets and evaluation perspectives.

## Gap evidence
- Current stack focuses on variance forecasts; no native conditional-quantile dynamics model.

## Proposal
Add a `CAViaRForecaster` (Adaptive/Symmetric/Asymmetric specifications) with optional conversion to volatility proxy where needed.

## Implementation notes
- Optimization via quantile regression loss.
- Add evaluation hooks for quantile calibration (hit tests, DQ test).
- Keep explicit target taxonomy extension for conditional quantile.

## Acceptance criteria
- [ ] CAViaR model class + tests for quantile coverage.
- [ ] Evaluation utilities for quantile forecast diagnostics.
- [ ] Documentation clarifying use alongside variance models.

## References
- Engle & Manganelli (2004), Journal of Business & Economic Statistics.
