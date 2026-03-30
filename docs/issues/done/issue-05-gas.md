# Issue 05 — Implement score-driven (GAS/DCS) volatility models

## Why
GAS models are observation-driven but structurally different from GARCH, updating states using scaled score of the conditional density.

## Gap evidence
- No score-driven volatility model currently in the executable model set.

## Proposal
Add `GASVolForecaster` supporting Normal and Student-t innovations.

## Implementation notes
- State update: `f_{t+1} = ω + A s_t + B f_t` where `s_t` is scaled score.
- Map state to variance via exponential link.
- Add distribution-aware likelihood for robust tails.

## Acceptance criteria
- [ ] GAS forecaster with fit/predict/update.
- [ ] Tests for score recursion stability and positive variance.
- [ ] Documentation of when GAS is preferable to GARCH.

## References
- Creal, Koopman, Lucas (2013).
- Harvey (2013).
