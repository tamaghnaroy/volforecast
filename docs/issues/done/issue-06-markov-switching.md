# Issue 06 — Add Markov-switching volatility models

## Why
Regime changes (calm/crisis) are central in volatility dynamics and not captured well by single-regime models.

## Gap evidence
- No regime-switching forecaster currently implemented.

## Proposal
Add `MSGARCHForecaster` (or `MSVolForecaster`) with 2+ latent regimes.

## Implementation notes
- Hidden Markov chain for regime transitions.
- Regime-conditional volatility equations (start with simple Gaussian variances).
- Inference via Hamilton filter / EM.

## Acceptance criteria
- [ ] Regime-switching model class with one-step and multi-step forecasts.
- [ ] Tests recovering regime persistence in synthetic regime-switch DGP.
- [ ] Benchmarks demonstrating stress-period behavior.

## References
- Hamilton & Susmel (1994).
- Gray (1996).
