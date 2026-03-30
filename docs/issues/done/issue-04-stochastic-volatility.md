# Issue 04 — Add Stochastic Volatility family (SV, SV-J)

## Why
SV models are foundational latent-volatility alternatives to observation-driven GARCH. They are fundamentally different in state-space structure and inference.

## Gap evidence
- Docs/knowledge graph mention SV-family concepts, but no runnable SV forecaster class exists.

## Proposal
Implement:
1. `SVForecaster` (latent log-vol AR(1))
2. `SVJForecaster` (SV with jump component)

## Implementation notes
- Provide at least one inference backend: particle filter / quasi-MLE / Bayesian sampler.
- Define clear computational modes (`fast`, `accurate`) for benchmark practicality.
- Return forecast mean and optionally predictive intervals.

## Acceptance criteria
- [ ] SV class with BaseForecaster-compatible API.
- [ ] Optional jump variant with explicit target declaration.
- [ ] Reproducible tests on synthetic SV DGP.

## References
- Taylor (1986).
- Harvey, Ruiz, Shephard (1994).
- Bates (1996) for jump extension.
