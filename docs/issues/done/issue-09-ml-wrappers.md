# Issue 09 — Add executable ML forecaster wrappers listed in knowledge graph

## Why
Knowledge graph and package docs reference LSTM/Transformer/RF volatility approaches, but users currently cannot instantiate equivalent forecasters in `volforecast.models`.

## Gap evidence
- No `LSTMForecaster`, `TransformerForecaster`, or `RandomForestForecaster` class exists in the model package.

## Proposal
Implement minimal, reproducible wrappers:
1. `LSTMVolForecaster`
2. `TransformerVolForecaster`
3. `RFVolForecaster`

## Implementation notes
- Keep strict feature-generation contracts (lagged returns/realized features).
- Add deterministic seeds and lightweight defaults suitable for CI.
- Make optional dependencies explicit via extras.

## Acceptance criteria
- [ ] Forecasters available via `volforecast.models` exports.
- [ ] Smoke tests run in CI with tiny datasets.
- [ ] Documentation updated to distinguish experimental ML models from econometric baselines.
