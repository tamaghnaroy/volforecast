# VolForecast Algorithm Gap Review and Issue Backlog

This backlog reviews currently implemented forecasting algorithms and proposes missing, materially different approaches for **conditional volatility (cvol)** forecasting.

## What is currently implemented

Implemented forecasters in `volforecast/models`:
- GARCH(1,1), EGARCH, GJR-GARCH, APARCH, CGARCH, Realized GARCH
- HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR

Implemented combiners in `volforecast/combination`:
- Equal Weight, Inverse MSE, AFTER, EWA, Fixed-Share, RL combiner

## Key gaps identified

1. **Foundational baseline missing:** ARCH(q), EWMA (RiskMetrics)
2. **Long-memory volatility not implemented:** FIGARCH / FIEGARCH
3. **Realized-measure alternative missing:** HEAVY
4. **Latent-state family missing in code (present only in knowledge graph/docs):** SV, SV-J, multi-factor SV
5. **Different dynamics class missing:** Score-driven (GAS/DCS) volatility
6. **Regime dynamics missing:** Markov-switching volatility
7. **Realized-data + macro horizon bridge missing:** GARCH-MIDAS / HAR-MIDAS
8. **Distributional/quantile-native volatility modeling missing:** CAViaR / quantile volatility
9. **ML models documented in knowledge graph but absent as runnable forecasters:** LSTM, Transformer, RF wrappers

## Draft issues

- [Issue 01: Add ARCH(q) and EWMA/RiskMetrics baselines](./issue-01-arch-ewma-baselines.md)
- [Issue 02: Implement FIGARCH forecaster for long memory](./issue-02-figarch.md)
- [Issue 03: Implement HEAVY model using realized measures](./issue-03-heavy.md)
- [Issue 04: Add Stochastic Volatility family (SV, SV-J)](./issue-04-stochastic-volatility.md)
- [Issue 05: Implement score-driven (GAS) volatility models](./issue-05-gas.md)
- [Issue 06: Add Markov-switching volatility models](./issue-06-markov-switching.md)
- [Issue 07: Implement MIDAS volatility models (GARCH-MIDAS / HAR-MIDAS)](./issue-07-midas.md)
- [Issue 08: Add quantile-native volatility methods (CAViaR)](./issue-08-caviar.md)
- [Issue 09: Add production ML wrappers promised by docs (LSTM/Transformer/RF)](./issue-09-ml-wrappers.md)

