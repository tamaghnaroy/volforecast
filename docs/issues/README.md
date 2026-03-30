# VolForecast Algorithm Gap Review and Issue Backlog

This backlog reviews currently implemented forecasting algorithms and proposes missing, materially different approaches for **conditional volatility (cvol)** forecasting.

## What is currently implemented

Implemented forecasters in `volforecast/models`:
- **GARCH family:** ARCH(q), EWMA/RiskMetrics, GARCH(1,1), EGARCH, GJR-GARCH, APARCH, CGARCH, FIGARCH, HEAVY, Realized GARCH, GARCH-MIDAS
- **HAR family:** HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR, **HAR-IV** (issue 13)
- **Stochastic Volatility:** SV (QML), SV-J (with jumps)
- **Score-driven:** GAS(1,1) with Normal and Student-t
- **Regime-switching:** Markov-Switching Volatility (K regimes), **MSGARCH** (issue 18)
- **Quantile:** CAViaR (SAV, Asymmetric Slope, Indirect GARCH)
- **ML wrappers:** Random Forest, LSTM, Transformer (PyTorch optional), **DeepVol/dilated causal CNN** (issue 12)
- **Multivariate:** **DCC-GARCH** (issue 10), **Copula-GARCH + EVT tails** (issue 15)
- **Rough Volatility:** **rBergomi, Rough Heston** (issue 11)

Realized measures in `volforecast/realized`:
- RV, BV, MedRV, MinRV, TSRV, Pre-Averaging, Realized Semivariance
- **Realized Kernel with Parzen, Bartlett, and cubic kernels + auto-BW** (issue 14)

Evaluation tools in `volforecast/evaluation`:
- MSE, QLIKE, MAE loss functions; Diebold-Mariano, MCS, Mincer-Zarnowitz tests
- **Conformal Prediction Intervals** — `SplitConformalVol`, `OnlineConformalVol` (issue 17)

Implemented combiners in `volforecast/combination`:
- Equal Weight, Inverse MSE, AFTER, EWA, Fixed-Share, RL combiner

## Open issues

None — all v1 and v2 identified gaps have been addressed.

## Done

- [Issue 01: Add ARCH(q) and EWMA/RiskMetrics baselines](./done/issue-01-arch-ewma-baselines.md)
- [Issue 02: Implement FIGARCH forecaster for long memory](./done/issue-02-figarch.md)
- [Issue 03: Implement HEAVY model using realized measures](./done/issue-03-heavy.md)
- [Issue 04: Add Stochastic Volatility family (SV, SV-J)](./done/issue-04-stochastic-volatility.md)
- [Issue 05: Implement score-driven (GAS) volatility models](./done/issue-05-gas.md)
- [Issue 06: Add Markov-switching volatility models](./done/issue-06-markov-switching.md)
- [Issue 07: Implement MIDAS volatility models (GARCH-MIDAS / HAR-MIDAS)](./done/issue-07-midas.md)
- [Issue 08: Add quantile-native volatility methods (CAViaR)](./done/issue-08-caviar.md)
- [Issue 09: Add production ML wrappers promised by docs (LSTM/Transformer/RF)](./done/issue-09-ml-wrappers.md)

