# VolForecast v2 Algorithm Roadmap

## Summary

This document identifies **materially new forecasting algorithm families** not yet present in `volforecast/models` as of v1. Candidates were determined by:

1. Reviewing the v1 backlog (`docs/issues/README.md`) — which exhausts univariate GARCH, HAR, SV, GAS, Markov-switching, CAViaR, and shallow ML wrappers.
2. Cross-referencing against the 2022–2025 academic literature on conditional volatility forecasting (quantitative finance, econometrics, and machine learning venues).

Each proposed issue is **materially different** from the existing model set, not a minor parameterization change. Issues are ordered by estimated implementation effort vs. empirical payoff.

---

## Proposed Issues

---

### Issue 10 — DCC-GARCH: Dynamic Conditional Correlation (Multivariate)

**Priority:** High

#### Why
All v1 models are strictly univariate. DCC-GARCH (Engle 2002) is the canonical extension to multiple assets, modeling time-varying correlations between volatilities in addition to individual conditional variances. It is the industry standard for multi-asset risk and portfolio construction, and is fundamentally incompatible with a single-asset output contract.

#### Gap evidence
- No multivariate forecaster exists in `volforecast/models`.
- `volforecast/combination/` operates on independent scalar forecasts and cannot recover co-volatility.

#### Proposal
Implement `DCCGARCHForecaster` under `volforecast/models/multivariate.py`:
1. Univariate GARCH marginals (reuse existing `GARCHForecaster`).
2. DCC correlation update equations (Engle 2002).
3. Return the conditional covariance matrix `H_t` as forecast output.

#### Implementation notes
- Accept a 2-D returns array; output a `(T, N, N)` covariance array.
- Expose scalar per-asset volatility as a derived property for compatibility with `BaseForecaster`.
- Engle and Sheppard (2001) DECO variant (Equicorrelation) is a useful restricted sub-model.
- Consider `rmgarch` (R) as reference implementation for numerical correctness.

#### References
- Engle, R. (2002). "Dynamic Conditional Correlation." *Journal of Business & Economic Statistics.*
- Engle, R. & Sheppard, K. (2001). "Theoretical and Empirical Properties of DCC." NBER WP 8554.

#### Acceptance criteria
- [x] `DCCGARCHForecaster` implementing a multivariate extension of `BaseForecaster`.
- [x] Returns conditional covariance matrices per step.
- [x] Unit tests with 2- and 5-asset portfolios.

---

### Issue 11 — Rough Volatility: rBergomi and Rough Heston

**Priority:** High

#### Why
Empirical volatility time-series exhibit a Hurst exponent H ≈ 0.1 (Gatheral et al. 2018) — far rougher than the Brownian case (H = 0.5) assumed by all GARCH/SV models. Rough volatility models (rBergomi, rough Heston) are non-Markovian and cannot be nested inside the GARCH or SV families. They reproduce the power-law decay of the autocorrelation of absolute returns and the VIX term structure simultaneously.

#### Gap evidence
- FIGARCH captures long-memory persistence (H > 0.5) but not rough persistence (H < 0.5).
- Existing SV models use standard Brownian driving noise.

#### Proposal
Implement under `volforecast/models/rough_vol.py`:
1. `RoughBergomiForecaster` — hybrid Euler/Cholesky simulation of the rBergomi forward variance curve.
2. `RoughHestonForecaster` — fractional ODE characterisation via Adam, Delemotte et al. (2022).

#### Implementation notes
- Use the fractional Brownian motion (fBm) hybrid scheme of Bennedsen, Lunde, Pakkanen (2017) for simulation efficiency.
- Calibrate H and η to historical RV series; option-surface calibration is out of scope for v2.
- `stochvol` or custom Numpy fBm routines; no heavy external dependency required.
- Forecast output: E[RV_{t+1} | F_t] via Monte Carlo paths.

#### References
- Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance.*
- Bayer, C., Friz, P., Gatheral, J. (2016). "Pricing under rough volatility." *Quantitative Finance.*
- El Euch, O., Rosenbaum, M. (2019). "The characteristic function of rough Heston models." *Mathematical Finance.*
- Bennedsen, M., Lunde, A., Pakkanen, M. (2017). "Hybrid scheme for BSDEs." *Finance and Stochastics.*

#### Acceptance criteria
- [x] `RoughBergomiForecaster` and `RoughHestonForecaster` with H ∈ (0, 0.5) constraint.
- [x] Simulation-based 1-step ahead RV forecast output.
- [x] Hurst exponent H recoverable from backtest diagnostics.

---

### Issue 12 — DeepVol: Dilated Causal CNN on Intraday Data

**Priority:** High

#### Why
DeepVol (Mugica et al. 2022, published *Quantitative Finance* 2024) uses **dilated causal convolutions** applied directly to intraday high-frequency return sequences — not to daily summaries. The architecture recovers both the level and dynamics of realized volatility without requiring an explicit realized measure aggregation step. This is architecturally distinct from LSTM/Transformer wrappers (which consume daily feature vectors) and from HAR (which uses pre-computed RV averages).

#### Gap evidence
- `LSTMVolForecaster` and `TransformerVolForecaster` in `volforecast/models/ml_wrappers.py` consume daily feature tensors.
- No model uses raw intraday returns as direct input.

#### Proposal
Implement `DeepVolForecaster` under `volforecast/models/deep_vol.py`:
- Encoder: stack of dilated causal 1-D convolutions (WaveNet-style receptive field).
- Decoder: linear head outputting next-day RV forecast.
- Input contract: `(T, M)` array of intraday returns (M = intraday bars, T = days).

#### Implementation notes
- PyTorch optional dependency (consistent with LSTM/Transformer).
- Default dilation schedule: 1, 2, 4, 8, 16 (covers ≥32 intraday bars).
- Normalise intraday returns per day to avoid scale shift.
- Reference: `arxiv 2210.04797`.

#### References
- Mugica, M., Trottier, D., Godin, F. (2022/2024). "DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions." *Quantitative Finance.*

#### Acceptance criteria
- [x] `DeepVolForecaster` accepting intraday bar data.
- [x] Graceful degradation or skip when PyTorch is absent.
- [x] Benchmark comparison vs. HAR-RV on same realized-measure targets.

---

### Issue 13 — HAR-IV: Implied Volatility Augmented HAR

**Priority:** Medium

#### Why
The standard HAR family uses only realized-measure inputs (RV, jumps, continuous variation). Bekaert and Hoerova (2014) and Prokopczuk et al. (2023) show that **options-implied volatility** (e.g., VIX, model-free IV) contains forward-looking information orthogonal to past realized variance and materially improves out-of-sample forecasts. HAR-IV is a distinct input regime, not a variant of HAR-RV-J.

#### Gap evidence
- All four HAR models in `volforecast/models/har.py` accept only realized-measure feature vectors.
- No forecaster accepts an options-derived IV column.

#### Proposal
Implement `HARIVForecaster` under `volforecast/models/har.py`:
- Extends `HARForecaster` with an additional IV regressor column.
- IV input can be VIX (for S&P 500), at-the-money IV, or model-free IV.
- Optional: HAR-IV-J (with jump component).

#### Implementation notes
- IV and RV must be on matched daily frequency; document alignment requirements.
- Allow `iv_column` parameter to specify which column in input DataFrame carries implied vol.
- Estimate by OLS; keep consistent with existing HAR implementation style.

#### References
- Prokopczuk, M., Symeonidis, L., Wese Simen, C. (2023). "Forecasting realized volatility: the role of implied volatility, leverage effect, overnight returns, and volatility of volatility." *Journal of Futures Markets.*
- Bekaert, G., Hoerova, M. (2014). "The VIX, the variance premium, and stock market volatility." *Journal of Econometrics.*

#### Acceptance criteria
- [x] `HARIVForecaster` with `iv_column` parameter.
- [x] Unit tests confirming IV regressor is used in forecast.
- [x] Documented alignment expectations for IV vs. RV inputs.

---

### Issue 14 — Realized Kernel Estimators (Noise-Robust HF Volatility)

**Priority:** Medium

#### Why
Standard realized variance sums squared intraday returns and is biased upward under microstructure noise. Realized Kernels (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008/2009) provide a **consistent, noise-robust** estimator of integrated variance using a bandwidth-selected kernel weighting of autocovariances. This is a materially different input to HAR and Realized GARCH models — not a new forecasting model, but a new **realized measure** that existing forecasters can consume.

#### Gap evidence
- `volforecast/realized/` (if present) or the input contract for `RealizedGARCHForecaster` does not compute or accept realized kernel estimates.
- Standard RV (sum of squared returns) is the only high-frequency input currently used.

#### Proposal
Implement under `volforecast/realized/kernels.py`:
1. `realized_kernel(returns_intraday, kernel='parzen', bandwidth=None)` — returns scalar RK estimate per day.
2. Auto-bandwidth selection (Barndorff-Nielsen et al. 2009 Rule of Thumb).
3. Supported kernels: Parzen, Bartlett-Priestley-Epanechnikov, cubic.

#### Implementation notes
- Numpy only; no heavy dependency.
- Can be used as a drop-in replacement for `realized_variance()` in feature construction pipelines.
- Expose as a standalone utility, not a `BaseForecaster` subclass.

#### References
- Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., Shephard, N. (2008). "Designing Realized Kernels to Measure the ex-post Variation of Equity Prices." *Econometrica.*
- Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., Shephard, N. (2009). "Realized Kernels in Practice: Trades and Quotes." *Econometrics Journal.*

#### Acceptance criteria
- [x] `realized_kernel()` function with Parzen and Bartlett kernels.
- [x] Unit tests confirming noise-robustness (bias reduction vs. naive RV on synthetic noisy data).
- [x] Integration with `RealizedGARCHForecaster` input pipeline.

---

### Issue 15 — Copula-GARCH with EVT Tails

**Priority:** Medium

#### Why
DCC-GARCH (Issue 10) models correlation with Gaussian or Student-t assumptions. Copula-GARCH separates the marginal GARCH volatility from the **dependence structure**, allowing asymmetric and heavy-tailed joint distributions via vine or elliptical copulas. Combined with Extreme Value Theory (EVT/GPD) tails, this is the standard approach for multi-asset VaR and Expected Shortfall in practice.

#### Gap evidence
- No copula-based joint model exists in `volforecast`.
- CAViaR provides univariate quantile regression; no multi-asset tail model exists.

#### Proposal
Implement `CopulaGARCHForecaster` under `volforecast/models/copula_garch.py`:
1. Fit marginal GARCH models (reuse existing forecasters).
2. Standardize residuals to PIT (probability integral transform).
3. Fit selected copula (Gaussian, Student-t, Clayton, Gumbel) to PIT residuals.
4. Optionally fit GPD tails above/below threshold per marginal (EVT step).
5. Simulate joint scenarios for portfolio VaR/ES.

#### Implementation notes
- `scipy.stats` provides copula sampling; `copulas` (SDV) is a heavier option.
- Start with elliptical copulas; vine copulas (R `VineCopula`) are out of scope for v2.
- The EVT tail fitting can be a separate utility in `volforecast/evaluation/`.

#### References
- Sklar, A. (1959). Copula theorem.
- Joe, H. (2014). *Dependence Modeling with Copulas.* CRC Press.
- McNeil, A., Frey, R., Embrechts, P. (2005). *Quantitative Risk Management.* Princeton UP.

#### Acceptance criteria
- [x] `CopulaGARCHForecaster` accepting multi-asset returns.
- [x] At least Gaussian and Student-t copula.
- [x] Portfolio VaR/ES output from joint simulation.

---

### Issue 17 — Conformal Prediction Intervals for Volatility

**Priority:** Medium

#### Why
All current forecasters produce point estimates. Existing interval methods (e.g., GARCH conditional variance ± z·σ) rely on distributional assumptions that are frequently violated. **Conformal prediction** (Vovk et al. 2005; Gibbs & Candès 2021 for time-series) provides **finite-sample, distribution-free coverage guarantees** by wrapping any base forecaster. This is materially different from the RL/EWA combination approaches in `volforecast/combination/` because it operates on residuals rather than weights.

#### Gap evidence
- No forecaster or combiner in `volforecast` produces calibrated prediction intervals.
- `volforecast/evaluation/` does not compute empirical coverage metrics.

#### Proposal
Implement under `volforecast/evaluation/conformal.py`:
1. `SplitConformalVol` — split-conformal wrapper around any `BaseForecaster`.
2. `OnlineConformalVol` — rolling conformal via Gibbs-Candès (2021) for non-exchangeable data.
3. `coverage_diagnostic(intervals, actuals)` — returns empirical coverage and interval width.

#### Implementation notes
- No additional heavy dependency; pure NumPy.
- `SplitConformalVol` uses a calibration split to compute nonconformity scores (absolute residuals).
- `OnlineConformalVol` adaptively updates the quantile level α_t each step.
- Integrates with `BenchmarkRunner` for coverage reporting.

#### References
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Gibbs, I., Candès, E. (2021). "Adaptive Conformal Inference Under Distribution Shift." *NeurIPS.*
- Zaffran, M. et al. (2022). "Adaptive Conformal Predictions for Time Series." *ICML.*

#### Acceptance criteria
- [x] `SplitConformalVol` wrapper compatible with any `BaseForecaster`.
- [x] `OnlineConformalVol` with adaptive α update.
- [x] `coverage_diagnostic` integrated into benchmark output.

---

### Issue 18 — MSGARCH: Markov-Switching GARCH Dynamics

**Priority:** Medium

#### Why
The existing `MSVolForecaster` (`markov_switching.py`) switches between K constant-variance regimes. **MSGARCH** (Haas, Mittnik, Paolella 2004; Augustyniak 2014) places a full GARCH(1,1) in each regime, so volatility persistence and shock response differ across regimes rather than just the level. This is widely used in risk management and is empirically superior to constant-regime models.

#### Gap evidence
- `MSVolForecaster` has fixed per-regime variance constants (no GARCH dynamics within regimes).
- MSGARCH appears in the academic literature as a materially different model class.

#### Proposal
Implement `MSGARCHForecaster` under `volforecast/models/markov_switching.py`:
- K-regime model, each with independent GARCH(1,1) parameters (ω_k, α_k, β_k).
- Hamilton filter for regime probabilities.
- Forecast: probability-weighted sum of regime-conditional GARCH forecasts.

#### Implementation notes
- Extend `MSVolForecaster` rather than rewrite; share Hamilton filter utilities.
- Initialise each regime's GARCH from a single-regime fit, then refine jointly.
- Guard against degenerate regimes (near-zero ω or α + β ≈ 1) with parameter bounds.

#### References
- Haas, M., Mittnik, S., Paolella, M. (2004). "A New Approach to Markov-Switching GARCH Models." *Journal of Financial Econometrics.*
- Augustyniak, M. (2014). "Maximum likelihood estimation of the Markov-switching GARCH model." *Computational Statistics & Data Analysis.*

#### Acceptance criteria
- [x] `MSGARCHForecaster` with K-regime GARCH dynamics per state.
- [x] Regime-conditional forecast and smoothed regime probabilities as outputs.
- [x] Unit tests confirming K=1 reduces to standard GARCH(1,1).

---

## Priority Summary

| Issue | Model Family | Priority | Effort | Key Novelty vs. v1 |
|-------|-------------|----------|--------|---------------------|
| 10 | DCC-GARCH | High | Medium | First multivariate forecaster |
| 11 | Rough Volatility (rBergomi / rough Heston) | High | High | Non-Markovian, H < 0.5 |
| 12 | DeepVol (Dilated Causal CNN) | High | Medium | Raw intraday input |
| 13 | HAR-IV | Medium | Low | Implied vol as regressor |
| 14 | Realized Kernels | Medium | Low | Noise-robust HF estimator |
| 15 | Copula-GARCH + EVT | Medium | Medium | Non-Gaussian joint tails |
| 17 | Conformal Prediction Intervals | Medium | Low | Distribution-free coverage |
| 18 | MSGARCH | Medium | Low | Per-regime GARCH dynamics |

## Suggested v2 Delivery Order

**Phase 1 (Quick wins + high impact):**
Issues 13, 14, 17, 18 — all low-effort, medium-to-high payoff, no new heavy dependencies.

**Phase 2 (Structural additions):**
Issues 10, 12, 15 — multivariate support, new input regimes, medium effort.

**Phase 3 (Research frontier):**
Issue 11 — high novelty, higher compute or dependency requirements.
