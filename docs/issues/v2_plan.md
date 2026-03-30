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
- [ ] `DCCGARCHForecaster` implementing a multivariate extension of `BaseForecaster`.
- [ ] Returns conditional covariance matrices per step.
- [ ] Unit tests with 2- and 5-asset portfolios.

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
- [ ] `RoughBergomiForecaster` and `RoughHestonForecaster` with H ∈ (0, 0.5) constraint.
- [ ] Simulation-based 1-step ahead RV forecast output.
- [ ] Hurst exponent H recoverable from backtest diagnostics.

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
- [ ] `DeepVolForecaster` accepting intraday bar data.
- [ ] Graceful degradation or skip when PyTorch is absent.
- [ ] Benchmark comparison vs. HAR-RV on same realized-measure targets.

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
- [ ] `HARIVForecaster` with `iv_column` parameter.
- [ ] Unit tests confirming IV regressor is used in forecast.
- [ ] Documented alignment expectations for IV vs. RV inputs.

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
- [ ] `realized_kernel()` function with Parzen and Bartlett kernels.
- [ ] Unit tests confirming noise-robustness (bias reduction vs. naive RV on synthetic noisy data).
- [ ] Integration with `RealizedGARCHForecaster` input pipeline.

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
- [ ] `CopulaGARCHForecaster` accepting multi-asset returns.
- [ ] At least Gaussian and Student-t copula.
- [ ] Portfolio VaR/ES output from joint simulation.

---

### Issue 16 — Neural SDE Volatility

**Priority:** Low

#### Why
Neural Stochastic Differential Equations (Neural SDEs, Kidger et al. 2021; Li et al. 2020) parameterize the drift and diffusion of a latent SDE with neural networks, learned end-to-end. Applied to volatility, this gives a continuous-time, flexible latent factor model that subsumes SV and rough volatility as special cases without prescribing the functional form of the dynamics. This is the leading research-frontier approach to volatility as of 2024.

#### Gap evidence
- Existing SV models have fixed (Ornstein-Uhlenbeck) dynamics.
- Existing ML wrappers (LSTM, Transformer) are discrete-time and do not model a latent diffusion process.

#### Proposal
Implement `NeuralSDEVolForecaster` under `volforecast/models/neural_sde.py` (PyTorch optional):
- Latent SDE with neural drift `f_θ(h_t)` and diffusion `g_θ(h_t)`.
- Observation model: `log(RV_t) = readout(h_t) + ε_t`.
- Train via the adjoint method or Euler-Maruyama simulation + KL regularisation.
- Reference implementation: `torchsde` library (Kidger et al.).

#### Implementation notes
- Hard dependency on `torch` and `torchsde`.
- Inference is simulation-based; expose `n_samples` parameter for forecast distribution.
- Treat as a research/experimental model; separate from production-grade models.

#### References
- Kidger, P., Foster, J., Li, X., Oberhauser, H., Lyons, T. (2021). "Neural SDEs as Infinite-Dimensional GANs." *ICML.*
- Li, X. et al. (2020). "Scalable Gradients and Variational Inference for Stochastic Differential Equations." *AISTATS.*

#### Acceptance criteria
- [ ] `NeuralSDEVolForecaster` (torch-optional) with `fit` / `predict` API.
- [ ] At minimum, 1-step ahead point forecast and predictive interval.
- [ ] Marked as `experimental` in docstring and `__init__.py`.

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
- [ ] `SplitConformalVol` wrapper compatible with any `BaseForecaster`.
- [ ] `OnlineConformalVol` with adaptive α update.
- [ ] `coverage_diagnostic` integrated into benchmark output.

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
- [ ] `MSGARCHForecaster` with K-regime GARCH dynamics per state.
- [ ] Regime-conditional forecast and smoothed regime probabilities as outputs.
- [ ] Unit tests confirming K=1 reduces to standard GARCH(1,1).

---

### Issue 19 — Diffusion / Score-Based Generative Volatility

**Priority:** Low

#### Why
Denoising diffusion probabilistic models (Ho et al. 2020) and score-based generative models (Song et al. 2020) have been adapted to financial time series (FinDiff 2023; GBM-Diffusion 2025). Applied to volatility, these models learn the **full conditional distribution** of the next volatility trajectory, not just its mean. Conditional generation under extreme/rare volatility regimes is a novel capability absent from all existing model classes.

#### Gap evidence
- No model in `volforecast` outputs a distributional forecast beyond point estimates or parametric intervals.
- Neural SDE (Issue 16) gives a continuous-time process; diffusion models give a discrete-time conditional generative model — complementary, not redundant.

#### Proposal
Implement `DiffusionVolForecaster` under `volforecast/models/diffusion_vol.py` (PyTorch):
- DDPM-style forward/reverse Markov chain on log-RV sequences.
- Condition on past T lags of log-RV (via cross-attention or concatenated context).
- Inference: reverse diffusion with N denoising steps → distributional forecast sample.

#### Implementation notes
- Hard dependency on `torch`; mark as `experimental`.
- Keep to a small U-Net or Transformer backbone to avoid compute prohibitiveness.
- Evaluate with CRPS (Continuous Ranked Probability Score) and WIS, not just RMSE.

#### References
- Ho, J., Jain, A., Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS.*
- Lim, H. et al. (2023). "FinDiff: Diffusion Models for Financial Tabular Data Generation." *ICAIF.*
- Tanaka et al. (2025). "A diffusion-based generative model for financial time series via GBM." *arXiv 2507.19003.*

#### Acceptance criteria
- [ ] `DiffusionVolForecaster` sampling N trajectories from conditional distribution.
- [ ] CRPS evaluation in benchmark runner.
- [ ] Marked as `experimental`.

---

### Issue 20 — LLM / Sentiment-Augmented Volatility

**Priority:** Low

#### Why
Large language models (FinBERT, FinGPT, GPT-4) can extract forward-looking volatility signals from financial news, earnings call transcripts, and central bank communication. Sentiment-augmented GARCH and HAR models demonstrate statistically significant improvements in out-of-sample forecasts (Ballinari & Behrendt 2021; Zhang et al. 2024). This is materially different from all existing models because it ingests **unstructured text** as an exogenous input.

#### Gap evidence
- No model in `volforecast` accepts text or pre-computed sentiment scores.
- All exogenous inputs (IV, realized jumps) are numerical time-series.

#### Proposal
Implement `SentimentGARCHForecaster` under `volforecast/models/sentiment_vol.py`:
- Augments GARCH(1,1) mean equation with a daily sentiment score Sₜ.
- Sentiment score is an external input (user-provided or via FinBERT API).
- Optional: `SentimentHARForecaster` extending HAR-IV similarly.

#### Implementation notes
- No LLM inference dependency in `volforecast`; sentiment scores are pre-computed externally.
- Accept `sentiment_series: pd.Series` aligned to the returns index.
- Validate alignment and handle missing sentiment days via forward-fill or zero.

#### References
- Ballinari, D., Behrendt, S. (2021). "Sentiment-augmented volatility forecasting." *Finance Research Letters.*
- Zhang, W. et al. (2024). "Large language models for financial sentiment and volatility prediction." *Journal of International Financial Markets.*

#### Acceptance criteria
- [ ] `SentimentGARCHForecaster` with external sentiment input.
- [ ] Unit test confirming zero-sentiment reduces to standard GARCH(1,1).
- [ ] Clear documentation that sentiment extraction is external to this library.

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
| 16 | Neural SDE | Low | High | Latent continuous-time SDE |
| 17 | Conformal Prediction Intervals | Medium | Low | Distribution-free coverage |
| 18 | MSGARCH | Medium | Low | Per-regime GARCH dynamics |
| 19 | Diffusion / Score-Based | Low | High | Full distributional forecast |
| 20 | LLM Sentiment-Augmented | Low | Low | Text as exogenous input |

## Suggested v2 Delivery Order

**Phase 1 (Quick wins + high impact):**
Issues 13, 14, 17, 18 — all low-effort, medium-to-high payoff, no new heavy dependencies.

**Phase 2 (Structural additions):**
Issues 10, 12, 15 — multivariate support, new input regimes, medium effort.

**Phase 3 (Research frontier):**
Issues 11, 16, 19, 20 — high novelty, higher compute or dependency requirements.
