# SOTA Architectural Feedback for AutoVolForecaster

AutoVolForecaster is already a strong 2010s-style volatility platform: the candidate pool spans the main classical families, `BaseForecaster` is flexible, `BenchmarkRunner` supports rolling OOS benchmarking, and the online combination layer is usable. It is not yet 2020-2025 SOTA. The missing pieces are mostly additive rather than structural: richer realized-volatility families, genuinely modern hybrid neural/econometric models, second-order online aggregation, multi-proxy evaluation, explicit drift handling, and production-grade uncertainty/calibration.

The good news is that the current architecture can absorb most of these upgrades without replacing the framework. The main work is in extending `volforecast/models`, tightening `volforecast/evaluation` and `volforecast/benchmark`, and upgrading `volforecast/combination`.

## 1. Model Coverage Gaps

### Overall assessment

The current pool is broad for a research library, but the 2020-2025 frontier moved in five directions that the repo does not yet cover well:

1. realized-volatility score-driven models;
2. neural/econometric hybrids instead of generic black-box ML wrappers;
3. attention/panel/commonality architectures rather than univariate sequence models only;
4. modern rough-volatility forecasting and calibration, not just simple moment-matching plus Monte Carlo;
5. interval- and density-aware forecasting layers, especially conformal calibration.

### Missing families and fit with the current API

| Gap | What it adds | Key paper(s) | Can `BaseForecaster` handle it? |
|---|---|---|---|
| Neural GARCH / GARCH-informed neural networks | Lets the model keep economically interpretable variance recursion while learning nonlinear residual structure, asymmetric effects, and feature interactions that plain GARCH misses. This is much closer to the current literature frontier than the repo's generic `LSTMVolForecaster` and `TransformerVolForecaster`. | Liu and So (2020), "A GARCH Model with Artificial Neural Networks" ([MDPI](https://doi.org/10.3390/info11100489)); Wei, Yang and Cui (2025), "Unified GARCH-Recurrent Neural Network in Financial Volatility Forecasting" ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5215228)); Petrosino et al. (2025), "A GARCH-temporal fusion transformer model..." ([Springer](https://link.springer.com/article/10.1007/s00521-025-11468-z)) | Yes. These are straightforward `BaseForecaster` subclasses. The main repo change is richer feature plumbing and warm-start support in `volforecast/models/ml_wrappers.py`. |
| Conformal prediction intervals for volatility | Adds distribution-free or approximately valid forecast intervals around any point forecaster, which is critical for live risk use. For volatility systems, the right design is not a separate model family but a calibration layer over any base model or combined forecast. | Xu and Xie (2023), "Conformal Prediction for Time Series" ([PubMed/IEEE](https://pubmed.ncbi.nlm.nih.gov/37819805/)); Xu and Xie (2023), "Sequential Predictive Conformal Inference for Time Series" ([PMLR](https://proceedings.mlr.press/v202/xu23r.html)); Fantazzini (2024), "Adaptive Conformal Inference for Computing Market Risk Measures" ([MDPI](https://www.mdpi.com/1911-8074/17/6/248)); Calleo (2025), "A note on adaptive conformal prediction for time series with structural breaks" ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5946791)) | Partly. `ForecastResult` can carry uncertainty in `variance` or `metadata`, but SOTA implementation wants explicit interval/quantile fields. This is a small API extension, not an architectural blocker. |
| Score-driven extensions beyond current `GASVolForecaster` | The current GAS implementation is univariate, return-based, and limited to Gaussian or Student innovations. The frontier uses realized-volatility DCS/GAS models with two components, leverage, seasonality, heteroskedastic measurement errors, and fat-tailed GB2/EGB2 distributions. These often beat HAR variants in realized-vol forecasting. | Harvey and Palumbo (2023), "Score-driven models for realized volatility" ([Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2023.01.029)); Cipollini, Gallo and Otranto (2023), "Time-varying variance and skewness in realized volatility measures" ([IJF](https://doi.org/10.1016/j.ijforecast.2022.02.009)) | Yes. This is the cleanest near-term addition: create `RealizedGASForecaster` or extend `gas.py` with realized-vol targets and richer score equations. |
| Rough-volatility calibration improvements | The current rough-vol classes are too stylized for SOTA use: Hurst via crude variogram, parameter calibration via simple moments, and expensive Monte Carlo-only prediction. The modern practice is quadratic rough Heston / Zumbach-aware forecasting and Markovian approximations that make rough models calibratable and deployable. | Tang, Rosenbaum and Zhou (2024), "Forecasting volatility with machine learning and rough volatility" ([SSRN/arXiv](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4626835)); Abi Jaber and Li (2024), "Volatility Models in Practice: Rough, Path-Dependent or Markovian?" ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4684016)); Ye, Fan and Kwok (2025), "VIX Term Structure in the Rough Heston Model via Markovian Approximation" ([HKUST](https://researchportal.hkust.edu.hk/en/publications/vix-term-structure-in-the-rough-heston-model-via-markovian-approx/)) | Yes, but with work. The API is fine; the implementation needs better state representation, faster calibration, and possibly optional auxiliary inputs such as term-structure features or Zumbach terms. |
| Realized GARCH-MIDAS hybrids | The repo has `RealizedGARCHForecaster` and `GARCHMIDASForecaster`, but not the modern hybrids that combine high-frequency measurement equations, long-run MIDAS structure, jump robustness, or higher moments. These are directly relevant to the current code because the library already has realized-measure and MIDAS plumbing. | Liu, Maheu and Yang (2021), "A realized EGARCH-MIDAS model with higher moments" ([ScienceDirect](https://doi.org/10.1016/j.frl.2019.101392)); Chevallier and Sanhaji (2023), "Jump-Robust Realized-GARCH-MIDAS-X Estimators..." ([MDPI](https://doi.org/10.3390/stats6040082)) | Yes. `fit(..., realized_measures=..., **kwargs)` already provides the right extension point. The missing piece is exogenous low-frequency driver support in the benchmark and auto-selection layers. |
| Transformer/attention-based realized-vol models | The repo's ML wrappers are generic lagged-feature learners. The 2020-2025 frontier uses attention models that exploit long-range temporal structure, exogenous covariates, and sometimes pooling across assets or sectors. These are materially different from a plain sequence-to-one Transformer. | Ramos-Perez, Alonso-Gonzalez and Nunez-Velazquez (2021), "Multi-Transformer..." ([MDPI](https://doi.org/10.3390/math9151794)); Frank (2023), "Forecasting realized volatility in turbulent times using temporal fusion transformers" ([RePEc](https://econpapers.repec.org/RePEc%3Azbw%3Aiwqwdp%3A032023)); Zhang et al. (2024), "Volatility Forecasting with Machine Learning and Intraday Commonality" ([SSRN/JFEC metadata](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4022147)) | Yes, with richer feature interfaces. The natural move is a new module, not more code in `ml_wrappers.py`, because TFT/panel/commonality models need covariates and grouped training. |
| Panel/commonality models for pooled realized volatility | Some of the best recent forecasting gains come from pooling information across assets and using common market volatility factors. The current architecture is strictly single-series in spirit, even though the API could pass extra regressors. | Zhang et al. (2024), "Volatility Forecasting with Machine Learning and Intraday Commonality" ([SSRN/JFEC metadata](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4022147)); Hu and Tsay (2023), "Forecasting realized volatility with machine learning: Panel data perspective" ([JEF](https://doi.org/10.1016/j.jempfin.2023.07.003)) | Mostly yes. `BaseForecaster` can stay unchanged, but the benchmark/auto layers need optional pooled features and cross-sectional training data. |

### Bottom line

The single biggest model-family omission is not "more ML" in general. It is realized-aware, hybrid, and pooled models:

- realized-volatility DCS/GAS;
- GARCH-informed neural hybrids;
- REGARCH-MIDAS-style hybrids;
- attention/commonality models;
- production-grade rough-volatility forecasters.

Those fit the current architecture better than a full framework rewrite.

## 2. Forecast Combination Frontier

### Regret bounds: are EWA / FixedShare / AFTER enough?

They are a good baseline, but not a frontier stack.

- `EWACombiner` is still the right default floor because it is simple, robust, and has the standard `O(sqrt(T log K))` adversarial regret.
- `FixedShareCombiner` is the right regime-switching floor because it tracks the best sequence of experts.
- `AFTERCombiner` is a useful classical adaptive aggregator, but it is still a first-order method.

That is not enough for a SOTA 2020-2025 architecture. The missing frontier pieces are:

1. second-order aggregation such as BOA, which adapts to loss variance and can achieve faster rates in easier environments;
2. adaptive-eta Hedge variants such as AdaHedge, which dominate fixed learning-rate EWA when the environment alternates between easy and adversarial;
3. stabilized online mirror descent / dual averaging, which gives more flexible geometry and discounting than plain Hedge.

Key references:

- Wintenberger (2017), "Optimal learning with Bernstein online aggregation" ([University of Copenhagen portal](https://researchprofiles.ku.dk/en/publications/optimal-learning-with-bernstein-online-aggregation/));
- de Rooij et al. (2014), "Follow the Leader If You Can, Hedge If You Must" ([JMLR](https://jmlr.org/beta/papers/v15/rooij14a.html));
- Fang, Harvey, Portella and Friedlander (2020), "Online mirror descent and dual averaging..." ([PMLR](https://proceedings.mlr.press/v119/fang20a.html)).

Recommendation:

- keep `EWA`, `FixedShare`, and `AFTER`;
- add `BOACombiner`, `AdaHedgeCombiner`, and a discounted/stabilized `OMDCombiner`;
- add discount factors and restart hooks to all online combiners.

### Conformal combination: should combined forecasts carry intervals?

Yes. A SOTA production forecaster should output:

- a point forecast;
- an interval or quantile band;
- a calibration/coverage diagnostic.

The right design is:

1. combine experts first to obtain the point forecast;
2. conformalize the combined residual process, not the expert intervals independently;
3. use time-series conformal methods such as EnbPI or SPCI on the combined forecast residuals;
4. reset or discount the calibration window after detected breaks.

This is especially important because the repo's current combination layer only optimizes point losses. That is acceptable for a baseline research library, not for a SOTA auto-forecaster.

### Meta-learning: should combiner selection itself be learned from data?

Yes. Hard-coded selection rules like:

- "small sample -> equal weight",
- "regime switching -> fixed share",
- "large T -> RL"

are too crude for a serious automated system.

The best low-risk design within the existing architecture is:

- treat combiners as experts;
- let each combiner produce its own combined forecast;
- run an outer BOA/AdaHedge/FixedShare layer over the combiners;
- optionally condition the prior weights on regime features such as rolling QLIKE dispersion, break scores, jump intensity, and proxy quality.

This is much safer than making `RLCombiner` the top-level selector. RL should remain opt-in research mode until it beats BOA/AdaHedge in rolling OOS tests.

### Stacking vs online: when should static stacking beat online updating?

Static stacking should beat pure online updating when:

- the regime is stable;
- there is a long calibration sample;
- the expert set is fixed;
- the deployment target matches the validation target and proxy;
- you can estimate constrained weights from rolling-origin OOS forecasts.

Online updating should beat static stacking when:

- there are breaks or volatility state changes;
- expert quality rotates over time;
- data arrive sequentially and retraining is expensive;
- the system must adapt before enough new data accumulate for a restack.

Best practice for this repo:

- learn a static stacked prior from historical OOS benchmark forecasts;
- initialize the online combiner at those weights;
- then fine-tune online with discounted BOA/AdaHedge/FixedShare.

That "stack-then-track" design is stronger than either static stacking or purely online weights alone.

## 3. Evaluation & Selection Rigor

### Is the current MCS + DM + Patton-robust stack sufficient?

It is a strong baseline. It is not SOTA by itself.

What it already does well:

- robust losses (`MSE`, `QLIKE`);
- pairwise DM testing;
- MCS-style survivor selection;
- basic MZ calibration checks.

What is missing:

- data-snooping control when the model pool gets large;
- conditional, state-dependent predictive ability tests;
- evaluation across multiple realized proxies rather than one main proxy;
- nested rolling-origin hyperparameter validation for neural and hybrid models;
- regime-specific and stress-period reporting.

### Should we add SPA?

Yes.

Why:

- once the candidate pool includes neural hybrids, realized-GAS variants, rough models, and multiple combiners, plain DM plus MCS is not enough protection against data snooping;
- Hansen's SPA is specifically designed to test whether the best observed model genuinely has superior predictive ability over a benchmark in a many-model search setting.

Key references:

- White (2000), "A Reality Check for Data Snooping" ([Econometric Society](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping));
- Hansen (2005), "A Test for Superior Predictive Ability" ([Taylor and Francis](https://doi.org/10.1198/073500105000000063)).

Recommendation:

- add SPA in `volforecast/evaluation/tests.py`;
- use it after coarse ranking and before final survivor set construction;
- report both SPA and MCS, because they answer different questions.

### Giacomini-White conditional predictive ability test?

Yes, especially once the system becomes regime-aware.

DM tells you whether one model is better on average. Giacomini-White tells you whether one model is better conditional on current information. For volatility, that matters because model superiority is often state-dependent:

- calm vs crisis periods;
- jump vs non-jump days;
- low vs high proxy noise periods;
- leverage-dominated vs symmetric episodes.

Key reference:

- Giacomini and White (2006), "Tests of Conditional Predictive Ability" ([Econometric Society](https://doi.org/10.1111/j.1468-0262.2006.00718.x)).

Recommendation:

- add GW CPA to `volforecast/evaluation/tests.py`;
- use conditioning instruments such as lagged RV, jump flags, sign of return, realized quarticity, and break-detector state.

### Should the proxy quality gate use multiple realized measures, not just RV?

Absolutely yes.

This is one of the biggest evaluation gaps in the current architecture.

The repo already implements:

- `BV`, `MedRV`, `MinRV`;
- realized kernel;
- TSRV;
- pre-averaging;
- jump decomposition.

But `BenchmarkRunner` still evaluates almost everything against a single `RV` stream. That throws away one of the library's biggest built-in advantages.

SOTA practice should be:

- use `RV` when the target is total quadratic variation;
- use `BV`/`MedRV`/`MinRV` when the target is continuous variation or jump-robust IV;
- use realized kernel / TSRV / pre-averaging when microstructure noise is material;
- check whether model rankings are stable across proxies, not just whether one proxy says so.

Recommendation:

- compute a proxy panel in `BenchmarkRunner`;
- score each model across all admissible proxies for its `TargetSpec`;
- rank by robust-loss stability or average rank, not only raw `RV` QLIKE.

### Cross-validation strategies for time series

To match SOTA methodology:

- keep rolling-origin / expanding-window OOS evaluation as the primary benchmark;
- add nested rolling-origin validation for hyperparameters and combiner tuning;
- use blocked or hv-block CV only for certain model-selection subproblems;
- use purged/embargoed folds whenever targets overlap across horizons or features use overlapping lookbacks;
- make purging mandatory for `horizon > 1`.

Key references:

- Bergmeir, Hyndman and Koo (2018), "A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction" ([author page](https://cbergmeir.com/publications/2018-01-01_bergmeir2018note/));
- Lopez de Prado (2018), *Advances in Financial Machine Learning* for purged/embargoed CV.

## 4. Online Adaptivity & Regime Handling

### How should the system handle nonstationarity beyond FixedShare?

FixedShare is necessary. It is not sufficient.

The SOTA design here is a two-layer adaptivity stack:

1. continuous adaptation in weights and calibration windows;
2. discrete resets/refits triggered by drift or change detection.

### Adaptive windowing

Recommended additions:

- ADWIN for adaptive loss-window selection;
- Page-Hinkley or CUSUM on combined forecast loss;
- detector-specific reset logic for conformal calibration windows.

Where to apply detectors:

- combined QLIKE / MSE;
- expert regret gaps;
- proxy signal-to-noise ratio;
- interval coverage error after conformalization;
- disagreement across realized proxies.

### Change-point detection triggering model refit

Yes. This should be explicit policy, not an informal suggestion.

Suggested actions after detection:

- reset online combiner weights toward equal or stacked priors;
- refit only the top `N` active experts first;
- re-run cheap experts immediately, expensive experts lazily;
- shrink the effective training window until stability returns.

This is more production-worthy than calendar-only `refit_every`.

### Regime-aware weight scheduling

Yes.

A practical implementation is:

- compute rolling regime features: recent loss dispersion, jump share, realized skewness, Hurst estimate, break score, proxy quality;
- map those features to combiner priors or combiner choice;
- optionally keep specialist experts active only in matching regimes.

This is implementable without changing `BaseForecaster`. It belongs in the combination/auto layer.

### Forgetting/discounting in the combination layer

This should be standard.

Recommended:

- discounted cumulative losses for all online combiners;
- time-varying learning rates;
- restartable BOA/AdaHedge;
- sleeping-expert logic for expensive or regime-specific models.

Without discounting, the online layer will react too slowly after structural breaks even if `FixedShare` is present.

## 5. Production Architecture Improvements

### Uncertainty quantification on combined forecasts

Required for a production-grade SOTA system.

Recommended design:

- extend `ForecastResult` with explicit `lower`, `upper`, and/or `quantiles`;
- support both model-based intervals and conformal intervals;
- store rolling empirical coverage and average interval width in diagnostics.

### Forecast calibration layer

This is missing and should be explicit.

For point forecasts:

- rolling MZ-style bias correction;
- monotone calibration or isotonic regression on OOS forecasts vs proxy;
- proxy-specific calibration for variance vs realized-measure targets.

For intervals:

- conformal recalibration with reset/discount logic.

Calibration often improves live accuracy more than adding yet another expert.

### Monitoring / drift detection in deployment

At minimum monitor:

- rolling QLIKE and MSE;
- conformal coverage gap;
- proxy SNR;
- disagreement across realized proxies;
- expert failure rate and stale-state count;
- weight concentration and expert turnover;
- latency and refit cost.

This belongs in the returned diagnostics artifact, not only logs.

### Computational budget management

A SOTA auto-forecaster should be budget-aware.

Recommended policy:

- always-on cheap experts: `EWMA`, `GARCH`, `HAR`, realized-GAS;
- conditionally-on medium experts: `HEAVY`, `RealizedGARCH`, `FIGARCH`, `MSGARCH`;
- gated expensive experts: rough-vol, TFT/panel models, RL combiners.

Also add:

- early elimination for experts that are persistently dominated in rolling OOS;
- cached realized-measure computation;
- multi-fidelity benchmarking before full refit.

### Warm-start and incremental refit strategies

This is one of the easiest production wins.

Recommended:

- carry forward filtered states for GARCH/SV/GAS/MSGARCH classes;
- support fixed-parameter filtering after the last full fit;
- add lightweight fine-tuning for neural models instead of full retraining;
- checkpoint combiner state, conformal state, and realized-feature caches.

## 6. Concrete Recommendations

Ranked by expected improvement to point forecast accuracy within the current architecture.

| Rank | Title | Description | Expected impact | Implementation complexity | Key reference | Module to modify or create |
|---|---|---|---|---|---|---|
| 1 | Add Realized-GAS / DCS Models | Implement a realized-volatility score-driven family with two-component dynamics, leverage, and fat-tailed GB2/EGB2 measurement distributions. | High | Moderate | Harvey and Palumbo (2023) | Modify `volforecast/models/gas.py` or create `volforecast/models/realized_gas.py`; update `volforecast/models/__init__.py` |
| 2 | Upgrade Online Combination to BOA + AdaHedge + Discounted OMD | Move from first-order expert aggregation only to second-order, variance-adaptive, discounted online aggregation with restart hooks. | High | Moderate | Wintenberger (2017); de Rooij et al. (2014); Fang et al. (2020) | Extend `volforecast/combination/online.py` |
| 3 | Add Change-Point-Triggered Reset and Refit Logic | Detect breaks on combined loss and proxy quality, then reset weights, shrink windows, and trigger selective expert refits. | High | Moderate | Giacomini and White (2006) for state dependence; Calleo (2025) for conformal resets | Modify `volforecast/benchmark/runner.py`; create `volforecast/combination/drift.py` |
| 4 | Add GARCH-Informed Neural Hybrids | Implement GARCH-GRU, GARCH-LSTM, and optionally GARCH-TFT so the neural layer learns nonlinear residual dynamics on top of interpretable variance recursion. | High | Hard | Wei, Yang and Cui (2025); Petrosino et al. (2025) | Create `volforecast/models/hybrid_neural.py`; optionally refactor `volforecast/models/ml_wrappers.py` |
| 5 | Implement REGARCH-MIDAS-X / Higher-Moment Hybrids | Combine realized-measure equations with MIDAS long-run structure, jump-robust proxies, and optional exogenous low-frequency drivers. | High | Hard | Liu, Maheu and Yang (2021); Chevallier and Sanhaji (2023) | Extend `volforecast/models/realized_garch.py` and `volforecast/models/midas.py`; create `volforecast/models/realized_midas.py` |
| 6 | Add Panel/Commonality Attention Models | Introduce pooled realized-vol models that use market/sector commonality and attention over cross-asset and exogenous features. | Medium-High | Hard | Zhang et al. (2024); Frank (2023) | Create `volforecast/models/panel_attention.py`; extend benchmark data plumbing for pooled features |
| 7 | Make Evaluation Multi-Proxy by Default | Score models against a target-aware panel of proxies (`RV`, `BV`, `MedRV`, realized kernel, TSRV, pre-averaging) and rank by proxy-stable robust loss. | Medium-High | Moderate | Patton (2011); Barndorff-Nielsen et al. (2008) | Modify `volforecast/benchmark/runner.py`, `volforecast/evaluation/proxy.py`, `volforecast/core/targets.py` |
| 8 | Replace Simplistic Rough-Vol Calibration | Upgrade rough-vol forecasters to Markovian/quadratic rough variants with Zumbach-aware dynamics and faster calibration. | Medium | Hard | Tang, Rosenbaum and Zhou (2024); Abi Jaber and Li (2024) | Rewrite `volforecast/models/rough_vol.py` |
| 9 | Add SPA and Giacomini-White to Model Selection | Protect against data snooping in large candidate pools and test conditional, state-dependent predictive superiority. | Medium | Moderate | Hansen (2005); Giacomini and White (2006) | Extend `volforecast/evaluation/tests.py`; surface results in `volforecast/benchmark/runner.py` |
| 10 | Add Rolling Forecast Calibration Layer | Apply rolling bias correction / monotone recalibration to individual or combined forecasts before deployment. | Medium | Easy-Moderate | Mincer and Zarnowitz (1969); practical forecasting literature | Create `volforecast/evaluation/calibration.py`; integrate in `volforecast/combination/online.py` and deployment wrapper |

### Final prioritization view

If the goal is "best accuracy gain per unit engineering effort", the first four changes should be:

1. realized-volatility DCS/GAS;
2. better online aggregation;
3. explicit drift/reset logic;
4. multi-proxy evaluation.

If the goal is "best long-run frontier position", add next:

5. GARCH-informed neural hybrids;
6. REGARCH-MIDAS-style hybrids;
7. pooled attention/commonality models;
8. upgraded rough-volatility implementations.

### Final verdict

AutoVolForecaster does not need a new architecture. It needs a stronger version of the current one:

- richer realized-aware model families;
- second-order online combination instead of only first-order expert advice;
- multi-proxy, data-snooping-robust evaluation;
- explicit drift detection and reset policies;
- a calibration/uncertainty layer on top of the combined forecaster.

Those changes are compatible with the existing `BaseForecaster`, `BenchmarkRunner`, and online combiner design, and they are enough to move the system from "strong classical auto-benchmark" to "credible 2020-2025 SOTA-style automated volatility forecaster."
