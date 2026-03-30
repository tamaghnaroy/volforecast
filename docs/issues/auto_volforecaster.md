# AutoVolForecaster — Design & Implementation Plan

## Overview

Build an `AutoVolForecaster` class (and a convenience `auto_fit()` function) that, given a time series
of returns (and optionally intraday data), **automatically selects, fits, evaluates, and combines**
the best volatility models from the `volforecast` library, returning a single calibrated
`BaseForecaster`-compatible object ready for production forecasting.

---

## Guiding Principles

- **Consistent model evaluation**: all scoring uses Patton-robust losses (MSE, QLIKE) via `evaluation.losses`.
- **Proxy awareness**: proxy quality is assessed via `evaluation.proxy.proxy_noise_correction`; rankings are only trusted when SNR is adequate.
- **Online-ready output**: the final forecaster supports `.update()` for streaming use.
- **Composability**: every phase delegates to existing `volforecast` primitives — no reimplementation.
- **Fail-safe**: any model that fails to fit is silently dropped; the ensemble always has at least one member (GARCH fallback). If `BenchmarkRunner` returns an empty suite (i.e. even the GARCH fallback failed to produce enough valid forecasts), `AutoVolForecaster.fit()` must raise a `RuntimeError("All models including GARCH fallback failed. Cannot proceed with model selection.")` with diagnostic info rather than silently returning an unusable object.

---

## Architecture

```
AutoVolForecaster
│
├── Phase 1 — DataProfiler
│   └── Characterises the input series (length, memory, jumps, leverage)
│
├── Phase 2 — CandidateSelector
│   └── Chooses which model classes to include given the profile
│
├── Phase 3 — BenchmarkRunner (existing)
│   └── Expanding-window OOS evaluation of all candidates
│
├── Phase 4 — ModelSelector
│   └── DM tests + MCS to prune dominated models
│
├── Phase 5 — CombinationLayer
│   └── Online combiner (EWA / FixedShare / AFTER / RL) over the MCS survivors
│
└── Phase 6 — AutoForecastResult
    └── Wraps the final forecaster + diagnostics report
```

---

## Phase 1 — Data Profiler (`volforecast/auto/profiler.py`)

### Goal
Characterise the input series so that Phase 2 can make data-driven model-family decisions.

### Inputs
- `returns: NDArray[np.float64]` — daily log-returns, shape `(T,)`
- `intraday_returns: NDArray[np.float64] | None` — shape `(T, n_intraday)`, optional
- `realized_measures: dict[str, NDArray] | None` — pre-computed RV/BV/etc., optional

### Computed features

| Feature | Method | Decision trigger |
|---|---|---|
| Series length `T` | len | Drops ML/SV/RL for `T < 500` |
| Long-memory indicator | Hurst exponent (R/S or DFA) | Includes `FIGARCHForecaster` if `H > 0.6`; includes rough-vol models if `H < 0.5` |
| Leverage test | Sign correlation of `r_t` and `r_t^2` | Includes `EGARCHForecaster`, `GJRGARCHForecaster` |
| Jump signature | BNS test from `realized.jumps.jump_decomposition` | Includes `HARJForecaster`, `HARCJForecaster` |
| Regime-switching indicator | ACF of squared returns + BDS test | Includes `MSVolForecaster` |
| Intraday data availability | `intraday_returns is not None` | Unlocks all HAR variants, `HEAVYForecaster`, `RealizedGARCHForecaster` |
| Tail heaviness | Excess kurtosis > 5 | Includes `SVForecaster`, `SVJForecaster` |

### Output dataclass
```python
@dataclass
class DataProfile:
    T: int
    has_intraday: bool
    has_realized: bool
    hurst_exp: float
    has_long_memory: bool        # H > 0.6
    has_rough_vol: bool          # H < 0.5 (rough volatility regime)
    has_leverage: bool
    jump_fraction: float         # fraction of days with significant jumps
    has_jumps: bool              # jump_fraction > 0.05
    excess_kurtosis: float
    heavy_tails: bool            # excess_kurtosis > 5
    has_regime_switching: bool
    rv: NDArray | None           # computed RV series if intraday provided
    bv: NDArray | None
    jv: NDArray | None
    cv: NDArray | None
```

---

## Phase 2 — Candidate Selector (`volforecast/auto/selector.py`)

### Goal
Map a `DataProfile` to an ordered list of `BaseForecaster` instances to evaluate.

### Candidate pools

**Always included (baseline)**
- `GARCHForecaster` — GARCH(1,1), universal baseline
- `EWMAForecaster` — RiskMetrics, zero-parameter benchmark
- `ARCHForecaster` — simplest parametric model

**Conditional on leverage**
- `EGARCHForecaster`
- `GJRGARCHForecaster`
- `APARCHForecaster`

**Conditional on long memory** (`H > 0.6`)
- `FIGARCHForecaster`
- `CGARCHForecaster`

**Conditional on roughness** (`H < 0.5` and `T >= 500`)
- `RoughBergomiForecaster`
- `RoughHestonForecaster`

**Conditional on intraday / realized data**
- `HARForecaster`
- `HARJForecaster` (if `has_jumps`)
- `HARCJForecaster` (if `has_jumps`)
- `SHARForecaster`
- `HEAVYForecaster`
- `RealizedGARCHForecaster`
- `GARCHMIDASForecaster` (if `T >= 500`; note: the current implementation constructs its low-frequency driver from rolling squared returns internally — no external macro regressors required)

**Conditional on heavy tails / regime**
- `SVForecaster` (if `T >= 500`)
- `SVJForecaster` (if `T >= 500 and has_jumps`)
- `GASVolForecaster`
- `MSVolForecaster` (if `has_regime_switching`)

**Conditional on `T >= 1000`**
- `RFVolForecaster`
- `LSTMVolForecaster` (if PyTorch available)
- `TransformerVolForecaster` (if PyTorch available)

### Output
`list[BaseForecaster]` — instantiated with sensible defaults, ready to pass to `BenchmarkRunner`.

---

## Phase 3 — Cross-Validated Benchmarking (existing `benchmark.runner.BenchmarkRunner`)

### Configuration
- **Window type**: expanding (default) or rolling
- **Initial training size**: `max(252, T // 3)` — at least one year or one-third of series
- **Refit frequency**: every 21 steps by default (monthly re-estimation)
- **Loss functions for ranking**: QLIKE primary, MSE secondary (both Patton-robust)

### Realized measure pipeline
The runner pre-computes (using `realized.measures`):
- `RV` — `realized_variance_series`
- `BV` — `bipower_variation` per day
- `CV = min(BV, RV)` — continuous variation proxy
- `JV = max(RV - BV, 0)` — jump variation proxy
- `RS_pos`, `RS_neg` — semi-variances

These are passed as `realized_measures` to every `forecaster.fit()` call, ensuring HAR/HEAVY/Realized-GARCH receive the correct inputs.

### Proxy quality check and effective loss selection
After benchmarking, run `evaluation.proxy.proxy_noise_correction(forecasts, rv_oos, loss_fn="QLIKE")` for every model.
If `signal_to_noise_ratio < 1` (proxy quality = "poor — rankings may be unreliable"), emit a warning and set `effective_loss = "MSE"`. Otherwise, `effective_loss = loss_fn` (user-supplied, default QLIKE).

This `effective_loss` is the single loss function used for **all downstream ranking and selection** in Phase 4 (sorting, DM pruning, MCS). It is stored in `ModelSelectionResult.primary_loss`.

### Output
`BenchmarkSuiteResult` — one `BenchmarkResult` per candidate. Also returns `effective_loss: str`.

---

## Phase 4 — Statistical Model Selection (`volforecast/auto/model_selection.py`)

### Goal
Prune dominated models using `evaluation.tests`, yielding a **Model Confidence Set (MCS)** of survivors.

### Steps

All steps below use `effective_loss` (determined by Phase 3's proxy quality check) as the loss function. This is QLIKE when proxy quality is adequate, MSE when it is poor.

1. **Rank by effective loss**: sort `BenchmarkSuiteResult.results` ascending by the `effective_loss` metric.

2. **Diebold-Mariano pruning**: for each pair (best model vs. model `k`), compute per-period losses using `effective_loss` for both models, then run
   `evaluation.tests.diebold_mariano_test(losses_best, losses_k, horizon=1, significance=0.10)`.
   Eliminate `k` if DM test p-value < 0.10 and model `k` has higher mean loss.

3. **MCS**: build a `(T, M)` loss matrix where each column is a model's per-period `effective_loss`, then run
   `evaluation.tests.model_confidence_set(loss_matrix, alpha=0.10)`.
   Retain only MCS survivors (those with `p_values > 0.10`).

4. **Mincer-Zarnowitz check**: for every survivor, run
   `evaluation.tests.mincer_zarnowitz_test(forecasts_oos, rv_oos)`.
   Flag models that are not MZ-efficient (`alpha ≠ 0, beta ≠ 1`) but do not drop them
   (combination may correct bias).

5. **Fallback**: if MCS reduces to zero survivors (degenerate case), keep the top-3 by QLIKE.

### Output
`ModelSelectionResult`:
```python
@dataclass
class ModelSelectionResult:
    mcs_survivors: list[BenchmarkResult]   # models in the MCS
    eliminated: list[BenchmarkResult]      # pruned models
    primary_loss: str                      # "QLIKE" or "MSE"
    proxy_quality: str                     # from proxy_noise_correction
    rankings: pd.DataFrame                 # full table (name, MSE, QLIKE, MZ_R2, in_mcs)
```

---

## Phase 5 — Forecast Combination Layer (`volforecast/auto/combination.py`)

### Goal
Wrap MCS survivors into an online combiner so the final forecaster adapts its weights over time.

### Combiner selection heuristic

| Condition | Recommended combiner |
|---|---|
| `len(survivors) == 1` | Passthrough — no combination needed |
| `T < 750` | `EqualWeightCombiner` (small sample, hard to learn weights) |
| `has_regime_switching` | `FixedShareCombiner` (adapts to best expert over regimes) |
| default | `EWACombiner` with auto learning rate |
| `T >= 1500` (opt-in) | `RLCombiner` (SimplePolicyGradient by default; requires explicit `train(expert_forecasts, realizations)` call on historical benchmark data before deployment) |

The combiner uses **QLIKE loss** internally for weight updates (most robust to proxy noise per Patton 2011). The public API's `loss_fn` parameter controls only benchmark ranking and DM/MCS selection, not the combiner's internal update rule.

### `CombinedForecaster` wrapper class
Implements `BaseForecaster` protocol:
- `fit()` — refits all component models, initialises combiner weights equally
- `predict(horizon)` — calls `predict()` on each component, passes their point forecasts through `combiner.combine(forecasts)`, and wraps the result in `ForecastResult(point=np.array([combined]), target_spec=TargetSpec.CONDITIONAL_VARIANCE, model_name=self.model_spec.name)` to satisfy the `BaseForecaster` contract. In v1, raises `NotImplementedError` if `horizon != 1`.
- `update(new_returns, new_realized)` — executes the correct online-learning sequence:
  1. Capture period-`t` forecasts from all components (these are the *pre-update* forecasts the combiner was scored on).
  2. Extract the realization from `new_realized["RV"]` (or squared return fallback).
  3. Call `combiner.update(forecasts_t, realization)` to update weights using those pre-update forecasts.
  4. Call `component.update(new_returns, new_realized)` on every component model so they are ready for period `t+1`.
  This ordering ensures no look-ahead contamination: the combiner scores pre-update expert forecasts against the period-`t` realization, and component models are updated *after* weight adjustment.
- `model_spec` — `ModelSpec(name="Auto[{survivor_names}]/{combiner_name}", abbreviation="AUTO", family="COMBO", target=VolatilityTarget.CONDITIONAL_VARIANCE)`

---

## Phase 6 — AutoForecastResult & Public API

### `AutoForecastResult` dataclass
```python
@dataclass
class AutoForecastResult:
    forecaster: BaseForecaster        # fitted, ready for .predict() / .update()
    profile: DataProfile
    selection: ModelSelectionResult
    combiner_name: str
    component_models: list[str]
    initial_weights: NDArray[np.float64]
    benchmark_summary: str            # BenchmarkSuiteResult.summary_table()
    proxy_quality: str
    warnings: list[str]
```

### Public API

**Class interface** (`volforecast/auto/auto.py`):
```python
class AutoVolForecaster:
    def __init__(
        self,
        model_families: list[str] | None = None,   # None = auto-select all
        combination_method: str = "auto",           # "auto"|"ewa"|"fixed_share"|"after"|"rl"|"equal"
        loss_fn: str = "QLIKE",
        window_type: str = "expanding",
        min_train: int | None = None,               # None = auto (max(252, T//3))
        refit_every: int = 21,
        mcs_alpha: float = 0.10,
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        returns: NDArray[np.float64],
        intraday_returns: NDArray[np.float64] | None = None,
        realized_measures: dict[str, NDArray] | None = None,
    ) -> AutoForecastResult: ...

    def predict(self, horizon: int = 1) -> ForecastResult: ...

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: dict[str, NDArray] | None = None,
    ) -> None: ...
```

**Convenience function**:
```python
def auto_fit(
    returns: NDArray[np.float64],
    intraday_returns: NDArray[np.float64] | None = None,
    **kwargs,
) -> AutoForecastResult: ...
```

### Module layout
```
volforecast/
└── auto/
    ├── __init__.py       # exports AutoVolForecaster, auto_fit, AutoForecastResult
    ├── auto.py           # AutoVolForecaster + auto_fit()
    ├── profiler.py       # DataProfiler + DataProfile
    ├── selector.py       # CandidateSelector
    ├── model_selection.py # ModelSelector + ModelSelectionResult
    └── combination.py    # CombinedForecaster
```

---

## Implementation Phases & Milestones

### Milestone 1 — Scaffold & DataProfiler
- [ ] Create `volforecast/auto/` package skeleton
- [ ] Implement `DataProfiler` with Hurst, leverage sign-correlation, jump fraction from `realized.jumps`
- [ ] Unit test: `DataProfiler` on synthetic GARCH(1,1) and HAR-J series from `benchmark.synthetic`

### Milestone 2 — CandidateSelector + BenchmarkRunner integration
- [ ] Implement `CandidateSelector` with full conditional logic table
- [ ] Wire `CandidateSelector → BenchmarkRunner` with realized measure pipeline
- [ ] Integration test: run on 2,000-obs synthetic series, confirm all models fit without error

### Milestone 3 — Statistical Model Selection
- [ ] Implement `ModelSelector` using existing DM test + MCS from `evaluation.tests`
- [ ] Add proxy quality gate using `evaluation.proxy`
- [ ] Unit test: confirm MCS correctly retains oracle model on known DGP

### Milestone 4 — CombinedForecaster + Combination heuristic
- [ ] Implement `CombinedForecaster` wrapping `BaseCombiner` subclasses
- [ ] Implement combiner selection heuristic
- [ ] Test: FixedShare combiner switches weight to better model after structural break

### Milestone 5 — AutoVolForecaster + AutoForecastResult
- [ ] Assemble full pipeline in `AutoVolForecaster.fit()`
- [ ] Implement `.predict()` and `.update()` delegation
- [ ] Build `AutoForecastResult` with `benchmark_summary` and `warnings`

### Milestone 6 — Tests, docs, and top-level export
- [ ] Add `tests/test_auto_volforecaster.py` covering happy path and edge cases (short series, no intraday, all models fail)
- [ ] Export `AutoVolForecaster`, `auto_fit`, `AutoForecastResult` from `volforecast/__init__.py`
- [ ] Add usage example to `README.md`

---

## Key Design Decisions & Rationale

- **No new loss functions**: reuse `evaluation.losses` throughout; do not invent alternatives.
- **Patton-robust ranking**: always rank by QLIKE or MSE (never MAE or MSE-log as primary criterion) to ensure proxy-noise-consistent rankings.
- **MCS over single-best selection**: combining MCS survivors almost always matches or beats single-model selection out-of-sample (Timmermann, 2006).
- **EWA as default combiner**: `O(sqrt(T log K))` regret bound is theoretically optimal for a fixed expert pool; `FixedShare` is preferred when regime-switching is detected because it tracks the *best sequence* of experts.
- **Minimal hyperparameter tuning**: the autoforecaster deliberately avoids grid search over model-specific parameters — models are fitted with their `__init__` defaults. The combination layer provides robustness against any single model's mis-specification.
- **Online-first design**: `CombinedForecaster.update()` is first-class, so the object can be deployed in production for daily rolling use without refit.

---

## Open Questions / Future Work

- **Multi-horizon forecasting**: current design is 1-step-ahead only; extend to `horizon > 1` via direct forecasting or model-specific multi-step methods.
- **CAViaR integration**: `CAViaRForecaster` targets quantiles, not variance — needs a separate tail-risk auto-selector track.
- **GARCH-MIDAS with macroeconomic regressors**: the current `GARCHMIDASForecaster` constructs its low-frequency driver from rolling squared returns internally (no external regressors needed). A future extension could accept external macro regressors via `fit(**kwargs)`, at which point the profiler should detect their presence and conditionally switch to the macro-driven variant.
- **Cross-asset panel**: extend `DataProfiler` to handle a panel of return series and fit a factor-structured volatility model.
- **Hyperparameter search**: add optional `tune=True` flag that runs a lightweight grid over `p, q` for GARCH and lag structures for HAR using the expanding-window losses.
