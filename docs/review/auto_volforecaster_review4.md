# AutoVolForecaster Review 4

## Findings

1. The advertised short-series path is still broken under the design's own default training-window rule. The public API sets `min_train=None` to mean `max(252, T // 3)` and Milestone 6 explicitly calls out short-series coverage (`docs/issues/auto_volforecaster.md:148-149`, `docs/issues/auto_volforecaster.md:268`, `docs/issues/auto_volforecaster.md:341`). But `BenchmarkRunner.run()` computes `n_oos = T - self.window_size` without validating that `window_size < T`, then immediately allocates arrays of length `n_oos` in both the main path and fallback path (`volforecast/benchmark/runner.py:151-152`, `volforecast/benchmark/runner.py:215`, `volforecast/benchmark/runner.py:306`). For any series with `T < 252`, the default auto setting will drive `n_oos` negative and the benchmark fails before selection starts.

2. A model that emits a constant forecast path can crash the benchmark during Mincer-Zarnowitz evaluation. `BenchmarkRunner` always calls `mincer_zarnowitz_test(forecasts_oos, proxies)` for every completed model (`volforecast/benchmark/runner.py:280`). In `mincer_zarnowitz_test`, the regression matrix is inverted directly via `np.linalg.inv(X.T @ X)` with no protection against singularity (`volforecast/evaluation/tests.py:205-220`). If `forecasts_oos` is constant, the forecast column is collinear with the intercept and this raises `LinAlgError`, aborting the run instead of marking the MZ diagnostic unavailable.

3. The fail-safe fallback can still rank a non-model proxy-copy series as `"GARCH(1,1)-fallback"`. The design says the ensemble always has at least one member via a GARCH fallback (`docs/issues/auto_volforecaster.md:18`). The runner does attempt that when all candidates fail (`volforecast/benchmark/runner.py:296-305`), but if the fallback fit/update/predict path fails at any OOS step it silently substitutes the previous realized proxy `realized["RV"][oos_idx - 1]` (`volforecast/benchmark/runner.py:317-331`) and still appends the result under the GARCH fallback label (`volforecast/benchmark/runner.py:343-350`). That reintroduces the same ranking contamination problem the earlier fail-safe refactor was meant to remove.

4. The `CombinedForecaster` design still does not actually match the `BaseForecaster` predict contract. Phase 5 says the wrapper "implements `BaseForecaster` protocol" and that `predict(horizon)` "returns `combiner.combine(forecasts)`" (`docs/issues/auto_volforecaster.md:227-229`), while the public API also types `predict()` as returning `ForecastResult` (`docs/issues/auto_volforecaster.md:281`). The base interface requires every forecaster's `predict()` to return a `ForecastResult`, not a bare float (`volforecast/core/base.py:23-44`, `volforecast/core/base.py:121-135`). As written, the wrapper spec is not implementable literally without violating the shared interface.

## Open Questions

- Should `AutoVolForecaster.loss_fn` control only model ranking/selection, or also the online combiner's internal loss updates? The public API exposes `loss_fn` (`docs/issues/auto_volforecaster.md:266`), but the combination layer currently hardcodes QLIKE updates (`docs/issues/auto_volforecaster.md:224`).
- Should `CombinedForecaster.update()` support batched updates, or should the auto forecaster explicitly narrow the contract to one-step streaming updates? `BaseForecaster.update()` accepts arrays of new observations (`volforecast/core/base.py:139-153`), but the combination design and combiner API are specified around a single forecast vector and single scalar realization (`docs/issues/auto_volforecaster.md:230-235`, `volforecast/combination/online.py:73-83`).

## Change Summary

The design is close, but it is not yet implementable end to end without cleanup. The remaining blockers are concrete: the default short-series path can fail before benchmarking begins, Mincer-Zarnowitz diagnostics can crash valid benchmark runs, the all-fail fallback still degrades into a proxy-copy pseudo-model, and the `CombinedForecaster.predict()` spec does not yet satisfy the repo's `BaseForecaster` contract. Resolve those and the implementation can proceed on a stable interface.

## Implementation Notes

**FIXED** — All 4 findings addressed:

1. **Short-series validation** — `BenchmarkRunner.run()` now validates `window_size < T` at the start. If violated, it clamps `window_size` to leave at least 10% OOS (min 10 obs) and logs a warning. This prevents negative `n_oos` for any series length.

2. **MZ crash on constant forecasts** — `mincer_zarnowitz_test()` now wraps the `np.linalg.inv(X.T @ X)` call in a `try/except LinAlgError`. On singular input (constant forecasts), it returns a degenerate `MZTestResult` with `efficient=False` and infinite standard errors instead of crashing.

3. **GARCH fallback proxy-copy contamination** — Fallback now initialises forecasts as `NaN` instead of substituting proxy values on failure. Only includes the fallback result if at least 50% of OOS steps produced valid (non-NaN) forecasts. Evaluation uses only the valid portion. If the fallback itself mostly fails, it logs an error and returns an empty suite rather than contaminating rankings.

4. **CombinedForecaster predict contract** — Design doc updated: `predict(horizon)` now explicitly wraps the combined float in `ForecastResult(point=np.array([combined]), interval=None)` to satisfy the `BaseForecaster` protocol.
