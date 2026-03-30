# AutoVolForecaster Review 5

## Findings

1. The fail-safe guarantee is still not true in the current code path, so the auto-forecaster can still reach Phase 4 with zero benchmark results. The design still promises that failed models are dropped and "the ensemble always has at least one member (GARCH fallback)" in `docs/issues/auto_volforecaster.md:18`. `BenchmarkRunner` repeats the same assumption in the fallback comment at `volforecast/benchmark/runner.py:303-305`, but the actual fallback only appends `"GARCH(1,1)-fallback"` if it produces at least `max(n_oos // 2, 10)` valid forecasts (`volforecast/benchmark/runner.py:345-371`). Otherwise it logs `"No models available"` and returns an empty suite (`volforecast/benchmark/runner.py:372-376`). Without an additional recovery rule in `AutoVolForecaster.fit()`, the advertised end-to-end fail-safe path is still missing.

2. The `CombinedForecaster` wrapper is still specified with impossible `BaseForecaster` return types. Phase 5 says `predict(horizon)` should wrap the combined value as `ForecastResult(point=np.array([combined]), interval=None)` in `docs/issues/auto_volforecaster.md:229`, but `ForecastResult` has no `interval` field and instead requires `target_spec` and `model_name` in `volforecast/core/base.py:23-44`. The same section only specifies `model_spec` as a family/name string pair in `docs/issues/auto_volforecaster.md:236`, while the required `ModelSpec` type needs at least `name`, `abbreviation`, `family`, and `target` in `volforecast/core/base.py:47-77`. A developer cannot implement the wrapper literally from the current spec without inventing missing contract details.

## Open Questions

- Should `AutoVolForecaster.predict()` explicitly reject `horizon != 1` in v1, or is multi-step support required now? The design says the wrapper combines a single scalar forecast in `docs/issues/auto_volforecaster.md:229`, the public API still exposes `predict(horizon: int = 1)` in `docs/issues/auto_volforecaster.md:281`, and the future-work section says the current design is 1-step-ahead only in `docs/issues/auto_volforecaster.md:360`.

- Should `loss_fn` govern only benchmark ranking, or also the live combiner update rule? The public API exposes `loss_fn` in `docs/issues/auto_volforecaster.md:266`, but the combination layer hardcodes QLIKE updates in `docs/issues/auto_volforecaster.md:224`.

## Change Summary

The design is close, but it is not yet implementable end to end for building an auto volatility forecaster from a single time series. Two blockers remain: the promised fail-safe path can still return zero benchmark survivors, and the `CombinedForecaster` contract is still inconsistent with the repo's actual `ForecastResult` and `ModelSpec` types. Resolve those, then implementation can proceed cleanly.

## Implementation Notes

**FIXED** — Both findings and both open questions addressed:

1. **Empty-suite recovery** — Design doc now specifies that `AutoVolForecaster.fit()` must raise `RuntimeError` with diagnostic info if `BenchmarkRunner` returns an empty suite (even after GARCH fallback). This is an explicit design-level contract, not a silent failure path. The runner already logs the error; the auto-forecaster layer will surface it to the user.

2. **ForecastResult/ModelSpec types** — `CombinedForecaster.predict()` spec now uses the correct types:
   - `ForecastResult(point=np.array([combined]), target_spec=TargetSpec.CONDITIONAL_VARIANCE, model_name=self.model_spec.name)`
   - `ModelSpec(name="Auto[...]/{combiner}", abbreviation="AUTO", family="COMBO", target=VolatilityTarget.CONDITIONAL_VARIANCE)`
   - `predict(horizon != 1)` raises `NotImplementedError` in v1.

3. **Open Q: loss_fn scope** — Resolved: `loss_fn` controls benchmark ranking and DM/MCS selection only. Combiner always uses QLIKE internally (documented explicitly).

4. **Open Q: horizon** — Resolved: v1 is 1-step-ahead only. `predict(horizon != 1)` raises `NotImplementedError`. Multi-horizon is deferred to future work.
