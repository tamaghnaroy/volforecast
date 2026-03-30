# AutoVolForecaster Review 2

## Findings

1. The design still overstates the data requirements for the benchmark phase. The overview says the pipeline works from returns with only optional intraday data ([auto_volforecaster.md:6](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L6)), but Phase 3 delegates to the existing `BenchmarkRunner`, which now requires either `intraday_returns` or fully `precomputed_realized` inputs and raises otherwise ([runner.py:131](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L131), [runner.py:133](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L133), [runner.py:168](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L168)). As written, a plain daily-returns-only call cannot complete the advertised benchmark-and-select workflow.

2. The plan depends on the "existing `BenchmarkRunner`", but that runner is currently broken after the recent fail-safe refactor. Evaluation uses `proxies` at the end of the run ([runner.py:239](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L239), [runner.py:251](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L251)), yet `proxies` is no longer defined anywhere in the function. Since Phase 3 is explicitly built around this runner, the implementation path is not just underspecified; it is currently non-executable until the benchmark code is fixed.

3. The revised `CombinedForecaster.update()` description still leaves component models without a valid update path. The doc says components are not updated inside `update()` and will instead be updated "during the next period's prediction cycle" ([auto_volforecaster.md:225](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L225)). But the shared interface separates `predict(horizon=1)` from `update(new_returns, new_realized)`; `predict()` receives no new observations, so it cannot perform the component updates promised by the design ([base.py:122](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/core/base.py#L122), [base.py:139](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/core/base.py#L139)). In practice this means component experts either stay stale after the first live step or the wrapper must violate the repo’s current API contract.

4. The RL combiner section does not match the actual implementation, which makes the auto-selection rule misleading. The doc recommends `RLCombiner` when `T >= 1500 and PyTorch available`, describing it as PPO-based ([auto_volforecaster.md:217](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L217)). In the code, `RLCombiner` does not depend on PyTorch, defaults to `SimplePolicyGradient`, and requires an explicit `train(expert_forecasts, realizations)` stage before it is meaningfully ready ([rl_combiner.py:243](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/combination/rl_combiner.py#L243), [rl_combiner.py:264](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/combination/rl_combiner.py#L264), [rl_combiner.py:279](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/combination/rl_combiner.py#L279)). The design never specifies where those historical expert forecasts come from or when RL training occurs, so the "auto" policy cannot be implemented faithfully.

5. The GARCH-MIDAS selector logic is still inconsistent with the model that actually exists in the repo. The issue doc treats `GARCHMIDASForecaster` as requiring macro or other low-frequency regressors ([auto_volforecaster.md:121](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L121), [auto_volforecaster.md:352](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L352)), but the implementation currently constructs its long-run driver from rolling squared returns internally, not external regressors ([midas.py:113](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/models/midas.py#L113), [midas.py:146](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/models/midas.py#L146), [midas.py:147](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/models/midas.py#L147)). That mismatch will lead the candidate selector to exclude or include MIDAS on the wrong basis.

6. The diagnostics output the design wants to expose will disagree with its own ranking rule. The proposal says QLIKE is the primary ranking loss ([auto_volforecaster.md:145](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L145), [auto_volforecaster.md:173](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L173)) and stores `benchmark_summary` from `BenchmarkSuiteResult.summary_table()` in the public result ([auto_volforecaster.md:242](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md#L242)). But `summary_table()` currently sorts results by `mse`, not `qlike` ([runner.py:73](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L73), [runner.py:82](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/benchmark/runner.py#L82)). That will surface a benchmark report whose ordering conflicts with the selection logic and confuse debugging.

## Open Questions

- If the intended user path is truly "returns only", what realized proxy is supposed to drive benchmarking, DM testing, and online combiner updates when neither intraday data nor precomputed realized measures are available?
- If RL remains in the auto-selection heuristic, should the design standardize a training data source for `RLCombiner.train()` or demote RL to an opt-in path until that contract is specified?

## Change Summary

This revision of the design fixed some API-signature issues from the earlier version, especially around DM and MCS calls. The remaining blockers are now mostly implementation-alignment problems: the benchmark dependency is currently broken, the live update semantics for combined experts are still inconsistent with `BaseForecaster`, and the selector logic for RL and MIDAS does not match the code that exists today.

## Implementation Notes

**FIXED** — All 6 findings have been addressed:

1. **Daily-returns-only fallback** — `BenchmarkRunner.run()` no longer raises when neither `intraday_returns` nor `precomputed_realized` is provided. It falls back to squared daily returns as a crude RV proxy (`r² ≈ RV` under zero-mean assumption), with `BV=RV`, `JV=0`, and symmetric semi-variance split. A log-level info message warns users about the noisy proxy.

2. **Undefined `proxies` variable** — Added `proxies = realized["RV"][self.window_size:]` before evaluation. The runner is now executable end-to-end.

3. **CombinedForecaster update semantics** — Document rewritten with explicit 4-step ordering:
   (1) capture pre-update forecasts from components,
   (2) extract realization,
   (3) `combiner.update(forecasts_t, realization)` — weights update on *pre-update* forecasts,
   (4) `component.update(new_returns, new_realized)` — components updated *after* weight adjustment.
   This matches `BaseForecaster.update()` contract and avoids look-ahead contamination.

4. **RLCombiner selector** — Changed from "PPO-based, PyTorch required" to accurate description: `SimplePolicyGradient` by default, opt-in only, requires explicit `train(expert_forecasts, realizations)` on historical benchmark data.

5. **GARCH-MIDAS selector** — Added note that the current implementation constructs its low-frequency driver from rolling squared returns internally. No external macro regressors are required. The open question was updated to describe this as a future extension path.

6. **`summary_table()` sort order** — Changed default sort from MSE to QLIKE (`sort_by: str = "qlike"`), matching the design's primary ranking loss. MSE sorting remains available via `sort_by="mse"`.
