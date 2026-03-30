# Publication Verdict 3 — Research Setup And Infrastructure Only

## Verdict

**Not fully publication-standard yet, but close.**

If `AutoVolForecaster` is modified and [run_rolling_forecast.py](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/run_rolling_forecast.py) is rerun, the current setup does produce a substantially complete quantitative research bundle: frozen data, environment capture, candidate audit, baseline comparison, robustness tables, and result plots.

That is strong research infrastructure.

The remaining issue is not forecast performance. It is that the research pipeline still does **not** guarantee that the final conclusions are automatically consistent, provenance-complete, and publication-safe.

## What Is Strong

- Frozen raw inputs exist under [data](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/data).
- Environment metadata is captured in [environment.txt](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/environment.txt).
- Candidate-level selection artifacts are saved under [audit](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/audit).
- External baseline comparisons are saved under [baselines](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/baselines).
- Proxy robustness outputs are saved under [robustness](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/robustness).
- The core `AutoVolForecaster` pipeline is covered by automated tests in [test_auto_volforecaster.py](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/tests/test_auto_volforecaster.py).

## Why It Is Not Publication-Standard Yet

1. **The analysis layer is not single-source-of-truth.** The narrative in [README.md](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/README.md) can drift from the quantitative audit files, and it already has. The current audit artifacts in [mcs_results.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/audit/mcs_results.csv) and [selection_log.txt](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/audit/selection_log.txt) do not match the README’s model-selection story. For publication, the written conclusion must be generated from the saved numeric outputs, not maintained manually.

2. **The script duplicates parts of the selection logic instead of reading the library’s final structured outputs directly.** [run_rolling_forecast.py](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/run_rolling_forecast.py) manually rebuilds benchmark tables, DM tests, and MCS audit trails in parallel to the actual `volforecast.auto` pipeline in [auto.py](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/auto/auto.py). That creates a genuine risk that the analysis says one thing while the library selected something slightly different.

3. **Provenance capture is still incomplete.** [environment.txt](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/environment.txt) records package versions, but not the git commit, dirty-worktree status, CLI arguments, or an explicit config snapshot of the experiment. For publication, you need to be able to prove exactly which code and settings generated the conclusions.

4. **The rerun path is reproducible only if frozen data is used deliberately.** The script supports `--from-frozen`, which is good, but defaulting to Bloomberg/FinAPI means a casual rerun can silently become a different experiment. Publication-grade infrastructure should default to the frozen dataset for conclusion-generating reruns.

5. **There are still small research-code quality issues in the analysis script that reduce trust.** For example, the baseline metric helper contains unused coverage logic built from synthetic signed square-root proxies rather than actual returns. That specific value is not currently driving the main comparison table, so it is not corrupting the current conclusion, but it is exactly the kind of loose edge that publication pipelines should eliminate.

## Bottom Line

The setup is now good enough for serious internal research and for credible positive or negative quantitative assessment on USDJPY. It is **not yet** robust enough to guarantee that every rerun yields publication-standard conclusions automatically.

The main remaining gap is infrastructure integrity, not model scope:

- conclusions are not auto-derived from a single canonical result object,
- provenance is not fully captured,
- the publication narrative can diverge from the saved audit tables.

Once those issues are fixed, the setup itself would be close to publication-standard even before discussing whether the model wins or loses empirically.
