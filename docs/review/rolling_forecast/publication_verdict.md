# Publication Verdict — USDJPY Rolling Forecast

## Verdict

**Not publication standard yet.**

This verdict does **not** penalize the work for being USDJPY-only. Under the current scope constraint, a single-asset study is acceptable. The issue is that the evidence bundle is still internal-validation grade rather than paper-grade: reproducibility is incomplete, the evaluation relies on a very noisy daily squared-return proxy, and the folder does not contain enough comparative/statistical outputs to fully substantiate the forecasting claims.

## Main Reasons

1. The artifact is not publication-reproducible as packaged. The execution depends on private/local infrastructure, including Bloomberg/FinAPI access and a hardcoded local library path in [run_rolling_forecast.py](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/run_rolling_forecast.py). The folder does not include a frozen raw dataset snapshot or environment manifest sufficient for independent reruns.

2. The empirical case is mostly descriptive rather than comparative. The README documents internal candidate selection and final holdout performance, but it does not provide formal holdout comparisons against external baselines such as standalone EWMA, standalone Component GARCH, simple rolling-vol baselines, or implied-vol-based benchmarks in a way that would support a publication claim ([README.md](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/README.md)).

3. The main evaluation proxy is daily squared return, and the README itself reports low proxy quality (`signal-to-noise ratio = 0.15`), which materially weakens inference unless supported by stronger robustness checks ([README.md](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/README.md)).

4. Key statistical claims are not fully auditable from the folder contents. The README states DM pruning, MCS survivors, and FIGARCH instability, but the folder does not include the underlying candidate-level benchmark tables, per-period loss arrays, DM test outputs, or MCS elimination records needed for full review.

5. The execution script is adequate for internal experimentation but not for publication-quality artifact hygiene. It mixes data access, modeling, evaluation, and plotting in one script and reaches into private object state (`avf._forecaster.weights`) to record weights ([run_rolling_forecast.py](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/run_rolling_forecast.py)).

## What Is Strong

- The methodology is coherent and the README is unusually clear for an internal research bundle.
- The saved CSV supports the headline holdout metrics reported in the documentation.
- The observed holdout behavior is plausible for USDJPY under a daily-data, GARCH-family-only setup.
- The online combination logic and the predict-observe-update protocol are described carefully and align with the current implementation.

From the saved CSV [usdjpy_rolling_forecast.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/usdjpy_rolling_forecast.csv), the recorded holdout results are internally consistent:

- `504` holdout observations
- `MSE ≈ 5.73e-09`
- `QLIKE ≈ -9.099`
- `±2σ coverage ≈ 94.6%`
- Final combiner weight concentrated in Component GARCH (`~96.7%`)

## Minimum Upgrades Needed For Publication Standard

1. Freeze and include the exact USDJPY dataset used for the run.
2. Save candidate-level benchmark outputs, DM/MCS statistics, and survivor-selection logs.
3. Add formal holdout comparisons against external baselines, not only the internal model pool.
4. Add robustness analysis for the low-SNR `r_t^2` proxy, ideally with stronger realized-volatility proxies when available.
5. Package the experiment with environment/version metadata and a clean reproducible entrypoint.
