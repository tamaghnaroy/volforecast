# Publication Verdict 2 — Quantitative Review Only

## Verdict

**Not publication standard yet.**

Under the USDJPY-only constraint, the current bundle is strong as an internal quantitative research package, but the quantitative evidence is still insufficient for a publication-grade forecasting claim.

## Quantitative Basis For The Verdict

1. **AutoVol does not dominate the strongest external baseline in the holdout.** In [baseline_comparison.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/baselines/baseline_comparison.csv), `Implied Vol 1M` beats `AutoVolForecaster` on both key losses:
   - `MSE`: `5.4659e-09` vs `5.7317e-09`
   - `QLIKE`: `-9.2114` vs `-9.0989`

2. **The formal holdout tests favor implied vol over AutoVol.** In [dm_vs_baselines.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/baselines/dm_vs_baselines.csv), Diebold-Mariano tests against `Implied Vol 1M` prefer the implied-vol baseline for both losses:
   - `MSE`: `p = 0.0215`, preferred = `model2`
   - `QLIKE`: `p = 0.0316`, preferred = `model2`

3. **The ranking is not robust across proxy constructions.** In [proxy_robustness.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/robustness/proxy_robustness.csv), `AutoVolForecaster` is best under daily, 5-day, and 10-day proxy constructions, but loses under the 21-day smoothed proxy:
   - `RV_21d` `MSE`: `Rolling Var 21d = 5.7352e-11` vs `AutoVolForecaster = 1.4448e-10`
   - `RV_21d` `QLIKE`: `Rolling Var 21d = -9.2273` vs `AutoVolForecaster = -9.2087`

4. **The proxy quality is quantitatively weak.** The README reports `signal-to-noise ratio = 0.15`, which means inference is being made with a low-quality daily proxy. That does not invalidate the experiment, but it reduces the evidentiary strength of the reported loss rankings unless stronger robustness checks are added.

5. **The internal audit files do not support a uniquely strong winner inside the training-period benchmark.** In [mcs_results.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/audit/mcs_results.csv), the MCS includes `GARCH(1,1)`, `EWMA`, `ARCH(1)`, and `Component GARCH(1,1)`, with only FIGARCH excluded. Quantitatively, this means the in-family model-selection evidence is much weaker than a “clear best model” story.

## Quantitative Tasks Required To Reach Publication Standard

## 1. Baseline Strengthening

### Task
Expand the holdout comparison table to include stronger, quantitatively relevant baselines and require AutoVol to beat them on predeclared metrics.

### Steps

1. Add standalone forecasts for:
   - `Component GARCH(1,1)`
   - `EWMA`
   - `GARCH(1,1)`
   - `HAR-RV` if a usable proxy can be defined
   - `Implied Vol 1M`
   - a naive `random-walk variance` baseline
   - rolling realized-volatility baselines at `5d`, `10d`, `21d`, `63d`

2. Compute for each model on the same holdout:
   - `MSE`
   - `QLIKE`
   - `MAE` as a secondary descriptive metric only
   - `MZ alpha`, `MZ beta`, `MZ R²`, `MZ efficient`

3. Require pairwise DM tests between AutoVol and every baseline under both:
   - `MSE`
   - `QLIKE`

4. Summarize the result as:
   - number of baselines beaten significantly
   - number of baselines tied
   - number of baselines lost to

### Publication criterion

AutoVol should beat or at least tie the strongest external baseline on the primary loss. At present it loses to implied vol.

## 2. Proxy Robustness Expansion

### Task
Demonstrate that model ranking is stable under multiple quantitatively defensible proxy constructions.

### Steps

1. Extend [proxy_robustness.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/robustness/proxy_robustness.csv) to include:
   - `r_t^2`
   - `5d` rolling mean of `r_t^2`
   - `10d` rolling mean
   - `21d` rolling mean
   - `63d` rolling mean

2. For each proxy, recompute:
   - loss rankings
   - DM tests
   - MCS

3. Record whether AutoVol remains:
   - best
   - in the MCS
   - significantly superior to key baselines

4. Compute a stability score:
   - fraction of proxy definitions under which AutoVol is rank 1
   - fraction under which AutoVol is in the MCS

### Publication criterion

The main conclusion should not depend materially on one proxy window. Right now the ranking flips under `RV_21d`.

## 3. Holdout Uncertainty Quantification

### Task
Add sampling uncertainty around the reported holdout metrics.

### Steps

1. Use block bootstrap on the holdout period to generate confidence intervals for:
   - `MSE`
   - `QLIKE`
   - AutoVol-minus-baseline loss differences

2. Use the same bootstrap to estimate confidence intervals for:
   - `MZ beta`
   - `MZ R²`
   - `coverage`

3. Report:
   - median
   - 2.5% quantile
   - 97.5% quantile

4. Repeat for at least:
   - `AutoVol`
   - `Implied Vol 1M`
   - `EWMA`
   - `Component GARCH(1,1)`

### Publication criterion

Point estimates alone are not enough. The paper-grade claim needs uncertainty bounds on the holdout advantage or lack thereof.

## 4. Rolling Subperiod Analysis

### Task
Check whether results are stable over economically distinct subperiods within the holdout.

### Steps

1. Split the 504-day holdout into at least:
   - first half / second half
   - pre-spike / spike / post-spike windows around the Jul–Aug 2024 episode

2. Recompute for each subperiod:
   - `MSE`
   - `QLIKE`
   - DM tests vs key baselines
   - mean annualized forecast volatility

3. Quantify whether AutoVol’s relative ranking changes by subperiod.

4. Produce a table of subperiod winners.

### Publication criterion

If the model only works in one segment and loses elsewhere, the publication claim must be narrowed.

## 5. Selection-Stage Quantification

### Task
Strengthen the internal model-selection evidence with complete, auditable statistics.

### Steps

1. Recompute candidate-level training-period benchmark results with:
   - per-period losses saved for every candidate
   - DM p-values saved for all relevant pairs
   - MCS inclusion frequencies under bootstrap resampling

2. Add effect sizes:
   - mean loss difference between top model and each alternative
   - percentage loss improvement over EWMA

3. Resolve the mismatch between the README narrative and [mcs_results.csv](/C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/review/rolling_forecast/audit/mcs_results.csv) quantitatively.

4. Add a final selection table with:
   - rank by effective loss
   - DM outcome
   - MCS status
   - final inclusion/exclusion decision

### Publication criterion

The selection story must be numerically consistent and fully auditable.

## 6. Economic Benchmarking Against Implied Vol

### Task
Because implied vol currently wins quantitatively, test whether AutoVol adds incremental predictive value rather than absolute dominance.

### Steps

1. Run Mincer-Zarnowitz-style encompassing regressions:
   - `proxy_t = α + β1 * AutoVol_t + β2 * IV_t + ε_t`

2. Test:
   - whether `β1` remains significant when implied vol is included
   - whether `β2` remains significant when AutoVol is included

3. Build simple forecast combinations:
   - `AutoVol + IV`
   - `EWMA + IV`
   - `CGARCH + IV`

4. Compare combined forecasts against standalone implied vol using:
   - `MSE`
   - `QLIKE`
   - DM tests

### Publication criterion

If AutoVol cannot beat implied vol, it can still be publishable if it adds statistically significant incremental information to implied vol.

## 7. Calibration Diagnostics

### Task
Extend calibration analysis beyond the current `±2σ` coverage summary.

### Steps

1. Compute standardized residuals:
   - `z_t = r_t / sqrt(forecast_var_t)`

2. Test whether `z_t` has:
   - mean near zero
   - variance near one
   - low autocorrelation in `z_t^2`

3. Report:
   - Ljung-Box tests on `z_t`
   - Ljung-Box tests on `z_t^2`
   - Jarque-Bera or kurtosis/skewness diagnostics

4. Compute empirical coverage for:
   - `±1σ`
   - `±1.5σ`
   - `±2σ`
   - `±3σ`

### Publication criterion

A publishable volatility forecast should be calibrated in more than one coarse coverage statistic.

## 8. Forecast Combination Attribution

### Task
Show quantitatively whether the combination layer adds value beyond the best single survivor.

### Steps

1. Compare holdout performance of:
   - combined AutoVol forecast
   - standalone `EWMA`
   - standalone `Component GARCH(1,1)`

2. Compute:
   - loss difference between combo and best component
   - DM test for combo vs best component

3. Quantify weight concentration:
   - average weight
   - final weight
   - entropy of weights over time

4. Report whether the combiner delivers:
   - statistically significant gain
   - or only a cosmetic averaging effect

### Publication criterion

If the combiner does not materially improve on the best component, the contribution claim should be reduced.

## 9. Out-of-Time Replication On USDJPY

### Task
Stay within the USDJPY-only constraint but create multiple non-overlapping forecast experiments on the same asset.

### Steps

1. Run at least 3 separate train/holdout splits on USDJPY, for example:
   - 2010–2018 train, 2019–2020 holdout
   - 2010–2021 train, 2022–2023 holdout
   - 2010–2023 train, 2023–2025 holdout

2. For each split, save:
   - baseline table
   - DM tests
   - MCS
   - robustness table

3. Aggregate across splits:
   - mean rank
   - median rank
   - number of wins vs each baseline

### Publication criterion

A single holdout episode on one asset is usually not enough for a publication-grade claim, even if single-asset scope is accepted.

## 10. Predeclared Decision Rule

### Task
Turn the evaluation into a quantitatively predeclared acceptance rule.

### Steps

1. Declare one primary metric:
   - either `QLIKE`
   - or `MSE` under low-SNR proxy conditions

2. Declare one primary benchmark:
   - implied vol
   - or best external statistical baseline

3. Define ex ante success as:
   - AutoVol significantly better on the primary metric
   - and not losing badly on the main robustness checks

4. Apply that rule unchanged to the saved experiments.

### Publication criterion

Without a predeclared rule, the result remains exploratory rather than publication-grade.

## Bottom Line

Quantitatively, the folder has improved a lot and is now a serious research artifact. But the current evidence still does not support a publication-standard claim because:

- AutoVol loses to implied vol on the main holdout,
- the ranking is not robust across all proxy definitions,
- the internal selection evidence is weaker than the narrative suggests.

The most direct path to publication standard is:

1. prove incremental value over implied vol,
2. demonstrate ranking robustness across proxy definitions and subperiods,
3. replicate across multiple USDJPY out-of-time splits,
4. attach uncertainty intervals and predeclared acceptance rules.
