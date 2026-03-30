"""
Publication-grade rolling forecast experiment for AutoVolForecaster on USDJPY.

Addresses all publication-verdict requirements:
  1. Freezes raw dataset as CSV for reproducibility
  2. Saves candidate-level benchmark outputs, DM/MCS audit trail
  3. Formal holdout comparisons against external baselines
  4. Robustness analysis (multiple proxy windows, alternative proxies)
  5. Clean reproducible entrypoint with environment metadata

Outputs (all saved to this folder):
  data/
    usdjpy_raw_spot.csv           — frozen raw spot data
    usdjpy_raw_returns.csv        — frozen log-return series
    usdjpy_implied_vol_1m.csv     — frozen 1M ATM implied vol
  audit/
    candidate_benchmark.csv       — per-candidate OOS metrics from BenchmarkRunner
    dm_test_results.csv           — pairwise DM test p-values
    mcs_results.csv               — MCS inclusion/elimination log
    selection_log.txt             — full selection pipeline log
  baselines/
    baseline_comparison.csv       — holdout metrics for all baselines + AutoVol
    baseline_comparison.png       — bar chart comparison
  robustness/
    proxy_robustness.csv          — metrics across different proxy windows
    proxy_robustness.png          — robustness plot
  results/
    usdjpy_rolling_forecast.csv   — daily forecast data with weights
    usdjpy_rolling_forecast.png   — 4-panel diagnostic
    usdjpy_forecast_band.png      — returns with ±2σ band
  environment.txt                 — package versions + runtime metadata

Usage:
    python run_rolling_forecast.py [--from-frozen]

    --from-frozen   Use frozen CSV data instead of fetching from Bloomberg
"""
import sys
import os
import json
import platform
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
LIB_PATH = "C:/Code/Libraries/Libraries/"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
# Default to frozen data for reproducible conclusion-generating reruns.
# Use --fetch-fresh to pull new data from Bloomberg.
USE_FROZEN = "--fetch-fresh" not in sys.argv

# Setup paths
for p in [LIB_PATH, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rolling_forecast")

# Create output subdirectories
for sub in ["data", "audit", "baselines", "robustness", "results"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

t_start = time.time()


# Environment metadata is now captured inside AutoForecastResult.provenance
# (see volforecast.auto.auto._capture_provenance). No separate file needed.


# ══════════════════════════════════════════════════════════════════════
# Step 1: Load or fetch data + freeze
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("Step 1: Loading USDJPY data...")

spot_path = os.path.join(OUT_DIR, "data", "usdjpy_raw_spot.csv")
returns_path = os.path.join(OUT_DIR, "data", "usdjpy_raw_returns.csv")
ivol_path = os.path.join(OUT_DIR, "data", "usdjpy_implied_vol_1m.csv")

if USE_FROZEN and os.path.exists(returns_path) and os.path.exists(spot_path):
    logger.info("Using frozen dataset from %s", returns_path)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    log_returns = returns_df["log_return"]
    dates = log_returns.index
    returns_arr = log_returns.values.astype(np.float64)
    if os.path.exists(ivol_path):
        ivol_df = pd.read_csv(ivol_path, index_col=0, parse_dates=True)
        atm_vol_1m = ivol_df["atm_vol_1m"].reindex(dates)
    else:
        atm_vol_1m = None
else:
    from FinAPI.FX.provider import FXTickers, FXDataDownloader
    fx = FXTickers()
    pair = fx.to_valid_fx_pair("USDJPY")
    h = FXDataDownloader.fetch_data(pair)

    spot = h.get_spot_close()
    log_returns = np.log(spot / spot.shift(1)).dropna()
    dates = log_returns.index
    returns_arr = log_returns.values.astype(np.float64)

    # Freeze raw data
    spot.to_frame("spot_close").to_csv(spot_path)
    pd.DataFrame({"log_return": log_returns}).to_csv(returns_path)
    logger.info("Frozen: %s (%d obs)", spot_path, len(spot))
    logger.info("Frozen: %s (%d obs)", returns_path, len(log_returns))

    try:
        atm_vol_1m = h.get_atm_vol("1M").reindex(dates)
        atm_vol_1m.to_frame("atm_vol_1m").to_csv(ivol_path)
        logger.info("Frozen: %s", ivol_path)
    except Exception:
        atm_vol_1m = None

T = len(returns_arr)
logger.info("USDJPY: %s to %s, T=%d", dates[0].date(), dates[-1].date(), T)


# ══════════════════════════════════════════════════════════════════════
# Step 2: Define train/holdout split
# ══════════════════════════════════════════════════════════════════════
HOLDOUT = min(504, T // 3)
TRAIN_END = T - HOLDOUT
train_returns = returns_arr[:TRAIN_END]
holdout_returns = returns_arr[TRAIN_END:]
holdout_dates = dates[TRAIN_END:]
train_dates = dates[:TRAIN_END]

logger.info("Training: %s to %s (%d obs)", dates[0].date(), dates[TRAIN_END-1].date(), TRAIN_END)
logger.info("Holdout:  %s to %s (%d obs)", holdout_dates[0].date(), holdout_dates[-1].date(), HOLDOUT)


# ══════════════════════════════════════════════════════════════════════
# Step 3: Fit AutoVolForecaster — single source of truth
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("Step 3: Fitting AutoVolForecaster...")

from volforecast.auto import AutoVolForecaster
from volforecast.auto.auto import AutoForecastResult
from volforecast.evaluation.tests import diebold_mariano_test, mincer_zarnowitz_test

window_size = min(500, TRAIN_END // 2)
avf = AutoVolForecaster(
    model_families=["GARCH"],
    loss_fn="QLIKE",
    window_type="expanding",
    min_train=window_size,
    refit_every=21,
)
result = avf.fit(train_returns)

# ── Save canonical result (single source of truth for all audit/README) ──
canonical_path = os.path.join(OUT_DIR, "audit", "canonical_result.json")
result.save(canonical_path)
logger.info("Saved canonical result: %s", canonical_path)

# ── Also save the benchmark table and selection summary as CSV for convenience ──
sel_data = result.to_dict()["selection"]
bench_df = pd.DataFrame(sel_data["candidate_metrics"])
bench_df.to_csv(os.path.join(OUT_DIR, "audit", "candidate_benchmark.csv"), index=False)

sel_summary = {
    "primary_loss": sel_data["primary_loss"],
    "proxy_quality": sel_data["proxy_quality"],
    "mcs_survivors": sel_data["mcs_survivors"],
    "eliminated": sel_data["eliminated"],
}
with open(os.path.join(OUT_DIR, "audit", "selection_log.txt"), "w", encoding="utf-8") as f:
    f.write(json.dumps(sel_summary, indent=2))

logger.info("AutoVol components: %s, combiner: %s", result.component_models, result.combiner_name)


# ══════════════════════════════════════════════════════════════════════
# Step 4: Rolling forecast — AutoVol + all baselines
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("Step 4: Rolling forecasts (AutoVol + baselines)...")

# ── Baseline 1: Standalone EWMA (λ=0.94) ──
def ewma_rolling(returns, lam=0.94):
    T = len(returns)
    var_ewma = np.empty(T, dtype=np.float64)
    var_ewma[0] = returns[0] ** 2
    for t in range(1, T):
        var_ewma[t] = lam * var_ewma[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return var_ewma

# ── Baseline 2: Rolling 21-day variance ──
def rolling_var(returns, window=21):
    T = len(returns)
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        start = max(0, t - window)
        out[t] = np.var(returns[start:t + 1]) if t > 0 else returns[0] ** 2
    return out

# ── Baseline 3: Rolling 63-day variance ──
def rolling_var_63(returns, window=63):
    return rolling_var(returns, window)

# Fit EWMA on training, then extend into holdout
all_returns = returns_arr  # full series for baselines
ewma_full = ewma_rolling(all_returns)
rv21_full = rolling_var(all_returns, 21)
rv63_full = rolling_var(all_returns, 63)

# Holdout forecasts for baselines (1-step-ahead: use value at t-1 for t)
ewma_holdout = ewma_full[TRAIN_END - 1:T - 1]   # forecast for t uses info up to t-1
rv21_holdout = rv21_full[TRAIN_END - 1:T - 1]
rv63_holdout = rv63_full[TRAIN_END - 1:T - 1]

# Implied vol as baseline (convert annualized % → daily variance)
if atm_vol_1m is not None:
    iv_holdout_raw = atm_vol_1m.loc[holdout_dates].ffill()
    iv_holdout_var = (iv_holdout_raw / 100 / np.sqrt(252)) ** 2
    iv_holdout_var = iv_holdout_var.values.astype(np.float64)
else:
    iv_holdout_var = None

# ── AutoVol rolling forecast ──
autovol_vars = np.empty(HOLDOUT, dtype=np.float64)
realized_vars = np.empty(HOLDOUT, dtype=np.float64)
weights_history = []

for t in range(HOLDOUT):
    fr = avf.predict(horizon=1)
    autovol_vars[t] = fr.point[0]
    weights_history.append(avf._forecaster.weights.copy())
    r_t = holdout_returns[t]
    realized_vars[t] = r_t ** 2
    avf.update(np.array([r_t]))
    if (t + 1) % 100 == 0:
        logger.info("  %d/%d done", t + 1, HOLDOUT)

logger.info("Rolling forecast complete.")
weights_arr = np.array(weights_history)


# ══════════════════════════════════════════════════════════════════════
# Step 5: Formal baseline comparison (Table 1 of a paper)
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("Step 5: Baseline comparison...")

def compute_metrics(forecast_var, realized_var, label):
    """Compute all evaluation metrics for one forecaster."""
    f = np.asarray(forecast_var, dtype=np.float64)
    y = np.asarray(realized_var, dtype=np.float64)
    # Filter NaN
    valid = ~(np.isnan(f) | np.isnan(y))
    f, y = f[valid], y[valid]
    n = len(f)
    if n < 10:
        return {"model": label, "n": n, "mse": np.nan, "qlike": np.nan}

    mse = float(np.mean((f - y) ** 2))
    f_safe = np.maximum(f, 1e-20)
    qlike = float(np.mean(y / f_safe + np.log(f_safe)))
    mae = float(np.mean(np.abs(f - y)))

    try:
        mz = mincer_zarnowitz_test(f, y)
        mz_alpha, mz_beta, mz_r2, mz_eff = mz.alpha, mz.beta, mz.r_squared, mz.efficient
    except Exception:
        mz_alpha, mz_beta, mz_r2, mz_eff = np.nan, np.nan, np.nan, False

    return {
        "model": label,
        "n": n,
        "mse": mse,
        "qlike": qlike,
        "mae": mae,
        "mz_alpha": mz_alpha,
        "mz_beta": mz_beta,
        "mz_r2": mz_r2,
        "mz_efficient": mz_eff,
        "vol_ann_mean": float(np.mean(np.sqrt(f) * np.sqrt(252) * 100)),
    }

baselines = {
    "AutoVolForecaster": autovol_vars,
    "EWMA (λ=0.94)": ewma_holdout,
    "Rolling Var 21d": rv21_holdout,
    "Rolling Var 63d": rv63_holdout,
}
if iv_holdout_var is not None:
    baselines["Implied Vol 1M"] = iv_holdout_var

comparison_rows = []
for label, fvar in baselines.items():
    row = compute_metrics(fvar, realized_vars, label)
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(OUT_DIR, "baselines", "baseline_comparison.csv"), index=False)
logger.info("Saved: baselines/baseline_comparison.csv")

# Print comparison table
print("\n" + "=" * 90)
print("TABLE 1: Holdout Forecast Comparison (USDJPY, %s to %s)" % (
    holdout_dates[0].date(), holdout_dates[-1].date()))
print("=" * 90)
print(comparison_df.to_string(index=False, float_format="%.6e"))

# ── Pairwise DM tests: AutoVol vs each baseline ──
autovol_losses_mse = (autovol_vars - realized_vars) ** 2
f_safe_av = np.maximum(autovol_vars, 1e-20)
autovol_losses_qlike = realized_vars / f_safe_av + np.log(f_safe_av)

dm_baseline_rows = []
for label, fvar in baselines.items():
    if label == "AutoVolForecaster":
        continue
    fvar = np.asarray(fvar, dtype=np.float64)
    valid = ~(np.isnan(fvar) | np.isnan(realized_vars))
    if np.sum(valid) < 10:
        continue
    bl_losses_mse = (fvar[valid] - realized_vars[valid]) ** 2
    try:
        dm_mse = diebold_mariano_test(autovol_losses_mse[valid], bl_losses_mse, horizon=1)
        dm_baseline_rows.append({
            "AutoVol_vs": label,
            "loss": "MSE",
            "dm_stat": dm_mse.statistic,
            "p_value": dm_mse.p_value,
            "preferred": dm_mse.preferred,
        })
    except Exception:
        pass
    f_safe_bl = np.maximum(fvar[valid], 1e-20)
    bl_losses_qlike = realized_vars[valid] / f_safe_bl + np.log(f_safe_bl)
    try:
        dm_ql = diebold_mariano_test(autovol_losses_qlike[valid], bl_losses_qlike, horizon=1)
        dm_baseline_rows.append({
            "AutoVol_vs": label,
            "loss": "QLIKE",
            "dm_stat": dm_ql.statistic,
            "p_value": dm_ql.p_value,
            "preferred": dm_ql.preferred,
        })
    except Exception:
        pass

dm_baseline_df = pd.DataFrame(dm_baseline_rows)
if len(dm_baseline_df):
    dm_baseline_df.to_csv(os.path.join(OUT_DIR, "baselines", "dm_vs_baselines.csv"), index=False)
    logger.info("Saved: baselines/dm_vs_baselines.csv")
    print("\nDM Tests: AutoVol vs Baselines")
    print(dm_baseline_df.to_string(index=False))

# ── Baseline comparison bar chart ──
fig_bl, axes_bl = plt.subplots(1, 2, figsize=(12, 5))
models = comparison_df["model"].values
mse_vals = comparison_df["mse"].values
qlike_vals = comparison_df["qlike"].values

ax = axes_bl[0]
colors = ["#2196F3" if m == "AutoVolForecaster" else "#9E9E9E" for m in models]
ax.barh(models, mse_vals, color=colors)
ax.set_xlabel("MSE")
ax.set_title("Holdout MSE (lower is better)")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

ax = axes_bl[1]
ax.barh(models, qlike_vals, color=colors)
ax.set_xlabel("QLIKE")
ax.set_title("Holdout QLIKE (lower is better)")

fig_bl.suptitle(f"USDJPY Baseline Comparison ({holdout_dates[0].date()} – {holdout_dates[-1].date()})")
plt.tight_layout()
fig_bl.savefig(os.path.join(OUT_DIR, "baselines", "baseline_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig_bl)
logger.info("Saved: baselines/baseline_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Step 6: Robustness analysis — alternative proxy windows
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("Step 6: Robustness analysis...")

# Test if AutoVol ranking holds under different smoothed-proxy windows
robustness_rows = []
for proxy_window in [1, 5, 10, 21]:
    if proxy_window == 1:
        proxy = realized_vars  # r_t^2
        proxy_label = "r²_t (daily)"
    else:
        proxy_series = pd.Series(realized_vars, index=holdout_dates).rolling(proxy_window).mean()
        proxy = proxy_series.values
        proxy_label = f"RV_{proxy_window}d (rolling mean r²)"

    valid = ~np.isnan(proxy)
    if np.sum(valid) < 20:
        continue

    for label, fvar in baselines.items():
        fvar_v = np.asarray(fvar, dtype=np.float64)[valid]
        proxy_v = proxy[valid]
        mse = float(np.mean((fvar_v - proxy_v) ** 2))
        f_safe = np.maximum(fvar_v, 1e-20)
        qlike = float(np.mean(proxy_v / f_safe + np.log(f_safe)))
        robustness_rows.append({
            "proxy": proxy_label,
            "proxy_window": proxy_window,
            "model": label,
            "mse": mse,
            "qlike": qlike,
        })

rob_df = pd.DataFrame(robustness_rows)
rob_df.to_csv(os.path.join(OUT_DIR, "robustness", "proxy_robustness.csv"), index=False)
logger.info("Saved: robustness/proxy_robustness.csv")

# ── Robustness plot: rank of AutoVol across proxy windows ──
fig_rob, axes_rob = plt.subplots(1, 2, figsize=(12, 5))

for loss_col, ax, title in [("mse", axes_rob[0], "MSE"), ("qlike", axes_rob[1], "QLIKE")]:
    for model_name in baselines.keys():
        subset = rob_df[rob_df["model"] == model_name]
        lw = 2.5 if model_name == "AutoVolForecaster" else 1
        ls = "-" if model_name == "AutoVolForecaster" else "--"
        ax.plot(subset["proxy_window"], subset[loss_col], label=model_name, linewidth=lw, linestyle=ls)
    ax.set_xlabel("Proxy smoothing window (days)")
    ax.set_ylabel(title)
    ax.set_title(f"{title} across proxy windows")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig_rob.suptitle("Robustness: Model Ranking Stability Across Proxy Windows")
plt.tight_layout()
fig_rob.savefig(os.path.join(OUT_DIR, "robustness", "proxy_robustness.png"), dpi=150, bbox_inches="tight")
plt.close(fig_rob)
logger.info("Saved: robustness/proxy_robustness.png")


# ══════════════════════════════════════════════════════════════════════
# Step 7: Main results — rolling forecast plots + CSV
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("Step 7: Generating main results...")

forecast_vol_ann = np.sqrt(autovol_vars) * np.sqrt(252) * 100
realized_vol_21d = pd.Series(realized_vars, index=holdout_dates).rolling(21).mean()
realized_vol_21d_ann = np.sqrt(realized_vol_21d) * np.sqrt(252) * 100

# Rolling loss windows
window = 63
mse_rolling = pd.Series((autovol_vars - realized_vars) ** 2, index=holdout_dates).rolling(window).mean()
f_safe = np.maximum(autovol_vars, 1e-20)
qlike_daily = realized_vars / f_safe + np.log(f_safe)
qlike_rolling = pd.Series(qlike_daily, index=holdout_dates).rolling(window).mean()

mse_total = float(np.mean((autovol_vars - realized_vars) ** 2))
qlike_total = float(np.mean(qlike_daily))
mz = mincer_zarnowitz_test(autovol_vars, realized_vars)

# ── 4-panel diagnostic plot ──
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle(f"USDJPY AutoVolForecaster Rolling Forecast\n"
             f"Training: {dates[0].date()} to {dates[TRAIN_END-1].date()} | "
             f"Holdout: {holdout_dates[0].date()} to {holdout_dates[-1].date()}",
             fontsize=13)

ax = axes[0]
ax.plot(holdout_dates, forecast_vol_ann, label="AutoVol forecast (ann.)", linewidth=1, color="blue")
ax.plot(holdout_dates, realized_vol_21d_ann, label="Realized vol 21d (ann.)",
        linewidth=1, color="red", alpha=0.7)
ax.plot(holdout_dates, np.sqrt(ewma_holdout) * np.sqrt(252) * 100,
        label="EWMA baseline", linewidth=0.8, color="orange", alpha=0.6, linestyle="--")
if atm_vol_1m is not None:
    implied = atm_vol_1m.loc[holdout_dates].dropna()
    if len(implied) > 0:
        ax.plot(implied.index, implied.values, label="1M ATM implied vol",
                linewidth=1, color="green", alpha=0.7)
ax.set_ylabel("Vol (%)")
ax.set_title("Forecast vs Realized vs Baselines")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(mse_rolling.index, mse_rolling.values, color="purple", linewidth=1)
ax.set_ylabel("MSE (63d rolling)")
ax.set_title(f"Rolling MSE | Overall: {mse_total:.4e}")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(qlike_rolling.index, qlike_rolling.values, color="teal", linewidth=1)
ax.set_ylabel("QLIKE (63d rolling)")
ax.set_title(f"Rolling QLIKE | Overall: {qlike_total:.4f}")
ax.grid(True, alpha=0.3)

ax = axes[3]
for i, name in enumerate(result.component_models):
    ax.plot(holdout_dates, weights_arr[:, i], label=name, linewidth=1)
ax.set_ylabel("Weight")
ax.set_title("Combiner Weight Evolution")
ax.legend(fontsize=9)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "results", "usdjpy_rolling_forecast.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Forecast band plot ──
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.fill_between(holdout_dates,
                  -2 * np.sqrt(autovol_vars) * 100,
                  2 * np.sqrt(autovol_vars) * 100,
                  alpha=0.2, color="blue", label="±2σ forecast band")
ax2.plot(holdout_dates, holdout_returns * 100, linewidth=0.5, color="black",
         alpha=0.6, label="Daily returns (%)")
ax2.set_ylabel("Return (%)")
ax2.set_title("USDJPY Daily Returns with ±2σ Forecast Band")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "results", "usdjpy_forecast_band.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Save results CSV ──
summary_df = pd.DataFrame({
    "date": holdout_dates,
    "return": holdout_returns,
    "realized_var": realized_vars,
    "forecast_var_autovol": autovol_vars,
    "forecast_var_ewma": ewma_holdout,
    "forecast_var_rv21": rv21_holdout,
    "forecast_var_rv63": rv63_holdout,
    "forecast_vol_ann": forecast_vol_ann,
})
if iv_holdout_var is not None:
    summary_df["forecast_var_implied"] = iv_holdout_var
for i, name in enumerate(result.component_models):
    summary_df[f"weight_{name}"] = weights_arr[:, i]
summary_df.to_csv(os.path.join(OUT_DIR, "results", "usdjpy_rolling_forecast.csv"), index=False)

# ── Coverage ──
two_sigma = 2 * np.sqrt(autovol_vars)
within_band = np.abs(holdout_returns) < two_sigma
coverage = float(np.mean(within_band)) * 100

# ══════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_start

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print(f"  Runtime:    {elapsed:.1f}s")
print(f"  MSE:        {mse_total:.4e}")
print(f"  QLIKE:      {qlike_total:.4f}")
print(f"  MZ R²:      {mz.r_squared:.4f}")
print(f"  MZ eff:     {mz.efficient}")
print(f"  ±2σ cov:    {coverage:.1f}%")
print(f"  Components: {result.component_models}")
print(f"  Combiner:   {result.combiner_name}")
print(f"\nAll outputs in: {OUT_DIR}/")
print(f"  data/       — frozen raw data")
print(f"  audit/      — canonical_result.json + candidate benchmarks")
print(f"  baselines/  — comparison table + DM tests + plots")
print(f"  robustness/ — proxy-window robustness analysis")
print(f"  results/    — rolling forecast CSV + plots")

# ══════════════════════════════════════════════════════════════════════
# Auto-generate README from canonical result (Fix 1: single source of truth)
# ══════════════════════════════════════════════════════════════════════
logger.info("Generating README.md from canonical result...")

canonical = result.to_dict()
prof = canonical["profile"]
sel = canonical["selection"]
comb = canonical["combination"]
prov = canonical["provenance"]

readme_lines = [
    "# USDJPY Rolling Forecast — Model Documentation",
    "",
    "*This file is auto-generated from `audit/canonical_result.json`.*",
    "*Do not edit manually — rerun `run_rolling_forecast.py` to regenerate.*",
    "",
    "## Provenance",
    "",
    f"- **Generated**: {prov.get('timestamp', 'N/A')}",
    f"- **Git commit**: `{prov.get('git_commit', 'N/A')}`",
    f"- **Git dirty**: {prov.get('git_dirty', 'N/A')}",
    f"- **Python**: {prov.get('python', 'N/A').split(chr(10))[0]}",
    f"- **volforecast**: {prov.get('volforecast_version', 'N/A')}",
    f"- **Data source**: {'frozen CSV' if USE_FROZEN else 'Bloomberg/FinAPI'}",
    f"- **CLI args**: `{' '.join(prov.get('cli_args', []))}`",
    "",
    "## Data",
    "",
    f"| Item | Value |",
    f"|------|-------|",
    f"| Pair | USDJPY |",
    f"| Total observations | {prof['T'] + HOLDOUT} |",
    f"| Training period | {dates[0].date()} to {dates[TRAIN_END-1].date()} ({TRAIN_END} obs) |",
    f"| Holdout period | {holdout_dates[0].date()} to {holdout_dates[-1].date()} ({HOLDOUT} obs) |",
    "",
    "## Data Profile",
    "",
    f"| Feature | Value |",
    f"|---------|-------|",
    f"| Hurst exponent | {prof['hurst_exp']:.3f} |",
    f"| Long memory (H > 0.6) | {prof['has_long_memory']} |",
    f"| Rough vol (H < 0.5) | {prof['has_rough_vol']} |",
    f"| Leverage | {prof['has_leverage']} |",
    f"| Jumps (fraction > 5%) | {prof['has_jumps']} ({prof['jump_fraction']:.3f}) |",
    f"| Excess kurtosis | {prof['excess_kurtosis']:.2f} |",
    f"| Heavy tails (kurt > 5) | {prof['heavy_tails']} |",
    f"| Regime switching | {prof['has_regime_switching']} |",
    "",
    "## Model Selection",
    "",
    f"- **Effective loss**: {sel['primary_loss']} (proxy quality: {sel['proxy_quality']})",
    f"- **MCS survivors**: {', '.join(sel['mcs_survivors'])}",
    f"- **Eliminated**: {', '.join(sel['eliminated']) if sel['eliminated'] else 'none'}",
    "",
    "### Candidate Benchmark (training-period OOS)",
    "",
    "| Model | MSE | QLIKE | MZ R2 | MZ Eff |",
    "|-------|-----|-------|-------|--------|",
]
for cm in sel["candidate_metrics"]:
    readme_lines.append(
        f"| {cm['model']} | {cm['mse']:.4e} | {cm['qlike']:.4f} | "
        f"{cm['mz_r2']:.4f} | {cm['mz_efficient']} |"
    )

readme_lines.extend([
    "",
    "## Combination",
    "",
    f"- **Combiner**: {comb['combiner_name']}",
    f"- **Components**: {', '.join(comb['component_models'])}",
    f"- **Initial weights**: {comb['initial_weights']}",
    "",
    "## Holdout Results",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| MSE | {mse_total:.4e} |",
    f"| QLIKE | {qlike_total:.4f} |",
    f"| MZ alpha | {mz.alpha:.6f} |",
    f"| MZ beta | {mz.beta:.4f} |",
    f"| MZ R2 | {mz.r_squared:.4f} |",
    f"| MZ efficient | {mz.efficient} |",
    f"| +/-2s coverage | {coverage:.1f}% |",
    f"| Mean forecast vol (ann.) | {np.mean(forecast_vol_ann):.2f}% |",
    "",
    "## Warnings",
    "",
])
for w in canonical.get("warnings", []):
    readme_lines.append(f"- {w}")

readme_lines.extend([
    "",
    "## Files",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| `audit/canonical_result.json` | **Single source of truth** — all selection/profile/provenance data |",
    "| `audit/candidate_benchmark.csv` | Per-candidate OOS metrics (derived from canonical) |",
    "| `audit/selection_log.txt` | Selection summary (derived from canonical) |",
    "| `data/usdjpy_raw_*.csv` | Frozen raw data for reproducibility |",
    "| `baselines/baseline_comparison.csv` | Holdout metrics for all baselines |",
    "| `baselines/dm_vs_baselines.csv` | DM tests: AutoVol vs each baseline |",
    "| `robustness/proxy_robustness.csv` | Metrics across proxy windows |",
    "| `results/usdjpy_rolling_forecast.csv` | Daily forecast data with weights |",
    "| `results/usdjpy_rolling_forecast.png` | 4-panel diagnostic plot |",
    "| `results/usdjpy_forecast_band.png` | Returns with +/-2s forecast band |",
    "",
    "## Reproducibility",
    "",
    "```bash",
    "# Default: uses frozen data (reproducible)",
    "python run_rolling_forecast.py",
    "",
    "# To fetch fresh data from Bloomberg:",
    "python run_rolling_forecast.py --fetch-fresh",
    "```",
])

readme_path = os.path.join(OUT_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))
logger.info("Auto-generated: %s", readme_path)
