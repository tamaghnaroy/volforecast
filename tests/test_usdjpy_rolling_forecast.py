"""
Rolling forecast test for AutoVolForecaster on USDJPY.

1. Fetch USDJPY spot data
2. Fit AutoVolForecaster on training window
3. Roll forward daily: predict → observe → update
4. Compute rolling diagnostics (MSE, QLIKE, weights, vol comparison)
5. Save plots and summary
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ── Setup library path ──
LIB_PATH = "C:/Code/Libraries/Libraries/"
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

# ── Output directory ──
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "review", "rolling_forecast"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Step 1: Fetch USDJPY data
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step 1: Fetching USDJPY data...")
print("=" * 70)

from FinAPI.FX.provider import FXTickers, FXDataDownloader

fx = FXTickers()
pair = fx.to_valid_fx_pair("USDJPY")
h = FXDataDownloader.fetch_data(pair)

spot = h.get_spot_close()
log_returns = np.log(spot / spot.shift(1)).dropna()
dates = log_returns.index
returns_arr = log_returns.values.astype(np.float64)
T = len(returns_arr)

# Also fetch implied vol for comparison
try:
    atm_vol_1m = h.get_atm_vol("1M").reindex(dates)
except Exception:
    atm_vol_1m = None

print(f"USDJPY: {dates[0].date()} to {dates[-1].date()}, T={T}")

# ══════════════════════════════════════════════════════════════════════
# Step 2: Fit on training window, then roll forward
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: Fitting AutoVolForecaster + rolling forecast...")
print("=" * 70)

from volforecast.auto import AutoVolForecaster

# Use last 2 years as holdout (~504 trading days)
HOLDOUT = min(504, T // 3)
TRAIN_END = T - HOLDOUT
train_returns = returns_arr[:TRAIN_END]
holdout_returns = returns_arr[TRAIN_END:]
holdout_dates = dates[TRAIN_END:]

print(f"Training: {dates[0].date()} to {dates[TRAIN_END-1].date()} ({TRAIN_END} obs)")
print(f"Holdout:  {holdout_dates[0].date()} to {holdout_dates[-1].date()} ({HOLDOUT} obs)")

# Fit on training data (GARCH family for speed)
avf = AutoVolForecaster(
    model_families=["GARCH"],
    loss_fn="QLIKE",
    window_type="expanding",
    min_train=min(500, TRAIN_END // 2),
    refit_every=21,
)
result = avf.fit(train_returns)

print(f"\nFit complete:")
print(f"  Components: {result.component_models}")
print(f"  Combiner: {result.combiner_name}")
print(f"  Proxy quality: {result.proxy_quality}")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Rolling forecast
# ══════════════════════════════════════════════════════════════════════
print(f"\nRolling {HOLDOUT} daily forecasts...")

forecast_vars = np.empty(HOLDOUT, dtype=np.float64)
realized_vars = np.empty(HOLDOUT, dtype=np.float64)
weights_history = []

for t in range(HOLDOUT):
    # Predict
    fr = avf.predict(horizon=1)
    forecast_vars[t] = fr.point[0]

    # Record weights
    weights_history.append(avf._forecaster.weights.copy())

    # Observe realization (squared return as proxy)
    r_t = holdout_returns[t]
    realized_vars[t] = r_t ** 2

    # Update
    avf.update(np.array([r_t]))

    if (t + 1) % 100 == 0:
        print(f"  {t+1}/{HOLDOUT} done")

print(f"Rolling forecast complete.")

# ══════════════════════════════════════════════════════════════════════
# Step 4: Compute diagnostics
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: Computing diagnostics...")
print("=" * 70)

# Convert to annualized vol %
forecast_vol_ann = np.sqrt(forecast_vars) * np.sqrt(252) * 100
realized_vol_ann = np.sqrt(realized_vars) * np.sqrt(252) * 100

# Smoothed realized vol (21-day rolling)
realized_vol_21d = pd.Series(realized_vars, index=holdout_dates).rolling(21).mean()
realized_vol_21d_ann = np.sqrt(realized_vol_21d) * np.sqrt(252) * 100

# MSE and QLIKE over time (rolling 63-day window)
window = 63
mse_rolling = pd.Series(
    (forecast_vars - realized_vars) ** 2, index=holdout_dates
).rolling(window).mean()

f_safe = np.maximum(forecast_vars, 1e-20)
qlike_daily = realized_vars / f_safe + np.log(f_safe)
qlike_rolling = pd.Series(qlike_daily, index=holdout_dates).rolling(window).mean()

# Overall metrics
mse_total = float(np.mean((forecast_vars - realized_vars) ** 2))
qlike_total = float(np.mean(qlike_daily))

# Mincer-Zarnowitz
from volforecast.evaluation.tests import mincer_zarnowitz_test
mz = mincer_zarnowitz_test(forecast_vars, realized_vars)

print(f"\nOverall metrics (holdout = {HOLDOUT} days):")
print(f"  MSE       = {mse_total:.4e}")
print(f"  QLIKE     = {qlike_total:.6f}")
print(f"  MZ alpha  = {mz.alpha:.6f}")
print(f"  MZ beta   = {mz.beta:.4f}")
print(f"  MZ R²     = {mz.r_squared:.4f}")
print(f"  MZ eff    = {mz.efficient}")

# Forecast vol stats
print(f"\nForecast vol (annualized %):")
print(f"  Mean  = {np.mean(forecast_vol_ann):.2f}%")
print(f"  Std   = {np.std(forecast_vol_ann):.2f}%")
print(f"  Min   = {np.min(forecast_vol_ann):.2f}%")
print(f"  Max   = {np.max(forecast_vol_ann):.2f}%")

# Weight evolution
weights_arr = np.array(weights_history)
weight_df = pd.DataFrame(
    weights_arr, index=holdout_dates,
    columns=result.component_models,
)

# ══════════════════════════════════════════════════════════════════════
# Step 5: Plots
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: Generating plots...")
print("=" * 70)

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle(f"USDJPY AutoVolForecaster Rolling Forecast\n"
             f"Training: {dates[0].date()} to {dates[TRAIN_END-1].date()} | "
             f"Holdout: {holdout_dates[0].date()} to {holdout_dates[-1].date()}",
             fontsize=13)

# ── Panel 1: Forecast vol vs realized vol vs implied vol ──
ax = axes[0]
ax.plot(holdout_dates, forecast_vol_ann, label="Forecast vol (ann.)", linewidth=1, color="blue")
ax.plot(holdout_dates, realized_vol_21d_ann, label="Realized vol 21d (ann.)",
        linewidth=1, color="red", alpha=0.7)
if atm_vol_1m is not None:
    implied = atm_vol_1m.loc[holdout_dates].dropna()
    if len(implied) > 0:
        ax.plot(implied.index, implied.values, label="1M ATM implied vol",
                linewidth=1, color="green", alpha=0.7)
ax.set_ylabel("Vol (%)")
ax.set_title("Forecast vs Realized vs Implied Volatility")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 2: Rolling MSE ──
ax = axes[1]
ax.plot(mse_rolling.index, mse_rolling.values, color="purple", linewidth=1)
ax.set_ylabel("MSE (63d rolling)")
ax.set_title(f"Rolling MSE | Overall: {mse_total:.4e}")
ax.grid(True, alpha=0.3)

# ── Panel 3: Rolling QLIKE ──
ax = axes[2]
ax.plot(qlike_rolling.index, qlike_rolling.values, color="teal", linewidth=1)
ax.set_ylabel("QLIKE (63d rolling)")
ax.set_title(f"Rolling QLIKE | Overall: {qlike_total:.4f}")
ax.grid(True, alpha=0.3)

# ── Panel 4: Weight evolution ──
ax = axes[3]
for i, name in enumerate(result.component_models):
    ax.plot(holdout_dates, weights_arr[:, i], label=name, linewidth=1)
ax.set_ylabel("Weight")
ax.set_title("Combiner Weight Evolution")
ax.legend(fontsize=9)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "usdjpy_rolling_forecast.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {plot_path}")

# ── Panel 5: Separate plot — forecast vs spot returns ──
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.fill_between(holdout_dates,
                  -2 * np.sqrt(forecast_vars) * 100,
                  2 * np.sqrt(forecast_vars) * 100,
                  alpha=0.2, color="blue", label="±2σ forecast band")
ax2.plot(holdout_dates, holdout_returns * 100, linewidth=0.5, color="black",
         alpha=0.6, label="Daily returns (%)")
ax2.set_ylabel("Return (%)")
ax2.set_title("USDJPY Daily Returns with ±2σ Forecast Band")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
band_path = os.path.join(OUT_DIR, "usdjpy_forecast_band.png")
fig2.savefig(band_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {band_path}")

# ── Save summary CSV ──
summary_df = pd.DataFrame({
    "date": holdout_dates,
    "return": holdout_returns,
    "realized_var": realized_vars,
    "forecast_var": forecast_vars,
    "forecast_vol_ann": forecast_vol_ann,
})
for i, name in enumerate(result.component_models):
    summary_df[f"weight_{name}"] = weights_arr[:, i]
csv_path = os.path.join(OUT_DIR, "usdjpy_rolling_forecast.csv")
summary_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ── Coverage check: % of returns within ±2σ band ──
two_sigma = 2 * np.sqrt(forecast_vars)
within_band = np.abs(holdout_returns) < two_sigma
coverage = float(np.mean(within_band)) * 100
print(f"\n±2σ coverage: {coverage:.1f}% (expected ~95.4%)")

print("\n" + "=" * 70)
print("USDJPY Rolling Forecast Test COMPLETE")
print("=" * 70)
