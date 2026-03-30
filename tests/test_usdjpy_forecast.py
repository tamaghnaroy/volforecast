"""
Test AutoVolForecaster on real USDJPY data fetched via fx-market-data skill.
"""
import sys
import os
import numpy as np
import pandas as pd

# ── Setup library path ──
LIB_PATH = "C:/Code/Libraries/Libraries/"
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

# ── Fetch USDJPY spot data ──
print("=" * 60)
print("Fetching USDJPY spot data from Bloomberg/cache...")
print("=" * 60)

from FinAPI.FX.provider import FXTickers, FXDataDownloader

fx = FXTickers()
pair = fx.to_valid_fx_pair("USDJPY")
h = FXDataDownloader.fetch_data(pair)

spot = h.get_spot_close()
print(f"Pair: {pair}")
print(f"Spot series: {spot.index[0].date()} to {spot.index[-1].date()}, {len(spot)} obs")
print(f"Last spot: {spot.iloc[-1]:.4f}")

# ── Compute daily log-returns ──
log_returns = np.log(spot / spot.shift(1)).dropna().values.astype(np.float64)
T = len(log_returns)
print(f"Log-returns: T={T}, mean={log_returns.mean():.6f}, std={log_returns.std():.6f}")

# ── Also fetch 1M ATM vol for comparison ──
try:
    atm_vol = h.get_atm_vol("1M")
    last_implied_vol = atm_vol.iloc[-1]
    print(f"1M ATM implied vol (latest): {last_implied_vol:.2f}%")
except Exception as e:
    print(f"Could not fetch ATM vol: {e}")
    last_implied_vol = None

# ── Run AutoVolForecaster ──
print("\n" + "=" * 60)
print("Running AutoVolForecaster on USDJPY returns...")
print("=" * 60)

from volforecast.auto import AutoVolForecaster

avf = AutoVolForecaster(
    model_families=["GARCH"],   # Start with GARCH family for speed
    loss_fn="QLIKE",
    window_type="expanding",
    min_train=min(500, T // 2),
    refit_every=21,
    mcs_alpha=0.10,
)

result = avf.fit(log_returns)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nProfile:")
print(f"  T = {result.profile.T}")
print(f"  Hurst = {result.profile.hurst_exp:.3f}")
print(f"  Long memory = {result.profile.has_long_memory}")
print(f"  Rough vol = {result.profile.has_rough_vol}")
print(f"  Leverage = {result.profile.has_leverage}")
print(f"  Heavy tails = {result.profile.heavy_tails} (kurtosis={result.profile.excess_kurtosis:.2f})")
print(f"  Regime switching = {result.profile.has_regime_switching}")

print(f"\nModel Selection:")
print(f"  Primary loss = {result.selection.primary_loss}")
print(f"  Proxy quality = {result.proxy_quality}")
print(f"  MCS survivors = {[s.model_name for s in result.selection.mcs_survivors]}")
print(f"  Eliminated = {[e.model_name for e in result.selection.eliminated]}")

print(f"\nCombination:")
print(f"  Combiner = {result.combiner_name}")
print(f"  Components = {result.component_models}")
print(f"  Weights = {result.initial_weights}")

# ── Produce forecast ──
forecast = avf.predict(horizon=1)
forecast_var = forecast.point[0]
forecast_vol_daily = np.sqrt(forecast_var)
forecast_vol_annual = forecast_vol_daily * np.sqrt(252) * 100  # annualized %

print(f"\n1-step-ahead forecast:")
print(f"  Conditional variance = {forecast_var:.10f}")
print(f"  Daily vol = {forecast_vol_daily:.6f}")
print(f"  Annualized vol = {forecast_vol_annual:.2f}%")

if last_implied_vol is not None:
    print(f"  1M ATM implied vol = {last_implied_vol:.2f}%")
    print(f"  Difference (forecast - implied) = {forecast_vol_annual - last_implied_vol:+.2f}%")

# ── Benchmark summary ──
print(f"\nBenchmark Summary:")
print(result.benchmark_summary)

# ── Warnings ──
if result.warnings:
    print(f"\nWarnings:")
    for w in result.warnings:
        print(f"  - {w}")

# ── Test online update ──
print(f"\nTesting online update with last return...")
last_return = log_returns[-1:]
avf.update(last_return)
forecast2 = avf.predict(horizon=1)
print(f"  Post-update forecast vol = {np.sqrt(forecast2.point[0]) * np.sqrt(252) * 100:.2f}%")

print("\n" + "=" * 60)
print("USDJPY AutoVolForecaster test COMPLETE")
print("=" * 60)
