# USDJPY Rolling Forecast — Model Documentation

*This file is auto-generated from `audit/canonical_result.json`.*
*Do not edit manually — rerun `run_rolling_forecast.py` to regenerate.*

## Provenance

- **Generated**: 2026-03-30T09:26:12.102789
- **Git commit**: `7d34674151ad2d153bd00ddfd80d6eaa93165cd8`
- **Git dirty**: True
- **Python**: 3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:37:18) [MSC v.1943 64 bit (AMD64)]
- **volforecast**: 0.1.0
- **Data source**: frozen CSV
- **CLI args**: `docs\review\rolling_forecast\run_rolling_forecast.py`

## Data

| Item | Value |
|------|-------|
| Pair | USDJPY |
| Total observations | 4078 |
| Training period | 2010-01-04 to 2023-09-14 (3574 obs) |
| Holdout period | 2023-09-15 to 2025-08-20 (504 obs) |

## Data Profile

| Feature | Value |
|---------|-------|
| Hurst exponent | 0.770 |
| Long memory (H > 0.6) | True |
| Rough vol (H < 0.5) | False |
| Leverage | False |
| Jumps (fraction > 5%) | False (0.000) |
| Excess kurtosis | 5.29 |
| Heavy tails (kurt > 5) | True |
| Regime switching | False |

## Model Selection

- **Effective loss**: MSE (proxy quality: poor — rankings may be unreliable)
- **MCS survivors**: EWMA (RiskMetrics), Component GARCH(1,1)
- **Eliminated**: GARCH(1,1), ARCH(1), FIGARCH(1,d,1)

### Candidate Benchmark (training-period OOS)

| Model | MSE | QLIKE | MZ R2 | MZ Eff |
|-------|-----|-------|-------|--------|
| EWMA (RiskMetrics) | 6.9131e-09 | -9.5431 | 0.0547 | False |
| Component GARCH(1,1) | 6.9168e-09 | -9.5358 | 0.0488 | False |
| GARCH(1,1) | 6.9978e-09 | 1001.4762 | 0.0396 | False |
| ARCH(1) | 7.1637e-09 | -9.3910 | 0.0181 | False |
| FIGARCH(1,d,1) | 8.2592e-09 | 3120431422478851.5000 | 0.0000 | False |

## Combination

- **Combiner**: EWA
- **Components**: EWMA (RiskMetrics), Component GARCH(1,1)
- **Initial weights**: [0.5, 0.5]

## Holdout Results

| Metric | Value |
|--------|-------|
| MSE | 5.7415e-09 |
| QLIKE | -9.0961 |
| MZ alpha | 0.000017 |
| MZ beta | 0.6152 |
| MZ R2 | 0.0202 |
| MZ efficient | True |
| +/-2s coverage | 94.4% |
| Mean forecast vol (ann.) | 9.67% |

## Warnings

- Proxy SNR=0.17 < 1. Falling back to MSE for ranking.
- EWMA (RiskMetrics) is not MZ-efficient (alpha=0.0000, beta=0.7252)
- Component GARCH(1,1) is not MZ-efficient (alpha=0.0000, beta=0.8297)

## Files

| File | Description |
|------|-------------|
| `audit/canonical_result.json` | **Single source of truth** — all selection/profile/provenance data |
| `audit/candidate_benchmark.csv` | Per-candidate OOS metrics (derived from canonical) |
| `audit/selection_log.txt` | Selection summary (derived from canonical) |
| `data/usdjpy_raw_*.csv` | Frozen raw data for reproducibility |
| `baselines/baseline_comparison.csv` | Holdout metrics for all baselines |
| `baselines/dm_vs_baselines.csv` | DM tests: AutoVol vs each baseline |
| `robustness/proxy_robustness.csv` | Metrics across proxy windows |
| `results/usdjpy_rolling_forecast.csv` | Daily forecast data with weights |
| `results/usdjpy_rolling_forecast.png` | 4-panel diagnostic plot |
| `results/usdjpy_forecast_band.png` | Returns with +/-2s forecast band |

## Reproducibility

```bash
# Default: uses frozen data (reproducible)
python run_rolling_forecast.py

# To fetch fresh data from Bloomberg:
python run_rolling_forecast.py --fetch-fresh
```