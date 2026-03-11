# VolForecast

**Research-grade Python volatility forecasting library** with modular experts, online aggregation, adaptive expert weighting, RL-based forecast combination, and academically grounded evaluation.

## Key Features

### Modular Expert Forecasters
All models implement a shared `BaseForecaster` interface with explicit target declaration, `fit/predict/update` API, and online streaming capability.

| Family | Models | Target | Reference |
|--------|--------|--------|-----------|
| **GARCH** | GARCH, EGARCH, GJR-GARCH, APARCH, CGARCH, Realized GARCH | Conditional Variance | Bollerslev (1986), Nelson (1991), GJR (1993), Hansen et al. (2012) |
| **HAR** | HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR | Integrated Variance / Continuous Variation | Corsi (2009), Andersen et al. (2007), Patton & Sheppard (2015) |
| **SV** | Stochastic Volatility (via knowledge graph) | Integrated Variance | Taylor (1986), Harvey et al. (1994) |
| **Combination** | Equal Weight, Inverse MSE, AFTER, EWA, Fixed-Share, RL-PPO | Any | Yang (2004), Vovk (1990), Herbster & Warmuth (1998) |

### Numba-Optimized Realized Measures
- **RV** — Realized Variance (Andersen et al., 2003)
- **BV** — Bipower Variation (Barndorff-Nielsen & Shephard, 2004)
- **MedRV** — Median Realized Variance (Andersen, Dobrev, Schaumburg, 2012)
- **MinRV** — Minimum Realized Variance (Andersen, Dobrev, Schaumburg, 2012)
- **RK** — Realized Kernel with Parzen kernel (BNHLS, 2008)
- **TSRV** — Two-Scale Realized Variance (Zhang, Mykland, Aït-Sahalia, 2005)
- **Pre-Averaging** — Jacod et al. (2009)
- **Realized Semi-Variances** — RS⁺, RS⁻ (BKS, 2010)

### Jump Decomposition
- **BNS Jump Test** — Barndorff-Nielsen & Shephard (2006) with tri-power quarticity
- **C/J Decomposition** — Separate continuous and jump variation
- **Threshold-based** jump variation with significance testing

### Academically Grounded Evaluation
- **Robust loss functions** (Patton, 2011): MSE and QLIKE are the only losses robust to proxy noise
- **Diebold-Mariano test** with HAC standard errors and Harvey-Leybourne-Newbold correction
- **Model Confidence Set** (Hansen, Lunde, Nason, 2011) with block bootstrap
- **Mincer-Zarnowitz** efficiency regression
- **Proxy noise correction**: attenuation bias correction, Hansen-Lunde adjustment
- **Explicit forecast-target mismatch** treatment

### Volatility Target Taxonomy
The library explicitly distinguishes:
- **Conditional Variance**: E[r²ₜ | Fₜ₋₁] — GARCH-family target
- **Integrated Variance**: ∫σ²ₛ ds — HAR/SV target
- **Continuous Variation**: IV without jumps
- **Jump Variation**: Σ(ΔJₛ)²
- **Quadratic Variation**: C + J

### Knowledge Graph
NetworkX-based directed graph encoding model families, assumptions, targets, computational properties, and seminal references with queryable relationships (`extends`, `targets`, `assumes`, `competes_with`).

### Online Aggregation & RL Combination
- **EWA** with theoretical regret bound O(√(T log K))
- **Fixed-Share** for tracking best expert sequence
- **AFTER** with minimax-optimal rate
- **RL-based** adaptive combination via policy gradient

## Installation

```bash
# Core install
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development only
pip install -e ".[dev]"
```

### Dependencies
- **Core**: numpy, scipy, numba, pandas, arch, statsmodels, scikit-learn, networkx, matplotlib
- **RL** (optional): gymnasium, stable-baselines3
- **ML** (optional): torch, lightning
- **C++** (optional): pybind11, cmake

## Quick Start

```python
import numpy as np
from volforecast.benchmark.synthetic import generate_garch_data
from volforecast.realized.measures import realized_variance, bipower_variation
from volforecast.models.garch import GARCHForecaster
from volforecast.models.har import HARForecaster
from volforecast.evaluation.losses import mse_loss, qlike_loss
from volforecast.evaluation.tests import diebold_mariano_test

# Generate synthetic data
data = generate_garch_data(T=2000, n_intraday=78, seed=42)

# Compute realized measures
rv = np.array([realized_variance(data.intraday_returns[t]) for t in range(2000)])
bv = np.array([bipower_variation(data.intraday_returns[t]) for t in range(2000)])

# Fit models on training data
garch = GARCHForecaster()
garch.fit(data.daily_returns[:1000])

har = HARForecaster()
har.fit(data.daily_returns[:1000], realized_measures={"RV": rv[:1000]})

# Out-of-sample forecasts
garch_fcasts = []
har_fcasts = []
for t in range(1000, 2000):
    garch_fcasts.append(garch.predict(horizon=1).point[0])
    garch.update(data.daily_returns[t:t+1])
    
    har_fcasts.append(har.predict(horizon=1).point[0])
    har.update(data.daily_returns[t:t+1], new_realized={"RV": rv[t:t+1]})

garch_fcasts = np.array(garch_fcasts)
har_fcasts = np.array(har_fcasts)
proxy = rv[1000:]

# Evaluate with robust loss functions
print(f"GARCH MSE:  {mse_loss(garch_fcasts, proxy):.2e}")
print(f"HAR   MSE:  {mse_loss(har_fcasts, proxy):.2e}")
print(f"GARCH QLIKE: {qlike_loss(garch_fcasts, proxy):.4f}")
print(f"HAR   QLIKE: {qlike_loss(har_fcasts, proxy):.4f}")

# Diebold-Mariano test
garch_losses = (garch_fcasts - proxy)**2
har_losses = (har_fcasts - proxy)**2
dm = diebold_mariano_test(garch_losses, har_losses)
print(f"DM test: stat={dm.statistic:.3f}, p={dm.p_value:.4f}, preferred={dm.preferred}")
```

## Online Forecast Combination

```python
from volforecast.combination.online import EWACombiner, FixedShareCombiner

# 3 experts: GARCH, HAR, and their average
combiner = FixedShareCombiner(n_experts=2, alpha=0.01)

for t in range(len(proxy)):
    forecasts = np.array([garch_fcasts[t], har_fcasts[t]])
    combined = combiner.combine(forecasts)
    combiner.update(forecasts, proxy[t])

print(f"Final weights: {combiner.weights}")
```

## Knowledge Graph

```python
from volforecast.knowledge import VolatilityKnowledgeGraph
from volforecast.core.targets import VolatilityTarget

kg = VolatilityKnowledgeGraph()
print(kg.summary())

# Find all models targeting integrated variance
iv_models = kg.get_models_for_target(VolatilityTarget.INTEGRATED_VARIANCE)
print(f"Models targeting IV: {iv_models}")

# Find what GJR-GARCH extends
ancestors = kg.get_ancestors("GJR")
print(f"GJR ancestors: {ancestors}")
```

## Proxy-Target Mismatch

```python
from volforecast.evaluation.proxy import proxy_noise_correction, attenuation_bias_correction

# Assess proxy quality
correction = proxy_noise_correction(garch_fcasts, proxy, loss_fn="MSE")
print(f"Proxy quality: {correction['proxy_quality']}")
print(f"Signal-to-noise ratio: {correction['signal_to_noise_ratio']:.2f}")
print(f"Raw MSE: {correction['raw_loss']:.2e}")
print(f"Adjusted MSE: {correction['adjusted_loss']:.2e}")
```

## Benchmark Suite

```python
from volforecast.benchmark.runner import BenchmarkRunner
from volforecast.benchmark.synthetic import generate_garch_data, generate_sv_data

data = generate_garch_data(T=2000, seed=42)
runner = BenchmarkRunner(
    forecasters=[GARCHForecaster(), HARForecaster()],
    window_type="expanding",
    window_size=500,
    refit_every=100,
)
results = runner.run(
    data.daily_returns, data.intraday_returns,
    true_variance=data.true_variance, dgp_name="GARCH DGP",
)
print(results.summary_table())
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Project Structure

```
volforecast/
├── core/           # Base interfaces, target taxonomy
│   ├── base.py     # BaseForecaster, ForecastResult, ModelSpec
│   └── targets.py  # VolatilityTarget enum, TargetSpec
├── realized/       # Numba-optimized realized measures
│   ├── measures.py # RV, BV, MedRV, MinRV, RK, TSRV, PA, RSV
│   └── jumps.py    # BNS test, C/J decomposition
├── models/         # Forecaster implementations
│   ├── garch.py    # GARCH, EGARCH, GJR, APARCH, CGARCH
│   ├── har.py      # HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR
│   └── realized_garch.py
├── combination/    # Online aggregation & RL
│   ├── online.py   # EW, InvMSE, AFTER, EWA, Fixed-Share
│   └── rl_combiner.py
├── evaluation/     # Academically grounded evaluation
│   ├── losses.py   # MSE, QLIKE, Patton robust family
│   ├── tests.py    # DM test, MZ regression, MCS
│   └── proxy.py    # Noise correction, attenuation bias
├── knowledge/      # Model knowledge graph
│   └── graph.py    # NetworkX-based taxonomy
└── benchmark/      # Reproducible experiments
    ├── synthetic.py # GARCH, jump-diffusion, Heston DGPs
    └── runner.py    # Expanding/rolling window benchmark
```

## Design Principles

1. **Correctness first**: All formulas match seminal references with explicit citations
2. **Explicit targets**: Every model declares what it forecasts (conditional variance vs. integrated variance vs. continuous variation)
3. **Proxy awareness**: Evaluation accounts for the fact that true volatility is latent
4. **Modularity**: Models are independent experts with a shared interface
5. **Speed**: Critical paths use Numba JIT; optional C++ via pybind11
6. **Online capability**: All models support streaming updates; combiners operate sequentially
7. **Reproducibility**: Seeded DGPs, deterministic benchmarks

## Key References

- Andersen, Bollerslev, Diebold, Labys (2003). Modeling and Forecasting Realized Volatility. *Econometrica*.
- Barndorff-Nielsen & Shephard (2004). Power and bipower variation. *JFE*.
- Barndorff-Nielsen & Shephard (2006). Econometrics of testing for jumps. *JFE*.
- Barndorff-Nielsen, Hansen, Lunde, Shephard (2008). Designing realized kernels. *Econometrica*.
- Bollerslev (1986). Generalized autoregressive conditional heteroskedasticity. *JoE*.
- Corsi (2009). A simple approximate long-memory model of realized volatility. *JFE*.
- Hansen & Lunde (2006). Consistent ranking of volatility models. *JoE*.
- Hansen, Huang, Shek (2012). Realized GARCH. *JAE*.
- Hansen, Lunde, Nason (2011). The Model Confidence Set. *Econometrica*.
- Herbster & Warmuth (1998). Tracking the Best Expert. *Machine Learning*.
- Nelson (1991). Conditional heteroskedasticity in asset returns. *Econometrica*.
- Patton (2011). Volatility forecast comparison using imperfect proxies. *JoE*.
- Patton & Sheppard (2015). Good volatility, bad volatility. *JFQA*.
- Vovk (1990). Aggregating strategies. *COLT*.
- Yang (2004). Combining forecasting procedures. *Econometric Theory*.
- Zhang, Mykland, Aït-Sahalia (2005). A tale of two time scales. *JASA*.

## License

MIT
