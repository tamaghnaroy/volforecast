# Issue 01 — Add ARCH(q) and EWMA/RiskMetrics baseline forecasters

## Why
ARCH and EWMA are foundational cvol benchmarks used in empirical comparisons. They are lightweight, interpretable, and essential for sanity-checking richer models.

## Gap evidence
- Current implemented model classes do not include ARCH or EWMA forecasters.
- Existing benchmark/evaluation stack would benefit from simple low-parameter baselines.

## Proposal
Add:
1. `ARCHForecaster(q=1..p)`
2. `EWMAForecaster(lambda_=0.94 or estimated)`

Both should implement `BaseForecaster` and declare `CONDITIONAL_VARIANCE` target.

## Implementation notes
- ARCH: either use `arch_model(..., vol='ARCH')` or implement direct recursion.
- EWMA: deterministic recursion `sigma2_t = λ sigma2_{t-1} + (1-λ) r_{t-1}^2`.
- Support `fit/predict/update/get_params` for rolling/online benchmarks.

## Acceptance criteria
- [ ] New forecaster classes with tests for fit/predict/update behavior.
- [ ] Added to model registry/docs.
- [ ] Included in benchmark examples.

## References
- Engle (1982) ARCH.
- RiskMetrics Technical Document (1996).
