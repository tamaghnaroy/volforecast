# Initial Expert Review: AutoVolForecaster Design Plan

## 1. Per-dimension scoring

### DIM-A: Model Family Breadth vs. SOTA Literature
**Score: 6/10**

The plan is materially broader than a classical ARCH/GARCH-only design. `Phase 1 - Data Profiler` and `Phase 2 - Candidate Selector` cover asymmetric GARCH, long-memory models, HAR-family realized-volatility models, HEAVY/Realized-GARCH, GAS, stochastic-volatility, regime-switching, and several ML families, with inclusion rules tied to stylized facts such as leverage, jumps, long memory, heavy tails, and intraday availability. That breadth is a real strength, and the conditional logic is more defensible than a static "throw everything in" pool. The ceiling is lower than SOTA because the candidate set still lacks explicit rough-volatility families, hybrid neural-statistical models, and any literature-maintenance mechanism, while `Key Design Decisions` explicitly chooses to avoid model-specific tuning and therefore narrows the reachable frontier.

### DIM-B: Statistical Model Selection Validity
**Score: 6/10**

`Phase 3 - Cross-Validated Benchmarking` and `Phase 4 - Statistical Model Selection` specify an expanding-window protocol, a pre-declared primary loss (`QLIKE`), pairwise Diebold-Mariano pruning, and an MCS filter, which is already stronger than raw rank-based selection. The design also avoids the worst mistake in this area by separating benchmarking from final combination and by stating explicit fallback rules. The weakness is that the statistical protocol is incomplete where it matters most: the document does not specify HAC lag selection, bootstrap details for MCS, Harvey-Leybourne-Newbold small-sample correction, SPA/Reality-Check-style protection against data snooping, or special handling for nested model comparisons. As written, the design is credible but not yet reproducible enough to justify a higher anchor.

### DIM-C: Forecast Combination Theory and Online Regret Bounds
**Score: 7/10**

`Phase 5 - Forecast Combination Layer` is one of the stronger parts of the design. The plan names theory-backed combiners (`EWA`, `FixedShare`, `AFTER`, `RL`), maps some of them to operating conditions, and the `Key Design Decisions` section explicitly appeals to regret bounds for `EWA` and tracking arguments for `FixedShare`. The `CombinedForecaster` interface also makes online combination operational rather than rhetorical because weights are updated after new realizations arrive. The score stops at 7 because `AFTER` appears in the architecture but not in the selection heuristic, `RLCombiner` is introduced without a benchmark protocol against theory-backed baselines, and the plan does not define weight constraints, turnover penalties, or failure behavior when the combiner assumptions are violated.

### DIM-D: Proxy Robustness and Patton-2011 Consistency
**Score: 7/10**

The document is explicitly proxy-aware from the `Guiding Principles` onward, and `Phase 3` operationalizes that with Patton-robust losses, a realized-measure pipeline, and an SNR gate via `evaluation.proxy.proxy_noise_correction`. The use of `RV`, `BV`, `CV`, `JV`, and semivariances is materially better than treating one noisy realized-variance series as truth, and the ranking rule at least reacts to measured proxy quality. The main limitation is that the entire ranking stack still centers on a single `rv_oos` series and a single coarse threshold (`SNR < 1`), while the proposed fallback to `MSE` under poor proxy quality is not well justified because it does not remove target-proxy mismatch. There is also no realized-kernel, medRV, or cross-proxy sensitivity analysis, so the plan is robust by baseline standards but not fully target-aware.

### DIM-E: Online and Streaming Adaptivity with Regret Guarantees
**Score: 6/10**

The plan clearly treats online use as a first-class deployment mode. `Guiding Principles` promises an updateable final forecaster, `Phase 5` defines sequential weight updates, and the public API exposes `.update()` directly on `AutoVolForecaster`, which is the right systems-level shape for live forecasting. The design also fixes an operational refit cadence (`refit_every=21`), which is better than leaving retraining behavior implicit. It does not reach the next anchor because the adaptivity is concentrated in the combiner: the document specifies no forgetting factors, restart logic, drift detector, or performance-triggered retraining policy for the base models, and it gives no latency or resource budget for online operation.

### DIM-F: Regime Detection and Structural Break Handling
**Score: 5/10**

There is some real regime logic here, not just buzzwords. `Phase 1` computes a regime-switching indicator using squared-return dependence and a BDS test, `Phase 2` can include `MSVolForecaster`, and `Phase 5` routes the ensemble toward `FixedShareCombiner` when regime switching is detected. That is enough to show that regime variation is acknowledged and does change model-pool and combiner behavior. The design still falls short of a strong score because it has no explicit change-point or break detector, no uncertainty-aware policy once a break is suspected, no adaptive window-length response, and no milestone dedicated to stress-period validation.

### DIM-G: Multi-Horizon Forecast Architecture
**Score: 0/10**

The plan explicitly states in `Open Questions / Future Work` that the current design is "1-step-ahead only." The public API exposes `predict(horizon: int = 1)`, but there is no phase that defines direct versus iterated construction, no horizon-specific feature pipeline, and no horizon-specific evaluation or selection rule. This is exactly the case described by the rubric's zero anchor: longer horizons are deferred rather than designed. As a result, the system cannot currently claim to be an automated volatility forecaster beyond the daily one-step setting.

### DIM-H: Computational Scalability and Complexity
**Score: 3/10**

The document contains a few practical cost controls, but they are thin. `Phase 2` gates expensive models by sample length and PyTorch availability, and `Phase 3` reduces refit frequency to every 21 steps, which helps contain repeated estimation cost. Those are reasonable heuristics, but they are not a compute design. There is no complexity model, no parallelization strategy, no memory budget, no early elimination schedule for bad candidates, and no graceful degradation plan if the full pool is too expensive for production use.

### DIM-I: Production Readiness and Software Engineering
**Score: 6/10**

The software shape is good: the plan defines a stable public API, a result dataclass, a clear module layout, deterministic seeding support through `random_state`, basic capability gating such as "if PyTorch available," and an explicit fallback guarantee that at least one model survives. The milestone plan also includes unit and integration tests rather than treating testing as an afterthought. The design is still only mid-level production-ready because failures are often "silently dropped," there is no serialization/versioning contract for trained artifacts, no explicit input validation schema, no drift monitoring after deployment, and no rollback story if online updates degrade forecast quality. It reads like a solid library feature plan, not yet a fully hardened production subsystem.

### DIM-J: Benchmarking Rigor and Reproducibility
**Score: 6/10**

`Phase 3` specifies an expanding-window benchmark with a defined minimum training size, refit frequency, and pre-declared losses, and the milestone plan includes synthetic DGP checks for the profiler and MCS behavior. That is a competent evaluation foundation, especially because it is leakage-aware in spirit and ties selection to out-of-sample losses rather than in-sample fit. The document does not earn a higher score because the empirical validation plan is too narrow: there is no cross-asset panel benchmark, no multi-horizon evaluation, no ablation matrix for each design choice, no uncertainty intervals, and no one-command reproducibility path for the full benchmark suite. It is reproducible in fragments, not yet as a research-grade benchmarking program.

## 2. Total score

**Total: 52/100**

This is a competent modern baseline design with several strong components, especially in candidate coverage, proxy-aware evaluation, and online combination. It is not yet SOTA-grade because the frontier gaps are structural rather than cosmetic: there is no multi-horizon architecture, no explicit rough-volatility track, incomplete break handling, and no production-credible compute plan.

## 3. SOTA gap analysis: five critical weaknesses vs. best published systems (2020-2024)

1. **No rough-volatility or universal pooled-learning track.**  
   Tang, Rosenbaum, and Zhou (2024) show that a universal LSTM trained on pooled assets and a parsimonious rough-volatility model with Zumbach-effect structure can both outperform or match traditional per-asset approaches. The current plan includes `FIGARCHForecaster`, `LSTMVolForecaster`, and `TransformerVolForecaster`, but all selection logic is still single-series and there is no explicit roughness estimator, no rough-volatility candidate, and no pooled training mode. The missing capability is a frontier model family that exploits cross-asset universality instead of treating each return series as an isolated forecasting problem.

2. **No direct representation learning from raw intraday data.**  
   Moreno-Pino and Zohren (2024, *DeepVol*) show that dilated causal convolutions on raw high-frequency returns can preserve predictive content that is lost when intraday data are collapsed into realized measures. By contrast, `Phase 3 - Realized measure pipeline` converts intraday data into `RV`, `BV`, `CV`, `JV`, and semivariances and then feeds only those handcrafted summaries into the candidate models. The missing capability is a model class that learns intraday representations end to end rather than relying exclusively on precomputed realized features.

3. **No horizon-aware architecture despite published gains at medium and long horizons.**  
   Souto and Moradi (2024) show with NBEATSx that realized-volatility forecasting benefits from explicitly horizon-aware design and that medium- and long-horizon gains can be substantial relative to standard HAR/GARCH/LSTM baselines. The AutoVolForecaster plan explicitly postpones multi-horizon forecasting to `Future Work`, so the current scoring, selection, and combination logic are all one-step-only. The missing capability is not just more output steps; it is a coherent direct-versus-iterated, horizon-specific forecasting framework.

4. **Break handling is much weaker than recent regime-switching realized-volatility systems.**  
   Huang, Wan, Li, and Luo (2024) integrate long memory, jumps, heterogeneity, and switching regimes in a combined HAR-type system, while Ding, Kambouroudis, and McMillan (2023) show that regime-switching HAR variants improve realized-volatility forecasting especially beyond the daily horizon. In the plan, regime information is used only to include `MSVolForecaster` and prefer `FixedShareCombiner`; there is no formal change-point detector, no regime posterior carried through the pipeline, and no break-conditioned benchmark slice. The missing capability is a full trigger-to-action regime framework rather than a light pre-screening heuristic.

5. **No hybrid neural-statistical candidates and almost no model-level tuning.**  
   Reisenhofer, Bayer, and Hautsch (2022) use HARNet to initialize a deep model from a HAR baseline, and Perez-Hernandez, Arevalo-de-Pablos, and Camacho-Minano (2024) as well as Amirshahi and Lahmiri (2023) show that hybrid ANN/GARCH-style systems can beat either component alone. The AutoVolForecaster plan explicitly states in `Key Design Decisions` that it will avoid grid search and rely on model defaults, then combine models only after each one has already produced a forecast. The missing capability is hybridization inside the model family itself, where structural econometric priors and learned nonlinearities are fused rather than merely ensembled at the end.

## 4. Ten actionable improvement recommendations ranked by expected score uplift

1. **Title: Add a horizon-aware direct multi-step forecasting track**  
   **Problem:** The design scores `0/10` on DIM-G because it is explicitly one-step-only, which also weakens the benchmarking story and narrows the product claim.  
   **Proposed solution:** Split the pipeline by horizon bucket (`1d`, `5d`, `22d`) and support direct multi-step candidates alongside iterated ones. Add horizon-specific feature construction, horizon-specific ranking tables, and temporal-aggregation checks so that model selection and combination are not reused unchanged across horizons.  
   **Literature refs:** Souto and Moradi (2024); Marcellino, Stock, and Watson (2006); Ben Taieb, Sorjamaa, and Bontempi (2012).  
   **Dimensions addressed:** DIM-G, DIM-J, DIM-A.

2. **Title: Introduce explicit rough-volatility and universal pooled-learning candidates**  
   **Problem:** The current family set misses one of the clearest frontier directions in 2020-2024 volatility forecasting: roughness-aware and pooled cross-asset models.  
   **Proposed solution:** Add a `RoughVolForecaster` family with roughness diagnostics in `DataProfiler`, and add an optional pooled-training mode for neural candidates that can borrow strength across assets while preserving per-asset inference. Expose this as a gated capability rather than a default so that single-series workflows remain lightweight.  
   **Literature refs:** Tang, Rosenbaum, and Zhou (2024); Bennedsen, Lunde, and Pakkanen (2022); Bayer, Friz, and Gatheral (2016).  
   **Dimensions addressed:** DIM-A, DIM-E, DIM-J.

3. **Title: Add explicit structural-break detection and break-response policies**  
   **Problem:** The current regime logic is mostly a pre-fit heuristic, so the system has no principled response when the data-generating process shifts.  
   **Proposed solution:** Add a change-point detector or break detector ahead of benchmarking, and connect detections to concrete actions: reset online combiner weights, shorten windows, increase refit frequency, and re-open previously excluded candidate families. Benchmark stress windows separately so break handling is measurable instead of aspirational.  
   **Literature refs:** Fryzlewicz (2014); Truong, Oudre, and Vayatis (2020); Huang et al. (2024); Ding, Kambouroudis, and McMillan (2023).  
   **Dimensions addressed:** DIM-F, DIM-E, DIM-J.

4. **Title: Add hybrid neural-statistical model families instead of only post-hoc ensembling**  
   **Problem:** Late-stage combination cannot recover capabilities that require structural priors inside the model architecture.  
   **Proposed solution:** Add at least two hybrid candidates, such as HAR-initialized convolutional models and GARCH-feature-to-neural models, then benchmark them alongside the current standalone families. Keep the existing ensemble layer, but let it combine both pure and hybrid experts.  
   **Literature refs:** Reisenhofer, Bayer, and Hautsch (2022); Perez-Hernandez, Arevalo-de-Pablos, and Camacho-Minano (2024); Amirshahi and Lahmiri (2023).  
   **Dimensions addressed:** DIM-A, DIM-C, DIM-J.

5. **Title: Upgrade proxy robustness with multi-proxy sensitivity and microstructure-robust measures**  
   **Problem:** Ranking still depends too heavily on one realized-volatility proxy and one coarse SNR gate.  
   **Proposed solution:** Extend the realized-measure pipeline to include realized kernels or other microstructure-robust measures where intraday data permit, and run sensitivity analysis across at least two proxy families before final selection. If rankings are unstable across proxies, downgrade confidence and widen the survivor set instead of simply switching to `MSE`.  
   **Literature refs:** Patton (2011); Hansen and Lunde (2006); Barndorff-Nielsen, Hansen, Lunde, and Shephard (2008).  
   **Dimensions addressed:** DIM-D, DIM-B, DIM-J.

6. **Title: Complete the forecast-comparison protocol with HAC/SPA/Reality-Check safeguards**  
   **Problem:** DM plus MCS is directionally right but underspecified for a large candidate search with repeated out-of-sample reuse.  
   **Proposed solution:** Define the exact DM variant, HAC lag rule, bootstrap scheme for MCS, and add SPA or White-style Reality Check protection when screening many models. Report effect sizes and confidence intervals in the ranking table so selection decisions are auditable rather than purely binary.  
   **Literature refs:** Diebold and Mariano (1995); Harvey, Leybourne, and Newbold (1997); Hansen (2005); White (2000); Hansen, Lunde, and Nason (2011).  
   **Dimensions addressed:** DIM-B, DIM-J.

7. **Title: Add a staged compute scheduler with early elimination**  
   **Problem:** The current design can become computationally unbounded as the candidate pool and backtest length grow.  
   **Proposed solution:** Run cheap baselines first, eliminate obviously dominated candidates on a small warm-up backtest, and reserve expensive ML/SV/regime models for assets whose profiler statistics justify them. Document asymptotic cost and expected wall-clock envelopes for CPU-only and GPU-enabled modes.  
   **Literature refs:** Rocklin (2015); Dean and Ghemawat (2008); Sculley et al. (2015).  
   **Dimensions addressed:** DIM-H, DIM-I, DIM-J.

8. **Title: Add a raw intraday encoder candidate**  
   **Problem:** Handcrafted realized measures discard intraday shape information that modern high-frequency models can exploit.  
   **Proposed solution:** Add one raw-intraday neural candidate, such as a dilated temporal convolution or similar architecture, behind a capability gate that activates only when intraday arrays are available and sufficiently long. Evaluate it against the handcrafted realized-measure families rather than replacing them.  
   **Literature refs:** Moreno-Pino and Zohren (2024); Reisenhofer, Bayer, and Hautsch (2022).  
   **Dimensions addressed:** DIM-A, DIM-H, DIM-J.

9. **Title: Add a panel/global training mode across assets**  
   **Problem:** The plan assumes one-series-at-a-time fitting, which leaves data efficiency on the table and can make ML candidates both weaker and slower.  
   **Proposed solution:** Add an optional global mode in which a shared model is trained on a panel of assets with asset embeddings or grouped normalization, while still returning asset-specific forecasters at inference time. Use it only when enough assets are available and benchmark it directly against the current single-series mode.  
   **Literature refs:** Zhu, Bai, He, and Liu (2023); Tang, Rosenbaum, and Zhou (2024).  
   **Dimensions addressed:** DIM-A, DIM-H, DIM-J.

10. **Title: Harden the production contract around artifacts, monitoring, and rollback**  
   **Problem:** The current design is library-ready, but not fully operationally safe once forecasts go live and models are updated online.  
   **Proposed solution:** Version fitted artifacts, store profiler statistics and benchmark metadata with the model object, add forecast-drift monitors, and define rollback behavior when the online combiner or updated base models degrade. Replace silent model drops with typed warnings and structured diagnostics.  
   **Literature refs:** Breck et al. (2017); Amershi et al. (2019); Sculley et al. (2015).  
   **Dimensions addressed:** DIM-I, DIM-E, DIM-J.

## References

- Amirshahi, B., and Lahmiri, S. (2023). Hybrid deep learning and GARCH-family models for forecasting volatility of cryptocurrencies.
- Amershi, S., et al. (2019). Software Engineering for Machine Learning: A Case Study.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., and Shephard, N. (2008). Designing Realized Kernels to Measure the Ex Post Variation of Equity Prices in the Presence of Noise.
- Bayer, C., Friz, P., and Gatheral, J. (2016). Pricing under Rough Volatility.
- Bennedsen, M., Lunde, A., and Pakkanen, M. S. (2022). Decoupling the Short- and Long-Term Behavior of Stochastic Volatility.
- Ben Taieb, S., Sorjamaa, A., and Bontempi, G. (2012). Recursive and Direct Multi-Step Forecasting: The Best of Both Worlds.
- Breck, E., et al. (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.
- Dean, J., and Ghemawat, S. (2008). MapReduce: Simplified Data Processing on Large Clusters.
- Diebold, F. X., and Mariano, R. S. (1995). Comparing Predictive Accuracy.
- Ding, Y., Kambouroudis, D. S., and McMillan, D. G. (2023). Forecasting Realised Volatility Using Regime-Switching Models.
- Fryzlewicz, P. (2014). Wild Binary Segmentation for Multiple Change-Point Detection.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability.
- Hansen, P. R., and Lunde, A. (2006). Realized Variance and Market Microstructure Noise.
- Hansen, P. R., Lunde, A., and Nason, J. M. (2011). The Model Confidence Set.
- Harvey, D., Leybourne, S., and Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors.
- Huang, Y., Wan, Z., Li, H., and Luo, Y. (2024). Forecasting Volatility Based on a New Combined HAR-Type Model with Long Memory and Switching Regime.
- Marcellino, M., Stock, J. H., and Watson, M. W. (2006). A Comparison of Direct and Iterated Multistep AR Methods for Forecasting Macroeconomic Time Series.
- Moreno-Pino, F., and Zohren, S. (2024). DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions.
- Patton, A. J. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies.
- Perez-Hernandez, F., Arevalo-de-Pablos, A., and Camacho-Minano, M. M. (2024). A hybrid model integrating artificial neural network with multiple GARCH-type models and EWMA for performing the optimal volatility forecasting of market risk factors.
- Reisenhofer, R., Bayer, X., and Hautsch, N. (2022). HARNet: A Convolutional Neural Network for Realized Volatility Forecasting.
- Rocklin, M. (2015). Dask: Parallel Computation with Blocked Algorithms and Task Scheduling.
- Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems.
- Souto, H. G., and Moradi, A. (2024). Introducing NBEATSx to realized volatility forecasting.
- Tang, S. H., Rosenbaum, M., and Zhou, C. (2024). Forecasting volatility with machine learning and rough volatility: example from the crypto-winter.
- Truong, C., Oudre, L., and Vayatis, N. (2020). Selective Review of Offline Change Point Detection Methods.
- White, H. (2000). A Reality Check for Data Snooping.
- Zhu, H., Bai, L., He, L., and Liu, Z. (2023). Forecasting realized volatility with machine learning: Panel data perspective.
