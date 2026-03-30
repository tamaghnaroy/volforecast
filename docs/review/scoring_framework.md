# SOTA Scoring Framework for Automated Volatility Forecasting Design

## Purpose

This rubric scores an automated volatility forecasting design on **10 equal-weight dimensions**. Each dimension is scored from **0 to 10**, for a total score out of **100**.

The rubric is intended for **design review**, not marketing claims. A score should only be awarded when the design specifies concrete mechanisms, decision rules, and evaluation procedures. Stated aspirations without implementation detail should not score above **5**.

## Scoring Rules

1. Score each dimension independently.
2. Use the **highest anchor whose criteria are materially satisfied**.
3. Intermediate scores (`1-2`, `4`, `6`, `8-9`) may be used only when the design clearly falls between two anchors.
4. To award `7` or `10`, the design should usually specify both:
   - a concrete algorithmic mechanism; and
   - a defensible validation protocol.
5. Equal weights are recommended unless a downstream review explicitly overrides them.

## Score Interpretation

| Total | Interpretation |
|---|---|
| `0-24` | Non-credible research sketch |
| `25-49` | Weak design with major scientific gaps |
| `50-69` | Competent baseline design, not SOTA |
| `70-84` | Strong modern design with limited frontier gaps |
| `85-100` | SOTA-grade design with credible production path |

---

## DIM-A: Model Family Breadth vs. SOTA Literature (2015-2024)

**Measures**  
Coverage of materially distinct volatility model families and whether the candidate set spans the main empirical regularities documented in recent literature: leverage, jumps, realized measures, long memory, roughness, latent volatility, nonlinear interactions, and hybrid/neural approaches.

**Why it matters**  
Automated forecasters are capped by the candidate set they search. If the design excludes major families, it cannot recover frontier performance under the data-generating processes those families were built for.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Candidate set is effectively one narrow family, such as only ARCH/GARCH variants. No realized-volatility, latent-volatility, regime, rough/fractional, or neural/hybrid models are considered. |
| `3` | Includes 2-3 classical families, typically GARCH-type plus one of HAR, SV, or basic ML. Coverage is still missing major SOTA families and there is no explicit rationale for family inclusion. |
| `5` | Includes a respectable modern baseline panel: asymmetric GARCH, realized-volatility models, at least one latent-volatility or regime model, and one ML family. However, rough/fractional volatility, hybrid neural-statistical models, or explicit family-selection diagnostics are missing. |
| `7` | Broad candidate coverage across parametric, realized, latent-state, regime-switching, and ML families, with data-driven inclusion logic tied to observed stylized facts. At least one rough/fractional or other frontier family is represented or explicitly justified as out of scope. |
| `10` | Near-frontier coverage across all major practically relevant families: asymmetric GARCH, HAR variants, HEAVY/Realized-GARCH, GAS, SV/SVJ, regime models, long-memory/rough volatility, and neural/hybrid models. The design also specifies principled exclusion rules, capability gating, and an update path for literature published after deployment. |

**Key references**
- Hansen, Huang, Shek (2012), *Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility*.
- Bayer, Friz, Gatheral (2016), *Pricing under Rough Volatility*.
- Tang, Rosenbaum, Zhou (2023), *Forecasting Volatility with Machine Learning and Rough Volatility*.

---

## DIM-B: Statistical Model Selection Validity

**Measures**  
Whether the design uses statistically valid out-of-sample forecast comparison procedures, including dependence-robust tests, multiple-model corrections, and pre-specified selection rules.

**Why it matters**  
Raw loss rankings are noisy and unstable. Without valid testing, automated model selection overstates evidence, especially when many models are compared repeatedly on the same sample.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Selection is based on in-sample fit, eyeballing, or raw out-of-sample rank only. No hypothesis testing or uncertainty quantification is specified. |
| `3` | Uses a single pairwise test or single summary metric, but ignores serial dependence, data snooping, or multiple comparisons. |
| `5` | Uses DM tests or MCS, but the protocol is incomplete: dependence correction, bootstrap details, nested-model issues, or multiplicity control are underspecified. |
| `7` | Uses a defensible forecast-comparison stack: pre-declared primary loss, HAC-robust DM testing, MCS or SPA-style screening, and explicit rules for pruning or retaining statistically indistinguishable models. |
| `10` | Full forecast-selection protocol is specified end to end: primary/secondary losses, dependence-robust tests, multiple-testing control, treatment of nested models or model pools, effect-size reporting, and a deterministic decision rule that can be reproduced without analyst discretion. |

**Key references**
- Diebold, Mariano (1995), *Comparing Predictive Accuracy*.
- Harvey, Leybourne, Newbold (1997), *Testing the Equality of Prediction Mean Squared Errors*.
- Hansen, Lunde, Nason (2011), *The Model Confidence Set*.

---

## DIM-C: Forecast Combination Theory and Online Regret Bounds

**Measures**  
Theoretical soundness of the combination layer, including whether weight updates are justified by online learning, expert-advice, or forecast-combination theory, and whether nonstationary tracking is handled.

**Why it matters**  
Combining forecasts often beats single-model selection, but ad hoc averaging is fragile. In automated systems, the combination rule should remain defensible under model turnover and regime change.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | No combination layer, or combination is an ad hoc average with no theory or validation. |
| `3` | Static weighted average or stacking is proposed, but weights are fit heuristically and there is no discussion of regret, switching, or forecast-error dependence. |
| `5` | Includes at least one standard combiner such as equal-weight, Bates-Granger, or EWA, but without a clear rule for when to prefer it or how it behaves under nonstationarity. |
| `7` | Combination layer is theory-backed and matched to use case, e.g. EWA for exp-concave loss, Fixed-Share for switching experts, AFTER for adaptive aggregation. Regret or tracking guarantees are stated and tied to operational choices. |
| `10` | The design supports multiple principled combiners, explicit selection or gating among them, weight constraints, switching/nonstationary guarantees, and clear fallback behavior when regret assumptions are violated. If RL or other high-capacity combiners are used, they are benchmarked against theory-backed baselines rather than replacing them by default. |

**Key references**
- Cesa-Bianchi, Lugosi (2006), *Prediction, Learning, and Games*.
- Herbster, Warmuth (1998), *Tracking the Best Expert*.
- Yang (2004), *Combining Forecasting Procedures: Some Theoretical Results*.

---

## DIM-D: Proxy Robustness and Patton-2011 Consistency

**Measures**  
Whether the design respects the distinction between latent volatility targets and noisy realized proxies, and whether evaluation/ranking uses loss functions and diagnostics that are robust to proxy noise.

**Why it matters**  
Volatility is usually unobserved. A design that treats a noisy proxy as truth can select the wrong model even when forecasts are otherwise well calibrated.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Uses non-robust losses on noisy proxies with no acknowledgement of measurement error or target-proxy mismatch. |
| `3` | Mentions QLIKE or MSE, but uses a single proxy mechanically and provides no diagnostics for proxy quality, microstructure noise, or target mismatch. |
| `5` | Uses Patton-consistent losses and acknowledges proxy quality, but relies on one realized-volatility proxy and does not define gating or fallback rules. |
| `7` | Uses robust losses, multiple realized proxies or jump/continuous decompositions where relevant, and explicit proxy-quality checks such as SNR or noise diagnostics that influence model ranking. |
| `10` | The design is target-aware and proxy-aware throughout: robust loss selection, microstructure-robust realized measures, sensitivity across proxies, explicit gating when proxies are too noisy, and ranking decisions that remain valid under the Patton consistency conditions. |

**Key references**
- Patton (2011), *Volatility Forecast Comparison Using Imperfect Volatility Proxies*.
- Hansen, Lunde (2006), *Realized Variance and Market Microstructure Noise*.
- Barndorff-Nielsen, Hansen, Lunde, Shephard (2008), *Designing Realized Kernels to Measure the Ex Post Variation of Equity Prices in the Presence of Noise*.

---

## DIM-E: Online and Streaming Adaptivity with Regret Guarantees

**Measures**  
Ability to update forecasts sequentially under streaming data, adapt to drift, and do so with explicit performance guarantees or clearly bounded adaptation logic.

**Why it matters**  
Volatility forecasting is intrinsically online. Systems that require repeated full refits or fixed historical regimes are brittle in live deployment.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Pure batch workflow. No incremental updates, no online interface, and no plan for post-deployment adaptation. |
| `3` | Periodic retraining is mentioned, but adaptation is effectively manual or calendar-based and does not react to forecast deterioration. |
| `5` | Supports online updates or warm starts, but without forgetting, restart, or formal treatment of drifting environments. |
| `7` | Streaming behavior is a first-class design feature: online updates, forgetting or discounting, explicit update cadence, and at least one regret- or tracking-based argument for why adaptation should remain stable. |
| `10` | Streaming-first design with anytime updates, restart/share logic, adaptive learning rates or window lengths, clear drift triggers, and formal guarantees or strong theory-backed surrogates for performance under nonstationarity. Resource budgets for online operation are also specified. |

**Key references**
- Cesa-Bianchi, Lugosi (2006), *Prediction, Learning, and Games*.
- Herbster, Warmuth (1998), *Tracking the Best Expert*.
- Hazan (2016), *Introduction to Online Convex Optimization*.

---

## DIM-F: Regime Detection and Structural Break Handling

**Measures**  
How the design detects latent regimes, structural breaks, or local nonstationarities, and how those detections change model selection, weighting, windows, or fallback logic.

**Why it matters**  
Volatility dynamics are not globally stationary. Designs that ignore breaks or regime shifts often look strong in average periods and fail exactly when robustness matters.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Assumes stationarity throughout. No regime logic, break detection, or adaptive windowing is specified. |
| `3` | Uses simple heuristics such as volatility thresholds or rolling windows, but there is no formal detection step and no clear policy response. |
| `5` | Includes one substantive mechanism such as Markov-switching, change-point detection, or adaptive windows, but the trigger/action mapping is incomplete or unvalidated. |
| `7` | Combines at least one latent-regime approach with one explicit break-handling mechanism, and specifies what changes after detection: model pool, weights, refit schedule, or window length. |
| `10` | Regime and break handling are deeply integrated: probabilistic or statistically validated detection, uncertainty-aware actions, adaptive horizons/windows, and regime-specific evaluation showing that stress-period performance is not an afterthought. |

**Key references**
- Hamilton, Susmel (1994), *Autoregressive Conditional Heteroskedasticity and Changes in Regime*.
- Fryzlewicz (2014), *Wild Binary Segmentation for Multiple Change-Point Detection*.
- Truong, Oudre, Vayatis (2020), *Selective Review of Offline Change Point Detection Methods*.

---

## DIM-G: Multi-Horizon Forecast Architecture

**Measures**  
Whether the design treats 1-step, medium-horizon, and long-horizon volatility forecasting as distinct tasks with coherent forecast construction, horizon-aligned features, and consistent evaluation.

**Why it matters**  
A strong 1-day design is not automatically strong at 5-day or 22-day horizons. Horizon mismatch is a common source of false confidence in automated forecasting systems.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Only 1-step forecasting is designed. Longer horizons are ignored or implicitly assumed equivalent. |
| `3` | Multi-horizon outputs are produced recursively, but there is no discussion of direct vs. iterated forecasting, aggregation consistency, or horizon-specific evaluation. |
| `5` | Supports multiple horizons, but uses the same features, tuning, and ranking logic across all horizons without testing coherence or degradation. |
| `7` | Horizon design is explicit: direct vs. iterated strategy is justified, HAR-style or other multi-scale features are used where appropriate, and evaluation is horizon-specific. Temporal aggregation consistency is checked or at least explicitly managed. |
| `10` | Unified multi-horizon architecture with horizon-aware model selection, coherent aggregation constraints, direct/iterated/multi-output tradeoffs documented, and separate evidence that short-, medium-, and long-horizon performance are all credible. |

**Key references**
- Marcellino, Stock, Watson (2006), *A Comparison of Direct and Iterated Multistep AR Methods for Forecasting Macroeconomic Time Series*.
- Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*.
- Ben Taieb, Sorjamaa, Bontempi (2012), *Recursive and Direct Multi-Step Forecasting: The Best of Both Worlds*.

---

## DIM-H: Computational Scalability and Complexity

**Measures**  
Whether the design specifies time and memory complexity, parallelization strategy, hardware assumptions, and graceful degradation as the model pool, data volume, or horizon count grows.

**Why it matters**  
Automated model search is easy to describe and expensive to run. A design that is statistically elegant but computationally unbounded is not deployable.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | No runtime, memory, or hardware considerations are given. |
| `3` | Runtime is mentioned qualitatively, but there is no complexity estimate, no batching strategy, and no prioritization of expensive candidates. |
| `5` | Includes some practical efficiency choices such as parallel model fitting, warm starts, or optional GPU use, but complexity is not tied to the candidate pool or deployment constraints. |
| `7` | Specifies at least rough asymptotic or empirical scaling, parallel evaluation strategy, memory-conscious data flow, and explicit treatment of expensive models through gating, early stopping, or capability checks. |
| `10` | End-to-end compute design is production credible: complexity estimates, scheduling policy, CPU/GPU separation, checkpointing, incremental recomputation, latency/throughput budgets, and graceful fallback when resource limits are hit. |

**Key references**
- Dean, Ghemawat (2008), *MapReduce: Simplified Data Processing on Large Clusters*.
- Rocklin (2015), *Dask: Parallel Computation with Blocked Algorithms and Task Scheduling*.
- Abadi et al. (2016), *TensorFlow: A System for Large-Scale Machine Learning*.

---

## DIM-I: Production Readiness and Software Engineering

**Measures**  
Software design quality: API consistency, failure handling, testability, observability, dependency management, reproducibility, and safe deployment behavior.

**Why it matters**  
Automated forecasting systems fail in operational details long before they fail in asymptotic theory. Production readiness determines whether a strong research design can survive live use.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | Research script only. No stable API, no tests, no input validation, and no failure handling. |
| `3` | Basic class/API structure exists, but operational concerns are thin: poor error handling, no monitoring, limited tests, and unclear dependency assumptions. |
| `5` | Modular API, some unit tests, and basic fallback behavior are present. However, serialization, compatibility contracts, reproducibility, and operational safeguards remain partial. |
| `7` | Production-oriented design: stable interfaces, explicit capability detection, fail-safe fallbacks, deterministic seeds, unit/integration tests, and clear diagnostics surfaced to users. |
| `10` | Full production posture: versioned artifacts, reproducible environments, strong test coverage, health checks, forecast-drift monitoring, rollback/fallback plans, and operational documentation sufficient for handoff to another engineering team. |

**Key references**
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.
- Breck et al. (2017), *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*.
- Amershi et al. (2019), *Software Engineering for Machine Learning: A Case Study*.

---

## DIM-J: Benchmarking Rigor and Reproducibility

**Measures**  
Quality of the empirical validation plan: out-of-sample protocol, leakage prevention, benchmark breadth, synthetic and empirical evaluation, ablations, and reproducible experiment control.

**Why it matters**  
Many volatility designs look strong only because the evaluation is permissive. Reproducible and leakage-safe benchmarking is the difference between a publishable claim and a trustworthy system.

**Rubric**

| Score | Criteria |
|---|---|
| `0` | In-sample or poorly defined evaluation only. No reproducibility controls. |
| `3` | Single train/test split with basic baselines. Leakage controls, seed management, and robustness checks are absent or vague. |
| `5` | Uses rolling or expanding out-of-sample evaluation with sensible baselines and some reproducibility controls, but little ablation, stress testing, or cross-dataset evidence. |
| `7` | Benchmark harness is methodical: leakage-safe rolling/expanding evaluation, strong baselines, ablations, sensitivity analysis, multiple datasets or DGPs, and reproducible configuration. |
| `10` | Evaluation is frontier-grade: pre-declared benchmark protocol, cross-asset and cross-horizon coverage, synthetic DGPs plus empirical panels, ablations of every major design choice, uncertainty intervals, and one-command reproducibility of the full benchmark suite. |

**Key references**
- White (2000), *A Reality Check for Data Snooping*.
- Hansen (2005), *A Test for Superior Predictive Ability*.
- Bergmeir, Hyndman, Koo (2018), *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction*.

---

## Recommended Review Output Format

For each dimension, reviewers should record:

| Field | Description |
|---|---|
| `Score` | Integer or half-point score from `0` to `10` |
| `Evidence` | Specific design elements that justify the score |
| `Missing for next anchor` | Minimal additions required to reach the next anchor |
| `Primary risk` | Main failure mode if the design is implemented as written |

This keeps scoring auditable and makes score changes traceable during debate or revision rounds.
