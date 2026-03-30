## 1. **Memory Architecture**

* How does vol forecaster explore temporal dependency — fixed window, exponential decay, attention mechanism?
* Doe the auto-volcaster test and distinguish **short-memory** (intraday shocks, Hurst exponent < 0.5) from **long-memory** (ARCH effects, Hurst exponent > 0.5)?
* Fractional integration handling — volatility is known to be long-memory as well as recent results show volatility is rough; does the algorithm respect this?

## 2. **Stationarity Assumptions**

* Does the algorithm explore trnsformations of vols -  raw vol, log-vol, or differenced series?
* How does it handle non-stationarity and structural breaks — does it adapt or assume away?

## 3. **Variance Decomposition**

* Does the model separate **continuous variation** from **jump components**?
* Is realized variance decomposed (e.g., bipower variation) to isolate jump-robust estimates?

## 4. **Asymmetry Encoding**

* Does the algorithm capture the **leverage effect** — that negative returns increase vol more than positive returns of equal magnitude?
* Is there an explicit asymmetric loss surface, or is it symmetric (a fundamental algorithmic flaw for vol)?

## 5. **Heteroscedasticity Modeling**

* Is conditional heteroscedasticity explicitly modeled, or is it a residual artifact?
* Does the algorithm distinguish **volatility-of-volatility** from volatility itself?

## 6. **Loss Function Alignment**

* What loss function is the algorithm trained on — MSE, QLIKE, log-likelihood?
* QLIKE is theoretically optimal for vol forecasting; MSE penalizes large errors symmetrically, which is misaligned with vol's skewed distribution
* Does it make corrections for the fact that the realized volatility can be different from predicted volatility even if predicted volatility is 100% accurate due to the statistical nature of the realized volatility with variance proportional to volatiity itself.

## 7. **Distributional Assumptions**

* Does the algorithm assume Gaussian innovations, or does it allow fat tails (Student-t, GED, NIG)?
* Is the conditional distribution parameterized or non-parametric?

## 8. **Aggregation Consistency**

* Are forecasts **temporally consistent** — does a 5-day forecast equal the aggregation of five 1-day forecasts under the model's own logic?
* Violation of this is a fundamental algorithmic incoherence

## 9. **Information Set Specification**

* What sigma-algebra does the model condition on — price only, volume, options surface, macro?
* Is the conditioning information set **non-anticipative** (no look-ahead leakage built into the algorithm)?

## 10. **Mean Reversion Dynamics**

* Does the algorithm encode vol mean-reversion explicitly, and to what target (unconditional mean, regime-conditional mean)?
* What is the implied half-life of a vol shock — is it algorithmically imposed or learned?


## 11. **Aggregation Mechanism**

* Is combination done via **simple averaging, weighted averaging, stacking, or learned gating**?
* Are weights static or **dynamic** — do they shift based on recent model performance or regime?
* Does the gating function itself have a memory structure, or is it memoryless?

## 12. **Diversity of the Ensemble**

* Are base models diverse in their **information sets** (price-only vs. options vs. macro), or just hyperparameter variants of the same architecture?
* Algorithmic diversity matters more than model count — ten correlated models add little over one
* Is diversity **enforced** (e.g., negative correlation training) or incidental?

## 13. **Bias-Variance Decomposition Across Members**

* Are high-variance/low-bias models (e.g., deep nets) being combined with low-variance/high-bias models (e.g., GARCH) intentionally?
* Does the aggregation scheme exploit this decomposition, or ignore it?

## 14. **Meta-Model Specification**

* In stacking, what is the meta-learner's input — raw forecasts, forecast errors, forecast uncertainty, or all three?
* Is the meta-model allowed to learn **when to trust which base model**, or just a fixed linear blend?
* Does the meta-learner overfit to base model idiosyncrasies rather than signal?

## 15. **Regime-Conditioned Weighting**

* Are model weights conditioned on latent regime (low/high vol, trending/mean-reverting)?
* Is regime detection endogenous to the aggregation algorithm, or an exogenous input?
* A mixture-of-experts architecture is fundamentally different from a flat ensemble here

## 16. **Temporal Stability of Weights**

* How frequently are aggregation weights refit — and does refit frequency introduce lookahead?
* Are weights allowed to change continuously (online learning) or in discrete windows?
* Weight instability can amplify rather than reduce forecast variance

## 17. **Error Correlation Structure**

* Does the aggregation algorithm explicitly model the **covariance of base model errors**?
* Optimal linear combination (Bates-Granger) requires error covariance — ignoring it is a specification gap
* Are errors assumed independent across members, and is that assumption tested?

## 18. **Propagation of Uncertainty**

* Does the meta-model propagate **distributional uncertainty** from base models, or collapse to a point estimate?
* In Bayesian model averaging (BMA), model weights are posterior probabilities — is the algorithm doing true BMA or an approximation?
* Is the ensemble's output uncertainty larger than any individual member's when members disagree — as it should be?

## 19. **Bagging Coherence for Time Series**

* Standard bagging assumes i.i.d. samples — does the algorithm use **block bootstrap** or **stationary bootstrap** to respect vol's autocorrelation structure?
* Are bootstrap replications drawn to preserve the long-memory and heteroscedasticity properties of the original series?

## 20. **Online vs. Offline Aggregation**

* Is the aggregation algorithm **online** (weights update with each new observation) or **offline** (batch refit)?
* For vol forecasting, offline aggregation with a long refit window will systematically underweight regime shifts
* Does the algorithm have a forgetting mechanism — exponential discounting, sliding window, or change-point detection?

## 21. **Forecast Horizon Consistency Across Members**

* Do all base models produce forecasts over the **same horizon with the same frequency**?
* Mixing a 1-day GARCH with a 5-day HAR without horizon alignment is an aggregation incoherence
* Does the meta-model account for differential horizon degradation rates across members?

## 22. **Collinearity Among Base Models**

* If base models share features or architectures, their forecasts are collinear — does the aggregation algorithm handle this (e.g., regularized stacking via ridge/lasso)?
* Unregularized stacking with correlated members is algorithmically equivalent to amplifying a single noisy signal
