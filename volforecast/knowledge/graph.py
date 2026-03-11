"""
Volatility model knowledge graph.

NetworkX-based directed graph encoding:
- Model nodes with metadata (family, target, assumptions, complexity, reference)
- Relationship edges (extends, targets, assumes, competes_with)
- Query interface for model discovery and comparison

Built from the Codex CLI Round 1 debate taxonomy validated against:
- Bollerslev (1986), Nelson (1991), Glosten et al. (1993), Ding et al. (1993)
- Baillie et al. (1996), Engle & Lee (1999), Shephard & Sheppard (2010)
- Hansen et al. (2012), Corsi (2009), Andersen et al. (2007)
- Barndorff-Nielsen & Shephard (2002, 2004, 2006)
- Patton (2011), Hansen & Lunde (2006)
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from volforecast.core.targets import VolatilityTarget


def _vt(name: str) -> VolatilityTarget:
    return VolatilityTarget[name]


# ──────────────────────────────────────────────
# Canonical model registry
# ──────────────────────────────────────────────
_MODELS: list[dict] = [
    # ── GARCH family ──
    dict(id="GARCH", name="GARCH(1,1)", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("stationary returns", "symmetric response", "finite 4th moment"),
         complexity="O(T) MLE", reference="Bollerslev (1986), JoE",
         extends=("ARCH",)),
    dict(id="EGARCH", name="Exponential GARCH", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("log-variance dynamics", "asymmetric news impact"),
         complexity="O(T) MLE", reference="Nelson (1991), Econometrica",
         extends=("GARCH",)),
    dict(id="GJR", name="GJR-GARCH(1,1)", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("leverage effect via indicator", "stationary"),
         complexity="O(T) MLE", reference="Glosten, Jagannathan, Runkle (1993), JoF",
         extends=("GARCH",)),
    dict(id="APARCH", name="Asymmetric Power ARCH", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("power transformation delta", "asymmetry gamma"),
         complexity="O(T) MLE, extra param delta", reference="Ding, Granger, Engle (1993), JIMF",
         extends=("GARCH", "GJR")),
    dict(id="FIGARCH", name="Fractionally Integrated GARCH", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("long memory d in (0,0.5)", "fractional differencing"),
         complexity="O(T*p_trunc) MLE", reference="Baillie, Bollerslev, Mikkelsen (1996), JoE",
         extends=("GARCH",)),
    dict(id="CGARCH", name="Component GARCH", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("permanent + transitory components", "mean-reverting"),
         complexity="O(T) MLE", reference="Engle & Lee (1999), in Engle (ed.)",
         extends=("GARCH",)),
    dict(id="HEAVY", name="High-frEquency bAsed VolatilitY", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("uses realized measure as regressor", "HEAVY recursion"),
         complexity="O(T) MLE", reference="Shephard & Sheppard (2010), JFE",
         extends=("GARCH",)),
    dict(id="RGARCH", name="Realized GARCH", family="GARCH",
         target="CONDITIONAL_VARIANCE",
         assumptions=("joint model returns + realized measure", "measurement eq"),
         complexity="O(T) MLE", reference="Hansen, Huang, Shek (2012), JAE",
         extends=("GARCH", "HEAVY")),

    # ── HAR family ──
    dict(id="HAR_RV", name="HAR-RV", family="HAR",
         target="INTEGRATED_VARIANCE",
         assumptions=("heterogeneous agents", "RV as proxy for IV", "linear"),
         complexity="O(T) OLS", reference="Corsi (2009), JFE",
         extends=()),
    dict(id="HAR_RV_J", name="HAR-RV-J", family="HAR",
         target="INTEGRATED_VARIANCE",
         assumptions=("HAR + jump component regressor", "jump persistence"),
         complexity="O(T) OLS", reference="Andersen, Bollerslev, Diebold (2007), JFE",
         extends=("HAR_RV",)),
    dict(id="HAR_RV_CJ", name="HAR-RV-CJ", family="HAR",
         target="CONTINUOUS_VARIATION",
         assumptions=("separate C and J dynamics", "BV proxy for C"),
         complexity="O(T) OLS", reference="Andersen, Bollerslev, Diebold (2007), JFE",
         extends=("HAR_RV", "HAR_RV_J")),
    dict(id="SHAR", name="Semi-variance HAR", family="HAR",
         target="INTEGRATED_VARIANCE",
         assumptions=("positive/negative semi-variances", "asymmetric response"),
         complexity="O(T) OLS", reference="Patton & Sheppard (2015), JFQA",
         extends=("HAR_RV",)),

    # ── Stochastic Volatility ──
    dict(id="SV", name="Stochastic Volatility", family="SV",
         target="INTEGRATED_VARIANCE",
         assumptions=("latent log-vol AR(1)", "Gaussian innovations"),
         complexity="O(T) MCMC or particle filter",
         reference="Taylor (1986); Harvey, Ruiz, Shephard (1994), RES",
         extends=()),
    dict(id="SVJ", name="SV with Jumps", family="SV",
         target="QUADRATIC_VARIATION",
         assumptions=("SV + Poisson jumps in returns", "jump intensity lambda"),
         complexity="O(T) MCMC", reference="Bates (1996), RFS",
         extends=("SV",)),
    dict(id="MSV", name="Multi-factor SV", family="SV",
         target="INTEGRATED_VARIANCE",
         assumptions=("K latent factors", "factor structure"),
         complexity="O(T*K) MCMC or SMC",
         reference="Chernov, Gallant, Ghysels, Tauchen (2003), JoE",
         extends=("SV",)),

    # ── Realized Measures (not forecasters, but estimation targets) ──
    dict(id="RV", name="Realized Variance", family="REALIZED_MEASURE",
         target="QUADRATIC_VARIATION",
         assumptions=("no microstructure noise at chosen freq", "sum of squared returns"),
         complexity="O(n)", reference="Andersen, Bollerslev, Diebold, Labys (2003), Econometrica",
         extends=()),
    dict(id="BV", name="Bipower Variation", family="REALIZED_MEASURE",
         target="CONTINUOUS_VARIATION",
         assumptions=("consecutive abs returns product", "jump-robust"),
         complexity="O(n)", reference="Barndorff-Nielsen & Shephard (2004), JFE",
         extends=("RV",)),
    dict(id="MedRV", name="Median Realized Variance", family="REALIZED_MEASURE",
         target="CONTINUOUS_VARIATION",
         assumptions=("median of 3 consecutive abs returns", "more robust than BV"),
         complexity="O(n)", reference="Andersen, Dobrev, Schaumburg (2012), JoE",
         extends=("BV",)),
    dict(id="MinRV", name="MinRV", family="REALIZED_MEASURE",
         target="CONTINUOUS_VARIATION",
         assumptions=("min of consecutive abs returns", "jump-robust"),
         complexity="O(n)", reference="Andersen, Dobrev, Schaumburg (2012), JoE",
         extends=("BV",)),
    dict(id="RK", name="Realized Kernel", family="REALIZED_MEASURE",
         target="INTEGRATED_VARIANCE",
         assumptions=("kernel-based noise correction", "Parzen or Bartlett kernel"),
         complexity="O(n*H)", reference="Barndorff-Nielsen, Hansen, Lunde, Shephard (2008), Ecta",
         extends=("RV",)),
    dict(id="TSRV", name="Two-Scale Realized Variance", family="REALIZED_MEASURE",
         target="INTEGRATED_VARIANCE",
         assumptions=("two time scales for noise correction",),
         complexity="O(n)", reference="Zhang, Mykland, Ait-Sahalia (2005), JASA",
         extends=("RV",)),
    dict(id="PA", name="Pre-Averaging Estimator", family="REALIZED_MEASURE",
         target="INTEGRATED_VARIANCE",
         assumptions=("local averaging to reduce noise", "window H ~ n^{1/2}"),
         complexity="O(n)", reference="Jacod, Li, Mykland, Podolskij, Vetter (2009), SPA",
         extends=("RV",)),

    # ── ML approaches ──
    dict(id="LSTM_VOL", name="LSTM Volatility", family="ML",
         target="INTEGRATED_VARIANCE",
         assumptions=("sequence model", "universal approximator", "data-hungry"),
         complexity="O(T*d^2) per epoch", reference="Kim & Won (2018), Complexity",
         extends=()),
    dict(id="TRANSFORMER_VOL", name="Transformer Volatility", family="ML",
         target="INTEGRATED_VARIANCE",
         assumptions=("self-attention mechanism", "positional encoding"),
         complexity="O(T^2*d) per epoch",
         reference="Wu, Xu, Wang, Long (2023), KBS / various",
         extends=()),
    dict(id="RF_VOL", name="Random Forest Volatility", family="ML",
         target="INTEGRATED_VARIANCE",
         assumptions=("ensemble of trees", "feature engineering required"),
         complexity="O(T*n_trees*depth)",
         reference="Luong & Dokuchaev (2018), JRFM",
         extends=()),

    # ── Forecast Combination ──
    dict(id="EQW", name="Equal Weight Combination", family="COMBO",
         target="CONDITIONAL_VARIANCE",
         assumptions=("all experts equally skilled", "diversification"),
         complexity="O(K)", reference="Bates & Granger (1969), OR Quarterly",
         extends=()),
    dict(id="INV_MSE", name="Inverse MSE Weighting", family="COMBO",
         target="CONDITIONAL_VARIANCE",
         assumptions=("static weights from rolling window",),
         complexity="O(K)", reference="Stock & Watson (2004), JBES",
         extends=("EQW",)),
    dict(id="AFTER", name="AFTER", family="COMBO",
         target="CONDITIONAL_VARIANCE",
         assumptions=("exponential re-weighting", "minimax rate"),
         complexity="O(K) online", reference="Yang (2004), Econometric Theory",
         extends=("INV_MSE",)),
    dict(id="EWA", name="Exponentially Weighted Average", family="COMBO",
         target="CONDITIONAL_VARIANCE",
         assumptions=("online learning", "Bayesian interpretation"),
         complexity="O(K) online", reference="Vovk (1990), COLT",
         extends=("EQW",)),
    dict(id="FIXED_SHARE", name="Fixed-Share Expert Aggregation", family="COMBO",
         target="CONDITIONAL_VARIANCE",
         assumptions=("allows expert switching", "mixing rate alpha"),
         complexity="O(K) online", reference="Herbster & Warmuth (1998), ML",
         extends=("EWA",)),
]

# ──────────────────────────────────────────────
# Assumption registry
# ──────────────────────────────────────────────
_ASSUMPTIONS: dict[str, str] = {
    "stationary returns": "Return process is covariance-stationary",
    "symmetric response": "Positive and negative shocks have equal impact on variance",
    "finite 4th moment": "E[r_t^4] < infinity required for QML consistency",
    "log-variance dynamics": "Variance modeled in log space, ensuring positivity",
    "asymmetric news impact": "Negative returns have larger impact on volatility",
    "leverage effect via indicator": "Asymmetry captured by I(r<0) indicator",
    "power transformation delta": "Variance raised to power delta/2",
    "long memory d in (0,0.5)": "Fractional integration parameter for persistence",
    "permanent + transitory components": "Two-component variance decomposition",
    "uses realized measure as regressor": "High-frequency data enters the model",
    "joint model returns + realized measure": "Simultaneous return and RV equations",
    "heterogeneous agents": "Daily, weekly, monthly agents drive volatility",
    "RV as proxy for IV": "Realized variance is consistent estimator of integrated variance",
    "linear": "Linear regression structure",
    "separate C and J dynamics": "Continuous and jump components modeled separately",
    "positive/negative semi-variances": "Upside vs downside realized semi-variances",
    "latent log-vol AR(1)": "Log-volatility follows autoregressive process",
    "Gaussian innovations": "Both return and volatility innovations are Gaussian",
    "SV + Poisson jumps in returns": "Jumps arrive as compound Poisson process",
    "K latent factors": "Multiple volatility factors with different persistence",
    "no microstructure noise at chosen freq": "Sampling frequency avoids noise contamination",
    "consecutive abs returns product": "Product of adjacent absolute returns",
    "median of 3 consecutive abs returns": "Robust to single outlier/jump",
    "kernel-based noise correction": "Realized kernel corrects for microstructure noise",
    "two time scales for noise correction": "Bias correction using fast/slow scales",
    "local averaging to reduce noise": "Pre-averaging over local windows",
    "sequence model": "Temporal dependencies via recurrent architecture",
    "self-attention mechanism": "Global dependencies via attention weights",
    "ensemble of trees": "Bagging of decision trees for robustness",
    "all experts equally skilled": "No prior information on expert quality",
    "exponential re-weighting": "Weights updated via exponential loss",
    "online learning": "Sequential weight updates without full retraining",
    "allows expert switching": "Best expert can change over time",
}


class VolatilityKnowledgeGraph:
    """Directed graph of volatility model knowledge.

    Nodes represent models, realized measures, and assumptions.
    Edges represent relationships: extends, targets, assumes, competes_with.
    """

    def __init__(self) -> None:
        self.G = nx.MultiDiGraph()
        self._build()

    def _build(self) -> None:
        """Populate graph from canonical registry."""
        # Add assumption nodes
        for key, desc in _ASSUMPTIONS.items():
            self.G.add_node(f"A:{key}", kind="assumption", description=desc)

        # Add target nodes
        for t in VolatilityTarget:
            self.G.add_node(f"T:{t.name}", kind="target", name=t.name)

        # Add model nodes and edges
        for m in _MODELS:
            mid = f"M:{m['id']}"
            self.G.add_node(
                mid,
                kind="model",
                name=m["name"],
                family=m["family"],
                target=m["target"],
                complexity=m["complexity"],
                reference=m["reference"],
            )
            # extends edges
            for parent in m.get("extends", ()):
                self.G.add_edge(mid, f"M:{parent}", relation="extends")
            # targets edge
            self.G.add_edge(mid, f"T:{m['target']}", relation="targets")
            # assumes edges
            for a in m.get("assumptions", ()):
                if a in _ASSUMPTIONS:
                    self.G.add_edge(mid, f"A:{a}", relation="assumes")

        # Add competition edges within families
        families: dict[str, list[str]] = {}
        for m in _MODELS:
            families.setdefault(m["family"], []).append(f"M:{m['id']}")
        for fam, members in families.items():
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    self.G.add_edge(a, b, relation="competes_with")
                    self.G.add_edge(b, a, relation="competes_with")

    def get_family(self, family: str) -> list[dict]:
        """Return all models in a family."""
        return [
            {**self.G.nodes[n]}
            for n in self.G.nodes
            if self.G.nodes[n].get("kind") == "model"
            and self.G.nodes[n].get("family") == family
        ]

    def get_models_for_target(self, target: VolatilityTarget) -> list[str]:
        """Return model IDs targeting a specific volatility object."""
        target_node = f"T:{target.name}"
        result = []
        for pred in self.G.predecessors(target_node):
            for _, edge_data in self.G[pred][target_node].items():
                if edge_data.get("relation") == "targets":
                    result.append(pred)
                    break
        return result

    def get_ancestors(self, model_id: str) -> list[str]:
        """Return models that the given model extends (transitively via BFS)."""
        mid = f"M:{model_id}" if not model_id.startswith("M:") else model_id
        visited: set[str] = set()
        queue = [mid]
        while queue:
            node = queue.pop(0)
            for _, target, data in self.G.out_edges(node, data=True):
                if data.get("relation") == "extends" and target not in visited:
                    visited.add(target)
                    queue.append(target)
        return sorted(visited)

    def summary(self) -> dict:
        """Return summary statistics of the knowledge graph."""
        models = [n for n in self.G.nodes if self.G.nodes[n].get("kind") == "model"]
        families = set(self.G.nodes[n].get("family") for n in models)
        return {
            "n_models": len(models),
            "n_families": len(families),
            "families": sorted(families),
            "n_assumptions": sum(
                1 for n in self.G.nodes if self.G.nodes[n].get("kind") == "assumption"
            ),
            "n_edges": self.G.number_of_edges(),
        }

    def to_dict(self) -> dict:
        """Export graph as node-link JSON-serializable dict."""
        return nx.node_link_data(self.G)
