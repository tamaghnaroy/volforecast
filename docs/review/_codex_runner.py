"""
Codex CLI runner using the stdin-pipe (-) invocation pattern from the skill docs.
This avoids gpt-5.4's greeting-turn problem: by piping via stdin, the prompt
arrives as the user's response to the greeting, so the model executes the task.

Usage: python _codex_runner.py <round>
  round = "smoke" | "initial" | "debate1" | "debate2" | "final"
"""
import subprocess
import sys
import os
import time

ROUND = sys.argv[1] if len(sys.argv) > 1 else "initial"
CODEX = r"C:\Users\tamaghna roy\AppData\Roaming\npm\codex.cmd"
CWD   = r"c:\Users\tamaghna roy\CascadeProjects\windsurf-project-3"

PROMPTS = {

# ─── Round 0a: build the scoring framework only ───────────────────────────────
"scoring_framework": """Write the file docs/review/scoring_framework.md.

Content: a rigorous, objective 10-dimension SOTA scoring rubric for automated \
volatility forecasting algorithm design. Each dimension is scored 0-10 with \
explicit rubric criteria at levels 0, 3, 5, 7, and 10. Include what each \
dimension measures, why it matters, rubric criteria, and 2-3 key academic \
references per dimension.

Dimensions to cover:
DIM-A: Model family breadth vs SOTA literature 2015-2024 (EGARCH, HAR variants, SV, Neural-GARCH, Rough Vol, etc.)
DIM-B: Statistical model selection validity (MCS, DM test, multiple-testing correction)
DIM-C: Forecast combination theory and online regret bounds (EWA, Fixed-Share, AFTER, RL)
DIM-D: Proxy robustness and Patton-2011 consistency (QLIKE, MSE, SNR gating)
DIM-E: Online and streaming adaptivity with regret guarantees
DIM-F: Regime detection and structural break handling (Markov-switching, CPD, adaptive windows)
DIM-G: Multi-horizon forecast architecture (direct vs iterated, HAR multi-step)
DIM-H: Computational scalability and complexity (O-notation, parallelism, GPU)
DIM-I: Production readiness and software engineering (API design, fail-safes, tests)
DIM-J: Benchmarking rigor and reproducibility (OOS discipline, synthetic DGPs, ablation)
""",

# ─── Round 0b: score the plan using the framework ────────────────────────────
"initial_review": """Read docs/issues/auto_volforecaster.md and docs/review/scoring_framework.md.

Write the file docs/review/initial_review.md containing a complete expert review \
of the AutoVolForecaster design plan using the 10-dimension scoring framework.

Structure:
1. Per-dimension score (0-10) with 3-5 sentences of evidence-based justification \
   citing specific sections of the plan. Be critical and precise.
2. Total score out of 100 (equal weights, 10 per dimension).
3. SOTA gap analysis: the 5 most critical weaknesses vs best published systems \
   2020-2024. Cite author, year, and the specific capability the plan lacks.
4. Ten actionable improvement recommendations ranked by expected score uplift. \
   Each must include: title, problem, proposed solution, literature refs, \
   and which dimension it addresses.
""",

# ─── Round 1: adversarial challenge + cooperative improvement request ─────────
"debate1": r"""
Read docs/review/initial_review.md and docs/review/cascade_judgements.md.

You are acting as an expert adversarial reviewer in a structured debate.
Your role is to:

1. CHALLENGE the judgements in cascade_judgements.md that are marked REJECT or
   PARTIAL-ACCEPT. For each such item, argue your case citing academic
   literature, theoretical principles, or empirical evidence. Be specific.

2. ACCEPT the judgements marked ACCEPT and acknowledge where you agree.

3. For each of the 10 improvement recommendations in initial_review.md, provide
   CONCRETE implementation guidance:
   - Pseudocode or algorithmic specification
   - Specific parameters or thresholds with empirical justification
   - Python class/function signatures that fit the existing volforecast BaseForecaster API
   - Literature reference (author, year, equation numbers where relevant)

4. Propose 3 additional improvements that were missed in the initial review,
   each addressing a specific SOTA gap not covered by the original 10.

Write your response to docs/review/debate_round1_codex.md
""",

# ─── Round 2: synthesis + revised scoring ────────────────────────────────────
"debate2": r"""
Read docs/review/debate_round1_codex.md and docs/review/cascade_counter.md.

You are now in the synthesis phase of the debate. Your role is:

1. For each point of disagreement in cascade_counter.md, either:
   a. CONCEDE: admit the counter-argument is correct and update your position
   b. MAINTAIN: defend your position with stronger evidence
   c. SYNTHESISE: propose a compromise that captures both perspectives

2. Produce a revised version of the plan with all agreed improvements
   incorporated. Write this to docs/review/revised_plan_draft.md as a
   complete, restructured version of auto_volforecaster.md that would score
   8+ on every dimension of the scoring framework.

3. Re-score the revised plan on all 10 dimensions and compute the new total.
   Justify every score change. Write scores to docs/review/revised_scores.md.

4. Identify any remaining gaps between the revised plan and true SOTA, and
   list them in docs/review/remaining_gaps.md.
""",

# ─── Round 3: final SOTA validation ──────────────────────────────────────────
"final": r"""
Read docs/review/revised_plan_draft.md and docs/review/remaining_gaps.md.

Perform a final SOTA validation:

1. Cross-check the revised plan against the following key SOTA systems and
   papers from 2022-2024:
   - GARCH-Informed Neural Networks (Ramos-Perez et al. 2022)
   - Conformal Prediction intervals for volatility (Angelopoulos et al. 2023)
   - Transformer-based realised variance forecasting (Lim et al. 2021)
   - Online convex optimisation for forecast combination (Cesa-Bianchi 2006, Orabona 2023 survey)
   - Rough volatility models (Gatheral, Jaisson, Rosenbaum 2018)
   - Neural GARCH (Ramos-Perez et al. 2023)
   - Score-driven models unification (Creal, Koopman, Lucas 2013; Harvey 2013)
   - Realised GARCH-MIDAS (Pan, Wang, Wu 2023)

2. For each gap still present, specify exactly what code addition would close it,
   with API signatures consistent with BaseForecaster.

3. Produce a final assessment: does the revised plan achieve SOTA quality?
   If yes, justify. If no, what single most impactful addition would tip it over.

4. Write the final SOTA-validated plan to docs/review/final_plan_sota.md

5. Write a one-page executive summary to docs/review/executive_summary.md
""",

# ─── Iterative review round (generic) ────────────────────────────────────────
"review4": """Read docs/issues/auto_volforecaster.md, volforecast/benchmark/runner.py, \
volforecast/combination/online.py, volforecast/evaluation/tests.py, \
volforecast/evaluation/proxy.py, volforecast/models/__init__.py, \
volforecast/core/base.py, and the previous reviews in docs/review/auto_volforecaster_review*.md.

You are an expert reviewer. Write docs/review/auto_volforecaster_review4.md containing:

## Findings
List only findings where the design doc or code has a concrete, \
verifiable inconsistency, missing implementation path, or behavioral bug. \
Each finding must cite specific file:line evidence. Do NOT repeat findings \
that have already been fixed (check the Implementation Notes in prior reviews). \
Do NOT suggest aspirational improvements that go beyond the goal of building \
a working auto vol forecaster for a given time series.

## Open Questions
List 0-3 genuine ambiguities that need a design decision.

## Change Summary
One paragraph assessment: is the design now implementable end-to-end? \
If yes, say so and recommend proceeding to implementation. \
If no, state the specific remaining blockers.
""",

"review5": """Read docs/issues/auto_volforecaster.md, volforecast/benchmark/runner.py, \
volforecast/evaluation/tests.py, volforecast/combination/online.py, \
volforecast/evaluation/proxy.py, volforecast/models/__init__.py, \
volforecast/core/base.py, and all docs/review/auto_volforecaster_review*.md files.

You are an expert reviewer on the final convergence round. \
Write docs/review/auto_volforecaster_review5.md containing:

## Findings
List ONLY findings where the design doc or code still has a concrete, \
verifiable inconsistency, missing implementation path, or behavioral bug \
that would prevent a developer from implementing the AutoVolForecaster \
end-to-end. Each finding must cite specific file:line evidence. \
Do NOT repeat findings already fixed in prior reviews (check Implementation Notes). \
Do NOT suggest aspirational SOTA improvements beyond the core goal.

## Open Questions
List 0-2 genuine ambiguities that need a design decision before coding starts.

## Change Summary
One paragraph: is the design now implementable end-to-end for the goal of \
building an auto vol forecaster for a given time series? If yes, explicitly \
recommend proceeding to implementation. If no, state the specific remaining blockers.
""",

"review6": """Read docs/issues/auto_volforecaster.md, volforecast/benchmark/runner.py, \
volforecast/evaluation/tests.py, volforecast/combination/online.py, \
volforecast/evaluation/proxy.py, volforecast/models/__init__.py, \
volforecast/core/base.py, and all docs/review/auto_volforecaster_review*.md files.

This is the FINAL convergence check. Write docs/review/auto_volforecaster_review6.md:

## Findings
List ONLY findings where the design doc or code STILL has a concrete bug or \
missing implementation path that would block a developer from coding the \
AutoVolForecaster. Check ALL Implementation Notes in reviews 1-5 to avoid \
repeating fixed issues. If there are no remaining blockers, state that explicitly.

## Change Summary
One paragraph: is the design now implementable? If yes, recommend proceeding \
to implementation and list the implementation order (which phase to code first).
""",

"sota_arch": """Read the full codebase: docs/issues/auto_volforecaster.md, \
volforecast/core/base.py, volforecast/core/targets.py, \
volforecast/models/__init__.py, volforecast/models/garch.py, \
volforecast/models/har.py, volforecast/models/rough_vol.py, \
volforecast/models/sv.py, volforecast/models/gas.py, \
volforecast/models/markov_switching.py, volforecast/models/midas.py, \
volforecast/models/ml_wrappers.py, volforecast/models/heavy.py, \
volforecast/models/realized_garch.py, volforecast/models/figarch.py, \
volforecast/combination/online.py, volforecast/combination/rl_combiner.py, \
volforecast/evaluation/losses.py, volforecast/evaluation/tests.py, \
volforecast/evaluation/proxy.py, volforecast/benchmark/runner.py, \
volforecast/benchmark/synthetic.py, volforecast/realized/measures.py, \
volforecast/realized/jumps.py, volforecast/knowledge/graph.py, \
and all docs/review/auto_volforecaster_review*.md, \
docs/review/scoring_framework.md.

You are a world-class expert in volatility forecasting, online learning, \
and production ML systems. Your goal: make the AutoVolForecaster \
architecture genuinely SOTA — matching or exceeding the best published \
automated volatility forecasting systems (2020-2025).

Write docs/review/sota_architectural_feedback.md containing:

# SOTA Architectural Feedback for AutoVolForecaster

## 1. Model Coverage Gaps
Which model families or recent innovations (2020-2025) are missing \
from the candidate pool? For each gap, cite the paper, explain what \
capability it adds, and whether the existing BaseForecaster API can \
accommodate it. Consider:
- Neural GARCH / GARCH-informed neural networks
- Conformal prediction intervals for volatility
- Score-driven (GAS/DCS) extensions beyond the current GASVolForecaster
- Rough volatility calibration improvements
- Realized GARCH-MIDAS hybrids
- Transformer/attention-based realized vol
- Any other frontier family the repo should include

## 2. Forecast Combination Frontier
How does the current combination layer compare to SOTA? Evaluate:
- Regret bounds: are EWA/FixedShare/AFTER sufficient or should we add \
  online mirror descent, AdaHedge, or BOA?
- Conformal combination: should combined forecasts carry prediction intervals?
- Meta-learning: should the combiner selection itself be learned from data?
- Stacking vs online: when should static stacking beat online updating?

## 3. Evaluation & Selection Rigor
What is missing to match SOTA evaluation methodology?
- Is the MCS + DM + Patton-robust stack sufficient?
- Should we add the Superior Predictive Ability (SPA) test?
- Giacomini-White conditional predictive ability test?
- Should the proxy quality gate use multiple realized measures, not just RV?
- Cross-validation strategies for time series (blocked, purged)?

## 4. Online Adaptivity & Regime Handling
How should the system handle nonstationarity beyond FixedShare?
- Adaptive windowing (ADWIN, Page-Hinkley)?
- Change-point detection triggering model refit?
- Regime-aware weight scheduling?
- Forgetting/discounting in the combination layer?

## 5. Production Architecture Improvements
What architectural changes would make this production-grade SOTA?
- Uncertainty quantification on combined forecasts
- Forecast calibration layer
- Monitoring / drift detection in deployment
- Computational budget management
- Warm-start and incremental refit strategies

## 6. Concrete Recommendations
Rank the top 10 most impactful changes by expected improvement to \
forecast accuracy, with:
- Title and 1-sentence description
- Expected impact (high/medium/low)
- Implementation complexity (easy/moderate/hard)
- Key reference (author, year)
- Which existing module to modify or create

Focus on changes that are implementable within the existing codebase \
architecture (BaseForecaster, BenchmarkRunner, online combiners). \
Do not suggest replacing the entire framework.
""",
}

SMOKE_PROMPT = (
    "Create the file docs/review/smoke_test.txt with exactly this content: "
    "codex-smoke-ok"
)
PROMPTS["smoke"] = SMOKE_PROMPT

# Map each round to the file it is expected to produce (relative to CWD)
EXPECTED_FILES = {
    "smoke":             "docs/review/smoke_test.txt",
    "scoring_framework": "docs/review/scoring_framework.md",
    "initial_review":    "docs/review/initial_review.md",
    "review4":           "docs/review/auto_volforecaster_review4.md",
    "review5":           "docs/review/auto_volforecaster_review5.md",
    "review6":           "docs/review/auto_volforecaster_review6.md",
    "sota_arch":         "docs/review/sota_architectural_feedback.md",
    "debate1":           "docs/review/debate_round1_codex.md",
    "debate2":           "docs/review/revised_plan_draft.md",
    "final":             "docs/review/final_plan_sota.md",
}

# Phrases that indicate the model greeted instead of working
_GREETINGS = [
    "what do you want me to work on",
    "what do you need changed",
    "how can i help",
    "what would you like me to",
    "what would you like to",
    "let me know what you",
]

def _is_greeting(text: str) -> bool:
    t = text.lower()
    return any(g in t for g in _GREETINGS)


def run_codex(prompt: str, label: str, timeout: int = 600) -> bool:
    """
    Invoke codex exec. Tries positional-arg first, then stdin-pipe.
    Success = expected output file exists after the call.
    """
    log_path = os.path.join(CWD, "docs", "review", f"codex_log_{label}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    expected = os.path.join(CWD, EXPECTED_FILES.get(label, "NONE"))

    base_args = [
        CODEX, "exec",
        "-m", "gpt-5.4",
        "-c", "model_reasoning_effort=high",   # xhigh causes over-cautious greeter; high executes
        "-s", "workspace-write",
        "--full-auto",
        "--skip-git-repo-check",
    ]

    print(f"\n{'='*60}")
    print(f"Codex round: {label}  →  {os.path.basename(expected)}")
    print(f"{'='*60}")

    def _attempt(tag: str, args: list, stdin_text: str) -> bool:
        print(f"[{tag}] running ...")
        try:
            r = subprocess.run(
                args, cwd=CWD,
                input=stdin_text, capture_output=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
            out = (r.stdout or "") + (r.stderr or "")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== {tag} ===\nexit={r.returncode}\n{out}\n")
            # Show last 600 chars of output
            snippet = out[-600:] if len(out) > 600 else out
            print(snippet)
            created = os.path.exists(expected)
            greeted = _is_greeting(out)
            print(f"  exit={r.returncode}  file_created={created}  greeted={greeted}")
            return created
        except subprocess.TimeoutExpired:
            print(f"[{tag}] TIMEOUT after {timeout}s")
            return False

    # Attempt 1: positional prompt arg (works for short/medium prompts)
    if _attempt("A1-positional", base_args + [prompt], ""):
        print(f"[round={label}] SUCCESS via A1")
        return True

    # Attempt 2: stdin pipe (catches greeting-then-wait flow)
    if _attempt("A2-stdin", base_args, prompt):
        print(f"[round={label}] SUCCESS via A2")
        return True

    print(f"[round={label}] FAILED. Log: {log_path}")
    return False


if __name__ == "__main__":
    all_rounds = list(PROMPTS.keys())
    if ROUND not in all_rounds:
        print(f"Unknown round '{ROUND}'. Choose from: {all_rounds}")
        sys.exit(1)

    t0 = time.time()
    ok = run_codex(PROMPTS[ROUND], ROUND)
    print(f"\nDone in {time.time()-t0:.1f}s  success={ok}")
