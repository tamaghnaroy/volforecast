# AutoVolForecaster Review 6

## Findings

1. The loss-selection contract is still internally inconsistent, so a developer cannot implement Phase 4 deterministically when proxy quality is poor. Phase 3 says that after `evaluation.proxy.proxy_noise_correction(...)`, if `signal_to_noise_ratio < 1` the pipeline should fall back to `MSE` as the ranking criterion ([docs/issues/auto_volforecaster.md](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md):163-164; [volforecast/evaluation/proxy.py](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/volforecast/evaluation/proxy.py):285-287). But Phase 4 still hardcodes QLIKE throughout model selection: it ranks by "primary loss (QLIKE)", runs Diebold-Mariano pruning on per-period QLIKE losses, and builds the MCS from per-period QLIKE losses ([docs/issues/auto_volforecaster.md](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md):178-184). The same design also says the public `loss_fn` parameter controls benchmark ranking and DM/MCS selection ([docs/issues/auto_volforecaster.md](C:/Users/tamaghna%20roy/CascadeProjects/windsurf-project-3/docs/issues/auto_volforecaster.md):224,266). Those instructions conflict on the exact loss to use in the poor-proxy path, and that choice changes which models survive selection, so the implementation path is still underspecified.

No other blocking issues were identified in the reviewed files.

## Change Summary

The design is very close, but it is not fully implementable until the poor-proxy loss rule is made consistent across benchmarking, DM pruning, and MCS. After that clarification, implementation should proceed in this order: Milestone 1 (scaffold plus `DataProfiler`), Milestone 2 (candidate selection plus `BenchmarkRunner` integration), Milestone 3 (model selection with the resolved loss rule), Milestone 4 (combination layer), Milestone 5 (public API and result wrapper), then Milestone 6 (tests, docs, and exports).

## Implementation Notes

**FIXED** — Single finding addressed:

1. **Loss-selection contract consistency** — Introduced `effective_loss` concept in Phase 3: proxy quality check now determines the loss function used for ALL downstream ranking and selection. When `signal_to_noise_ratio < 1`, `effective_loss = "MSE"`; otherwise `effective_loss = loss_fn` (user-supplied, default QLIKE). Phase 4 steps 1-3 (ranking, DM pruning, MCS) now explicitly reference `effective_loss` instead of hardcoding QLIKE. The `effective_loss` is stored in `ModelSelectionResult.primary_loss`. This resolves the conflict between the poor-proxy fallback rule and the selection logic.

**STATUS: Design is now implementable end-to-end.** Recommended implementation order per codex-cli: Milestone 1 → 2 → 3 → 4 → 5 → 6.
