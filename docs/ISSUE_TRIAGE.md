# Issue Triage

Use the issue templates under `.github/ISSUE_TEMPLATE/` so reports include enough context for small, focused fixes.

## Labels

Core type labels:

- `bug`: reproducible broken behavior.
- `feature`: new or expanded behavior.
- `regression`: behavior that used to work and now fails.

Area labels:

- `cleanup`: cleanup planning, masks, candidate scoring, inpainting, cleanup lab, and cleanup QA.
- `ocr`: OCR extraction, OCR confidence, source text edits, and re-OCR workflows.
- `translation`: translation provider behavior, glossary, memory, consistency, and review queues.
- `typeset`: rendered text fit, overflow, readability, and style/layout quality.
- `qa`: review queues, validation reports, ratings, or evaluator workflows.
- `frontend`: React UI and editor interactions.
- `backend`: Python engine, API, persistence, and pipeline orchestration.
- `source-sync`: source providers, imports, and chapter sync.
- `memory`: series memory, glossary, names, blocked mappings, and chapter TM.
- `test`: fixtures, test coverage, and regression checks.
- `documentation`: docs, runbooks, and contributor guidance.
- `packaging`: release, installer, startup, and system diagnostics.

Priority labels:

- `p0`: core trust or data-loss risk.
- `p1`: high-impact workflow or quality issue.
- `p2`: important maintenance or usability issue.
- `p3`: lower urgency or release hygiene.

## Cleanup Failure Reports

Attach cleanup lab artifacts when possible:

- `report.json`
- `original_crop.png`
- `text_mask.png`
- `container_mask.png`
- `cleanup_mask.png`
- `overlay.png`
- `cleaned_crop.png`

Use one or more canonical failure classes when known:

- `bad_text_mask`
- `bad_container_mask`
- `unsafe_mask_rejection`
- `bad_cleanup_routing`
- `bad_flat_fill`
- `bad_inpaint_backend`
- `halo_residual`
- `leftover_glyphs`
- `art_damage`
- `overcleanup`
- `undercleanup`
- `needs_manual_mask`
