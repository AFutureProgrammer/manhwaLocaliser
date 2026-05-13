# Cleanup Lab Notes

Use this file to record cleanup experiments, problematic regions, and observations before changing production cleanup behavior.

## Experiment template

- **Case**: page/region or fixture path
- **Input**: raw image and region metadata used
- **Observed output**: mask area, strategy, skip reason, artifacts reviewed
- **Expected output**: what should have happened
- **Hypothesis**: likely planner or mask issue
- **Next step**: lab-only experiment or proposed production change

## Current coupling notes

- `build_cleanup_plan()` accepts any block-like object, but expects several `OCRBlock`-style attributes such as `bbox()`, `boxes`, `bubble_role`, `detector_source`, `region_kind`, `bubble_bbox`, `bubble_mask`, and optional overrides.
- `.ml_state.json` loading through `ChapterManager.load_state()` can run migrations and save state, so the lab reads `.ml_state.json` directly instead.
- Container masks are local to `container_bbox`; text and cleanup masks are full-page masks.
- The existing executor writes to a passed-in copy, so the lab preserves the raw input image and writes only debug outputs.
