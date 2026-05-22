# Cleanup Lab

`tools/cleanup_lab` is an isolated side project for experimenting with cleanup geometry, masks, and inpainting behavior without running the backend server or changing the main app pipeline.

It imports the existing cleanup planner/executor from `backend/core/cleanup_plan.py`, builds a lightweight region object from either a standalone JSON fixture or `.ml_state.json`, runs one selected region, and writes debug artifacts to a lab output folder.

## What it is for

- Test cleanup/mask planning on a raw page image.
- Inspect text mask, container/bubble mask, final cleanup mask, and cleaned preview.
- Iterate on problematic regions before deciding whether production cleanup code should change.
- Keep raw, cleaned, and typeset concepts separate. The lab only creates a cleaned preview; it does not typeset.

## What it intentionally does not do

- It does not refactor or integrate with the main app cleanup flow.
- It does not start the backend server.
- It does not run OCR, translation, Naver/source sync, YOLO detection, memory, or typesetting.
- It does not mutate chapter `.ml_state.json`.
- It does not download or integrate LaMa.
- It does not write under a chapter folder unless you explicitly pass an `--out` path there.

## Standalone fixture mode

```powershell
python tools/cleanup_lab/cleanup_lab.py ^
  --image path\to\page.png ^
  --regions tools\cleanup_lab\sample_region_fixture.json ^
  --region-id R-01 ^
  --out tools\cleanup_lab\outputs\page6_R01
```

The fixture can also include `page_image`, allowing you to omit `--image`.

## Chapter `.ml_state.json` mode

```powershell
python tools/cleanup_lab/cleanup_lab.py ^
  --chapter path\to\chapter_folder ^
  --page 6 ^
  --region-id R-01 ^
  --out tools\cleanup_lab\outputs\page6_R01
```

`--page` is one-based by default. Add `--zero-based-page` if you want `--page 0` to mean the first page.

## Output files

Each run writes the following files to the output directory:

- `original_crop.png`: crop around the selected region and derived masks.
- `text_mask.png`: proposed glyph/text mask crop.
- `container_mask.png`: proposed bubble/container mask crop when available.
- `cleanup_mask.png`: final destructive cleanup mask crop.
- `overlay.png`: color overlay for quick inspection.
- `cleaned_crop.png`: cleaned crop preview.
- `cleaned_page_preview.png`: full-page cleaned preview.
- `report.json`: metrics, strategy, confidence, skip reason, artifact paths, and raw planner debug metrics.

Reports also include `failure_classes` and `failure_class`, using the canonical cleanup failure taxonomy:
`bad_text_mask`, `bad_container_mask`, `unsafe_mask_rejection`, `bad_cleanup_routing`,
`bad_flat_fill`, `bad_inpaint_backend`, `halo_residual`, `leftover_glyphs`, `art_damage`,
`overcleanup`, `undercleanup`, and `needs_manual_mask`.

## Batch QA manifest mode

Run a manifest of cleanup cases without starting the app:

```powershell
python tools/cleanup_lab/cleanup_lab.py ^
  --manifest tools\cleanup_lab\fixtures\cleanup_qa_manifest.json ^
  --out tools\cleanup_lab\outputs\cleanup_qa
```

Each case gets its own output folder with `original_crop.png`, `text_mask.png`, `container_mask.png`,
`cleanup_mask.png`, `overlay.png`, `cleaned_crop.png`, `cleaned_page_preview.png`, and `report.json`.
The batch folder also gets `review_manifest.json`, which records expected outcome, actual outcome,
failure classes, backend used, mask quality, inpaint quality, and final pass/fail.

The runner prints a concise table with case id, strategy, method, failure class, mask ratios, result,
and output folder. When using this workflow, keep iterating on one failing case at a time: inspect
the images and `report.json`, classify the failure, patch only the responsible cleanup path, rerun
the same case, then add or update a focused regression test before moving to the next case. If a
case cannot be made visually correct, leave an exact blocker in the review manifest notes or issue
comment with the artifact path and evidence.

The default manifest covers plain bubble, colored bubble, textured bubble, dark caption, and
text-over-art/SFX protected behavior. Current cleanup regressions are covered by focused tests in
`backend/core/test_cleanup_pipeline.py`, including colored local fill, dark caption fill, textured
halftone routing, and SFX protection cases.

## Fixture format

Use a small JSON file with only the fields needed for the experiment:

```json
{
  "page_image": "path/to/page.png",
  "regions": [
    {
      "id": "R-01",
      "bbox": [131, 565, 438, 181],
      "text": "optional OCR text",
      "role": "thought",
      "kind": "TEXTURED_BUBBLE",
      "yolo_kind": "narration",
      "detector": "yolo",
      "detector_confidence": 0.807,
      "bubble_bbox": [0, 475, 690, 362],
      "safe_rect": [0, 475, 690, 362]
    }
  ]
}
```

If you have reliable OCR/text polygons, add `boxes`, `ocr_boxes`, or `text_boxes`. If those are absent, cleanup uses CV-driven no-bbox text mask candidates inside the supplied region bbox.

## Creating a fixture from a problematic region

1. Copy the raw page image path.
2. Copy the region bbox from the UI, `.ml_state.json`, or an existing debug report.
3. Add optional metadata that affects cleanup classification: `role`, `kind`, `yolo_kind`, `detector`, `bubble_bbox`, and `safe_rect`.
4. Keep geometry minimal. Do not add Qwen/OCR geometry unless you specifically want to test behavior with those boxes.
5. Run the CLI and inspect `overlay.png`, `cleanup_mask.png`, and `report.json`.

## Future LaMa support

The lab is intentionally shaped around a small CLI adapter and artifact writer. A future LaMa backend can be added behind the lab runner or existing cleanup backend abstraction for experiments first, without changing the main app cleanup pipeline or requiring LaMa in normal app runs.
