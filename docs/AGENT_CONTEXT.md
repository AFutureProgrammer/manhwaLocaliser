# Agent Context for Manhwa Localiser

This document is a high-signal repo map for future agents. It is intentionally detailed but not always-loaded. Use it to reduce token use: read the smallest relevant section, then inspect the exact files named there.

Last verified locally: 2026-05-18.

## Why This Exists

Codex discovers root `AGENTS.md` automatically and concatenates instruction files until a configured size cap is reached. The OpenAI Codex docs describe `AGENTS.md` as the project-specific instruction chain and note a default combined size limit of 32 KiB. The public AGENTS.md format describes it as a README for agents. Context-engineering research also frames these files as a way to keep agents aligned with project structure, code style, build/test workflows, and policies.

This repo uses that pattern:

- `AGENTS.md`: short, always-loaded operating rules.
- `docs/AGENT_CONTEXT.md`: detailed architecture, file map, command map, and task routing.

Sources used for this structure:

- OpenAI Codex AGENTS.md guide: https://developers.openai.com/codex/guides/agents-md
- AGENTS.md project/format overview: https://github.com/agentsmd/agents.md
- Context Engineering for AI Agents in Open-Source Software: https://arxiv.org/abs/2510.21413

## Fast Orientation

Manhwa Localiser is a Windows-oriented desktop app for localising Korean manhwa chapters. It uses a Python backend with pywebview, a React/Vite frontend, OCR/region detection, translation, cleanup/inpainting, typesetting, source sync, and per-series translation memory.

Primary runtime:

- `launcher.py`: starts the desktop app. Production loads `frontend/dist/index.html`; dev mode points pywebview at Vite.
- `backend/api.py`: pywebview bridge. Public methods are callable from JavaScript as `window.pywebview.api.*`.
- `backend/engine.py`: main orchestrator for chapter state, detection, OCR, translation, cleanup, typesetting, source sync, memory, exports, and bootstrap payloads.
- `frontend/src/api.ts`: typed JavaScript bridge wrapper and dev stub.
- `frontend/src/App.tsx`: main editor UI.

Primary persisted state:

- Chapter state: `.ml_state.json` inside imported chapter folders.
- Series catalogue: `series_db.json` at repo root.
- Config: `model_config.json` at repo root.
- Translation memory: `series_memory/` by default.
- Cleanup/debug artifacts: `debug_cleanup*/` and `tools/cleanup_lab/outputs/`/`runs/`.

## Read This First By Task

Use this table before opening source files.

| Task | Start Here | Then Inspect |
| --- | --- | --- |
| pywebview API behavior | `backend/api.py` | matching `LocalizerEngine` method in `backend/engine.py`, matching frontend call in `frontend/src/api.ts` |
| Bootstrap/state shape | `backend/engine.py` `get_bootstrap()` and `_bootstrap_region_entry()` | `frontend/src/types.ts`, consumers in `frontend/src/App.tsx` |
| Region model or persisted `.ml_state.json` | `backend/core/regions.py` | `backend/core/project.py`, migration helpers |
| Detection/OCR | `backend/engine.py` detection/OCR methods | `backend/core/ocr.py`, `backend/core/config.py` |
| Translation | `backend/engine.py` translation methods | `backend/core/translation.py`, `backend/core/deepseek_translate.py`, `backend/core/text_utils.py`, `memory/` |
| Cleanup strategy/masks | `backend/core/cleanup_plan.py` | `backend/core/cleanup.py`, `backend/engine.py` cleanup entrypoints, `tools/cleanup_lab/README.md` |
| Typesetting | `backend/engine.py` typeset/render methods | `backend/core/typesetting.py`, `backend/core/regions.py` `TextStyle` |
| RAW-style matching | `backend/engine.py` RAW-style helpers | `backend/core/test_raw_style_matching.py`, `backend/core/raw_style_fixtures/` |
| Source sync/Naver | `backend/core/sources/` | `backend/core/project.py` `SeriesDB`, frontend browse/detail components |
| Memory/glossary/name consistency | `memory/__init__.py` | `memory/models.py`, `memory/retrieval.py`, engine memory integration |
| Frontend editor behavior | `frontend/src/App.tsx` | `frontend/src/types.ts`, `frontend/src/api.ts` |
| Series browser/detail UI | `frontend/src/components/BrowseModal.tsx` | `frontend/src/components/SeriesDetailPanel.tsx`, `frontend/src/api_sync.ts`, `frontend/src/types_sync.ts` |
| Cleanup experiments | `tools/cleanup_lab/README.md` | `tools/cleanup_lab/cleanup_lab.py`, targeted fixture/report |

## Command Map

Use PowerShell from repo root unless noted.

```powershell
# Backend syntax/import validation required after backend changes
python -m compileall backend memory

# Focused backend tests
python -m unittest backend.core.test_cleanup_pipeline
python -m unittest backend.core.test_raw_style_matching

# Frontend build required after frontend changes
cd frontend
npm run build

# Frontend dev server for launcher --dev
cd frontend
npm run dev

# Desktop app, production build
python launcher.py

# Desktop app, Vite dev mode
python launcher.py --dev

# Cleanup lab sample pattern
python tools/cleanup_lab/cleanup_lab.py --image path\to\page.png --regions tools\cleanup_lab\sample_region_fixture.json --region-id R-01 --out tools\cleanup_lab\outputs\case_name
```

## Git And Dirty Worktree Caution

As of this document creation, the repo had existing modified/untracked files not created by this documentation task, including:

- `.gitignore`
- `backend/core/test_cleanup_pipeline.py`
- `backend/core/test_raw_style_matching.py`
- `backend/engine.py`
- `frontend/src/App.tsx`
- `frontend/src/types.ts`
- `run_deepseek_local.bat`
- `.venv/`
- `backend/core/raw_style_fixtures/`
- `handoff.json`

Do not revert or overwrite these unless the user explicitly asks. `handoff.json` describes in-progress RAW-style QA work and should be treated as user/project state.

## High-Level Architecture

```text
React UI
  frontend/src/App.tsx
  frontend/src/components/*
  frontend/src/api.ts
        |
        | window.pywebview.api.*
        v
Pywebview bridge
  backend/api.py
        |
        v
LocalizerEngine
  backend/engine.py
        |
        +-- state/persistence: backend/core/project.py, backend/core/regions.py
        +-- config: backend/core/config.py, model_config.json
        +-- source sync: backend/core/sources/*
        +-- detection/OCR: backend/core/ocr.py
        +-- translation: backend/core/translation.py, backend/core/deepseek_translate.py, memory/*
        +-- cleanup: backend/core/cleanup_plan.py, backend/core/cleanup.py, backend/core/sam2_mask.py
        +-- typesetting: backend/core/typesetting.py and engine render methods
```

## Backend File Map

### `launcher.py`

Entrypoint for the desktop app.

- `python launcher.py`: loads built `frontend/dist/index.html`.
- `python launcher.py --dev`: points pywebview at `http://localhost:5173`.
- Creates `LocalizerEngine`, wraps it in `PywebviewAPI`, creates a pywebview window, and starts the event loop.
- If production build is missing, it tells the user to run `cd frontend && npm install && npm run build`.

### `backend/api.py`

Thin pywebview adapter. This is the only backend file that should import `webview`.

Rules from the file itself:

- Public methods become `window.pywebview.api.methodName(...)`.
- Arguments and returns must be JSON-serialisable.
- Methods return `{"ok": true, ...}` or `{"ok": false, "error": "..."}`.
- Progress and busy events are pushed into JS via `window.dispatchEvent`.

Main API areas:

- Lifecycle/bootstrap: `get_bootstrap`.
- Chapter/navigation: `open_chapter_folder`, `import_chapter`, `go_to_page`.
- Pipeline: `run_step`, `run_all`, `run_current_page`, `continue_run_all`, `detect_all`.
- Export/images: `export_project`, `reveal_export_folder`, `get_page_image`.
- Region edits: `update_region`, `update_region_bbox`, `add_region`, `delete_region`, `undo`.
- Region single-step work: `ocr_region`, `translate_region`, cleanup preview/apply/rerun/delete.
- Cleanup tools: SAM2 mask proposal, cleanup debug, mask QA label/export/train, candidate compare/apply.
- YOLO tools: export finetune dataset, set train class, train detector, training status.
- Sources/series: source list, browse, select, sync metadata/chapters, import chapter, delete series, thumbnails.
- Memory: list/add/update/delete series names and glossary.
- Config: get/update model config.

When adding an API method, update all three surfaces:

- `backend/api.py`
- `backend/engine.py`
- `frontend/src/api.ts`

If it affects frontend state, also update `frontend/src/types.ts`.

### `backend/engine.py`

Main application orchestrator. It is very large, so search before reading.

Important groups:

- Initialization/model config: constructor, `_init_models`, `ModelConfig.load`.
- Progress: `_notify`, `_set_progress`, `_begin_active_operation`, `_end_active_operation`.
- RAW-style matching: `_analyze_raw_style_for_block`, `_apply_raw_style_match`, `_raw_match_quality_summary`, `_raw_style_bootstrap_payload`.
- Page/chapter state: `_load_page_into_working_state`, `_flush_working_state_to_page`, `_with_page_context`.
- Cross-page region support: methods around `_cross_page_*`, `_build_stitched_*`, `_update_cross_page_metadata`.
- Undo and edits: `_push_undo_snapshot`, `undo_last_edit`, `update_region_field`, `update_region_bbox`, `add_region`, `delete_region`.
- Cleanup patches: `_run_selected_region_cleanup`, preview/compare/apply/rerun/delete cleanup methods, cleanup patch encode/decode helpers.
- Memory: `_init_memory`, `_retrieve_batch_context`, `_build_prompt_prefix`, `_post_translate`, list/add/update/delete memory methods.
- Series/source sync: list/browse/update/sync/import/delete source series methods.
- Detection/OCR: `detect_current_page`, `_detect_regions`, `ocr_current_page`, `_ocr_one_region`, PaddleOCR/Qwen/EasyOCR helpers.
- Translation: `translate_current_page`, `_translate_texts`, `_translate_single_region_text`, DeepSeek/Ollama helpers.
- Cleanup/typeset pipeline: `cleanup_current_page`, `typeset_current_page`, `_typeset_image`.
- Bootstrap/export/images: `get_bootstrap`, `_bootstrap_region_entry`, `get_page_image_b64`, `export_chapter`.

Risk notes:

- Many UI features derive from bootstrap shape. Any backend field rename can break TypeScript or runtime UI.
- `update_region_field` is a choke point for many editor controls; preserve existing field behavior.
- SFX/text-over-art cleanup defaults are conservative. Do not make destructive cleanup more aggressive unless the task asks.
- Cross-page regions have owner/secondary metadata. Geometry edits must keep page-local and composite bboxes consistent.
- Raw/cleaned/typeset artifacts are separate concepts. Do not collapse them into one output path.

### `backend/core/regions.py`

Data model for OCR/region state.

Key classes:

- `RegionKind`: region classification enum.
- `BackgroundKind`: background classification enum.
- `TextStyle`: render style with colors, outline, shadow, glow, reflection, gradient, rotation, role/font info.
- `RegionReview`: review state.
- `RegionOverride`: manual overrides for cleanup, placement, erase-only, skip-typeset, and style.
- `OCRBlock`: core region object with text, translation, geometry, confidence, detector metadata, cleanup fields, style, overrides, memory hits, review state, and serialization helpers.

Serialization helpers:

- `_block_to_dict`
- `_block_from_dict`
- `_apply_block_dict`

When adding fields to `OCRBlock`, consider:

- Default value in dataclass/class init.
- `_block_to_dict` and `_block_from_dict`.
- `ChapterManager` migrations if old `.ml_state.json` files need compatibility.
- Bootstrap mapping in `backend/engine.py`.
- Frontend type in `frontend/src/types.ts`.

### `backend/core/project.py`

Chapter and series persistence.

Key classes:

- `ChapterPage`: raw image path and per-page artifacts/state.
- `ChapterManager`: loads chapter folders, persists `.ml_state.json`, debounces saves, handles artifacts, migrations, navigation, and run-all checkpoints.
- `SeriesDB`: manages `series_db.json`, series metadata, chapter records, source IDs, stats, and delete/update workflows.

Risk notes:

- `ChapterManager.load_state()` can run migrations and save state. Tools that only inspect `.ml_state.json` should read JSON directly to avoid unintended migration writes.
- Artifact relpaths are stored in state; preserve relative/absolute expectations.
- Series source identity and memory keys are part of continuity across imports.

### `backend/core/config.py`

`ModelConfig` dataclass persisted to `model_config.json`.

Important config groups:

- Ollama model names and keep-alive.
- Detector backend and YOLO settings.
- OCR backend: cascade, qwen_vl, paddleocr, easyocr.
- Cleanup backend and many safety thresholds.
- SAM2 mask-assist settings.
- SFX pipeline toggle.
- Translation provider: Ollama or DeepSeek.
- Optional Klein cleanup backend.

Never put actual API keys in `model_config.json`. DeepSeek reads the key from the env var named by `deepseek_api_key_env`.

### `backend/core/ocr.py`

Detection and OCR helpers.

Key items:

- `group_ocr_blocks`: groups OCR text blocks.
- `build_mask`: builds a mask from OCR geometry.
- `OCRProcessor`: OCR processing wrapper.
- `RegionDetector`: abstract detector shape.
- `OCRRegionDetector`: OCR-derived region detector.
- `YoloV8RegionDetector`: ONNX YOLO detector.

### `backend/core/translation.py`

Local translation clients.

Key items:

- `OllamaClient`: local Ollama HTTP client.
- `NLLBTranslator`: translation helper.

DeepSeek support lives in `backend/core/deepseek_translate.py`:

- Reads API key from environment.
- Sends batch translation requests to DeepSeek-compatible HTTP endpoint.
- Raises typed config/API errors.

Text cleanup and heuristics live in `backend/core/text_utils.py`.

### `backend/core/cleanup_plan.py`

Main cleanup planner and executor. This is the largest shared cleanup file.

Key concepts:

- `CleanupPlan`: per-region plan/result/debug object.
- `CleanupPolicy`: config-derived safety and mode thresholds.
- `ONNXEngine`/`LamaTilingEngine`: optional LaMa inpainting support.
- `SystemMonitor`, `PhotoshopBridge`, `BatchEngine`: support utilities.
- `build_cleanup_plan`: classifies region, builds candidates, selects strategy, stores masks/metrics.
- `execute_cleanup_plan`: applies selected cleanup plan to an image.
- `summarize_cleanup_plan`: serializes plan summary for UI/debug.

Important strategy/candidate areas:

- Background classification and mask quality metrics.
- OCR contrast, multichannel threshold, edge component, no-bbox CV, dark-caption, and container-first glyph mask candidates.
- Container mask and bubble fill helpers.
- Residual retry and cleanup quality validation.
- Flat fill, color plane, gradient IDW, Telea/NS, external inpaint backend execution.

Risk notes:

- This file contains safety gates to avoid destructive cleanup over art/SFX. Preserve gate intent.
- Cleanup masks may be full-image while container masks may be local to a bbox. Confirm coordinate space before editing.
- Candidate scoring is intentionally multi-factor; avoid replacing it with a single metric.

### `backend/core/cleanup.py`

Older/direct cleanup utilities and visual style analysis.

Key items:

- Text mask/bubble mask builders.
- Bubble/background classification helpers.
- Render helpers for plate, shadow, glow, gradient text.
- `classify_region`, `decide_cleanup_strategy`, `erase_text_region`, `compute_placement`, `extract_block_colors`.
- YOLO-derived text mask helper.

### `backend/core/sam2_mask.py`

Optional SAM2 mask-assist integration.

Key items:

- Config resolution for `external/sam2` and checkpoint paths.
- Import-path isolation to avoid conflicting packages.
- Device selection.
- Status/load/propose mask APIs.

SAM2 is not an inpainting backend here. It proposes masks for existing cleanup/manual-mask flows.

### `backend/core/sources/`

Source provider system.

- `base.py`: `SourceProvider` abstract contract. Providers return JSON-safe dicts/lists and should be stateless.
- `__init__.py`: lazy registry. `get_provider(name)` and `list_providers()`.
- `naver.py`: Naver Comic provider. Supports manifest-first data and some public live endpoints; raw image sync requires manifest/local folder/provider configuration.

Naver controls:

- `MANHWA_NAVER_DISABLED=1`: disable provider loading.
- `MANHWA_NAVER_MODE`: `manifest`, `live_public`, or `disabled`.

### `backend/core/test_cleanup_pipeline.py`

Focused cleanup pipeline unittest suite with synthetic pages/regions.

Use when changing:

- Cleanup plan classification.
- Mask candidate generation.
- Cleanup execution.
- Cleanup safety gates.
- SAM2/manual mask interactions.
- Typeset/cleanup interactions covered by synthetic assertions.

### `backend/core/test_raw_style_matching.py`

Focused RAW-style matching unittest suite.

Use when changing:

- RAW style analysis.
- RAW style fixture metadata.
- Auto/proposal/review behavior.
- Bootstrap RAW QA fields.

### `backend/engineOG.py` and `memory/engine.py`

Legacy/reference engines. Do not update unless the task explicitly targets them. Current runtime imports `backend.engine.LocalizerEngine`.

## Memory System

The `memory/` package is intended to be portable and JSON-serialisable.

Public API from `memory/__init__.py`:

- Stores: `GlossaryStore`, `NameMemory`, `ChapterTM`, `BlockedMappingStore`.
- Retrieval: `retrieve_batch`.
- Consistency checks: `check_name_drift`, `check_glossary_drift`, `check_blocked_output`.
- Promotion workflow: `approve_entry`, `reject_entry`, `mark_reviewed`, `promote_entry_to_series`, `promote_entry_to_global`.
- Dataclasses: `GlossaryEntry`, `NameEntry`, `ChapterTMEntry`, `BlockedMappingEntry`, `RetrievalResult`, `ConsistencyWarning`.

Disk layout:

```text
<memory_root>/
  _global/
    glossary.json
    names.json
    blocked.json
  <series_slug>/
    glossary.json
    names.json
    blocked.json
    .chapter_tm/
      <chapter_id>/tm.json
```

Rules:

- Global memory should be rare and only for cross-series conventions.
- Series memory owns character names and series-specific terms.
- Chapter TM is provisional machine output until reviewed/promoted.
- Legacy maps are not auto-applied; migration is explicit.

## Frontend File Map

### `frontend/src/api.ts`

Typed wrapper around `window.pywebview.api.*`.

Rules from the file:

- This is the only frontend file that references `window.pywebview`.
- Every call returns a typed `ApiResponse`; callers should not need try/catch.
- If pywebview is unavailable in Vite dev mode, a stub returns empty state.

When backend API changes, update the `PywebviewBridge` type, wrapper method, and stub.

### `frontend/src/types.ts`

Frontend data contracts mirroring `backend/engine.py get_bootstrap()`.

Important interfaces:

- `Region`
- `CleanupPatchInfo`
- `CleanupPreviewResponse`
- `Sam2MaskResponse`
- `CleanupDebugResponse`
- `CleanupQaAnalysis`
- `CleanupCandidate`
- `SeriesMemory`
- `PageSummary`
- `Chapter`
- `Series`
- `Issue`
- `Meta`
- `Bootstrap`
- `RawStyleMatch`
- `RawMatchQa`
- Source/series sync types
- `ProgressEvent`

If a backend bootstrap/API field is added, keep this file in sync.

### `frontend/src/App.tsx`

Main editor. Large file; search for component/function names before reading.

Major sections visible by symbol names:

- Global CSS and icons.
- Pipeline step/status UI.
- Page image cache helpers.
- Region geometry/style helpers.
- `TopBar`
- `SettingsModal`
- `PageThumb`
- `SeriesThumb`
- `CoverLightbox`
- `LeftPanel`
- `ContinuousPage`
- `ContinuousReader`
- `CanvasArea`
- `InspectorTab`
- `LayersTab`
- `ReviewTab`
- `MemoryTab`
- `RightPanel`
- `StatusBar`

Risk notes:

- `App.tsx` contains both UI and editor behavior; prefer targeted edits.
- Keep type changes aligned with `types.ts`.
- Region updates usually flow through `api.updateRegion` / backend `update_region`.
- Image mode/cache changes must invalidate the right page cache keys.

### `frontend/src/components/BrowseModal.tsx`

Source browser UI.

Handles source selection, search, series cards, thumbnails, saved/sample badges, source links, and select/import actions.

### `frontend/src/components/SeriesDetailPanel.tsx`

Series detail/sync UI.

Handles metadata fields, sync controls, chapter list, stats, danger-zone delete controls, badges, and source/chapter actions.

### `frontend/src/api_sync.ts` and `frontend/src/types_sync.ts`

Source/series sync wrappers and narrow sync-related types. Keep aligned with `api.ts`/`types.ts` if source sync contracts change.

## Data And Artifact Map

Common repo-root files/directories:

- `model_config.json`: runtime model/backend/config settings.
- `series_db.json`: series catalogue and chapter metadata.
- `handoff.json`: current/in-progress project notes; read before touching files listed there.
- `terminal.txt`: large terminal log; avoid reading unless needed.
- `run_deepseek_local.bat`: local helper batch file.
- `yolov8n.pt`: model artifact; ignored pattern covers `*.pt`.

Ignored/generated/heavy directories:

- `.venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `external/` model/vendor content
- `series_memory/`
- `source_naver_comic_*/`
- `debug_cleanup*/`
- `tools/cleanup_lab/outputs/`
- `tools/cleanup_lab/runs/`
- `__pycache__/`

Do not include generated artifacts in documentation examples unless the task specifically asks.

## Pipeline Summary

Typical full-page pipeline:

1. Detect regions.
2. OCR detected regions.
3. Translate region text.
4. Cleanup original text from image.
5. Typeset translated text.
6. Export or show raw/cleaned/typeset images.

Backend entrypoints:

- Single current page: `LocalizerEngine.run_current_page_steps`.
- All pages: `LocalizerEngine.run_all_steps`.
- Continue interrupted all-pages run: `LocalizerEngine.continue_run_all_steps`.
- Step-by-step: `detect_current_page`, `ocr_current_page`, `translate_current_page`, `cleanup_current_page`, `typeset_current_page`.

Frontend/API entrypoints:

- `api.runStep("detect" | "ocr" | "translate" | "cleanup" | "typeset")`
- `api.runCurrentPage()`
- `api.runAll()`
- `api.continueRunAll()`

## State Contract Pattern

The common update pattern is:

1. Frontend calls a method in `frontend/src/api.ts`.
2. `backend/api.py` wraps the call, guards busy state if needed, and returns `ok` response.
3. `backend/engine.py` mutates state and returns `get_bootstrap()`.
4. React replaces/synchronizes UI state from the returned bootstrap.
5. `ChapterManager` persists `.ml_state.json` directly or through debounced save.

For a new persisted region field, update:

- `OCRBlock` and serialization in `backend/core/regions.py`.
- Migrations/default restoration in `backend/core/project.py` if old state must load safely.
- Bootstrap entry in `backend/engine.py`.
- Frontend type in `frontend/src/types.ts`.
- UI/API code if user-editable.
- Tests if behavior affects cleanup, RAW QA, pipeline, or rendering.

## Cleanup Safety Notes

Cleanup has the highest risk of visible regressions.

Preserve these distinctions:

- `raw`: original page image.
- `cleaned`: text removed/filled but no translated text.
- `typeset`: translated text rendered on top.
- text/glyph mask: pixels believed to be source text.
- container/bubble mask: surrounding speech bubble or caption region.
- cleanup mask: destructive edit mask.
- manual mask: user-provided or SAM2-assisted mask, still routed through cleanup logic.

Before changing cleanup behavior:

- Identify the exact candidate/strategy path.
- Use cleanup lab for isolated experiments when possible.
- Confirm mask coordinate space: full-image vs crop-local vs container-local.
- Preserve safety gates around SFX, text-over-art, busy backgrounds, translucent captions, and large masks.
- Prefer adding a focused regression to `backend/core/test_cleanup_pipeline.py` over relying on visual inspection only.

## RAW-Style QA Notes

The current worktree contains in-progress RAW-style QA changes according to `handoff.json`.

Important planned concepts from that handoff:

- Small cropped fixtures under `backend/core/raw_style_fixtures/`.
- Deterministic tests in `backend/core/test_raw_style_matching.py`.
- RAW match QA metadata in bootstrap region serialization.
- Review statuses: `unreviewed`, `accepted`, `rejected`.
- Stale RAW match marker when bboxes/crops change.
- SFX/text-over-art should remain proposal-only and not become more aggressive.

Before editing this area, read:

- `handoff.json`
- `backend/engine.py` RAW-style helpers around `_raw_style_*`
- `backend/core/test_raw_style_matching.py`
- `frontend/src/types.ts` `RawStyleMatch` / `RawMatchQa`
- `frontend/src/App.tsx` review/issue UI

## Source Sync Notes

Source sync is provider-based.

Provider contract:

- Implement `SourceProvider`.
- Return JSON-safe dicts/lists.
- Keep provider imports lazy and optional.
- Use snake_case keys.
- Raise/return clear errors for unsupported operations.

Naver provider notes:

- Manifest schema is documented in `backend/core/sources/naver.py`.
- Source IDs and memory keys are stable identity; avoid casual renames.
- Raw image sync may rely on local manifest/folder paths.

## Frontend Design/Behavior Notes

This app is an editor/operations UI, not a marketing page.

Preserve:

- Dense but readable controls.
- Fast page/region navigation.
- Predictable inspector/review/memory panels.
- Minimal layout shift around canvas, thumbnails, status, and inspector controls.
- Existing dark UI conventions unless the task asks for redesign.

When adding UI:

- Keep `types.ts` and backend bootstrap aligned.
- Use existing button/input/panel styling patterns in `App.tsx` or component file.
- Avoid adding explanatory in-app text where a control label, tooltip, badge, or status value is enough.
- Build state for disabled/loading/error paths when calling backend APIs.

## Best Practices For Future Markdown Context

Use this pattern for new docs:

- Keep `AGENTS.md` short enough to be always loaded.
- Put large maps, contracts, and recipes in `docs/` and link to them.
- Prefer concrete file paths, commands, and known caveats over generic advice.
- Add "read this first by task" routing so future agents avoid broad scans.
- Include validation commands next to the subsystem they validate.
- Record generated/heavy directories to avoid accidental token blowups.
- Preserve current public API contracts in documentation examples.
- Update this file when architecture or command contracts materially change.

## Common Mistakes To Avoid

- Reading `terminal.txt` or generated cleanup run directories without a specific reason.
- Reverting dirty files that existed before your task.
- Editing `backend/engine.py` broadly when a smaller helper file owns the behavior.
- Adding backend bootstrap fields without updating `frontend/src/types.ts`.
- Adding frontend API calls without updating the pywebview bridge type and dev stub.
- Treating SAM2 as an inpainting backend.
- Treating chapter TM as approved glossary/name memory.
- Mutating `.ml_state.json` through `ChapterManager.load_state()` from inspection-only tooling.
- Making SFX cleanup/typeset behavior more aggressive by accident.
- Collapsing raw, cleaned, and typeset images into one artifact path.
