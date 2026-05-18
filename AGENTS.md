# AGENTS.md

This file is the small, always-loaded instruction layer for Codex and other coding agents. Keep it concise. Put detailed architecture and task-routing context in `docs/AGENT_CONTEXT.md` so agents can open it only when needed.

## Working Rules

- Work small. Prefer minimal diffs over rewrites.
- Before editing, inspect only files needed for the task.
- Do not refactor unrelated code.
- Do not rewrite full files unless unavoidable.
- Preserve existing public API contracts.
- Preserve existing frontend/backend behavior unless the task says otherwise.
- For bug fixes, identify the exact code path first and change the smallest viable surface.
- Do not add new features while fixing a bug.
- Treat existing uncommitted changes as user work unless you made them in the current turn.

## Repo Context

- Read `docs/AGENT_CONTEXT.md` before broad backend, frontend, pipeline, cleanup, OCR, source-sync, or memory work.
- For narrow edits, open only the relevant section of `docs/AGENT_CONTEXT.md` and the files listed there.
- Avoid scanning generated outputs under `tools/cleanup_lab/runs/`, `tools/cleanup_lab/outputs/`, `debug_cleanup*/`, `frontend/dist/`, `frontend/node_modules/`, `series_memory/`, `.venv/`, and `external/` unless the task explicitly targets them.
- `backend/engine.py`, `backend/core/cleanup_plan.py`, and `frontend/src/App.tsx` are large shared surfaces. Search for the exact method or type before reading large ranges.

## Validation

- Backend changes: run `python -m compileall backend memory`.
- Frontend changes: run `npm run build` in `frontend/`.
- Cleanup pipeline changes: also consider `python -m unittest backend.core.test_cleanup_pipeline`.
- RAW-style matching changes: also consider `python -m unittest backend.core.test_raw_style_matching`.
- Documentation-only changes do not require compile/build unless they change code examples that must be executable.

## Final Response

- List changed files.
- Give a short summary.
- List tests/checks run.
- Do not dump full files.
