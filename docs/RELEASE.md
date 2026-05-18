# Release Checklist

Use this checklist before sharing or reinstalling a release candidate.

## One-Command Check

From the repo root:

```powershell
python tools/pre_release_check.py
```

The checker runs backend compile validation, focused tests, frontend build, secret scanning, launcher checks, and system diagnostics. Missing required assets and tracked secrets are reported clearly.

## Required Files

- `launcher.py`
- `backend/`
- `memory/`
- `frontend/dist/`
- `model_config.json`
- Required model assets configured in `model_config.json`, such as the YOLO model when `detector_backend` is `yolo`

## Optional Assets

- SAM2 checkout and checkpoints when `sam2_enabled` is used
- iopaint or LaMa backends
- PaddleOCR service
- local source manifests
- local memory data under `series_memory/`

## Exclude From Release Archives

- `.env` and `.env.*`
- `run_deepseek_local.local.bat`
- `.venv/`
- `frontend/node_modules/`
- `frontend/dist/.vite/`
- `series_memory/` unless explicitly exporting user data
- `source_naver_comic_*/`
- `debug_cleanup*/`
- `tools/cleanup_lab/outputs/`
- `tools/cleanup_lab/runs/`
- `external/` caches or checkpoints unless intentionally bundled
- `*.pt`, `*.pth`, and `*.ckpt` unless they are an intentional release asset

## Windows Launcher Checks

- `python launcher.py` works after `frontend/dist/` is built.
- `python launcher.py --dev` points at the Vite dev server.
- `run_deepseek_local.bat` does not contain a real key and requires `DEEPSEEK_API_KEY` to be set externally.

## Version Notes

Record the release date, commit, validation command output, required model asset versions, and known optional backends that were not bundled.
