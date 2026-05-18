# Contributor Workflow

This project is a Windows-oriented desktop app with a Python backend, a React/Vite frontend, optional local model assets, and ignored local config.

## Setup

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the Python packages required by the workflows you use. The focused validation tests currently require `numpy`, `opencv-python`, and `Pillow`. Runtime features may also require `pywebview`, `requests`, OCR/model packages, and optional backend-specific dependencies.

Frontend setup:

```powershell
cd frontend
npm install
cd ..
```

## Running

Production mode loads the built frontend:

```powershell
cd frontend
npm run build
cd ..
python launcher.py
```

Development mode uses Vite:

```powershell
cd frontend
npm run dev
```

In another terminal:

```powershell
python launcher.py --dev
```

DeepSeek mode reads the key from the environment variable named by `ModelConfig.deepseek_api_key_env`, which defaults to `DEEPSEEK_API_KEY`. Do not put real keys in tracked files.

## Validation

Run backend syntax validation after backend changes:

```powershell
python -m compileall backend memory
```

Run focused backend suites when touching cleanup or RAW-style behavior:

```powershell
python -m unittest backend.core.test_cleanup_pipeline
python -m unittest backend.core.test_raw_style_matching
```

Run the frontend build after frontend changes:

```powershell
cd frontend
npm run build
cd ..
```

Run local dependency diagnostics:

```powershell
python tools/system_check.py
```

Run release checks before packaging:

```powershell
python tools/pre_release_check.py
```

## Cleanup Lab

Use `tools/cleanup_lab/README.md` for isolated cleanup experiments. Keep run outputs under ignored output directories, and attach `report.json` plus mask/preview PNGs to cleanup issues.

## Secrets And Local Config

Tracked files must not contain API keys. Use environment variables, ignored `.env` files, or ignored local launcher scripts such as `run_deepseek_local.local.bat`.

Safe examples:

```powershell
$env:DEEPSEEK_API_KEY="..."
python launcher.py --dev
```

Ignored local files:

- `.env`
- `.env.*`
- `model_config.local.json`
- `run_deepseek_local.local.bat`

## Small Diffs

Keep changes scoped to the issue. Preserve public API contracts unless the issue explicitly changes them. Treat unrelated modified files as user work.
