# Secret Handling

The project must never store real API keys in tracked files.

## Rules

- Read API keys from environment variables.
- Keep local launcher scripts and `.env` files ignored.
- Keep examples as placeholders only.
- Do not paste secrets into issue reports, debug logs, screenshots, or release notes.
- Redact values that look like API keys before sharing diagnostics.

## DeepSeek

DeepSeek uses the environment variable named by `deepseek_api_key_env` in `model_config.json`. The default is `DEEPSEEK_API_KEY`.

PowerShell example:

```powershell
$env:DEEPSEEK_API_KEY="..."
python launcher.py --dev
```

Batch launcher reminder:

```bat
if "%DEEPSEEK_API_KEY%"=="" echo Set DEEPSEEK_API_KEY before launching.
python launcher.py --dev
```

Keep any batch file containing the real value outside git, for example `run_deepseek_local.local.bat`.

## Checks

Run:

```powershell
python tools/pre_release_check.py --skip-build --skip-tests --skip-compile
```

The check scans tracked text files for common secret patterns and fails if it finds one.
