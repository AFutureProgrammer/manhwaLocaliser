# Offline And Degraded Mode

The app should keep local review work usable when optional network or model services are unavailable.

## Expected Offline Workflows

These workflows should work without remote APIs:

- open an existing imported chapter
- inspect and manually edit regions
- view existing raw, cleaned, and typeset page states
- rerender typeset output when local fonts and images are present
- run cleanup lab with local OpenCV-style paths
- inspect QA reports and cleanup artifacts
- export existing rendered outputs when required artifacts are present

## Expected Degraded Behavior

Unavailable optional services should be visible but non-fatal:

- Missing DeepSeek key should block DeepSeek translation only.
- Missing PaddleOCR service should affect PaddleOCR OCR only.
- Missing iopaint or LaMa should be reported as optional backend unavailable.
- Missing SAM2 should disable mask proposal, not normal cleanup.
- Missing model assets should produce direct remediation hints.

Run:

```powershell
python tools/system_check.py
```

Use `python tools/system_check.py --json` when attaching diagnostics to an issue.
