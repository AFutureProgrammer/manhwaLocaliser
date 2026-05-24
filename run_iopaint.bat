@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=external\iopaint-venv\Scripts\python.exe"
set "SETUP_PS1=tools\setup_iopaint.ps1"

if not exist "%VENV_PY%" (
    echo IOPaint venv is missing or broken. Running setup...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SETUP_PS1%"
    if errorlevel 1 (
        echo Setup failed. See messages above.
        pause
        exit /b 1
    )
)

"%VENV_PY%" -c "import pathlib,importlib.util;s=importlib.util.find_spec('iopaint');r=pathlib.Path(list(s.submodule_search_locations)[0]);assert (r/'__main__.py').is_file() and (r/'cli.py').is_file()" 2>nul
if errorlevel 1 (
    echo IOPaint install looks corrupted. Re-running setup...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SETUP_PS1%"
    if errorlevel 1 (
        echo Setup failed.
        pause
        exit /b 1
    )
)

echo Starting IOPaint inpainting service on http://127.0.0.1:8080 ...
"%VENV_PY%" -m iopaint start --model=lama --host=127.0.0.1 --port=8080 --no-inbrowser
if errorlevel 1 (
    echo IOPaint exited with error %errorlevel%.
    pause
    exit /b %errorlevel%
)
pause
