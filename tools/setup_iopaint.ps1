# Creates or repairs external/iopaint-venv for the IOPaint cleanup backend.
# Usage (from manhwaLocaliser):  powershell -ExecutionPolicy Bypass -File tools\setup_iopaint.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $Root "external\iopaint-venv"
$PyExe = Join-Path $VenvDir "Scripts\python.exe"

function Test-IopaintInstall {
    param([string]$Python)
    if (-not (Test-Path $Python)) { return $false }
    $code = @"
import importlib.util
import pathlib
spec = importlib.util.find_spec("iopaint")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)
root = pathlib.Path(list(spec.submodule_search_locations)[0])
for name in ("__init__.py", "__main__.py", "cli.py"):
    if not (root / name).is_file():
        raise SystemExit(2)
from iopaint import entry_point  # noqa: F401
import yaml  # noqa: F401
from huggingface_hub import hf_hub_download  # noqa: F401
"@
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $Python -c $code *>$null
    $ok = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $prev
    return $ok
}

function Find-PythonLauncher {
    foreach ($ver in @("3.11", "3.12", "3.10")) {
        try {
            $out = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
        } catch { }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return (Get-Command python).Source
    }
    throw "No suitable Python found. Install Python 3.10+ or use the py launcher."
}

function New-IopaintVenv {
    param([string]$BasePython)
    if (Test-Path $VenvDir) {
        Write-Host "Removing broken IOPaint venv at external\iopaint-venv ..."
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
    $parent = Split-Path $VenvDir -Parent
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    Write-Host "Creating venv with $BasePython ..."
    & $BasePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

function Install-IopaintViaPip {
    Write-Host "Installing IOPaint via pip (includes PyTorch; may take several minutes) ..."
    & $PyExe -m pip install --upgrade pip
    & $PyExe -m pip install "iopaint>=1.6.0"
    if ($LASTEXITCODE -ne 0) { throw "pip install iopaint failed" }
}

Push-Location $Root
try {
    if (Test-IopaintInstall $PyExe) {
        Write-Host "IOPaint venv is already OK: $PyExe"
        exit 0
    }

    $reuse = $false
    if (Test-Path $PyExe) {
        Write-Host "Existing venv failed verification; reinstalling IOPaint packages ..."
        $reuse = $true
    } else {
        $basePy = Find-PythonLauncher
        New-IopaintVenv $basePy
    }

    Install-IopaintViaPip

    if (-not (Test-IopaintInstall $PyExe)) {
        throw "IOPaint install verification failed after setup."
    }

    Write-Host ""
    Write-Host "IOPaint venv ready."
    Write-Host "  Python: $PyExe"
    & $PyExe -m iopaint --help | Select-Object -First 6
    Write-Host ""
    Write-Host "Start the server with:  .\run_iopaint.bat"
} finally {
    Pop-Location
}
