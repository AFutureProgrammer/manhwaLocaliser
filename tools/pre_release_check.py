from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][^'\"]{12,}['\"]"),
    re.compile(r"(?i)set\s+['\"]?[A-Z0-9_]*(API_KEY|SECRET|TOKEN)['\"]?\s*=\s*[^%\s][^&|]{12,}"),
]
TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".env",
    ".example",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yml",
    ".yaml",
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "debug_cleanup",
    "debug_cleanup_cases",
    "debug_cleanup_runs",
    "external",
    "frontend/dist",
    "frontend/node_modules",
    "node_modules",
    "series_memory",
    "tools/cleanup_lab/outputs",
    "tools/cleanup_lab/runs",
}


@dataclass
class Result:
    name: str
    ok: bool
    message: str


def _run(name: str, command: list[str], cwd: pathlib.Path = ROOT) -> Result:
    executable = shutil.which(command[0])
    if executable is None and sys.platform.startswith("win") and not command[0].lower().endswith(".cmd"):
        executable = shutil.which(command[0] + ".cmd")
    if executable is None:
        return Result(name, False, f"command not found: {command[0]}")
    proc = subprocess.run([executable, *command[1:]], cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode == 0:
        return Result(name, True, "passed")
    return Result(name, False, f"failed with exit code {proc.returncode}")


def _git_files() -> list[pathlib.Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    files: list[pathlib.Path] = []
    for line in proc.stdout.splitlines():
        rel = line.strip().replace("\\", "/")
        if not rel:
            continue
        parts = set(rel.split("/"))
        if parts & SKIP_PARTS:
            continue
        if any(rel == item or rel.startswith(item + "/") for item in SKIP_PARTS if "/" in item):
            continue
        files.append(ROOT / rel)
    return files


def _looks_text(path: pathlib.Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith(".env")


def _scan_secrets() -> Result:
    hits: list[str] = []
    for path in _git_files():
        if not path.exists() or not _looks_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except Exception:
                continue
        except Exception:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(str(path.relative_to(ROOT)))
                break
    if hits:
        return Result("secret_scan", False, "possible tracked secrets in: " + ", ".join(sorted(hits)))
    return Result("secret_scan", True, "no tracked secret patterns found")


def _check_gitignore() -> Result:
    path = ROOT / ".gitignore"
    if not path.exists():
        return Result("gitignore", False, ".gitignore is missing")
    text = path.read_text(encoding="utf-8", errors="replace")
    required = [
        ".env",
        ".env.*",
        ".venv/",
        "run_deepseek_local.local.bat",
        "series_memory/",
        "debug_cleanup*/",
        "frontend/node_modules/",
        "frontend/dist/",
        "external/sam2_checkpoints/",
    ]
    missing = [entry for entry in required if entry not in text]
    if missing:
        return Result("gitignore", False, "missing ignore entries: " + ", ".join(missing))
    return Result("gitignore", True, "required ignore entries present")


def _check_launcher() -> Result:
    path = ROOT / "run_deepseek_local.bat"
    if not path.exists():
        return Result("deepseek_launcher", True, "launcher is absent")
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"sk-[A-Za-z0-9_-]{20,}", text):
        return Result("deepseek_launcher", False, "run_deepseek_local.bat contains an inline key")
    if "DEEPSEEK_API_KEY" not in text:
        return Result("deepseek_launcher", False, "launcher does not check DEEPSEEK_API_KEY")
    return Result("deepseek_launcher", True, "launcher requires external DEEPSEEK_API_KEY")


def _system_check(allow_missing_assets: bool) -> Result:
    command = [sys.executable, "tools/system_check.py", "--json"]
    if not allow_missing_assets:
        command.extend(["--strict-release", "--fail-on-required"])
    proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        return Result("system_check", False, "system diagnostics reported required failures")
    try:
        checks = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return Result("system_check", False, "system diagnostics did not return JSON")
    warnings = sum(1 for check in checks if check.get("status") == "warn")
    failures = sum(1 for check in checks if check.get("status") == "fail")
    if failures and not allow_missing_assets:
        return Result("system_check", False, f"{failures} required checks failed")
    return Result("system_check", True, f"diagnostics completed with {warnings} warnings and {failures} failures")


def _print_results(results: Iterable[Result]) -> bool:
    ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.message}")
        ok = ok and result.ok
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Manhwa Localiser release candidate.")
    parser.add_argument("--skip-build", action="store_true", help="Skip frontend npm build.")
    parser.add_argument("--skip-compile", action="store_true", help="Skip backend compileall.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip focused unittest suites.")
    parser.add_argument(
        "--allow-missing-assets",
        action="store_true",
        help="Do not fail when configured release model assets are absent locally.",
    )
    args = parser.parse_args()

    results: list[Result] = [
        _scan_secrets(),
        _check_gitignore(),
        _check_launcher(),
        _system_check(allow_missing_assets=args.allow_missing_assets),
    ]
    if not args.skip_compile:
        results.append(_run("backend_compile", [sys.executable, "-m", "compileall", "backend", "memory"]))
    if not args.skip_tests:
        results.append(_run("cleanup_tests", [sys.executable, "-m", "unittest", "backend.core.test_cleanup_pipeline"]))
        results.append(_run("raw_style_tests", [sys.executable, "-m", "unittest", "backend.core.test_raw_style_matching"]))
    if not args.skip_build:
        results.append(_run("frontend_build", ["npm", "run", "build"], cwd=ROOT / "frontend"))

    return 0 if _print_results(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
