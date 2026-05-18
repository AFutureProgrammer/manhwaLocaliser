from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model_config.json"


@dataclass
class Check:
    name: str
    status: str
    message: str
    remediation: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "remediation": self.remediation,
        }


def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_config() -> tuple[dict[str, Any], list[Check]]:
    if not CONFIG_PATH.exists():
        return {}, [
            Check(
                "model_config",
                "fail",
                "model_config.json is missing.",
                "Create model_config.json or run the app once to write defaults.",
            )
        ]
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [
            Check(
                "model_config",
                "fail",
                f"model_config.json is not valid JSON: {exc}",
                "Fix the JSON syntax before starting the app.",
            )
        ]
    return data, [Check("model_config", "ok", "model_config.json is present and parseable.")]


def _module_check(module: str, package: str, required: bool = True) -> Check:
    if importlib.util.find_spec(module) is not None:
        return Check(f"python_package:{package}", "ok", f"{package} is importable.")
    status = "fail" if required else "warn"
    return Check(
        f"python_package:{package}",
        status,
        f"{package} is not importable.",
        f"Install {package} in the active Python environment.",
    )


def _path_check(name: str, path: pathlib.Path, required: bool, remediation: str) -> Check:
    if path.exists():
        return Check(name, "ok", f"{_rel(path)} exists.")
    return Check(
        name,
        "fail" if required else "warn",
        f"{_rel(path)} is missing.",
        remediation,
    )


def _env_check(config: dict[str, Any]) -> Check:
    provider = str(config.get("translation_provider", "ollama") or "ollama").lower()
    key_env = str(config.get("deepseek_api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY")
    if provider != "deepseek":
        return Check("deepseek_key", "ok", "DeepSeek is not the active translation provider.")
    if os.environ.get(key_env):
        return Check("deepseek_key", "ok", f"{key_env} is set.")
    return Check(
        "deepseek_key",
        "warn",
        f"{key_env} is not set; DeepSeek translation will be unavailable.",
        f"Set {key_env} in the shell or switch translation_provider to ollama.",
    )


def collect_checks(strict_release: bool = False) -> list[Check]:
    config, checks = _load_config()

    checks.extend(
        [
            _module_check("webview", "pywebview", required=False),
            _module_check("PIL", "Pillow", required=True),
            _module_check("cv2", "opencv-python", required=True),
            _module_check("numpy", "numpy", required=True),
            _module_check("requests", "requests", required=True),
            _path_check(
                "frontend_package",
                ROOT / "frontend" / "package.json",
                True,
                "Restore frontend/package.json.",
            ),
            _path_check(
                "frontend_dist",
                ROOT / "frontend" / "dist" / "index.html",
                strict_release,
                "Run: cd frontend && npm run build",
            ),
            _path_check(
                "launcher",
                ROOT / "launcher.py",
                True,
                "Restore launcher.py.",
            ),
        ]
    )

    if config:
        detector_backend = str(config.get("detector_backend", "") or "").lower()
        yolo_path = str(config.get("yolo_model_path", "") or "").strip()
        if detector_backend == "yolo" and yolo_path:
            checks.append(
                _path_check(
                    "yolo_model",
                    ROOT / yolo_path,
                    strict_release,
                    "Bundle the configured YOLO model or change detector_backend for this release.",
                )
            )
        sam2_required = str(config.get("sam2_required", "false") or "false").lower() == "true"
        sam2_path = str(config.get("sam2_model_path", "") or "").strip()
        sam2_checkpoint = str(config.get("sam2_checkpoint_path", "") or "").strip()
        if sam2_path:
            checks.append(
                _path_check(
                    "sam2_model_path",
                    ROOT / sam2_path,
                    sam2_required,
                    "Install SAM2 or disable sam2_required.",
                )
            )
        if sam2_checkpoint:
            checks.append(
                _path_check(
                    "sam2_checkpoint",
                    ROOT / sam2_checkpoint,
                    sam2_required,
                    "Install the configured SAM2 checkpoint or disable sam2_required.",
                )
            )
        checks.append(_env_check(config))

    writable_targets = [ROOT, ROOT / "tools" / "cleanup_lab"]
    for target in writable_targets:
        if os.access(target, os.W_OK):
            checks.append(Check(f"writable:{_rel(target)}", "ok", f"{_rel(target)} is writable."))
        else:
            checks.append(
                Check(
                    f"writable:{_rel(target)}",
                    "fail",
                    f"{_rel(target)} is not writable.",
                    "Fix directory permissions before running local workflows.",
                )
            )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Manhwa Localiser dependencies.")
    parser.add_argument("--json", action="store_true", help="Print JSON diagnostics.")
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Treat release assets such as frontend/dist and configured model files as required.",
    )
    parser.add_argument(
        "--fail-on-required",
        action="store_true",
        help="Exit non-zero when required checks fail.",
    )
    args = parser.parse_args()

    checks = collect_checks(strict_release=args.strict_release)
    if args.json:
        print(json.dumps([check.to_dict() for check in checks], indent=2))
    else:
        for check in checks:
            line = f"[{check.status.upper()}] {check.name}: {check.message}"
            print(line)
            if check.remediation:
                print(f"       {check.remediation}")

    if args.fail_on_required or args.strict_release:
        return 1 if any(check.status == "fail" for check in checks) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
