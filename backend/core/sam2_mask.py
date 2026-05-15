from __future__ import annotations

import base64
import importlib
import os
import pathlib
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_SAM2_MODEL_PATH = "external/sam2"
DEFAULT_SAM2_CHECKPOINT_PATH = "external/sam2_checkpoints/sam2.1_hiera_tiny.pt"

_LOCK = threading.Lock()
_PREDICTOR: Any = None
_CACHE_KEY: Optional[Tuple[str, str, str]] = None
_STATUS: Dict[str, Any] = {
    "status": "not_loaded",
    "loaded": False,
    "error": "",
    "device": "",
    "model_path": "",
    "checkpoint_path": "",
}


@dataclass(frozen=True)
class Sam2Config:
    model_path: str = DEFAULT_SAM2_MODEL_PATH
    checkpoint_path: str = DEFAULT_SAM2_CHECKPOINT_PATH
    device: str = "auto"


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_path(value: str, fallback: str) -> pathlib.Path:
    raw = str(value or "").strip() or fallback
    path = pathlib.Path(raw)
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def _venv_site_packages(model_path: pathlib.Path) -> List[pathlib.Path]:
    candidates: List[pathlib.Path] = []
    lib_dir = model_path / ".venv" / "Lib" / "site-packages"
    if lib_dir.exists():
        candidates.append(lib_dir)
    py_root = model_path / ".venv" / "lib"
    if py_root.exists():
        candidates.extend(p for p in py_root.glob("python*/site-packages") if p.exists())
    return candidates


def _allowed_import_roots(model_path: pathlib.Path) -> List[pathlib.Path]:
    return [model_path.resolve(), *[p.resolve() for p in _venv_site_packages(model_path)]]


def _module_file(module: Any) -> str:
    return str(getattr(module, "__file__", "") or "")


def _path_is_under(path: str, roots: List[pathlib.Path]) -> bool:
    if not path:
        return False
    try:
        p = pathlib.Path(path).resolve()
    except Exception:
        return False
    for root in roots:
        try:
            p.relative_to(root)
            return True
        except Exception:
            continue
    return False


def _torch_namespace_origins() -> Dict[str, str]:
    origins: Dict[str, str] = {}
    for name in ("torch", "torchvision", "sam2"):
        mod = sys.modules.get(name)
        if mod is not None:
            origins[name] = _module_file(mod) or "<loaded-without-file>"
    return origins


def _purge_conflicting_imports(model_path: pathlib.Path) -> Dict[str, str]:
    """Remove stale torch/torchvision/SAM2 modules imported from the wrong env."""
    roots = _allowed_import_roots(model_path)
    removed: Dict[str, str] = {}
    prefixes = ("torch", "torchvision", "sam2")
    for name, mod in list(sys.modules.items()):
        if not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        origin = _module_file(mod)
        partial_torchvision = name == "torchvision" and not hasattr(mod, "extension")
        if partial_torchvision or not _path_is_under(origin, roots):
            removed[name] = origin or "<partial>"
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return removed


def _sam2_import_error(exc: Exception, *, retry_removed: Optional[Dict[str, str]] = None) -> str:
    details = _torch_namespace_origins()
    parts = [f"Could not import torch/SAM2: {exc}"]
    if retry_removed:
        top_level = {
            name: origin
            for name, origin in retry_removed.items()
            if name in {"torch", "torchvision", "sam2"}
        }
        parts.append(f"Retried after clearing conflicting modules: {top_level or retry_removed}")
    if details:
        parts.append(f"Loaded module origins after failure: {details}")
    parts.append("Restart the app after changing torch/torchvision installs if this persists.")
    return " ".join(parts)


def _prepare_import_path(model_path: pathlib.Path) -> None:
    for path in [model_path, *_venv_site_packages(model_path)]:
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    for site_packages in _venv_site_packages(model_path):
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.exists() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass


def _select_device(torch: Any, requested: str) -> str:
    device = str(requested or "auto").strip().lower()
    if device == "auto":
        # Avoid probing CUDA on auto. On Windows, torch.cuda.is_available() can
        # load incompatible cuDNN DLLs before Python can catch a clean fallback.
        return "cpu"
    if device == "cuda" and not bool(getattr(torch.cuda, "is_available", lambda: False)()):
        raise RuntimeError("SAM2 device is set to cuda, but CUDA is not available to torch.")
    if device == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(getattr(mps, "is_available", lambda: False)()):
            raise RuntimeError("SAM2 device is set to mps, but MPS is not available to torch.")
    if device not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"Unsupported SAM2 device: {requested!r}. Use auto, cpu, cuda, or mps.")
    return device


def _is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("cuda", "cudnn", "cublas", "cufft", "nvrtc"))


def _config_for_checkpoint(checkpoint_path: pathlib.Path) -> str:
    name = checkpoint_path.name.lower()
    if "sam2.1" in name:
        if "hiera_t" in name or "tiny" in name:
            return "configs/sam2.1/sam2.1_hiera_t.yaml"
        if "hiera_s" in name or "small" in name:
            return "configs/sam2.1/sam2.1_hiera_s.yaml"
        if "hiera_b" in name or "base" in name:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        if "hiera_l" in name or "large" in name:
            return "configs/sam2.1/sam2.1_hiera_l.yaml"
    if "hiera_s" in name or "small" in name:
        return "configs/sam2/sam2_hiera_s.yaml"
    if "hiera_b" in name or "base" in name:
        return "configs/sam2/sam2_hiera_b+.yaml"
    if "hiera_l" in name or "large" in name:
        return "configs/sam2/sam2_hiera_l.yaml"
    return "configs/sam2/sam2_hiera_t.yaml"


def _cfg(config: Any) -> Sam2Config:
    return Sam2Config(
        model_path=str(getattr(config, "sam2_model_path", "") or DEFAULT_SAM2_MODEL_PATH),
        checkpoint_path=str(getattr(config, "sam2_checkpoint_path", "") or DEFAULT_SAM2_CHECKPOINT_PATH),
        device=str(getattr(config, "sam2_device", "auto") or "auto"),
    )


def status(config: Any = None) -> Dict[str, Any]:
    cfg = _cfg(config) if config is not None else Sam2Config()
    model_path = _resolve_path(cfg.model_path, DEFAULT_SAM2_MODEL_PATH)
    checkpoint_path = _resolve_path(cfg.checkpoint_path, DEFAULT_SAM2_CHECKPOINT_PATH)
    with _LOCK:
        loaded = _PREDICTOR is not None and _CACHE_KEY == (str(model_path), str(checkpoint_path), str(cfg.device or "auto"))
        current = dict(_STATUS)
    same_target = (
        str(current.get("model_path") or "") == str(model_path)
        and str(current.get("checkpoint_path") or "") == str(checkpoint_path)
    )
    if loaded:
        current.update({"status": "ready", "loaded": True, "error": ""})
    elif current.get("status") == "loading":
        current.update({"loaded": False})
    elif not model_path.exists():
        current.update({"status": "missing", "loaded": False, "error": f"SAM2 model path does not exist: {model_path}"})
    elif not checkpoint_path.exists():
        current.update({"status": "missing", "loaded": False, "error": f"SAM2 checkpoint does not exist: {checkpoint_path}"})
    elif same_target and current.get("status") in {"failed", "import_failed"}:
        current.update({"loaded": False})
    else:
        current.update({"status": "available", "loaded": False, "error": ""})
    current.update({
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
        "device": current.get("device") or str(cfg.device or "auto"),
    })
    return current


def load(config: Any) -> Dict[str, Any]:
    global _PREDICTOR, _CACHE_KEY, _STATUS
    cfg = _cfg(config)
    model_path = _resolve_path(cfg.model_path, DEFAULT_SAM2_MODEL_PATH)
    checkpoint_path = _resolve_path(cfg.checkpoint_path, DEFAULT_SAM2_CHECKPOINT_PATH)
    requested_device = str(cfg.device or "auto").strip().lower()
    key = (str(model_path), str(checkpoint_path), requested_device)
    with _LOCK:
        if _PREDICTOR is not None and _CACHE_KEY == key:
            return dict(_STATUS)
        _STATUS = {
            "status": "loading",
            "loaded": False,
            "error": "",
            "device": requested_device,
            "model_path": str(model_path),
            "checkpoint_path": str(checkpoint_path),
        }
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"SAM2 model path does not exist: {model_path}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint_path}")
            _prepare_import_path(model_path)
            try:
                torch = importlib.import_module("torch")
                build_mod = importlib.import_module("sam2.build_sam")
                predictor_mod = importlib.import_module("sam2.sam2_image_predictor")
            except Exception as exc:
                removed = _purge_conflicting_imports(model_path)
                if removed:
                    try:
                        torch = importlib.import_module("torch")
                        build_mod = importlib.import_module("sam2.build_sam")
                        predictor_mod = importlib.import_module("sam2.sam2_image_predictor")
                    except Exception as retry_exc:
                        _STATUS.update({"status": "import_failed", "error": _sam2_import_error(retry_exc, retry_removed=removed)})
                        return dict(_STATUS)
                else:
                    _STATUS.update({"status": "import_failed", "error": _sam2_import_error(exc)})
                    return dict(_STATUS)
            device = _select_device(torch, requested_device)
            try:
                sam_model = build_mod.build_sam2(_config_for_checkpoint(checkpoint_path), str(checkpoint_path), device=device)
            except Exception as exc:
                if requested_device == "auto" and device == "cuda" and _is_cuda_runtime_error(exc):
                    device = "cpu"
                    sam_model = build_mod.build_sam2(_config_for_checkpoint(checkpoint_path), str(checkpoint_path), device=device)
                else:
                    raise
            _PREDICTOR = predictor_mod.SAM2ImagePredictor(sam_model)
            _CACHE_KEY = key
            _STATUS = {
                "status": "ready",
                "loaded": True,
                "error": "",
                "device": device,
                "model_path": str(model_path),
                "checkpoint_path": str(checkpoint_path),
            }
            return dict(_STATUS)
        except Exception as exc:
            _PREDICTOR = None
            _CACHE_KEY = None
            _STATUS.update({"status": "failed", "loaded": False, "error": f"Could not load embedded SAM2: {exc}"})
            return dict(_STATUS)


def _points(values: Any) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if not isinstance(values, list):
        return points
    for item in values:
        try:
            if isinstance(item, dict):
                points.append((float(item.get("x")), float(item.get("y"))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
        except Exception:
            continue
    return points


def _encode_mask_png(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8))
    if not ok:
        raise RuntimeError("Could not encode SAM2 mask as PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def propose_mask(
    image_cv: np.ndarray,
    bbox: Tuple[int, int, int, int],
    positive_clicks: Any = None,
    negative_clicks: Any = None,
    mode: str = "cleanup",
    config: Any = None,
) -> Dict[str, Any]:
    state = load(config)
    if state.get("status") != "ready":
        return {"ok": False, "status": state.get("status", "failed"), "error": state.get("error") or "Embedded SAM2 is not ready."}
    if image_cv is None or not hasattr(image_cv, "shape") or len(image_cv.shape) < 2:
        return {"ok": False, "status": "error", "error": "SAM2 received no valid image."}
    try:
        x, y, w, h = [int(v) for v in bbox]
        ih, iw = image_cv.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + max(1, w)), min(ih, y + max(1, h))
        if x2 <= x1 or y2 <= y1:
            return {"ok": False, "status": "error", "error": "SAM2 received an invalid bbox."}
        pos = _points(positive_clicks)
        neg = _points(negative_clicks)
        if not pos:
            pos = [(x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0)]
        point_coords = np.array([*pos, *neg], dtype=np.float32)
        point_labels = np.array([*[1] * len(pos), *[0] * len(neg)], dtype=np.int32)
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        with _LOCK:
            _PREDICTOR.set_image(rgb)
            masks, scores, _ = _PREDICTOR.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
        if masks is None or len(masks) == 0:
            return {"ok": False, "status": "error", "error": "SAM2 prediction returned no masks."}
        best = int(np.argmax(scores)) if scores is not None and len(scores) else 0
        mask = (masks[best] > 0).astype(np.uint8) * 255
        confidence = float(scores[best]) if scores is not None and len(scores) else 0.0
        return {
            "ok": True,
            "status": "ready",
            "mask_b64": _encode_mask_png(mask),
            "bbox": [0, 0, int(iw), int(ih)],
            "confidence": confidence,
            "reason": f"embedded_sam2_{str(mode or 'cleanup')}_proposal",
            "backend": "embedded",
        }
    except Exception as exc:
        return {"ok": False, "status": "failed", "error": f"SAM2 prediction failed: {exc}"}
