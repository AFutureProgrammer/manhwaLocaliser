"""HTTP client for IOPaint 1.x /api/v1/inpaint (JSON + base64 payloads)."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import requests
from PIL import Image


def _png_b64_from_bgr(img_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64_from_mask(mask: np.ndarray) -> str:
    gray = mask
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray > 0, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_iopaint_inpaint(
    url: str,
    img_bgr: np.ndarray,
    mask: np.ndarray,
    timeout: float = 8.0,
) -> np.ndarray:
    """
    Call IOPaint inpaint API. Returns a full-frame BGR image the same size as img_bgr.
    """
    payload = {
        "image": _png_b64_from_bgr(img_bgr),
        "mask": _png_b64_from_mask(mask),
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("iopaint_invalid_response")
    if decoded.shape[:2] != img_bgr.shape[:2]:
        raise RuntimeError("iopaint_shape_mismatch")
    return decoded
