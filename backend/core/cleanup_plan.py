"""
backend/core/cleanup_plan.py
────────────────────────────
Manhwa-aware cleanup planner.

Separates detection concepts that must NOT be collapsed:
  region_bbox        – detector/editor container hint only
  text_bbox          – tight bounding box of source text
  text_mask          – actual Korean glyph stroke pixels
  outline_shadow_mask – nearby outline/shadow/glow belonging to source text
  container_mask     – bubble/caption interior shape (constraint, not cleanup area)
  cleanup_mask       – final destructive mask (= text_mask | outline_shadow_mask,
                       intersected with container_mask when available)
  typeset_box        – translated text placement region (not touched here)

Hard invariants enforced here:
  • region_bbox  is NEVER used as cleanup_mask.
  • bg_color     is NEVER used as fill colour for non-caption regions.
  • If mask confidence is too low, plan.cleanup_strategy = "skip".
  • Raw image is immutable; execute_cleanup_plan() writes only to a copy.

──────────────────────────────────────────────────────────────────────────────
Fix log (applied in this file)
  FIX-1  Lab chroma offset:            a/b centred at 128, not 0.
  FIX-2  NumPy chained-index assign:   use sub-view write-back in
                                       classify_background_model.
  FIX-3  Text-mask confidence scoring: stroke-ratio replaces plain coverage
                                       in all three candidate methods.
  FIX-4  Container seed isolation:     text_mask threaded into
                                       _container_color_fill so glyph
                                       pixels are excluded from bg sampling.
  FIX-5  SFX/text_on_art containers:  container inference disabled for those
                                       region classes in build_cleanup_plan.
  FIX-6  Mask normalisation helper:   normalize_mask_to_image() added and
                                       used everywhere global_cm is built.
  FIX-7  flat_fill variance gate:     falls back to TELEA when bg spread
                                       exceeds 22 gray levels.
  FIX-8  IDW pixel cap:               gradient_reconstruct_idw bails to
                                       TELEA when fill pixel count > 8 000.
──────────────────────────────────────────────────────────────────────────────
Audit note (engine.py / cleanup.py):
  All destructive pixel writes must go through erase_text_region_planned().
  Search the rest of the codebase for:
    flat_fill_region(  texture_clone_region(  erase_text_region(
    block.bg_color     bg_color               bbox.*fill
  Any survivor that reaches pixels without a CleanupPlan is the old bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from backend.core.constants import debug_print


# ──────────────────────────────────────────────────────────────────────────────
# CleanupPlan dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanupPlan:
    """All geometry, masks, decisions, and metrics for one region's cleanup."""

    # ── Identity ─────────────────────────────────────────────────────────────
    page_index:      int = -1
    region_id:       str = ""
    detector_source: str = "ocr"      # "ocr" | "yolo" | "manual"
    yolo_class:      str = ""          # "dialogue" | "sfx" | ...
    region_class:    str = "unknown"   # speech_bubble | sfx | caption_box | ...

    # ── Input geometry ────────────────────────────────────────────────────────
    region_bbox:    Optional[Tuple[int, int, int, int]] = None
    ocr_boxes:      List[Any] = field(default_factory=list)
    ocr_confidence: float = 0.0

    # ── Derived geometry ──────────────────────────────────────────────────────
    text_bbox:             Optional[Tuple[int, int, int, int]] = None
    text_mask:             Optional[np.ndarray] = None   # full-image uint8 0/255
    text_mask_confidence:  float = 0.0
    text_mask_reason:      str = ""
    outline_shadow_mask:   Optional[np.ndarray] = None
    halo_mask:             Optional[np.ndarray] = None
    container_mask:        Optional[np.ndarray] = None   # bubble interior (local ROI)
    container_bbox:        Optional[Tuple[int, int, int, int]] = None
    container_confidence:  float = 0.0
    container_reason:      str = ""

    # ── Final cleanup ─────────────────────────────────────────────────────────
    cleanup_mask:            Optional[np.ndarray] = None  # full-image uint8 0/255
    cleanup_mask_confidence: float = 0.0

    # ── Strategy ──────────────────────────────────────────────────────────────
    background_model: str = "unknown"
    # flat_light | flat_colored | smooth_gradient | translucent_gradient |
    # halftone_texture | busy_art | dark_bubble | unknown
    cleanup_strategy: str = "skip"
    # flat_fill | gradient_fill | texture_clone | mask_inpaint | skip | review
    inpaint_method:   str = "none"
    # telea | ns | idw_lab | local_sample | skip
    cleanup_backend: str = "opencv"
    iopaint_url: str = ""
    skip_reason:  str = ""
    cache_key:    str = ""

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug_metrics: Dict[str, Any] = field(default_factory=dict)
    cleanup_debug_artifacts: bool = False
    cleanup_debug_dir: str = ""

    def log(self) -> None:
        debug_print(
            f"CleanupPlan: page={self.page_index} region={self.region_id} "
            f"class={self.region_class!r} detector={self.detector_source!r} "
            f"yolo_class={self.yolo_class!r} "
            f"region_bbox={self.region_bbox} text_bbox={self.text_bbox} "
            f"text_mask_conf={self.text_mask_confidence:.2f} "
            f"text_mask_reason={self.text_mask_reason!r} "
            f"container_conf={self.container_confidence:.2f} "
            f"container_reason={self.container_reason!r} "
            f"bg_model={self.background_model!r} "
            f"strategy={self.cleanup_strategy!r} "
            f"inpaint={self.inpaint_method!r} "
            f"skip_reason={self.skip_reason!r} "
            f"metrics={self.debug_metrics}"
        )


@dataclass
class CleanupPolicy:
    cleanup_mode: str = "balanced"

    auto_clean_sfx: bool = False
    auto_typeset_sfx: bool = False
    auto_clean_text_over_art: bool = False
    auto_clean_busy_background: bool = False

    require_review_for_tier2: bool = True
    allow_gradient_fill: bool = True
    allow_texture_inpaint: bool = True

    sfx_experimental_cleanup_mode: str = "off"
    busy_background_cleanup_mode: str = "off"

    t1_text_conf: float = 0.45
    t1_container_conf: float = 0.40
    t1_max_mask_region_ratio: float = 0.18
    t1_max_border_touch: float = 0.15

    t2_text_conf: float = 0.25
    t2_max_mask_region_ratio: float = 0.28
    t2_max_border_touch: float = 0.35

    cleanup_solid_bubble_fill_enabled: bool = True
    cleanup_solid_bubble_min_container_confidence: float = 0.60
    cleanup_solid_bubble_max_mask_container_ratio: float = 0.15
    cleanup_solid_bubble_max_rectangularity: float = 0.45
    cleanup_flat_fill_ladder_enabled: bool = True
    cleanup_flat_fill_max_growth_px: int = 10
    cleanup_flat_fill_retry_extra_growth_px: int = 2
    cleanup_flat_fill_ring_px: int = 3
    cleanup_flat_fill_max_ring_gray_std: float = 14.0
    cleanup_flat_fill_max_ring_chroma_std: float = 12.0
    cleanup_flat_fill_max_ring_edge_density: float = 0.08
    cleanup_halo_mask_enabled: bool = True
    cleanup_halo_max_px: int = 2
    cleanup_residual_retry_enabled: bool = True
    cleanup_residual_retry_dilate_px: int = 1
    cleanup_allow_grouped_inpaint: bool = False
    cleanup_manual_review_only: bool = False
    cleanup_min_container_confidence: float = 0.0
    cleanup_max_mask_container_ratio: float = 0.50
    cleanup_max_mask_region_ratio: float = 0.28
    cleanup_max_border_touch_ratio: float = 0.35
    cleanup_max_rectangularity: float = 0.88
    cleanup_allow_translucent_caption: bool = False
    cleanup_allow_texture_inpaint: bool = False
    cleanup_prefer_iopaint_for_texture: bool = False
    cleanup_prefer_iopaint_for_translucent: bool = False
    cleanup_risky_action: str = "skip"
    cleanup_fallback_backend: str = "telea"
    cleanup_verbose_logs: bool = False
    cleanup_show_diagnostics: bool = False

    @classmethod
    def from_config(cls, cfg: Any) -> "CleanupPolicy":
        def _bool(name: str, default: bool) -> bool:
            value = getattr(cfg, name, default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        mode = str(getattr(cfg, "cleanup_mode", "balanced") or "balanced").strip().lower()
        if mode not in {"conservative", "balanced", "aggressive"}:
            mode = "balanced"
        policy = cls(
            cleanup_mode=mode,
            auto_clean_sfx=_bool("auto_clean_sfx", False),
            auto_typeset_sfx=_bool("auto_typeset_sfx", False),
            auto_clean_text_over_art=_bool("auto_clean_text_over_art", False),
            auto_clean_busy_background=_bool("auto_clean_busy_background", False),
            require_review_for_tier2=_bool("require_review_for_tier2", True),
            allow_gradient_fill=_bool("allow_gradient_fill", True),
            allow_texture_inpaint=_bool("allow_texture_inpaint", True),
            sfx_experimental_cleanup_mode=str(
                getattr(cfg, "sfx_experimental_cleanup_mode", "off") or "off"
            ).strip().lower(),
            busy_background_cleanup_mode=str(
                getattr(cfg, "busy_background_cleanup_mode", "off") or "off"
            ).strip().lower(),
            cleanup_solid_bubble_fill_enabled=_bool("cleanup_solid_bubble_fill_enabled", True),
            cleanup_flat_fill_ladder_enabled=_bool("cleanup_flat_fill_ladder_enabled", True),
            cleanup_halo_mask_enabled=_bool("cleanup_halo_mask_enabled", True),
            cleanup_residual_retry_enabled=_bool("cleanup_residual_retry_enabled", True),
            cleanup_allow_grouped_inpaint=_bool("cleanup_allow_grouped_inpaint", False),
            cleanup_manual_review_only=_bool("cleanup_manual_review_only", False),
            cleanup_allow_translucent_caption=_bool("cleanup_allow_translucent_caption", False),
            cleanup_allow_texture_inpaint=_bool("cleanup_allow_texture_inpaint", False),
            cleanup_prefer_iopaint_for_texture=_bool("cleanup_prefer_iopaint_for_texture", False),
            cleanup_prefer_iopaint_for_translucent=_bool("cleanup_prefer_iopaint_for_translucent", False),
            cleanup_verbose_logs=_bool("cleanup_verbose_logs", False),
            cleanup_show_diagnostics=_bool("cleanup_show_diagnostics", False),
        )
        try:
            policy.cleanup_solid_bubble_min_container_confidence = float(
                getattr(cfg, "cleanup_solid_bubble_min_container_confidence", 0.60) or 0.60
            )
        except Exception:
            policy.cleanup_solid_bubble_min_container_confidence = 0.60
        try:
            policy.cleanup_solid_bubble_max_mask_container_ratio = float(
                getattr(cfg, "cleanup_solid_bubble_max_mask_container_ratio", 0.15) or 0.15
            )
        except Exception:
            policy.cleanup_solid_bubble_max_mask_container_ratio = 0.15
        try:
            policy.cleanup_halo_max_px = int(
                getattr(cfg, "cleanup_halo_max_px", 2) or 2
            )
        except Exception:
            policy.cleanup_halo_max_px = 2
        for name, default in (
            ("cleanup_flat_fill_max_growth_px", 10),
            ("cleanup_flat_fill_retry_extra_growth_px", 2),
            ("cleanup_flat_fill_ring_px", 3),
        ):
            try:
                setattr(policy, name, max(0, int(getattr(cfg, name, default) or default)))
            except Exception:
                setattr(policy, name, int(default))
        try:
            policy.cleanup_residual_retry_dilate_px = int(
                getattr(cfg, "cleanup_residual_retry_dilate_px", 1) or 1
            )
        except Exception:
            policy.cleanup_residual_retry_dilate_px = 1
        for name, default in (
            ("cleanup_min_container_confidence", 0.0),
            ("cleanup_solid_bubble_max_rectangularity", 0.45),
            ("cleanup_max_mask_container_ratio", 0.50),
            ("cleanup_max_mask_region_ratio", 0.28),
            ("cleanup_max_border_touch_ratio", 0.35),
            ("cleanup_max_rectangularity", 0.88),
            ("cleanup_flat_fill_max_ring_gray_std", 14.0),
            ("cleanup_flat_fill_max_ring_chroma_std", 12.0),
            ("cleanup_flat_fill_max_ring_edge_density", 0.08),
        ):
            try:
                setattr(policy, name, float(getattr(cfg, name, default) or default))
            except Exception:
                setattr(policy, name, float(default))
        policy.cleanup_risky_action = str(
            getattr(cfg, "cleanup_risky_action", "skip") or "skip"
        ).strip().lower()
        if policy.cleanup_risky_action not in {"skip", "review", "attempt"}:
            policy.cleanup_risky_action = "skip"
        policy.cleanup_fallback_backend = str(
            getattr(cfg, "cleanup_fallback_backend", "telea") or "telea"
        ).strip().lower()
        if policy.cleanup_fallback_backend not in {"telea", "ns", "iopaint"}:
            policy.cleanup_fallback_backend = "telea"
        if policy.sfx_experimental_cleanup_mode not in {"off", "tight_mask", "telea"}:
            policy.sfx_experimental_cleanup_mode = "off"
        if policy.busy_background_cleanup_mode not in {"off", "tight_mask", "telea"}:
            policy.busy_background_cleanup_mode = "off"
        policy._apply_mode_thresholds()
        return policy

    def _apply_mode_thresholds(self) -> None:
        if self.cleanup_mode == "conservative":
            self.t1_text_conf = 0.55
            self.t1_container_conf = 0.50
            self.t1_max_mask_region_ratio = 0.15
            self.t1_max_border_touch = 0.10
            self.t2_text_conf = 0.40
            self.t2_max_mask_region_ratio = 0.22
            self.t2_max_border_touch = 0.25
        elif self.cleanup_mode == "aggressive":
            self.t1_text_conf = 0.35
            self.t1_container_conf = 0.30
            self.t1_max_mask_region_ratio = 0.22
            self.t1_max_border_touch = 0.18
            self.t2_text_conf = 0.18
            self.t2_max_mask_region_ratio = 0.35
            self.t2_max_border_touch = 0.45
        else:
            self.cleanup_mode = "balanced"
            self.t1_text_conf = 0.45
            self.t1_container_conf = 0.40
            self.t1_max_mask_region_ratio = 0.18
            self.t1_max_border_touch = 0.15
            self.t2_text_conf = 0.25
            self.t2_max_mask_region_ratio = 0.28
            self.t2_max_border_touch = 0.35


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _policy_with_region_override(policy: CleanupPolicy, block: Any) -> CleanupPolicy:
    override = getattr(block, "override", None)
    if override is None:
        return policy
    p = replace(policy)
    for attr, policy_attr, coerce in (
        ("cleanup_halo_max_px", "cleanup_halo_max_px", _coerce_optional_int),
        ("cleanup_residual_retry_enabled", "cleanup_residual_retry_enabled", _coerce_optional_bool),
        ("cleanup_residual_retry_dilate_px", "cleanup_residual_retry_dilate_px", _coerce_optional_int),
        ("cleanup_min_container_confidence", "cleanup_min_container_confidence", _coerce_optional_float),
        ("cleanup_max_mask_container_ratio", "cleanup_max_mask_container_ratio", _coerce_optional_float),
        ("cleanup_max_mask_region_ratio", "cleanup_max_mask_region_ratio", _coerce_optional_float),
        ("cleanup_max_border_touch_ratio", "cleanup_max_border_touch_ratio", _coerce_optional_float),
        ("cleanup_max_rectangularity", "cleanup_max_rectangularity", _coerce_optional_float),
        ("cleanup_allow_texture_inpaint", "cleanup_allow_texture_inpaint", _coerce_optional_bool),
        ("cleanup_allow_translucent_caption", "cleanup_allow_translucent_caption", _coerce_optional_bool),
    ):
        value = coerce(getattr(override, attr, None))
        if value is not None:
            setattr(p, policy_attr, value)
    if bool(getattr(override, "cleanup_allow_texture_inpaint", False)):
        p.allow_texture_inpaint = True
    return p


def _cleanup_override_mode(block: Any) -> str:
    override = getattr(block, "override", None)
    return str(getattr(override, "cleanup_override_mode", "") or "").strip().lower()


# ──────────────────────────────────────────────────────────────────────────────
# FIX-6: Mask normalisation helper
# ──────────────────────────────────────────────────────────────────────────────

def normalize_mask_to_image(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Paste a local ROI mask into a full-image-sized canvas.

    Safe against shape mismatches caused by stale state, old migrations, or
    YOLO bbox clipping at image edges.  Every place that previously built a
    global_cm array by hand now calls this instead.

    Parameters
    ----------
    mask        : local uint8 mask (H_local × W_local).
    bbox        : (x, y, w, h) in full-image coordinates.
    image_shape : (H_img, W_img[, ...]) – only first two dims are used.

    Returns
    -------
    Full-image uint8 mask with mask pasted at bbox, zeros elsewhere.
    """
    h_img, w_img = image_shape[:2]
    out = np.zeros((h_img, w_img), dtype=np.uint8)
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    clip_h, clip_w = y2 - y1, x2 - x1
    if clip_h <= 0 or clip_w <= 0:
        return out
    local = mask[:clip_h, :clip_w]
    if local.shape[0] != clip_h or local.shape[1] != clip_w:
        local = cv2.resize(
            local, (clip_w, clip_h), interpolation=cv2.INTER_NEAREST
        )
    out[y1:y2, x1:x2] = local
    return out


def _mask_preserves_cleanup(
    cleanup_mask: Optional[np.ndarray],
    limiter_mask: Optional[np.ndarray],
    min_ratio: float = 0.82,
) -> Tuple[bool, float]:
    if cleanup_mask is None or limiter_mask is None:
        return False, 0.0
    cleanup_px = int(np.count_nonzero(cleanup_mask))
    if cleanup_px <= 0:
        return False, 0.0
    kept_px = int(np.count_nonzero((cleanup_mask > 0) & (limiter_mask > 0)))
    ratio = kept_px / max(1, cleanup_px)
    return ratio >= min_ratio or cleanup_px - kept_px <= 64, float(ratio)


# ──────────────────────────────────────────────────────────────────────────────
# FIX-3 helper: stroke-ratio area score
# ──────────────────────────────────────────────────────────────────────────────

def _stroke_area_score(stroke_ratio: float) -> float:
    """
    Score the plausibility of a text mask based on ink-density.

    Korean glyph strokes typically cover 8–45 % of their OCR polygon area.
    Masks much sparser than that are probably noise; masks much denser are
    probably whole-bbox fills.  A pure coverage score ("more filled = better")
    penalises thin-stroke glyphs unfairly.

    Returns a value in [0, 1].
    """
    if 0.06 <= stroke_ratio <= 0.55:
        return 1.0
    if stroke_ratio < 0.06:
        return stroke_ratio / 0.06
    # stroke_ratio > 0.55
    return max(0.0, 1.0 - (stroke_ratio - 0.55) / 0.35)


# ──────────────────────────────────────────────────────────────────────────────
# Background model classifier
# ──────────────────────────────────────────────────────────────────────────────

_BG_MODELS = (
    "flat_light", "flat_colored", "smooth_gradient", "translucent_gradient",
    "halftone_texture", "busy_art", "dark_bubble", "unknown",
)


def classify_background_model(
    img_cv: np.ndarray,
    region_bbox: Tuple[int, int, int, int],
    container_mask: Optional[np.ndarray] = None,
    container_bbox: Optional[Tuple[int, int, int, int]] = None,
    container_confidence: float = 1.0,
    exclude_mask: Optional[np.ndarray] = None,
    text_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify the background type inside a region.

    Uses: local variance, gradient magnitude, Laplacian fine-scale energy,
    saturation variance, edge density, mean brightness, color neutrality.

    Returns (model_name, metrics_dict).
    """
    h_img, w_img = img_cv.shape[:2]
    x, y, w, h = region_bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return "unknown", {}

    roi = img_cv[y1:y2, x1:x2].copy()
    sample_mask = np.ones(roi.shape[:2], dtype=bool)
    sample_source = "region"

    # Prefer bubble/container interior when available.
    if container_mask is not None and container_bbox is not None:
        bx, by, bw, bh = container_bbox
        cm_x1 = max(0, bx - x1);  cm_y1 = max(0, by - y1)
        cm_x2 = min(x2 - x1, bx - x1 + bw)
        cm_y2 = min(y2 - y1, by - y1 + bh)
        lm_x1 = max(0, x1 - bx);  lm_y1 = max(0, y1 - by)
        lm_x2 = lm_x1 + (cm_x2 - cm_x1)
        lm_y2 = lm_y1 + (cm_y2 - cm_y1)
        if (
            cm_x2 > cm_x1 and cm_y2 > cm_y1
            and lm_x2 <= container_mask.shape[1]
            and lm_y2 <= container_mask.shape[0]
        ):
            local_cm = container_mask[lm_y1:lm_y2, lm_x1:lm_x2] > 0
            next_mask = np.zeros_like(sample_mask)
            next_mask[cm_y1:cm_y2, cm_x1:cm_x2] = local_cm
            fill_ratio = float(np.count_nonzero(local_cm)) / max(1, local_cm.size)
            trusted_container = (
                float(container_confidence or 0.0) >= 0.45
                and fill_ratio < 0.92
            )
            if np.any(next_mask) and trusted_container:
                sample_mask = next_mask
                sample_source = "container"

    excluded_text_px = 0
    if exclude_mask is not None:
        ex_roi = exclude_mask[y1:y2, x1:x2]
        if ex_roi.shape[:2] == sample_mask.shape:
            kernel = np.ones((5, 5), dtype=np.uint8)
            ex_roi = cv2.dilate((ex_roi > 0).astype(np.uint8), kernel, iterations=1) > 0
            excluded_text_px = int(np.count_nonzero(ex_roi & sample_mask))
            sample_mask &= ~ex_roi

    # Low-confidence full-rect containers are usually detector boxes, not
    # bubble interiors.  For classification, trim the ROI border so speech
    # bubble outlines do not masquerade as halftone texture.
    if sample_source == "region":
        min_dim = min(sample_mask.shape[:2])
        erode_px = max(3, min(12, int(round(min_dim * 0.08))))
        if erode_px > 0 and sample_mask.shape[0] > erode_px * 2 and sample_mask.shape[1] > erode_px * 2:
            interior = np.zeros_like(sample_mask)
            interior[erode_px:-erode_px, erode_px:-erode_px] = True
            trimmed = sample_mask & interior
            if int(np.count_nonzero(trimmed)) >= 32:
                sample_mask = trimmed
                sample_source = "region_inner"

    # Fallback to a ring around text when no reliable container/background sample exists.
    if int(np.count_nonzero(sample_mask)) < 16 and text_bbox is not None:
        tx, ty, tw, th = text_bbox
        pad_outer = 12
        pad_inner = 3
        ox1, oy1 = max(x1, tx - pad_outer), max(y1, ty - pad_outer)
        ox2, oy2 = min(x2, tx + tw + pad_outer), min(y2, ty + th + pad_outer)
        ix1, iy1 = max(x1, tx - pad_inner), max(y1, ty - pad_inner)
        ix2, iy2 = min(x2, tx + tw + pad_inner), min(y2, ty + th + pad_inner)
        ring = np.zeros_like(sample_mask)
        if ox2 > ox1 and oy2 > oy1:
            ring[oy1 - y1:oy2 - y1, ox1 - x1:ox2 - x1] = True
        if ix2 > ix1 and iy2 > iy1:
            ring[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = False
        if exclude_mask is not None:
            ex_roi = exclude_mask[y1:y2, x1:x2]
            if ex_roi.shape[:2] == ring.shape:
                ring &= ~(ex_roi > 0)
        if np.any(ring):
            sample_mask = ring

    sample_px = int(np.count_nonzero(sample_mask))
    if sample_px < 8:
        sample_mask = np.ones(roi.shape[:2], dtype=bool)
        sample_px = int(np.count_nonzero(sample_mask))

    sampled = roi[sample_mask]
    median_bgr = np.median(sampled.reshape(-1, 3), axis=0).astype(np.uint8)
    analysis_roi = roi.copy()
    analysis_roi[~sample_mask] = median_bgr

    gray = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv  = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab  = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    sample_gray = gray[sample_mask]
    sample_hsv = hsv[sample_mask]
    sample_lab = lab[sample_mask]

    # 1. Mean brightness
    mean_brightness = float(np.mean(sample_gray))

    # 2. Local variance (texture strength)
    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    local_var = float(np.std((gray - blur)[sample_mask]))

    # 3. Gradient magnitude
    gx       = cv2.Sobel(lab[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    gy       = cv2.Sobel(lab[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2)[sample_mask]))

    # 4. Fine-scale Laplacian energy → halftone detection
    lap      = cv2.Laplacian(gray, cv2.CV_32F)
    sample_lap_abs = np.abs(lap[sample_mask])
    fine_std = float(np.std(lap[sample_mask]))
    fine_edge_density = float(np.mean(sample_lap_abs > 18.0))
    fine_strong_density = float(np.mean(sample_lap_abs > 32.0))

    # 5. Saturation variance
    sat_var = float(np.std(sample_hsv[:, 1]))

    # 6. Edge density
    edges        = cv2.Canny(roi, 40, 120)
    edge_density = float(edges[sample_mask].sum()) / max(1, sample_px * 255)

    # 7. Color neutrality (a/b channels in Lab)
    a_mean     = float(np.mean(sample_lab[:, 1]))
    b_mean     = float(np.mean(sample_lab[:, 2]))
    a_std      = float(np.std(sample_lab[:, 1]))
    b_std      = float(np.std(sample_lab[:, 2]))
    chroma_offset = max(abs(a_mean - 128.0), abs(b_mean - 128.0))
    is_neutral = a_std < 8.0 and b_std < 8.0 and chroma_offset < 10.0

    # 8. Monotone gradient test
    n_bands    = max(3, min(8, gray.shape[0] // 8))
    band_means = [
        float(np.mean(gray[
            int(i * gray.shape[0] / n_bands):int((i + 1) * gray.shape[0] / n_bands)
        ]))
        for i in range(n_bands)
    ]
    d = np.diff(band_means) if len(band_means) > 1 else np.array([0.0])
    monotone_score = float(
        max(np.sum(d > 2.0), np.sum(d < -2.0))
    ) / max(1, len(d))
    spread     = float(max(band_means) - min(band_means)) if band_means else 0.0
    is_gradient = monotone_score > 0.5 and spread > 15.0

    metrics = {
        "mean_brightness": round(mean_brightness, 1),
        "local_var":       round(local_var, 2),
        "grad_mag":        round(grad_mag, 2),
        "fine_std":        round(fine_std, 2),
        "fine_edge_density": round(fine_edge_density, 5),
        "fine_strong_density": round(fine_strong_density, 5),
        "sat_var":         round(sat_var, 2),
        "edge_density":    round(edge_density, 5),
        "a_mean":          round(a_mean, 2),
        "b_mean":          round(b_mean, 2),
        "a_std":           round(a_std, 2),
        "b_std":           round(b_std, 2),
        "chroma_offset":   round(chroma_offset, 2),
        "monotone_score":  round(monotone_score, 2),
        "spread":          round(spread, 1),
        "sample_px":       sample_px,
        "excluded_text_px": excluded_text_px,
        "sample_source":    sample_source,
        "container_confidence": round(float(container_confidence or 0.0), 2),
    }

    # ── Classification rules (priority order) ─────────────────────────────
    # Pass 10 halftone guard: checked inline in P1/P2 so bright halftone/
    # screentone backgrounds are not silently overridden by the flat_light rules.
    # C11: require repeated fine edges, not just a high Laplacian ratio from
    # bubble outlines or missed glyph edges in otherwise plain white bubbles.
    _halftone_guard = (
        fine_std > 10.0
        and fine_std / max(1.0, local_var) > 0.70
        and edge_density > 0.050
        and fine_edge_density > 0.040
        and fine_strong_density > 0.015
    )

    if (
        mean_brightness > 225.0
        and is_neutral
        and sat_var < 10.0
        and spread < 32.0
        and edge_density < 0.065
        and local_var < 34.0
    ):
        model = "halftone_texture" if _halftone_guard else "flat_light"
    elif mean_brightness > 205.0 and local_var < 18.0 and edge_density < 0.04:
        if _halftone_guard:
            model = "halftone_texture"
        elif spread < 18.0:
            model = "flat_light"
        else:
            model = "smooth_gradient"
    elif _halftone_guard:
        model = "halftone_texture"
    elif edge_density > 0.055 and local_var > 18.0:
        model = "busy_art"
    elif is_gradient:
        if edge_density > 0.025 or sat_var > 30.0:
            model = "translucent_gradient"
        else:
            model = "smooth_gradient"
    elif mean_brightness < 70.0 and local_var < 18.0:
        model = "dark_bubble"
    elif local_var < 14.0 and edge_density < 0.025:
        model = "flat_light" if is_neutral else "flat_colored"
    else:
        model = "unknown"

    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Text mask candidates
# ──────────────────────────────────────────────────────────────────────────────

def _text_bbox_from_boxes(
    boxes: List[Any],
    img_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Union bounding box over all OCR polygon boxes."""
    h_img, w_img = img_shape[:2]
    pts = []
    for box in boxes:
        try:
            arr = np.array(box, dtype=np.float32)
            if arr.ndim == 2:
                pts.append(arr)
        except Exception:
            pass
    if not pts:
        return None
    all_pts = np.concatenate(pts, axis=0)
    x1 = int(np.clip(np.min(all_pts[:, 0]), 0, w_img - 1))
    y1 = int(np.clip(np.min(all_pts[:, 1]), 0, h_img - 1))
    x2 = int(np.clip(np.max(all_pts[:, 0]), 0, w_img))
    y2 = int(np.clip(np.max(all_pts[:, 1]), 0, h_img))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    pad: int,
    img_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    h_img, w_img = img_shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def _clip_mask_to_bbox(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
) -> np.ndarray:
    out = np.zeros(img_shape[:2], dtype=np.uint8)
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_shape[1], x + w), min(img_shape[0], y + h)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2 - x1, y2 - y1)


def _bbox_close(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
    tol: int = 6,
) -> bool:
    if a is None or b is None:
        return False
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def _bbox_list(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[List[int]]:
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def _mask_px(mask: Optional[np.ndarray]) -> Optional[int]:
    if mask is None:
        return None
    return int(np.count_nonzero(mask))


def compute_cleanup_effectiveness_metrics(
    raw: np.ndarray,
    cleaned: np.ndarray,
    cleanup_mask: Optional[np.ndarray],
    text_mask: Optional[np.ndarray] = None,
    *,
    attempted: bool = False,
    intended_skip: bool = False,
    manual_label: Optional[Dict[str, Any]] = None,
    validation_source: str = "metric_only",
) -> Dict[str, Any]:
    manual_label = manual_label or {}
    h = min(int(raw.shape[0]), int(cleaned.shape[0]))
    w = min(int(raw.shape[1]), int(cleaned.shape[1]))
    if h <= 0 or w <= 0:
        changed = np.zeros((0, 0), dtype=bool)
    else:
        raw_cmp = raw[:h, :w]
        cleaned_cmp = cleaned[:h, :w]
        changed = (
            np.any(raw_cmp != cleaned_cmp, axis=2)
            if raw_cmp.ndim == 3 and cleaned_cmp.ndim == 3
            else raw_cmp != cleaned_cmp
        )
    diff_px = int(np.count_nonzero(changed))
    total_px = int(max(1, changed.size))
    near_identical_px = int(max(2, min(12, int(total_px * 0.00001))))
    mask_full = None
    if cleanup_mask is not None and cleanup_mask.shape[:2] == changed.shape[:2]:
        mask_full = cleanup_mask[:h, :w] > 0
    text_px = int(np.count_nonzero(text_mask)) if text_mask is not None else 0
    cleanup_px = int(np.count_nonzero(cleanup_mask)) if cleanup_mask is not None else 0
    if mask_full is not None:
        inside = int(np.count_nonzero(changed & mask_full))
        outside = int(np.count_nonzero(changed & ~mask_full))
    else:
        inside = 0
        outside = diff_px
    manual_success = str(manual_label.get("manual_visual_success", "")).strip().lower() in {"1", "true", "yes", "success"}
    manual_partial = str(manual_label.get("manual_visual_partial", "")).strip().lower() in {"1", "true", "yes", "partial"}
    if manual_label.get("cleanup_effective") not in (None, ""):
        manual_success = manual_success or str(manual_label.get("cleanup_effective")).strip().lower() in {"1", "true", "yes"}
    did_attempt = bool(attempted or (cleanup_px > 0 and not intended_skip))
    cleaned_same = bool(diff_px == 0)
    near_identical = bool(diff_px <= near_identical_px)
    effective = False
    reason = ""
    source = validation_source or "metric_only"
    if manual_success or manual_partial:
        source = "manual_visual_label" if validation_source == "metric_only" else "mixed"
    if intended_skip and not did_attempt:
        reason = str(manual_label.get("cleanup_failure_reason") or "intentionally_skipped")
    elif cleanup_px <= 0:
        reason = "empty_cleanup_mask"
    elif did_attempt and near_identical:
        reason = "cleanup_executed_but_no_pixel_change"
    elif mask_full is None:
        reason = "mask_alignment_unknown"
    elif inside <= 0 and diff_px > near_identical_px:
        reason = "cleanup_changed_outside_cleanup_mask"
    elif manual_partial:
        reason = str(manual_label.get("cleanup_failure_reason") or "cleanup_residual_text_remains")
    elif manual_success:
        effective = True
    elif did_attempt and inside > 0 and diff_px > near_identical_px:
        effective = True
    else:
        reason = str(manual_label.get("cleanup_failure_reason") or "cleanup_not_validated")
    if effective:
        reason = ""
    return {
        "raw_cleaned_diff_px": diff_px,
        "raw_cleaned_diff_ratio": float(diff_px / max(1, total_px)),
        "diff_inside_cleanup_mask_px": inside,
        "diff_outside_cleanup_mask_px": outside,
        "cleanup_mask_px": cleanup_px,
        "text_mask_px": text_px,
        "cleaned_same_as_raw": cleaned_same,
        "near_identical_raw_cleaned": near_identical,
        "near_identical_tolerance_px": near_identical_px,
        "cleanup_effective": bool(effective),
        "cleanup_failure_reason": reason,
        "cleanup_validation_source": source,
        "manual_visual_success": bool(manual_success),
        "manual_visual_partial": bool(manual_partial),
    }


def _cleanup_changed_mask(
    raw: np.ndarray,
    cleaned: np.ndarray,
    cleanup_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, int, int, int]:
    h = min(int(raw.shape[0]), int(cleaned.shape[0]))
    w = min(int(raw.shape[1]), int(cleaned.shape[1]))
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=bool), 0, 0, 0
    raw_cmp = raw[:h, :w]
    cleaned_cmp = cleaned[:h, :w]
    changed = (
        np.any(raw_cmp != cleaned_cmp, axis=2)
        if raw_cmp.ndim == 3 and cleaned_cmp.ndim == 3
        else raw_cmp != cleaned_cmp
    )
    if cleanup_mask is not None and cleanup_mask.shape[:2] == changed.shape[:2]:
        changed = changed & (cleanup_mask[:h, :w] > 0)
    changed_px = int(np.count_nonzero(changed))
    total_px = int(max(1, changed.size))
    near_identical_px = int(max(2, min(12, int(total_px * 0.00001))))
    return changed, changed_px, total_px, near_identical_px


def _cleanup_safe_interior_mask(plan: CleanupPlan, shape: Tuple[int, ...], erode_px: int = 3) -> Optional[np.ndarray]:
    if plan.container_mask is None or plan.container_bbox is None:
        return None
    try:
        safe = normalize_mask_to_image(plan.container_mask, plan.container_bbox, shape)
    except Exception:
        return None
    if safe is None or not np.any(safe):
        return None
    safe = (safe > 0).astype(np.uint8) * 255
    if erode_px > 0:
        k = max(3, int(erode_px) * 2 + 1)
        safe = cv2.erode(safe, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return safe if np.any(safe) else None


def _component_stats(mask: np.ndarray, limit: int = 8) -> List[Dict[str, Any]]:
    if mask is None or mask.size == 0 or not np.any(mask):
        return []
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    rows: List[Dict[str, Any]] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        rows.append({
            "area": area,
            "bbox": [x, y, w, h],
            "rectangularity": round(float(area) / float(max(1, w * h)), 4),
            "aspect": round(float(max(w, h)) / float(max(1, min(w, h))), 4),
        })
    rows.sort(key=lambda item: int(item.get("area", 0)), reverse=True)
    return rows[:max(0, int(limit))]


def _bg_bgr_for_verifier(raw: np.ndarray, plan: CleanupPlan, mask: Optional[np.ndarray]) -> np.ndarray:
    residual = plan.debug_metrics.get("residual_score", {}) or {}
    bg_bgr = residual.get("sampled_bg_bgr")
    if isinstance(bg_bgr, list) and len(bg_bgr) == 3:
        return np.array(bg_bgr, dtype=np.float32)
    sampled, _metrics = _sample_container_bg_metrics(raw, plan, mask)
    if sampled is not None:
        return sampled.astype(np.float32)
    estimated, _conf = _estimate_plain_bg_color(
        raw,
        None,
        mask,
        plan.region_bbox,
        allow_dark=(plan.background_model == "dark_bubble"),
    )
    return estimated.astype(np.float32)


def _raw_glyph_support_mask(raw: np.ndarray, plan: CleanupPlan, bg_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = float(0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2])
    dist = np.sqrt(np.sum((raw.astype(np.float32) - bg_bgr[None, None, :]) ** 2, axis=2))
    edge = cv2.Canny(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), 40, 120).astype(np.float32)
    if plan.background_model == "dark_bubble":
        contrast = gray > bg_gray + 28.0
    else:
        contrast = gray < bg_gray - 22.0
    support = ((dist > 34.0) & contrast) | (edge > 0)
    return support.astype(np.uint8) * 255


def _detect_cleanup_residual_components(
    raw: np.ndarray,
    cleaned: np.ndarray,
    plan: CleanupPlan,
    mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    empty = {
        "residual_component_count": 0,
        "residual_component_px": 0,
        "residual_component_bboxes": [],
        "residual_component_authoritative_count": 0,
        "residual_component_authoritative_px": 0,
        "residual_component_stats": [],
        "residual_component_verdict": "",
        "residual_confidence": "",
        "residual_verifier_reason": "",
        "residual_retry_safe": False,
        "residual_retry_rejection_reason": "",
    }
    if (
        plan.region_class not in {"speech_bubble", "caption_box"}
        or plan.background_model not in {"flat_light", "flat_colored", "dark_bubble"}
        or plan.cleanup_strategy != "flat_fill"
        or mask is None
        or not np.any(mask)
        or plan.debug_metrics.get("cleanup_mask_rejected", False)
    ):
        return empty
    safe = _cleanup_safe_interior_mask(plan, cleaned.shape, erode_px=3)
    if safe is None:
        return {**empty, "residual_retry_rejection_reason": "no_safe_interior"}
    bg_bgr = _bg_bgr_for_verifier(raw, plan, mask)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = float(0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2])
    dist = np.sqrt(np.sum((cleaned.astype(np.float32) - bg_bgr[None, None, :]) ** 2, axis=2))
    edge = cv2.Canny(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), 35, 110).astype(np.float32)
    if plan.background_model == "dark_bubble":
        contrast = gray > bg_gray + 32.0
    else:
        contrast = gray < bg_gray - 24.0
    suspect = ((dist > 38.0) & contrast) | ((dist > 48.0) & (edge > 0))
    suspect &= safe > 0
    raw_support = _raw_glyph_support_mask(raw, plan, bg_bgr)
    if plan.text_mask is not None:
        text_near = cv2.dilate(
            (plan.text_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
    else:
        text_near = cv2.dilate(
            (mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
    suspect &= ((raw_support > 0) | (text_near > 0))
    if not np.any(suspect):
        return empty
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(suspect.astype(np.uint8), 8)
    kept = np.zeros(suspect.shape, dtype=np.uint8)
    high_confidence = np.zeros(suspect.shape, dtype=np.uint8)
    component_rows: List[Dict[str, Any]] = []
    safe_active = safe > 0
    safe_bbox = _mask_bbox(safe)
    rejection = ""
    if mask is not None and np.any(mask):
        distance_to_mask = cv2.distanceTransform((mask <= 0).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        distance_to_mask = np.full(suspect.shape, 9999.0, dtype=np.float32)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 1 or area > 420 or max(w, h) > 46:
            rejection = "component_size"
            continue
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        fill = float(area) / float(max(1, w * h))
        if aspect > 10.0 or (fill > 0.88 and area > 80):
            rejection = "non_glyph_shape"
            continue
        comp = labels == label
        if safe_bbox is not None:
            sx, sy, sw, sh = safe_bbox
            if x <= sx + 1 or y <= sy + 1 or x + w >= sx + sw - 1 or y + h >= sy + sh - 1:
                rejection = "touches_safe_border"
                continue
        support_ratio = float(np.count_nonzero(comp & (raw_support > 0))) / float(max(1, area))
        near_ratio = float(np.count_nonzero(comp & (text_near > 0))) / float(max(1, area))
        if support_ratio < 0.20 and near_ratio < 0.65:
            rejection = "weak_raw_glyph_support"
            continue
        kept[comp & safe_active] = 255
        cleanup_overlap = int(np.count_nonzero(comp & (mask > 0)))
        mask_distance = float(np.min(distance_to_mask[comp])) if np.any(comp) else 9999.0
        color_distance_mean = float(dist[comp].mean())
        color_distance_max = float(dist[comp].max())
        edge_mean = float(edge[comp].mean())
        edge_px = int(np.count_nonzero(edge[comp]))
        confidence = 0.35
        if near_ratio >= 0.65:
            confidence += 0.45
        if mask_distance <= 8.0:
            confidence += 0.30
        elif mask_distance <= 16.0:
            confidence += 0.12
        if mask_distance <= 16.0 and support_ratio >= 0.80 and (color_distance_mean >= 90.0 or edge_mean >= 50.0):
            confidence += 0.25
        if cleanup_overlap > 0:
            confidence += 0.20
        confidence = min(1.0, confidence)
        is_high_confidence = bool(confidence >= 0.65)
        if is_high_confidence:
            high_confidence[comp & safe_active] = 255
        component_rows.append({
            "area": area,
            "bbox": [x, y, w, h],
            "color_distance_mean": round(float(color_distance_mean), 3),
            "color_distance_max": round(float(color_distance_max), 3),
            "edge_mean": round(float(edge_mean), 3),
            "edge_px": edge_px,
            "raw_support_ratio": round(float(support_ratio), 4),
            "text_near_ratio": round(float(near_ratio), 4),
            "cleanup_mask_overlap_px": cleanup_overlap,
            "distance_to_cleanup_mask_px": round(float(mask_distance), 3),
            "residual_confidence": round(float(confidence), 3),
            "residual_component_verdict": "high_confidence_residual" if is_high_confidence else "low_confidence_texture_or_antialias",
        })
    component_px = int(np.count_nonzero(kept))
    if component_px <= 0:
        return {**empty, "residual_retry_rejection_reason": rejection or "no_kept_components"}
    high_confidence_px = int(np.count_nonzero(high_confidence))
    authoritative = bool(high_confidence_px > 0)
    retry_mask = cv2.dilate(
        high_confidence if authoritative else kept,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    retry_mask = np.where((retry_mask > 0) & (safe > 0), 255, 0).astype(np.uint8)
    retry_px = int(np.count_nonzero(retry_mask & ~(mask > 0)))
    mask_px = int(np.count_nonzero(mask))
    retry_safe = bool(authoritative and 0 < retry_px <= max(96, min(700, int(mask_px * 0.12))))
    component_rows.sort(key=lambda row: int(row.get("area", 0)), reverse=True)
    verdict = "high_confidence_residual" if authoritative else "low_confidence_texture_or_antialias"
    result = {
        "residual_component_count": len(_component_stats(kept, limit=32)),
        "residual_component_px": component_px,
        "residual_component_bboxes": [row["bbox"] for row in _component_stats(kept, limit=8)],
        "residual_component_authoritative_count": len(_component_stats(high_confidence, limit=32)),
        "residual_component_authoritative_px": high_confidence_px,
        "residual_component_stats": component_rows[:16],
        "residual_component_verdict": verdict,
        "residual_confidence": "high" if authoritative else "low",
        "residual_verifier_reason": "component_glyph_residual" if authoritative else "residual_component_low_confidence",
        "residual_retry_safe": retry_safe,
        "residual_retry_rejection_reason": "" if retry_safe else ("low_confidence_residual_components" if not authoritative else "retry_growth_too_large_or_empty"),
        "residual_retry_mask": retry_mask if retry_safe else None,
        "residual_retry_px": retry_px,
    }
    return result


def _changed_component_patch_analysis(
    raw: np.ndarray,
    cleaned: np.ndarray,
    plan: CleanupPlan,
    changed: np.ndarray,
) -> Dict[str, Any]:
    if (
        changed is None
        or changed.size == 0
        or not np.any(changed)
        or plan.background_model not in {"flat_light", "flat_colored", "dark_bubble"}
        or plan.cleanup_strategy != "flat_fill"
    ):
        return {
            "fill_patch_component_count": 0,
            "fill_patch_component_bboxes": [],
            "fill_patch_reason": "",
        }
    bg_bgr = _bg_bgr_for_verifier(raw, plan, plan.cleanup_mask)
    raw_support = _raw_glyph_support_mask(raw, plan, bg_bgr) > 0
    safe = _cleanup_safe_interior_mask(plan, raw.shape, erode_px=2)
    safe_bbox = _mask_bbox(safe) if safe is not None else None
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
    patch_components = np.zeros(changed.shape, dtype=np.uint8)
    reasons: List[str] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 64:
            continue
        comp = labels == label
        fill_ratio = float(area) / float(max(1, w * h))
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        raw_ratio = float(np.count_nonzero(comp & raw_support)) / float(max(1, area))
        side_band = False
        border_band = False
        if safe_bbox is not None:
            sx, sy, sw, sh = safe_bbox
            side_band = bool(x <= sx + 4 or x + w >= sx + sw - 4)
            border_band = bool(y <= sy + 4 or y + h >= sy + sh - 4)
        blocky = bool(fill_ratio >= 0.78 and area >= 180)
        long_bar = bool(aspect >= 5.0 and area >= 120)
        oversized = bool(plan.container_mask is not None and area >= max(240, int(np.count_nonzero(plan.container_mask) * 0.18)))
        if raw_ratio <= 0.24 and ((blocky and (side_band or border_band or oversized)) or (long_bar and (side_band or border_band))):
            patch_components[comp] = 255
            if blocky:
                reasons.append("blocky_changed_component")
            if side_band:
                reasons.append("side_band_changed_component")
            if border_band:
                reasons.append("border_band_changed_component")
            if long_bar:
                reasons.append("long_bar_changed_component")
    rows = _component_stats(patch_components, limit=8)
    reason = ",".join(dict.fromkeys(reasons))
    return {
        "fill_patch_component_count": len(rows),
        "fill_patch_component_bboxes": [row["bbox"] for row in rows],
        "fill_patch_reason": reason,
    }


def _cleanup_visual_quality_result(
    raw: np.ndarray,
    cleaned: np.ndarray,
    plan: CleanupPlan,
) -> Dict[str, Any]:
    changed, changed_px, total_px, _near_identical_px = _cleanup_changed_mask(raw, cleaned, plan.cleanup_mask)
    quality = plan.debug_metrics.get("quality", {}) or {}
    mask_region_ratio = float(quality.get("mask_region_ratio", 0.0) or 0.0)
    mask_container_ratio = float(quality.get("mask_container_ratio", 0.0) or 0.0)
    rectangularity = float(quality.get("rectangularity", 0.0) or 0.0)
    border_touch_ratio = float(quality.get("border_touch_ratio", 0.0) or 0.0)
    long_bar_score = float(quality.get("long_bar_score", 0.0) or 0.0)
    changed_bbox = None
    changed_rectangularity = 0.0
    if changed_px > 0:
        bbox = _mask_bbox(changed.astype(np.uint8) * 255)
        if bbox is not None:
            bx, by, bw, bh = bbox
            changed_bbox = [int(bx), int(by), int(bw), int(bh)]
            changed_rectangularity = float(changed_px) / float(max(1, bw * bh))
    component_analysis = _changed_component_patch_analysis(raw, cleaned, plan, changed)
    fill_patch_visible = bool(
        changed_px > 0
        and (
            (
                rectangularity >= 0.92
                and border_touch_ratio >= 0.28
                and (mask_container_ratio >= 0.38 or mask_region_ratio >= 0.30)
            )
            or (
                changed_rectangularity >= 0.90
                and changed_px / max(1, total_px) >= 0.025
                and (mask_region_ratio >= 0.28 or mask_container_ratio >= 0.35)
            )
            or (
                long_bar_score > 32.0
                and border_touch_ratio >= 0.35
                and mask_region_ratio >= 0.18
            )
            or int(component_analysis.get("fill_patch_component_count", 0) or 0) > 0
        )
    )
    failure_reason = "cleanup_fill_patch_visible" if fill_patch_visible else ""
    return {
        "visual_quality_ok": not fill_patch_visible,
        "visual_quality_failure_reason": failure_reason,
        "fill_patch_visible": fill_patch_visible,
        "changed_mask_px": int(changed_px),
        "changed_mask_bbox": changed_bbox,
        "changed_mask_rectangularity": round(float(changed_rectangularity), 4),
        **component_analysis,
    }


def _outcome_bool(outcome: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = outcome.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "success"}:
        return True
    if text in {"0", "false", "no", "off", "failure", "failed"}:
        return False
    return bool(value)


def _outcome_reason(outcome: Dict[str, Any]) -> str:
    return str(
        outcome.get("cleanup_failure_reason")
        or outcome.get("proposal_failure_reason")
        or outcome.get("visual_quality_failure_reason")
        or ""
    )


def clamp_cleanup_outcome_fields(outcome: Dict[str, Any]) -> Dict[str, Any]:
    cleanup_reason = str(outcome.get("cleanup_failure_reason") or outcome.get("visual_quality_failure_reason") or "")
    proposal_reason = str(outcome.get("proposal_failure_reason") or "")
    reason = cleanup_reason or proposal_reason
    proposal_valid = _outcome_bool(outcome, "proposal_valid", False)
    residual_text_visible = bool(
        _outcome_bool(outcome, "residual_text_visible", False)
        or reason in {"cleanup_residual_text_remains", "cleanup_ocr_text_remains"}
    )
    fill_patch_visible = bool(
        _outcome_bool(outcome, "fill_patch_visible", False)
        or reason in {"cleanup_fill_patch_visible", "fill_patch_visible"}
    )
    visual_quality_ok = bool(
        _outcome_bool(outcome, "visual_quality_ok", True)
        and reason not in {"visual_quality_failed", "cleanup_fill_patch_visible", "fill_patch_visible"}
    )
    gate_violation = _outcome_bool(outcome, "gate_violation", False)

    if fill_patch_visible:
        visual_quality_ok = False
        outcome["visual_quality_ok"] = False
        if not outcome.get("visual_quality_failure_reason"):
            outcome["visual_quality_failure_reason"] = "cleanup_fill_patch_visible"

    can_be_effective = bool(
        proposal_valid
        and not residual_text_visible
        and visual_quality_ok
        and not fill_patch_visible
        and not gate_violation
    )
    if not can_be_effective:
        outcome["cleanup_effective"] = False
        if not proposal_valid:
            outcome["production_patch_accepted"] = False
            if not outcome.get("proposal_failure_reason"):
                outcome["proposal_failure_reason"] = proposal_reason or "cleanup_proposal_invalid"
        if residual_text_visible:
            outcome["cleanup_partial"] = True
            if not cleanup_reason:
                outcome["cleanup_failure_reason"] = "cleanup_residual_text_remains"
                cleanup_reason = "cleanup_residual_text_remains"
        if fill_patch_visible and not outcome.get("cleanup_failure_reason"):
            outcome["cleanup_failure_reason"] = "cleanup_fill_patch_visible"
            cleanup_reason = "cleanup_fill_patch_visible"
        if not visual_quality_ok and not cleanup_reason:
            outcome["cleanup_failure_reason"] = str(
                outcome.get("visual_quality_failure_reason") or "visual_quality_failed"
            )
            cleanup_reason = str(outcome.get("cleanup_failure_reason") or "")
        if gate_violation and not cleanup_reason:
            outcome["cleanup_failure_reason"] = "cleanup_gate_violation"
        if not outcome.get("cleanup_failure_reason") and outcome.get("proposal_failure_reason"):
            outcome["cleanup_failure_reason"] = str(outcome.get("proposal_failure_reason") or "")
    else:
        outcome["cleanup_effective"] = True
        outcome.setdefault("cleanup_partial", False)
    outcome["proposal_valid"] = bool(proposal_valid)
    outcome["residual_text_visible"] = bool(residual_text_visible)
    outcome["fill_patch_visible"] = bool(fill_patch_visible)
    outcome["visual_quality_ok"] = bool(visual_quality_ok)
    outcome["gate_violation"] = bool(gate_violation)
    return outcome


def mark_cleanup_proposal_blocked(plan: CleanupPlan, reason: str) -> None:
    failure = str(reason or "cleanup_proposal_blocked")
    plan.debug_metrics.setdefault("proposal_original_strategy", plan.cleanup_strategy)
    plan.debug_metrics.setdefault("proposal_original_inpaint_method", plan.inpaint_method)
    plan.debug_metrics["diagnostic_only"] = True
    plan.debug_metrics["destructive_cleanup_executed"] = False
    plan.debug_metrics["production_patch_accepted"] = False
    plan.debug_metrics["proposal_valid"] = False
    plan.debug_metrics["proposal_failure_reason"] = failure
    plan.debug_metrics["gate_violation"] = False
    plan.debug_metrics["diagnostic_cleanup_ran"] = False
    plan.debug_metrics["text_removed"] = False
    plan.debug_metrics["residual_text_visible"] = False
    plan.debug_metrics["visual_quality_ok"] = False
    plan.debug_metrics["fill_patch_visible"] = False
    plan.debug_metrics["cleanup_effective"] = False
    plan.cleanup_strategy = "skip"
    plan.inpaint_method = "skip"
    plan.skip_reason = failure


def validate_cleanup_proposal(
    raw: np.ndarray,
    cleaned: np.ndarray,
    plan: CleanupPlan,
    *,
    destructive_allowed: bool = True,
    production_patch_accepted: bool = False,
    validation_source: str = "production",
) -> Dict[str, Any]:
    if not isinstance(plan.debug_metrics.get("quality", None), dict):
        _refresh_cleanup_quality(plan, raw)
    intended_skip = bool(plan.cleanup_strategy in ("skip", "review"))
    mask_present = bool(plan.cleanup_mask is not None and np.any(plan.cleanup_mask))
    attempted = bool(destructive_allowed and not intended_skip and mask_present)
    metrics = compute_cleanup_effectiveness_metrics(
        raw,
        cleaned,
        plan.cleanup_mask,
        plan.text_mask,
        attempted=attempted,
        intended_skip=intended_skip,
        validation_source=validation_source,
    )
    pixel_cleanup_executed = bool(
        int(metrics.get("raw_cleaned_diff_px", 0) or 0)
        > int(metrics.get("near_identical_tolerance_px", 0) or 0)
        and int(metrics.get("diff_inside_cleanup_mask_px", 0) or 0) > 0
    )
    destructive_cleanup_executed = bool(pixel_cleanup_executed)
    residual_score = plan.debug_metrics.get("residual_score", {}) or {}
    skip_reason = str(plan.skip_reason or "")
    residual_text_visible = bool(
        residual_score.get("bad", False)
        or int(plan.debug_metrics.get("residual_component_authoritative_count", 0) or 0) > 0
        or skip_reason in {"cleanup_residual_text_remains", "cleanup_ocr_text_remains"}
    )
    visual = _cleanup_visual_quality_result(raw, cleaned, plan)
    visual_quality_ok = bool(visual.get("visual_quality_ok", True))
    review_required = bool(plan.debug_metrics.get("review_required_after_cleanup", False))
    force_mode = str(plan.debug_metrics.get("cleanup_override_mode", "") or "") in {
        "force_allow", "force_solid", "force_telea", "force_ns", "force_iopaint"
    }
    production_gate_blocked = bool(
        not destructive_allowed
        or intended_skip
        or bool(skip_reason)
        or review_required
        or bool(plan.debug_metrics.get("cleanup_mask_rejected", False))
        or not mask_present
    )
    gate_violation = bool(production_patch_accepted and production_gate_blocked)
    text_removed = bool(destructive_cleanup_executed)
    failure_reason = str(plan.debug_metrics.get("proposal_failure_reason", "") or "")
    if not failure_reason:
        if not destructive_allowed:
            failure_reason = "destructive_cleanup_not_allowed"
        elif intended_skip:
            failure_reason = skip_reason or "cleanup_strategy_skip_or_review"
        elif not mask_present:
            failure_reason = "empty_cleanup_mask"
        elif review_required:
            failure_reason = skip_reason or "cleanup_requires_review"
        elif not destructive_cleanup_executed:
            failure_reason = str(metrics.get("cleanup_failure_reason") or "destructive_cleanup_not_executed")
        elif gate_violation:
            failure_reason = skip_reason or "cleanup_gate_violation"
    cleanup_failure_reason = ""
    if residual_text_visible:
        cleanup_failure_reason = "cleanup_residual_text_remains"
    elif not visual_quality_ok:
        cleanup_failure_reason = str(visual.get("visual_quality_failure_reason") or "visual_quality_failed")
    elif not destructive_cleanup_executed:
        cleanup_failure_reason = str(metrics.get("cleanup_failure_reason") or "destructive_cleanup_not_executed")
    elif gate_violation:
        cleanup_failure_reason = "cleanup_gate_violation"
    proposal_valid = bool(
        destructive_allowed
        and not production_gate_blocked
        and destructive_cleanup_executed
        and text_removed
        and not residual_text_visible
        and visual_quality_ok
    )
    if not proposal_valid and not failure_reason:
        failure_reason = "cleanup_proposal_invalid"
    if not visual_quality_ok and not skip_reason:
        plan.skip_reason = str(visual.get("visual_quality_failure_reason") or "visual_quality_failed")
        plan.debug_metrics["review_required_after_cleanup"] = True
    cleanup_effective = bool(proposal_valid and destructive_cleanup_executed)
    proposal = {
        **metrics,
        **visual,
        "diagnostic_only": bool(not proposal_valid),
        "diagnostic_cleanup_ran": bool(not proposal_valid and destructive_cleanup_executed),
        "destructive_cleanup_executed": bool(destructive_cleanup_executed),
        "production_patch_accepted": bool(production_patch_accepted and proposal_valid),
        "proposal_valid": bool(proposal_valid),
        "proposal_failure_reason": "" if proposal_valid else failure_reason,
        "cleanup_failure_reason": "" if proposal_valid else cleanup_failure_reason,
        "gate_violation": bool(gate_violation),
        "forced_cleanup_override": bool(force_mode),
        "text_removed": bool(text_removed),
        "residual_text_visible": bool(residual_text_visible),
        "cleanup_effective": bool(cleanup_effective),
    }
    clamp_cleanup_outcome_fields(proposal)
    plan.debug_metrics.update(proposal)
    return proposal


def cleanup_production_patch_allowed(plan: CleanupPlan) -> bool:
    metrics = plan.debug_metrics or {}
    return bool(
        metrics.get("proposal_valid", False)
        and metrics.get("cleanup_effective", False)
        and metrics.get("destructive_cleanup_executed", False)
        and not metrics.get("diagnostic_only", True)
        and not metrics.get("gate_violation", False)
        and not metrics.get("residual_text_visible", False)
        and not metrics.get("fill_patch_visible", False)
        and plan.cleanup_strategy not in ("skip", "review")
        and plan.cleanup_mask is not None
        and np.any(plan.cleanup_mask)
    )


def mark_production_patch_accepted(plan: CleanupPlan, accepted: bool) -> None:
    plan.debug_metrics["production_patch_accepted"] = bool(accepted and cleanup_production_patch_allowed(plan))


def _candidate_source_from_reason(reason: str) -> str:
    text = str(reason or "")
    if text.startswith("existing_mask"):
        return "legacy_block_text_mask"
    if text.startswith("ocr_contrast"):
        return "ocr_polygon"
    if text.startswith("multichannel"):
        return "cv_threshold"
    if text.startswith("edge_components"):
        return "edge_component"
    if text.startswith("container_first"):
        return "container_first_glyph"
    if text.startswith("dark_caption"):
        return "dark_caption_path"
    if text.startswith("region_cv_no_bbox"):
        return "fallback_cv_no_bbox"
    if text in {"none", "no_candidates", ""}:
        return "none"
    return "fallback"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(v) for v in value.shape],
            "mask_px": int(np.count_nonzero(value)) if value.ndim >= 2 else int(value.size),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cleanup_mask_metrics(
    mask: Optional[np.ndarray],
    region_bbox: Tuple[int, int, int, int],
    text_bbox: Optional[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    rx, ry, rw, rh = region_bbox
    region_area = int(max(1, rw * rh))
    text_area = int(max(1, (text_bbox[2] * text_bbox[3]) if text_bbox else 0))
    mask_area = int(np.count_nonzero(mask)) if mask is not None else 0
    return {
        "region_area": region_area,
        "text_bbox_area": 0 if text_bbox is None else text_area,
        "mask_area": mask_area,
        "mask_region_ratio": round(mask_area / max(1, region_area), 4),
        "mask_text_ratio": round(mask_area / max(1, text_area), 4) if text_bbox else 0.0,
        "mask_bbox": _mask_bbox(mask) if mask is not None else None,
    }


def _compute_mask_quality_metrics(
    mask: Optional[np.ndarray],
    container_mask: Optional[np.ndarray],
    region_bbox: Tuple[int, int, int, int],
    text_bbox: Optional[Tuple[int, int, int, int]] = None,
    safety_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """Compute concise safety/shape metrics for a full-image cleanup mask.

    Pass 1 (border-collision fix):
        ``safety_bbox`` — if provided, border-touch checks evaluate against
        this rectangle instead of ``region_bbox``. Callers pass a validated
        ``container_bbox`` here for high-confidence speech/caption bubbles so
        a tight YOLO ``region_bbox`` does not falsely flag border collision.
        Size ratios still use ``region_bbox`` — that is a size check, not a
        border check, and must remain strict.
    """
    rx, ry, rw, rh = region_bbox
    region_area = int(max(1, rw * rh))
    mask_area = int(np.count_nonzero(mask)) if mask is not None else 0
    container_area = (
        int(np.count_nonzero(container_mask)) if container_mask is not None else 0
    )
    sx, sy, sw, sh = safety_bbox if safety_bbox is not None else region_bbox
    safety_source = "container" if safety_bbox is not None and safety_bbox != region_bbox else "region"
    empty = {
        "mask_area": mask_area,
        "region_area": region_area,
        "container_area": container_area,
        "mask_region_ratio": 0.0,
        "mask_container_ratio": 0.0,
        "mask_bbox": None,
        "mask_bbox_area": 0,
        "rectangularity": 0.0,
        "border_touch_ratio": 0.0,
        "component_count": 0,
        "largest_component_ratio": 0.0,
        "long_bar_score": 0.0,
        "safety_bbox_source": safety_source,
    }
    if mask is None or mask_area <= 0:
        return empty

    mask_u8 = (mask > 0).astype(np.uint8)
    mb = _mask_bbox(mask_u8)
    if mb is None:
        return empty
    mx, my, mw, mh = mb
    mask_bbox_area = int(max(1, mw * mh))

    tol = 5
    touches = 0
    touches += 1 if abs(mx - sx) <= tol else 0
    touches += 1 if abs(my - sy) <= tol else 0
    touches += 1 if abs((mx + mw) - (sx + sw)) <= tol else 0
    touches += 1 if abs((my + mh) - (sy + sh)) <= tol else 0

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    largest = max(areas) if areas else 0

    metrics = {
        "mask_area": mask_area,
        "region_area": region_area,
        "container_area": container_area,
        "mask_region_ratio": round(mask_area / max(1, region_area), 4),
        "mask_container_ratio": (
            round(mask_area / max(1, container_area), 4) if container_area else 0.0
        ),
        "mask_bbox": mb,
        "mask_bbox_area": mask_bbox_area,
        "rectangularity": round(mask_area / max(1, mask_bbox_area), 4),
        "border_touch_ratio": round(touches / 4.0, 3),
        "component_count": len(areas),
        "largest_component_ratio": round(largest / max(1, mask_area), 4),
        "long_bar_score": round(max(mw / max(1, mh), mh / max(1, mw)), 3),
        "safety_bbox_source": safety_source,
    }
    if text_bbox is not None:
        metrics["text_bbox_area"] = int(max(1, text_bbox[2] * text_bbox[3]))
    return metrics


def _select_safety_bbox(
    plan: CleanupPlan,
) -> Optional[Tuple[int, int, int, int]]:
    """Return the validated container_bbox to use as the border-touch reference,
    or None to fall back to region_bbox.

    Pass 1 rule: speech_bubble / caption_box with container_confidence >= 0.60
    evaluate border-touch against the container, not the tight YOLO region_bbox.
    SFX / text_on_art / busy_art stay strict (returns None).
    """
    if (
        plan.region_class in ("speech_bubble", "caption_box")
        and plan.container_bbox is not None
        and float(plan.container_confidence or 0.0) >= 0.60
    ):
        return tuple(int(v) for v in plan.container_bbox)  # type: ignore[return-value]
    return None


def _translucent_detail_score(metrics: Dict[str, Any]) -> float:
    """Small score for mild translucent panels, larger for visible art detail."""
    try:
        edge_density = float(metrics.get("edge_density", 0.0) or 0.0)
        local_var = float(metrics.get("local_var", 0.0) or 0.0)
        fine_density = float(metrics.get("fine_edge_density", 0.0) or 0.0)
        sat_var = float(metrics.get("sat_var", 0.0) or 0.0)
        spread = float(metrics.get("spread", 0.0) or 0.0)
    except Exception:
        return 0.0
    score = 0.0
    score += max(0.0, (edge_density - 0.030) / 0.030)
    score += max(0.0, (local_var - 16.0) / 18.0)
    score += max(0.0, (fine_density - 0.020) / 0.035)
    score += max(0.0, (sat_var - 24.0) / 30.0)
    score += max(0.0, (spread - 34.0) / 34.0)
    return round(float(score), 3)


def _try_adopt_existing_safe_rect_container(
    img_cv: np.ndarray,
    plan: CleanupPlan,
) -> bool:
    if (
        plan.region_class not in {"speech_bubble", "caption_box"}
        or plan.background_model not in {"flat_light", "flat_colored", "dark_bubble"}
        or plan.container_mask is not None
        or float(plan.text_mask_confidence or 0.0) < 0.45
    ):
        return False
    safe_rect = plan.debug_metrics.get("cleanup_safe_rect_existing")
    safe_conf = float(plan.debug_metrics.get("cleanup_safe_rect_existing_confidence", 0.0) or 0.0)
    if not isinstance(safe_rect, list) or len(safe_rect) != 4 or safe_conf < 0.70:
        return False
    h_img, w_img = img_cv.shape[:2]
    x, y, w, h = [int(v) for v in safe_rect]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return False
    safe_bbox = (x1, y1, x2 - x1, y2 - y1)
    safe_area = max(1, safe_bbox[2] * safe_bbox[3])
    region_area = max(1, int(plan.region_bbox[2] * plan.region_bbox[3]))
    if safe_area < region_area or safe_area > max(region_area * 8, 120000):
        return False
    text_bbox = plan.text_bbox or (_mask_bbox(plan.text_mask) if plan.text_mask is not None else None)
    if text_bbox is None:
        return False
    tx, ty, tw, th = [int(v) for v in text_bbox]
    if tx < x1 or ty < y1 or tx + tw > x2 or ty + th > y2:
        return False
    local = img_cv[y1:y2, x1:x2]
    sample = np.ones(local.shape[:2], dtype=bool)
    if plan.text_mask is not None:
        tm = plan.text_mask[y1:y2, x1:x2] > 0
        tm = cv2.dilate(
            tm.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ) > 0
        sample &= ~tm
    if int(np.count_nonzero(sample)) < 64:
        return False
    gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(local, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(local, 40, 120)
    gray_std = float(np.std(gray[sample]))
    sat_std = float(np.std(hsv[:, :, 1][sample]))
    edge_density = float(edges[sample].sum()) / max(1, int(np.count_nonzero(sample)) * 255)
    if gray_std > 18.0 or sat_std > 24.0 or edge_density > 0.035:
        plan.debug_metrics["safe_rect_container_rejected"] = {
            "reason": "not_flat",
            "gray_std": round(gray_std, 3),
            "sat_std": round(sat_std, 3),
            "edge_density": round(edge_density, 5),
        }
        return False
    plan.container_bbox = safe_bbox
    plan.container_mask = np.full((safe_bbox[3], safe_bbox[2]), 255, dtype=np.uint8)
    plan.container_confidence = float(min(0.72, max(0.60, safe_conf)))
    plan.container_reason = "existing_cleanup_safe_rect_flat_container"
    plan.debug_metrics["safe_rect_container_adopted"] = {
        "bbox": _bbox_list(safe_bbox),
        "confidence": round(float(plan.container_confidence), 4),
        "gray_std": round(gray_std, 3),
        "sat_std": round(sat_std, 3),
        "edge_density": round(edge_density, 5),
    }
    return True


def _easy_cleanup_large_mask_allowed(
    plan: CleanupPlan,
    policy: CleanupPolicy,
    quality: Dict[str, Any],
) -> bool:
    """Return True when a large raw region ratio is only a tight-bbox artifact."""
    bg_model = str(plan.background_model or "")
    region_class = str(plan.region_class or "")
    mask_region_ratio = float(quality.get("mask_region_ratio", 0.0) or 0.0)
    mask_container_ratio = float(quality.get("mask_container_ratio", 0.0) or 0.0)
    rectangularity = float(quality.get("rectangularity", 0.0) or 0.0)
    border_touch_ratio = float(quality.get("border_touch_ratio", 0.0) or 0.0)

    flat_bg = bg_model in {"flat_light", "flat_colored", "dark_bubble"}
    mild_translucent = (
        bg_model in {"smooth_gradient", "translucent_gradient"}
        and policy.allow_gradient_fill
        and (
            bg_model != "translucent_gradient"
            or policy.cleanup_allow_translucent_caption
        )
        and float(plan.debug_metrics.get("translucent_detail_score", 0.0) or 0.0) < 1.25
    )
    if not (flat_bg or mild_translucent):
        plan.debug_metrics["easy_cleanup_eligible"] = False
        return False

    max_container_ratio = 0.68 if flat_bg else 0.34
    max_region_ratio = 0.50 if flat_bg else 0.42
    max_rectangularity = 0.92 if flat_bg else 0.62
    if region_class in {"speech_bubble", "caption_box"}:
        if (
            plan.container_bbox is None
            or plan.container_mask is None
            or float(plan.container_confidence or 0.0) < 0.45
        ):
            plan.debug_metrics["easy_cleanup_eligible"] = False
            return False
        allowed = (
            0.0 < mask_container_ratio <= max_container_ratio
            and mask_region_ratio <= max_region_ratio
            and border_touch_ratio <= max(0.55, policy.t2_max_border_touch)
            and (
                rectangularity <= max_rectangularity
                or mask_container_ratio <= 0.22
            )
        )
        plan.debug_metrics["easy_cleanup_eligible"] = bool(allowed)
        return bool(allowed)

    if region_class == "text_on_art" and flat_bg:
        allowed = (
            0.0 < mask_region_ratio <= 0.40
            and rectangularity <= 0.70
            and border_touch_ratio <= policy.t2_max_border_touch
        )
        plan.debug_metrics["easy_cleanup_eligible"] = bool(allowed)
        return bool(allowed)

    plan.debug_metrics["easy_cleanup_eligible"] = False
    return False


def _reject_unsafe_cleanup_mask(
    plan: CleanupPlan,
    cleanup: np.ndarray,
    policy: Optional[CleanupPolicy] = None,
) -> Optional[str]:
    policy = policy or CleanupPolicy()

    def _reject(reason: str) -> Optional[str]:
        mode = str(plan.debug_metrics.get("cleanup_override_mode", "") or "")
        if mode == "force_allow" or bool(plan.debug_metrics.get("cleanup_allow_low_confidence", False)):
            plan.debug_metrics["safety_override"] = f"attempt:{reason}"
            return None
        if policy.cleanup_risky_action == "attempt":
            plan.debug_metrics["safety_override"] = f"attempt:{reason}"
            return None
        if policy.cleanup_risky_action == "review" or mode == "force_review":
            plan.skip_reason = reason
            plan.debug_metrics["safety_override"] = f"review:{reason}"
            plan.debug_metrics["review_required_after_cleanup"] = True
            return None
        return reason

    metrics = _cleanup_mask_metrics(cleanup, plan.region_bbox, plan.text_bbox)
    plan.debug_metrics["mask"] = metrics
    quality_container = None
    if plan.container_mask is not None and plan.container_bbox is not None:
        quality_container = normalize_mask_to_image(
            plan.container_mask, plan.container_bbox, cleanup.shape
        )
    # Pass 1: for high-confidence speech/caption bubbles, evaluate border-touch
    # against the validated container_bbox instead of the tight YOLO region_bbox.
    # SFX / text_on_art / busy_art keep the strict region_bbox gate.
    safety_bbox = _select_safety_bbox(plan)
    quality = _compute_mask_quality_metrics(
        cleanup,
        quality_container,
        plan.region_bbox,
        plan.text_bbox,
        safety_bbox=safety_bbox,
    )
    plan.debug_metrics["quality"] = quality
    mask_area = int(metrics["mask_area"])
    if mask_area <= 0:
        return "empty_cleanup_mask"

    if int(quality.get("component_count", 0) or 0) == 0:
        return "empty_cleanup_mask"

    mask_region_ratio = float(quality.get("mask_region_ratio", metrics["mask_region_ratio"]) or 0.0)
    mask_container_ratio = float(quality.get("mask_container_ratio", 0.0) or 0.0)
    mask_text_ratio = float(metrics["mask_text_ratio"])
    mask_box = quality.get("mask_bbox") or metrics.get("mask_bbox")
    border_touch_ratio = float(quality.get("border_touch_ratio", 0.0) or 0.0)
    rectangularity = float(quality.get("rectangularity", 0.0) or 0.0)
    long_bar_score = float(quality.get("long_bar_score", 0.0) or 0.0)
    easy_large_mask = _easy_cleanup_large_mask_allowed(plan, policy, quality)
    flat_bg = str(plan.background_model or "") in {"flat_light", "flat_colored", "dark_bubble"}

    if plan.region_class in ("speech_bubble", "sfx", "text_on_art"):
        if (
            plan.region_class == "speech_bubble"
            and plan.container_bbox is not None
            and float(plan.container_confidence or 0.0) >= 0.35
            and mask_container_ratio > policy.cleanup_max_mask_container_ratio
            and not easy_large_mask
        ):
            return _reject(f"cleanup_mask_too_large_container_ratio({mask_container_ratio:.2f})")
        if mask_region_ratio > policy.cleanup_max_mask_region_ratio:
            # Pass 1b: container-aware large-mask safety gate.
            # For YOLO speech bubbles the region_bbox is the tight text crop, so
            # mask_region_ratio is computed against a rectangle that may cover only
            # a fraction of the actual bubble area.  When a validated container is
            # present and mask_container_ratio is small (mask is only a thin slice
            # of the real bubble interior) and the mask is non-rectangular and
            # doesn't touch the container border, the over-size signal is an
            # artefact of bbox tightness — not a dangerous full-region fill.
            # sfx / text_on_art are NOT eligible; they have no container.
            _container_override = (
                plan.region_class == "speech_bubble"
                and plan.container_bbox is not None
                and float(plan.container_confidence or 0.0) >= 0.60
                and quality.get("safety_bbox_source") == "container"
                and mask_container_ratio <= 0.12
                and rectangularity <= 0.35
                and border_touch_ratio <= policy.t2_max_border_touch
            )
            if _container_override:
                debug_print(
                    f"[CLEANUP_OVERRIDE] container_large_mask_override=True "
                    f"page={plan.page_index} region={plan.region_id} "
                    f"mask_region_ratio={mask_region_ratio:.4f} "
                    f"mask_container_ratio={mask_container_ratio:.4f} "
                    f"container_conf={float(plan.container_confidence or 0.0):.2f} "
                    f"rectangularity={rectangularity:.3f} "
                    f"border_touch_ratio={border_touch_ratio:.3f}"
                )
            elif easy_large_mask:
                plan.debug_metrics["safety_override"] = (
                    f"easy_cleanup_large_mask({mask_region_ratio:.2f})"
                )
            else:
                return _reject(f"cleanup_mask_too_large_region_ratio({mask_region_ratio:.2f})")
        if border_touch_ratio > policy.cleanup_max_border_touch_ratio:
            return _reject(f"cleanup_mask_border_collision({border_touch_ratio:.2f})")
        easy_flat_rect_band = (
            easy_large_mask
            and flat_bg
            and mask_container_ratio <= 0.68
            and rectangularity <= 0.92
            and mask_region_ratio <= 0.50
            and border_touch_ratio <= policy.t2_max_border_touch
        )
        if (
            rectangularity > policy.cleanup_max_rectangularity
            and mask_region_ratio > 0.18
            and not (easy_large_mask and mask_container_ratio <= 0.35)
            and not easy_flat_rect_band
        ):
            return _reject(f"cleanup_mask_rectangular_fill({rectangularity:.2f})")
        if long_bar_score > 20.0 and mask_area > max(32, int(plan.region_bbox[2] * plan.region_bbox[3] * 0.02)):
            return _reject(f"cleanup_mask_long_bar({long_bar_score:.1f})")
        if mask_text_ratio > (3.20 if easy_large_mask else 1.85):
            return _reject(f"cleanup_mask_too_large_text_ratio({mask_text_ratio:.2f})")
        # Pass 1: skip the bbox-equality gate when safety evaluated against
        # the container (mask touching the tight region_bbox is expected for
        # validated bubbles).
        if safety_bbox is None and _bbox_close(mask_box, plan.region_bbox, tol=8):
            debug_print(
                f"cleanup_mask_bbox_equals_region_bbox page={plan.page_index} "
                f"region={plan.region_id} mask_bbox={mask_box} "
                f"region_bbox={plan.region_bbox}"
            )
            return _reject("cleanup_mask_bbox_matches_region_bbox")
    elif plan.region_class == "caption_box":
        if (
            plan.container_bbox is not None
            and float(plan.container_confidence or 0.0) >= 0.35
            and mask_container_ratio > policy.cleanup_max_mask_container_ratio
            and not easy_large_mask
        ):
            return _reject(f"caption_cleanup_mask_too_large_container({mask_container_ratio:.2f})")
        if mask_region_ratio > min(0.45, policy.cleanup_max_mask_region_ratio * 1.6):
            # Pass 1b: same container-aware override for caption boxes.
            _container_override = (
                plan.container_bbox is not None
                and float(plan.container_confidence or 0.0) >= 0.60
                and quality.get("safety_bbox_source") == "container"
                and mask_container_ratio <= 0.12
                and rectangularity <= 0.35
                and border_touch_ratio <= policy.t2_max_border_touch
            )
            if _container_override:
                debug_print(
                    f"[CLEANUP_OVERRIDE] container_large_mask_override=True "
                    f"page={plan.page_index} region={plan.region_id} "
                    f"mask_region_ratio={mask_region_ratio:.4f} "
                    f"mask_container_ratio={mask_container_ratio:.4f} "
                    f"container_conf={float(plan.container_confidence or 0.0):.2f} "
                    f"rectangularity={rectangularity:.3f} "
                    f"border_touch_ratio={border_touch_ratio:.3f}"
                )
            elif easy_large_mask:
                plan.debug_metrics["safety_override"] = (
                    f"easy_caption_large_mask({mask_region_ratio:.2f})"
                )
            else:
                return _reject(f"caption_cleanup_mask_too_large({mask_region_ratio:.2f})")
        if (
            rectangularity > max(0.88, policy.cleanup_max_rectangularity)
            and mask_region_ratio > 0.28
            and not (easy_large_mask and mask_container_ratio <= 0.35)
        ):
            return _reject(f"caption_cleanup_mask_full_rect({rectangularity:.2f})")
        if safety_bbox is None and _bbox_close(mask_box, plan.region_bbox, tol=6):
            return _reject("caption_cleanup_mask_bbox_matches_region_bbox")
    return None


def _refresh_cleanup_quality(plan: CleanupPlan, img_cv: np.ndarray) -> Dict[str, Any]:
    quality_container = None
    if plan.container_mask is not None and plan.container_bbox is not None:
        quality_container = normalize_mask_to_image(
            plan.container_mask, plan.container_bbox, img_cv.shape
        )
    quality_mask = plan.cleanup_mask if plan.cleanup_mask is not None else plan.text_mask
    # Pass 1: use the same safety_bbox selection as _reject_unsafe_cleanup_mask
    # so that final debug metrics / tier classification are consistent with the
    # rejection decision (border_touch_ratio reflects container bounds, not the
    # tight YOLO region_bbox, for validated speech/caption bubbles).
    safety_bbox = _select_safety_bbox(plan)
    quality = _compute_mask_quality_metrics(
        quality_mask,
        quality_container,
        plan.region_bbox,
        plan.text_bbox,
        safety_bbox=safety_bbox,
    )
    plan.debug_metrics["quality"] = quality
    return quality


def _derive_safe_rect_from_container_mask(
    container_mask: Optional[np.ndarray],
    container_bbox: Optional[Tuple[int, int, int, int]],
    region_class: str,
    image_shape: Tuple[int, ...],
) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
    """
    Derive a conservative text-placement rectangle from a validated container.

    The returned rect is in full-image coordinates. It is intentionally based
    on the bubble/caption interior, never the cleanup/text mask.
    """
    if container_mask is None or container_bbox is None:
        return None, 0.0, "safe_rect_no_container"

    h_img, w_img = image_shape[:2]
    cx, cy, cw, ch = [int(v) for v in container_bbox]
    if cw < 24 or ch < 16:
        return None, 0.0, "safe_rect_container_too_small"

    try:
        if container_mask.shape[:2] == (h_img, w_img):
            full_mask = (container_mask > 0).astype(np.uint8)
        else:
            full_mask = normalize_mask_to_image(container_mask, container_bbox, (h_img, w_img)) > 0
            full_mask = full_mask.astype(np.uint8)
    except Exception as exc:
        return None, 0.0, f"safe_rect_mask_normalize_failed:{exc}"

    x1, y1 = max(0, cx), max(0, cy)
    x2, y2 = min(w_img, cx + cw), min(h_img, cy + ch)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, "safe_rect_bbox_out_of_bounds"

    roi = (full_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if not np.any(roi):
        return None, 0.0, "safe_rect_empty_container"
    is_caption = str(region_class or "").lower() == "caption_box"
    fill_ratio = float(np.count_nonzero(roi)) / float(max(1, roi.size))
    if not is_caption and fill_ratio > 0.92:
        return None, 0.25, "safe_rect_rejected_full_rect_speech_container"

    labels, stats_count = None, 0
    try:
        num, labels, stats, _cent = cv2.connectedComponentsWithStats(roi, 8)
        stats_count = num
        if num <= 1:
            return None, 0.0, "safe_rect_no_component"
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = (labels == largest_idx).astype(np.uint8) * 255
    except Exception:
        component = roi.astype(np.uint8) * 255

    min_dim = max(1, min(x2 - x1, y2 - y1))
    if is_caption:
        erode_px = max(2, min(4, int(round(min_dim * 0.025))))
    else:
        erode_px = max(4, min(8, int(round(min_dim * 0.040))))
    k = max(3, erode_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(component, kernel, iterations=1)
    if not np.any(eroded):
        eroded = component
        erode_px = 0

    local_bbox = _mask_bbox(eroded)
    if local_bbox is None:
        return None, 0.0, "safe_rect_eroded_empty"
    lx, ly, lw, lh = local_bbox
    gx, gy = x1 + lx, y1 + ly

    inset_min = 4 if is_caption else 8
    inset_ratio = 0.05 if is_caption else 0.08
    inset_x = max(inset_min, int(round(lw * inset_ratio)))
    inset_y = max(inset_min, int(round(lh * inset_ratio)))
    rx = max(0, gx + inset_x)
    ry = max(0, gy + inset_y)
    rw = max(1, lw - inset_x * 2)
    rh = max(1, lh - inset_y * 2)
    if rx + rw > w_img:
        rw = max(1, w_img - rx)
    if ry + rh > h_img:
        rh = max(1, h_img - ry)
    if rw < 24 or rh < 16:
        return None, 0.0, "safe_rect_too_small"

    eroded_full = np.zeros((h_img, w_img), dtype=np.uint8)
    eroded_full[y1:y2, x1:x2] = (eroded > 0).astype(np.uint8)
    rect_mask = eroded_full[ry:ry + rh, rx:rx + rw] > 0
    rect_area = max(1, rw * rh)
    coverage = float(np.count_nonzero(rect_mask)) / float(rect_area)
    required_coverage = 0.70 if is_caption else 0.80
    if coverage < required_coverage:
        return None, min(0.44, coverage), f"safe_rect_low_coverage:{coverage:.2f}"

    area_ratio = float(rect_area) / float(max(1, cw * ch))
    confidence = 0.35 + min(1.0, coverage) * 0.45 + min(area_ratio, 0.75) * 0.25
    if not is_caption and area_ratio > 0.85:
        confidence *= 0.60
    if not is_caption and stats_count <= 2 and area_ratio > 0.80:
        confidence *= 0.75
    confidence = float(max(0.0, min(1.0, confidence)))
    if confidence < 0.45:
        return None, confidence, f"safe_rect_low_confidence:{confidence:.2f}"

    reason = (
        f"safe_rect_ok(coverage={coverage:.2f},area={area_ratio:.2f},"
        f"erode={erode_px})"
    )
    return (int(rx), int(ry), int(rw), int(rh)), confidence, reason


def classify_cleanup_tier(
    plan: CleanupPlan,
    policy: Optional[CleanupPolicy] = None,
    mode: Optional[str] = None,
) -> Tuple[int, str, str]:
    """
    Map a completed CleanupPlan to the Pass 2 balanced cleanup tier.

    Returns (tier, status, reason). Tier 1 is auto-safe, Tier 2 is processed
    but review-worthy, Tier 3 is skipped/unsafe.
    """
    if policy is None:
        policy = CleanupPolicy(cleanup_mode=str(mode or "balanced").strip().lower())
        policy._apply_mode_thresholds()
    quality = plan.debug_metrics.get("quality", {}) or {}
    mask_region_ratio = float(quality.get("mask_region_ratio", 0.0) or 0.0)
    border_touch_ratio = float(quality.get("border_touch_ratio", 0.0) or 0.0)
    strategy = plan.cleanup_strategy or "skip"
    skip_reason = plan.skip_reason or ""
    ran_cleanup = (
        strategy not in ("skip", "review")
        and plan.cleanup_mask is not None
        and bool(np.any(plan.cleanup_mask))
    )

    if not ran_cleanup:
        if plan.region_class == "sfx" and strategy == "skip":
            return 3, "skipped_sfx_default_policy", "SFX cleanup is disabled by default."
        if plan.region_class == "text_on_art" and strategy == "skip":
            return 3, "skipped_text_over_art_default_policy", "Text-over-art cleanup is disabled by default."
        if plan.background_model == "busy_art" and strategy == "skip":
            return 3, "skipped_busy_background_default_policy", "Busy-background cleanup is disabled by default."
        if "texture_inpaint_disabled" in skip_reason:
            return 3, "skipped_texture_inpaint_disabled", skip_reason
        if "no_ocr_text" in skip_reason or "no_text" in skip_reason:
            return 3, "skipped_no_text_signal", skip_reason or "No cleanup text signal."
        if "too_large" in skip_reason or "bbox_matches" in skip_reason:
            return 3, "skipped_large_mask", skip_reason
        if "border_collision" in skip_reason:
            return 3, "skipped_border_collision", skip_reason
        if plan.text_mask_confidence < policy.t2_text_conf:
            return 3, "skipped_mask_low_confidence", f"Text mask confidence below {policy.t2_text_conf:.2f}."
        return 3, "skipped_unknown", skip_reason or "Cleanup skipped or sent to review."

    if plan.region_class == "sfx":
        return 2, "experimental_sfx_cleanup_review", "Experimental SFX cleanup requires review."
    if plan.region_class == "text_on_art":
        return 2, "experimental_art_cleanup_review", "Experimental text-over-art cleanup requires review."
    if plan.background_model == "busy_art":
        return 2, "experimental_busy_background_cleanup_review", "Experimental busy-background cleanup requires review."
    if bool(plan.debug_metrics.get("review_required_after_cleanup", False)):
        return 2, "cleanup_forced_review", skip_reason or "Cleanup was marked for review by override or safety policy."
    if skip_reason == "cleanup_residual_text_remains":
        return 2, "cleanup_residual_text_remains", "Residual text-like pixels remain after cleanup."

    if plan.text_mask_confidence < policy.t2_text_conf:
        return 3, "skipped_mask_low_confidence", f"Text mask confidence below {policy.t2_text_conf:.2f}."
    if skip_reason or mask_region_ratio >= policy.t2_max_mask_region_ratio:
        if bool(plan.debug_metrics.get("easy_cleanup_eligible", False)):
            return 2, "auto_cautious_cleaned_review", "Easy cleanup applied with container-aware large-mask override."
        status = "skipped_large_mask" if mask_region_ratio >= policy.t2_max_mask_region_ratio else "skipped_unknown"
        return 3, status, skip_reason or f"Mask region ratio {mask_region_ratio:.3f} is unsafe."

    gradient_error = plan.debug_metrics.get("gradient_fit_error")
    is_tier1_strategy = strategy in ("flat_fill", "caption_plain_fill")
    if strategy == "gradient_fill" and gradient_error is not None:
        is_tier1_strategy = float(gradient_error) < 10.0
    has_container = plan.container_confidence >= policy.t1_container_conf or plan.region_class == "caption_box"
    if (
        is_tier1_strategy
        and plan.text_mask_confidence >= policy.t1_text_conf
        and has_container
        and mask_region_ratio < policy.t1_max_mask_region_ratio
        and border_touch_ratio < policy.t1_max_border_touch
    ):
        if strategy == "gradient_fill":
            return 1, "auto_safe_cleaned", "Gradient color-plane cleanup passed safe mask gates."
        return 1, "auto_safe_cleaned", "Flat/plain cleanup passed safe mask gates."

    if strategy == "gradient_fill":
        return 2, "auto_cautious_cleaned_review", "Gradient cleanup applied; review requested unless color-plane fit is very strong."

    if (
        plan.text_mask_confidence >= policy.t2_text_conf
        and mask_region_ratio < policy.t2_max_mask_region_ratio
        and border_touch_ratio < policy.t2_max_border_touch
    ):
        return 2, "auto_cautious_cleaned_review", "Cleanup applied but did not meet all Tier 1 gates."

    if bool(plan.debug_metrics.get("easy_cleanup_eligible", False)):
        return 2, "auto_cautious_cleaned_review", "Easy cleanup applied with container-aware safety gates."

    return 3, "skipped_unknown", "Cleanup outcome did not meet safe or cautious gates."


def _write_cleanup_metadata_to_block(
    block: Any,
    plan: CleanupPlan,
    img_cv: np.ndarray,
    policy: Optional[CleanupPolicy] = None,
    cleanup_mode: str = "balanced",
) -> int:
    """Persist cleanup outcome and reusable cleanup geometry on the OCRBlock."""
    policy = policy or CleanupPolicy(cleanup_mode=str(cleanup_mode or "balanced").strip().lower())
    policy._apply_mode_thresholds()
    quality = _refresh_cleanup_quality(plan, img_cv)
    tier, status, reason = classify_cleanup_tier(plan, policy=policy)
    block.cleanup_tier = tier
    block.cleanup_status = status
    block.cleanup_reason = reason
    block.cleanup_meta = {
        "tier": int(tier),
        "status": str(status),
        "reason": str(reason),
        "review_required": bool(tier == 2 and policy.require_review_for_tier2),
        "diagnostic_only": bool(plan.debug_metrics.get("diagnostic_only", False)),
        "diagnostic_cleanup_ran": bool(plan.debug_metrics.get("diagnostic_cleanup_ran", False)),
        "destructive_cleanup_executed": bool(plan.debug_metrics.get("destructive_cleanup_executed", False)),
        "production_patch_accepted": bool(plan.debug_metrics.get("production_patch_accepted", False)),
        "proposal_valid": bool(plan.debug_metrics.get("proposal_valid", False)),
        "proposal_failure_reason": str(plan.debug_metrics.get("proposal_failure_reason", "") or ""),
        "cleanup_failure_reason": str(plan.debug_metrics.get("cleanup_failure_reason", "") or ""),
        "gate_violation": bool(plan.debug_metrics.get("gate_violation", False)),
        "residual_text_visible": bool(plan.debug_metrics.get("residual_text_visible", False)),
        "visual_quality_ok": bool(plan.debug_metrics.get("visual_quality_ok", True)),
        "fill_patch_visible": bool(plan.debug_metrics.get("fill_patch_visible", False)),
        "cleanup_effective": bool(plan.debug_metrics.get("cleanup_effective", False)),
    }

    if plan.container_bbox is not None and plan.container_confidence >= 0.40:
        block.cleanup_container_bbox = tuple(int(v) for v in plan.container_bbox)
        block.cleanup_container_confidence = float(plan.container_confidence)
    safe_rect, safe_conf, safe_reason = _derive_safe_rect_from_container_mask(
        plan.container_mask,
        plan.container_bbox,
        plan.region_class,
        img_cv.shape,
    )
    plan.debug_metrics["cleanup_safe_rect"] = safe_rect
    plan.debug_metrics["cleanup_safe_rect_confidence"] = float(safe_conf)
    plan.debug_metrics["cleanup_safe_rect_reason"] = safe_reason
    if safe_rect is not None and safe_conf >= 0.45:
        block.cleanup_safe_rect = tuple(int(v) for v in safe_rect)
        block.cleanup_safe_rect_confidence = float(safe_conf)
    if plan.text_bbox is not None and getattr(block, "computed_text_bbox", None) is None:
        block.computed_text_bbox = tuple(int(v) for v in plan.text_bbox)

    if (
        tier == 2
        and plan.cleanup_strategy not in ("skip", "review")
        and hasattr(block, "flag")
        and not bool(getattr(block, "is_flagged", False))
    ):
        block.flag(
            "auto_cautious_cleanup",
            {
                "tier": 2,
                "strategy": plan.cleanup_strategy,
                "text_conf": round(float(plan.text_mask_confidence), 3),
                "cont_conf": round(float(plan.container_confidence), 3),
                "mask_ratio": round(float(quality.get("mask_region_ratio", 0.0) or 0.0), 3),
            },
        )

    return tier


def _candidate_ocr_contrast(
    img_cv: np.ndarray,
    boxes: List[Any],
    region_bbox: Tuple[int, int, int, int],
    pad: int = 4,
) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Method A: OCR polygon contrast.

    For each OCR box, sample the border ring as the background estimate, then
    threshold interior pixels by Lab distance from that background.  Works for
    any text colour on any bubble colour.
    """
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    rx1, ry1 = max(0, rx), max(0, ry)
    rx2, ry2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, 0.0, "empty_region"

    lab_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2Lab).astype(np.float32)
    mask    = np.zeros(img_cv.shape[:2], dtype=np.uint8)

    if not boxes:
        return None, 0.0, "no_ocr_boxes"

    covered_area = 0
    for box in boxes:
        try:
            pts = np.array(box, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
        except Exception:
            continue

        bx, by, bw, bh = cv2.boundingRect(pts)
        bx1, by1 = max(0, bx - pad), max(0, by - pad)
        bx2, by2 = min(w_img, bx + bw + pad), min(h_img, by + bh + pad)
        if bx2 <= bx1 or by2 <= by1:
            continue

        crop_lab          = lab_img[by1:by2, bx1:bx2]
        crop_h, crop_w    = crop_lab.shape[:2]

        # Background estimate: border ring of the crop
        border_mask        = np.zeros((crop_h, crop_w), dtype=bool)
        bord               = max(1, int(min(crop_h, crop_w) * 0.15))
        border_mask[:bord, :]  = True
        border_mask[-bord:, :] = True
        border_mask[:, :bord]  = True
        border_mask[:, -bord:] = True
        border_pixels = crop_lab[border_mask]
        if border_pixels.shape[0] < 5:
            border_pixels = crop_lab.reshape(-1, 3)[:10]
        bg_lab = np.median(border_pixels, axis=0)

        diff   = np.sqrt(np.sum((crop_lab - bg_lab[None, None, :]) ** 2, axis=2))
        border_diff = np.sqrt(
            np.sum((border_pixels - bg_lab[None, :]) ** 2, axis=1)
        )
        thresh     = max(20.0, float(np.percentile(border_diff, 75)) * 2.0)
        text_local = np.where(diff > thresh, 255, 0).astype(np.uint8)

        # Keep only components that overlap the OCR polygon
        poly_mask_local          = np.zeros((crop_h, crop_w), dtype=np.uint8)
        shifted                  = pts.copy()
        shifted[:, 0]           -= bx1
        shifted[:, 1]           -= by1
        cv2.fillPoly(poly_mask_local, [shifted], 255)
        poly_dilated = cv2.dilate(
            poly_mask_local,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        text_local = cv2.bitwise_and(text_local, poly_dilated)

        kern       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        text_local = cv2.morphologyEx(text_local, cv2.MORPH_OPEN, kern)

        mask[by1:by2, bx1:bx2] = cv2.bitwise_or(
            mask[by1:by2, bx1:bx2], text_local
        )
        covered_area += int(np.count_nonzero(poly_mask_local))

    if not np.any(mask):
        return None, 0.0, "no_contrast_pixels"

    mask_in_region = int(np.count_nonzero(mask[ry1:ry2, rx1:rx2]))
    ocr_area       = max(1, covered_area)

    # FIX-3: use stroke-ratio scoring instead of plain coverage fraction.
    stroke_ratio = mask_in_region / ocr_area

    # Reject masks that are suspiciously dense (likely the whole OCR box fill).
    if stroke_ratio > 0.90:
        return None, 0.2, "full_box_coverage_suspect"

    area_score = _stroke_area_score(stroke_ratio)
    confidence = float(np.clip(area_score * 0.85, 0.0, 0.85))
    return mask, confidence, f"ocr_contrast(stroke={stroke_ratio:.2f})"


def _candidate_multichannel_threshold(
    img_cv: np.ndarray,
    boxes: List[Any],
    region_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Method B: Multi-channel thresholding.

    Converts to Lab + HSV, tries Otsu threshold on L, saturation, and chroma,
    picks the candidate with best OCR overlap.  Handles dark, white, red, and
    coloured text.
    """
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, "empty_region"

    roi     = img_cv[y1:y2, x1:x2]
    lab     = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_h, roi_w = roi.shape[:2]

    # Build OCR box mask within roi for scoring
    ocr_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
    if boxes:
        for box in boxes:
            try:
                pts         = np.array(box, dtype=np.int32).copy()
                pts[:, 0]   = np.clip(pts[:, 0] - x1, 0, roi_w - 1)
                pts[:, 1]   = np.clip(pts[:, 1] - y1, 0, roi_h - 1)
                cv2.fillPoly(ocr_local, [pts], 255)
            except Exception:
                pass
    if not np.any(ocr_local):
        return None, 0.0, "no_ocr_boxes"

    L_ch  = lab[:, :, 0]
    a_ch  = lab[:, :, 1]
    b_ch  = lab[:, :, 2]
    sat   = hsv[:, :, 1]

    # FIX-1: OpenCV Lab encodes a/b as unsigned bytes centred at 128, not 0.
    # Without subtracting 128, almost every pixel gets a huge spurious chroma
    # value, making the channel useless for coloured-text detection.
    chroma = np.sqrt(
        (a_ch.astype(np.float32) - 128.0) ** 2 +
        (b_ch.astype(np.float32) - 128.0) ** 2
    ).clip(0, 255).astype(np.uint8)

    candidates: List[Tuple[np.ndarray, float, str]] = []
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    for ch, name, invert_pairs in [
        (L_ch,  "L",      [(False, True)]),
        (sat,   "Sat",    [(True, False)]),
        (chroma,"Chroma", [(True, False), (False, True)]),
    ]:
        ch_u8 = np.clip(ch, 0, 255).astype(np.uint8)
        for inv in [p[0] for p in invert_pairs] + [p[1] for p in invert_pairs]:
            src      = 255 - ch_u8 if inv else ch_u8
            _, thr   = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr      = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kern)
            in_ocr   = int(np.count_nonzero(cv2.bitwise_and(thr, ocr_local)))
            ocr_area = max(1, int(np.count_nonzero(ocr_local)))
            # FIX-3: store raw OCR-overlap for stroke-ratio scoring below.
            raw_score = in_ocr / ocr_area
            if raw_score > 0.04:
                candidates.append((thr, raw_score, f"{name}{'_inv' if inv else ''}"))

    if not candidates:
        return None, 0.0, "no_threshold_channel_worked"

    candidates.sort(key=lambda t: -t[1])
    best_thr, best_raw, best_name = candidates[0]

    # Project back to full image
    mask     = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = best_thr

    # Reject full-rectangle results
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    mask_area = int(np.count_nonzero(best_thr))
    if mask_area > 0.88 * bbox_area:
        return None, 0.1, f"full_rect_{best_name}"

    # FIX-3: stroke-ratio scoring (mask pixels vs OCR polygon area).
    ocr_area     = max(1, int(np.count_nonzero(ocr_local)))
    stroke_ratio = int(np.count_nonzero(best_thr)) / ocr_area
    area_score   = _stroke_area_score(stroke_ratio)
    confidence   = float(np.clip(area_score * 0.80, 0.0, 0.80))
    return mask, confidence, f"multichannel_{best_name}(stroke={stroke_ratio:.2f})"


def _candidate_edge_components(
    img_cv: np.ndarray,
    boxes: List[Any],
    region_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Method C: Edge + connected component filtering.

    Runs Canny on the region, closes edges into glyph clusters, labels
    connected components, and keeps only those that overlap the OCR box area
    and have plausible size.
    """
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, "empty_region"

    roi          = img_cv[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    gray_roi     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # OCR box mask in ROI coords
    ocr_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
    if boxes:
        for box in boxes:
            try:
                pts         = np.array(box, dtype=np.int32).copy()
                pts[:, 0]   = np.clip(pts[:, 0] - x1, 0, roi_w - 1)
                pts[:, 1]   = np.clip(pts[:, 1] - y1, 0, roi_h - 1)
                cv2.fillPoly(ocr_local, [pts], 255)
            except Exception:
                pass
    if not np.any(ocr_local):
        return None, 0.0, "no_ocr_boxes"

    ocr_area      = int(np.count_nonzero(ocr_local))
    char_size_est = max(64, ocr_area // max(1, len(boxes) * 3))

    # Multi-scale Canny union
    edges1 = cv2.Canny(gray_roi, 20, 60)
    edges2 = cv2.Canny(gray_roi, 40, 120)
    edges  = cv2.bitwise_or(edges1, edges2)

    kern5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kern3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kern5, iterations=2)
    dilated = cv2.dilate(closed, kern3, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    good = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for i in range(1, n):
        ca = int(stats[i, cv2.CC_STAT_AREA])
        if ca < 16:
            continue
        max_area = max(rw * rh * 0.70, char_size_est * 80)
        if ca > max_area:
            continue
        comp    = np.where(labels == i, 255, 0).astype(np.uint8)
        overlap = int(np.count_nonzero(cv2.bitwise_and(comp, ocr_local)))
        if overlap < 4:
            continue
        good = cv2.bitwise_or(good, comp)

    if not np.any(good):
        return None, 0.0, "no_filtered_components"

    if int(np.count_nonzero(good)) > 0.88 * (roi_w * roi_h):
        return None, 0.1, "full_rect_components"

    # FIX-3: stroke-ratio scoring.
    covered      = int(np.count_nonzero(cv2.bitwise_and(good, ocr_local)))
    stroke_ratio = covered / max(1, ocr_area)
    area_score   = _stroke_area_score(stroke_ratio)
    confidence   = float(np.clip(area_score * 0.75, 0.0, 0.75))

    mask         = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = good
    return mask, confidence, f"edge_components(stroke={stroke_ratio:.2f})"


def _candidate_region_cv_no_bbox(
    img_cv: np.ndarray,
    region_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], float, str]:
    """Build a conservative glyph mask when Qwen read text but no box."""
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, "empty_region"

    roi = img_cv[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 8 or roi_w < 8:
        return None, 0.0, "region_too_small"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    border_w = max(2, min(8, roi_w // 8, roi_h // 8))
    border = np.zeros((roi_h, roi_w), dtype=bool)
    border[:border_w, :] = True
    border[-border_w:, :] = True
    border[:, :border_w] = True
    border[:, -border_w:] = True
    bg_lab = np.median(lab[border], axis=0)
    bg_gray = float(np.median(gray[border]))
    bg_sat = float(np.median(hsv[:, :, 1][border]))
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    bg_chroma = float(np.median(chroma[border]))
    diff = np.linalg.norm(lab - bg_lab, axis=2)

    dark = gray < max(90, bg_gray - 34)
    light = (gray > min(245, bg_gray + 42)) & (diff > 18)
    saturated = (hsv[:, :, 1] > max(45, bg_sat + 24)) & (chroma > bg_chroma + 12)
    contrast = diff > 26
    edges = cv2.Canny(gray, 40, 120)
    edge_band = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    raw = ((dark | light | saturated | contrast) & (edge_band | (diff > 38))).astype(np.uint8) * 255
    high_conf = ((diff > 46.0) | (np.abs(gray - bg_gray) > 34.0)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    # Keep tiny high-contrast punctuation that a 2x2 open would erase.
    raw = cv2.bitwise_or(opened, cv2.bitwise_and(raw, high_conf))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    region_area = max(1, roi_w * roi_h)
    min_area = max(2, int(region_area * 0.00006))
    max_area = max(24, int(region_area * 0.08))
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw > roi_w * 0.88 or bh > roi_h * 0.88:
            continue
        density = area / max(1, bw * bh)
        tiny_dense_punctuation = (
            area <= max(48, int(region_area * 0.003))
            and bw <= max(8, int(roi_w * 0.06))
            and bh <= max(8, int(roi_h * 0.16))
        )
        if density < 0.04 or (density > 0.82 and area > 16 and not tiny_dense_punctuation):
            continue
        kept[labels == label] = 255

    area = int(np.count_nonzero(kept))
    if area <= 0:
        return None, 0.0, "no_cv_components"
    ratio = area / region_area
    conf = 0.48
    if 0.003 <= ratio <= 0.14:
        conf = 0.58
    elif ratio > 0.22:
        conf = 0.22

    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = kept
    return mask, conf, f"region_cv_no_bbox(area={ratio:.3f})"


def _candidate_dark_caption_light_text(
    img_cv: np.ndarray,
    region_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], float, str]:
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, "dark_caption_empty_region"

    roi = img_cv[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 12 or roi_w < 24:
        return None, 0.0, "dark_caption_too_small"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    border_w = max(2, min(8, roi_w // 10, roi_h // 8))
    border = np.zeros((roi_h, roi_w), dtype=bool)
    border[:border_w, :] = True
    border[-border_w:, :] = True
    border[:, :border_w] = True
    border[:, -border_w:] = True
    bg_gray = float(np.median(gray[border]))
    bg_lab = np.median(lab[border], axis=0)
    if bg_gray > 118.0:
        return None, 0.0, f"dark_caption_not_dark(bg={bg_gray:.1f})"

    lab_dist = np.sqrt(np.sum((lab - bg_lab[None, None, :]) ** 2, axis=2))
    light_core = ((gray > max(126.0, bg_gray + 52.0)) & (lab_dist > 20.0)).astype(np.uint8) * 255
    if int(np.count_nonzero(light_core)) < 8:
        return None, 0.0, "dark_caption_no_light_text"

    light_core = cv2.morphologyEx(light_core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    near_text = cv2.dilate(
        light_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    dark_outline = ((gray < max(36.0, bg_gray - 10.0)) & (near_text > 0)).astype(np.uint8) * 255
    raw = cv2.bitwise_or(near_text, dark_outline)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    region_area = max(1, roi_w * roi_h)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > max(80, int(region_area * 0.24)):
            continue
        if bw > roi_w * 0.92 or bh > roi_h * 0.82:
            continue
        kept[labels == label] = 255

    area = int(np.count_nonzero(kept))
    if area <= 0:
        return None, 0.0, "dark_caption_no_components"
    ratio = area / region_area
    if ratio > 0.26:
        return None, 0.12, f"dark_caption_mask_too_large(area={ratio:.3f})"
    if ratio < 0.006:
        conf = 0.24
    elif ratio <= 0.22:
        conf = 0.68
    else:
        conf = 0.42

    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = kept
    return mask, conf, f"dark_caption_light_text(area={ratio:.3f},bg={bg_gray:.1f})"


def _candidate_container_first_glyph_mask(
    img_cv: np.ndarray,
    block: Any,
    region_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], float, str]:
    """Infer a light bubble/caption interior first, then keep only glyph strokes."""
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, "container_first_empty_region"

    crop = img_cv[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch < 8 or cw < 8:
        return None, 0.0, "container_first_too_small"

    container = None
    container_conf = 0.0
    bub_mask = getattr(block, "bubble_mask", None)
    bub_bbox = getattr(block, "bubble_bbox", None)
    if bub_mask is not None and bub_bbox is not None:
        bx, by, bw, bh = bub_bbox
        if abs(bx - x1) <= 3 and abs(by - y1) <= 3:
            try:
                local = bub_mask
                if local.shape != (ch, cw):
                    local = cv2.resize(local, (cw, ch), interpolation=cv2.INTER_NEAREST)
                fill = float(np.count_nonzero(local)) / max(1, ch * cw)
                if 0.18 <= fill <= 0.98:
                    container = (local > 0).astype(np.uint8) * 255
                    container_conf = 0.72 if fill < 0.92 else 0.48
            except Exception:
                container = None

    if container is None:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        light = (
            ((hsv[:, :, 2] > 178) & (hsv[:, :, 1] < 58))
            | ((gray > 188) & (hsv[:, :, 1] < 75))
            | ((lab[:, :, 0] > 178) & (hsv[:, :, 1] < 68))
        ).astype(np.uint8) * 255
        light = cv2.morphologyEx(
            light,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=2,
        )
        n, labels, stats, cent = cv2.connectedComponentsWithStats(light, 8)
        cx, cy = cw / 2.0, ch / 2.0
        best = 0
        best_score = -1.0
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(32, int(ch * cw * 0.08)):
                continue
            bx = int(stats[label, cv2.CC_STAT_LEFT])
            by = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if bw < max(6, cw * 0.25) or bh < max(6, ch * 0.20):
                continue
            dist = abs(float(cent[label][0]) - cx) / max(1, cw) + abs(float(cent[label][1]) - cy) / max(1, ch)
            score = area / max(1, ch * cw) - dist * 0.35
            if score > best_score:
                best_score = score
                best = label
        if best > 0:
            container = np.where(labels == best, 255, 0).astype(np.uint8)
            container = cv2.morphologyEx(
                container,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            fill = float(np.count_nonzero(container)) / max(1, ch * cw)
            container_conf = float(np.clip(0.42 + fill * 0.45, 0.0, 0.78))

    if container is None or not np.any(container):
        return None, 0.0, "container_first_no_container"

    erode_px = max(2, min(4, int(min(cw, ch) * 0.035)))
    eroded = cv2.erode(
        container,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)),
        iterations=1,
    )
    if int(np.count_nonzero(eroded)) < 24:
        return None, 0.0, "container_first_no_safe_interior"

    safe = eroded > 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    bg_bgr = np.median(crop[safe].reshape(-1, 3), axis=0)
    bg_gray = float(np.median(gray[safe]))
    bg_lab = np.median(lab[safe].reshape(-1, 3), axis=0)
    lab_dist = np.sqrt(np.sum((lab - bg_lab[None, None, :]) ** 2, axis=2))
    dark = gray.astype(np.float32) < (bg_gray - 25.0)
    contrast = lab_dist > 21.0
    not_dark_bg = gray > 24
    raw = ((dark | contrast) & safe & not_dark_bg).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    boundary = cv2.bitwise_xor(container, eroded)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    container_area = int(np.count_nonzero(eroded))
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > max(80, int(container_area * 0.12)):
            continue
        if bw > cw * 0.72 or bh > ch * 0.72:
            continue
        long_bar = max(bw / max(1, bh), bh / max(1, bw))
        density = area / max(1, bw * bh)
        if long_bar > 18.0 and density < 0.35:
            continue
        comp = labels == label
        boundary_touch = int(np.count_nonzero(comp & (boundary > 0)))
        if boundary_touch / max(1, area) > 0.32:
            continue
        kept[comp] = 255

    if not np.any(kept):
        return None, 0.0, "container_first_no_glyphs"

    kept = cv2.dilate(
        kept,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    kept = cv2.bitwise_and(kept, eroded)
    glyph_area = int(np.count_nonzero(kept))
    stroke_ratio = glyph_area / max(1, container_area)
    if stroke_ratio > 0.35:
        return None, 0.0, f"container_first_bad_stroke(stroke={stroke_ratio:.3f})"
    if 0.03 <= stroke_ratio <= 0.25:
        stroke_score = 1.0
    elif stroke_ratio < 0.03:
        stroke_score = stroke_ratio / 0.03
    else:
        stroke_score = max(0.0, 1.0 - ((stroke_ratio - 0.25) / 0.25))
    if stroke_score <= 0.0:
        return None, 0.0, f"container_first_bad_stroke(stroke={stroke_ratio:.3f})"

    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = kept
    conf = float(np.clip(0.85 * container_conf * stroke_score, 0.0, 0.86))
    return mask, conf, f"container_first_glyph(stroke={stroke_ratio:.3f},container={container_conf:.2f})"


def build_text_mask_candidates(
    img_cv: np.ndarray,
    region_bbox: Tuple[int, int, int, int],
    boxes: List[Any],
    existing_mask: Optional[np.ndarray] = None,
    text_bbox: Optional[Tuple[int, int, int, int]] = None,
    block: Optional[Any] = None,
    region_class: str = "",
    debug_metrics: Optional[Dict[str, Any]] = None,
) -> List[Tuple[np.ndarray, float, str]]:
    """
    Run all text mask candidate methods and return a list of
    (mask, confidence, reason) sorted by confidence descending.

    If existing_mask is provided and is already tight (not full-bbox),
    it is included as a candidate with the appropriate confidence score.
    """
    results: List[Tuple[np.ndarray, float, str]] = []
    h_img, w_img = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    clip_bbox = _expand_bbox(text_bbox, 8, img_cv.shape) if text_bbox is not None else region_bbox
    candidate_rows: List[Dict[str, Any]] = []

    def _note_candidate(
        reason: str,
        conf: float,
        mask: Optional[np.ndarray],
        accepted: bool,
        rejection_reason: str = "",
    ) -> None:
        row = {
            "reason": str(reason or ""),
            "source": _candidate_source_from_reason(str(reason or "")),
            "confidence": round(float(conf or 0.0), 4),
            "mask_px": int(np.count_nonzero(mask)) if mask is not None else 0,
            "accepted": bool(accepted),
            "rejection_reason": str(rejection_reason or ""),
        }
        candidate_rows.append(row)

    def _record_scores() -> None:
        if debug_metrics is None:
            return
        accepted = {str(reason) for _mask, _conf, reason in results}
        for row in candidate_rows:
            row["selected"] = False
            if row["reason"] in accepted and row["accepted"]:
                row["available_for_selection"] = True
        debug_metrics["text_mask_candidate_scores"] = candidate_rows

    # Include existing mask as a candidate if it looks tight
    if existing_mask is not None and np.any(existing_mask):
        x1, y1 = max(0, rx), max(0, ry)
        x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
        existing_mask = _clip_mask_to_bbox(existing_mask, clip_bbox, img_cv.shape)
        existing_area = int(np.count_nonzero(existing_mask[y1:y2, x1:x2]))
        existing_cov  = existing_area / max(1, (x2 - x1) * (y2 - y1))
        reason = f"existing_mask(cov={existing_cov:.2f})"
        if existing_cov < 0.85:
            results.append((existing_mask, 0.60, reason))
            _note_candidate(reason, 0.60, existing_mask, True)
        else:
            _note_candidate(
                reason,
                0.0,
                existing_mask,
                False,
                "legacy_block_text_mask_bbox_like_or_full_rectangle",
            )

    if text_bbox is None:
        if region_class == "caption_box":
            try:
                mask_dc, conf_dc, reason_dc = _candidate_dark_caption_light_text(img_cv, region_bbox)
                if mask_dc is not None and conf_dc > 0.05:
                    mask_dc = _clip_mask_to_bbox(mask_dc, clip_bbox, img_cv.shape)
                    if np.any(mask_dc):
                        results.append((mask_dc, conf_dc, reason_dc))
                        _note_candidate(reason_dc, conf_dc, mask_dc, True)
                    else:
                        _note_candidate(reason_dc, conf_dc, mask_dc, False, "empty_after_clip")
                else:
                    _note_candidate(reason_dc, conf_dc, mask_dc, False, "missing_or_low_confidence")
            except Exception as exc:
                debug_print(f"build_text_mask_candidates: dark_caption failed: {exc}")
                _note_candidate("dark_caption_exception", 0.0, None, False, str(exc))
        if (
            block is not None
            and getattr(block, "detector_source", "") == "yolo"
            and region_class != "sfx"
        ):
            try:
                mask_cf, conf_cf, reason_cf = _candidate_container_first_glyph_mask(
                    img_cv, block, region_bbox
                )
                if mask_cf is not None and conf_cf > 0.05:
                    mask_cf = _clip_mask_to_bbox(mask_cf, clip_bbox, img_cv.shape)
                    if np.any(mask_cf):
                        results.append((mask_cf, conf_cf, reason_cf))
                        _note_candidate(reason_cf, conf_cf, mask_cf, True)
                    else:
                        _note_candidate(reason_cf, conf_cf, mask_cf, False, "empty_after_clip")
                else:
                    _note_candidate(reason_cf, conf_cf, mask_cf, False, "missing_or_low_confidence")
            except Exception as exc:
                debug_print(
                    f"build_text_mask_candidates: container_first failed: {exc}"
                )
                _note_candidate("container_first_glyph_exception", 0.0, None, False, str(exc))
        try:
            mask_n, conf_n, reason_n = _candidate_region_cv_no_bbox(img_cv, region_bbox)
            if mask_n is not None and conf_n > 0.05:
                mask_n = _clip_mask_to_bbox(mask_n, clip_bbox, img_cv.shape)
                if np.any(mask_n):
                    results.append((mask_n, conf_n, reason_n))
                    _note_candidate(reason_n, conf_n, mask_n, True)
                else:
                    _note_candidate(reason_n, conf_n, mask_n, False, "empty_after_clip")
            else:
                _note_candidate(reason_n, conf_n, mask_n, False, "missing_or_low_confidence")
        except Exception as exc:
            debug_print(f"build_text_mask_candidates: no_bbox_cv failed: {exc}")
            _note_candidate("region_cv_no_bbox_exception", 0.0, None, False, str(exc))
        results.sort(key=lambda t: -t[1])
        _record_scores()
        return results

    # Method A: OCR contrast
    try:
        mask_a, conf_a, reason_a = _candidate_ocr_contrast(
            img_cv, boxes, region_bbox
        )
        if mask_a is not None and conf_a > 0.05:
            mask_a = _clip_mask_to_bbox(mask_a, clip_bbox, img_cv.shape)
            if np.any(mask_a):
                results.append((mask_a, conf_a, reason_a))
                _note_candidate(reason_a, conf_a, mask_a, True)
            else:
                _note_candidate(reason_a, conf_a, mask_a, False, "empty_after_clip")
        else:
            _note_candidate(reason_a, conf_a, mask_a, False, "missing_or_low_confidence")
    except Exception as exc:
        debug_print(f"build_text_mask_candidates: method_A failed: {exc}")
        _note_candidate("ocr_contrast_exception", 0.0, None, False, str(exc))

    # Method B: multi-channel threshold
    try:
        mask_b, conf_b, reason_b = _candidate_multichannel_threshold(
            img_cv, boxes, region_bbox
        )
        if mask_b is not None and conf_b > 0.05:
            mask_b = _clip_mask_to_bbox(mask_b, clip_bbox, img_cv.shape)
            if np.any(mask_b):
                results.append((mask_b, conf_b, reason_b))
                _note_candidate(reason_b, conf_b, mask_b, True)
            else:
                _note_candidate(reason_b, conf_b, mask_b, False, "empty_after_clip")
        else:
            _note_candidate(reason_b, conf_b, mask_b, False, "missing_or_low_confidence")
    except Exception as exc:
        debug_print(f"build_text_mask_candidates: method_B failed: {exc}")
        _note_candidate("multichannel_exception", 0.0, None, False, str(exc))

    # Method C: edge + components
    try:
        mask_c, conf_c, reason_c = _candidate_edge_components(
            img_cv, boxes, region_bbox
        )
        if mask_c is not None and conf_c > 0.05:
            mask_c = _clip_mask_to_bbox(mask_c, clip_bbox, img_cv.shape)
            if np.any(mask_c):
                results.append((mask_c, conf_c, reason_c))
                _note_candidate(reason_c, conf_c, mask_c, True)
            else:
                _note_candidate(reason_c, conf_c, mask_c, False, "empty_after_clip")
        else:
            _note_candidate(reason_c, conf_c, mask_c, False, "missing_or_low_confidence")
    except Exception as exc:
        debug_print(f"build_text_mask_candidates: method_C failed: {exc}")
        _note_candidate("edge_components_exception", 0.0, None, False, str(exc))

    results.sort(key=lambda t: -t[1])
    _record_scores()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Outline / shadow mask expansion
# ──────────────────────────────────────────────────────────────────────────────

def build_outline_shadow_mask(
    img_cv: np.ndarray,
    text_mask: np.ndarray,
    container_mask: Optional[np.ndarray] = None,
    container_bbox: Optional[Tuple[int, int, int, int]] = None,
    radius: int = 4,
) -> Optional[np.ndarray]:
    """
    Expand text_mask to include outline/shadow/glow strokes.

    Only includes pixels that are:
      - Within `radius` pixels of the text_mask
      - Visually different from their local background (outline/shadow,
        not bubble interior)
      - Inside container_mask when available

    Returns a full-image mask or None if the expansion is trivial.
    """
    if not np.any(text_mask):
        return None

    h_img, w_img = img_cv.shape[:2]

    kern     = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    expanded       = cv2.dilate(text_mask, kern, iterations=1)
    expansion_zone = cv2.bitwise_and(expanded, cv2.bitwise_not(text_mask))
    if not np.any(expansion_zone):
        return None

    gray     = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred  = cv2.GaussianBlur(gray, (radius * 2 + 1, radius * 2 + 1), 0)
    contrast = np.abs(gray - blurred)

    outline_cand = np.where(
        (expansion_zone > 0) & (contrast > 18.0), 255, 0
    ).astype(np.uint8)

    # FIX-6: use normalize_mask_to_image instead of manual canvas build.
    if container_mask is not None and container_bbox is not None:
        global_cm    = normalize_mask_to_image(
            container_mask, container_bbox, img_cv.shape
        )
        outline_cand = cv2.bitwise_and(outline_cand, global_cm)

    if not np.any(outline_cand):
        return None
    return outline_cand


def build_text_halo_mask(
    img_cv: np.ndarray,
    text_mask: np.ndarray,
    container_mask: Optional[np.ndarray] = None,
    container_bbox: Optional[Tuple[int, int, int, int]] = None,
    max_px: int = 2,
    region_bbox: Optional[Tuple[int, int, int, int]] = None,
    debug_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Catch high-contrast anti-aliased glyph pixels just outside text_mask."""
    if debug_metrics is not None:
        debug_metrics["halo_added_px"] = 0
        debug_metrics["halo_ratio_to_text_mask"] = 0.0
        debug_metrics["halo_rejected_reason"] = ""
    if text_mask is None or not np.any(text_mask):
        if debug_metrics is not None:
            debug_metrics["halo_rejected_reason"] = "empty_text_mask"
        return None

    radius = max(1, min(4, int(max_px or 1)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    zone = cv2.bitwise_and(
        cv2.dilate((text_mask > 0).astype(np.uint8) * 255, kernel, iterations=1),
        cv2.bitwise_not((text_mask > 0).astype(np.uint8) * 255),
    )
    if not np.any(zone):
        if debug_metrics is not None:
            debug_metrics["halo_rejected_reason"] = "empty_halo_zone"
        return None

    global_cm = None
    if container_mask is not None and container_bbox is not None:
        global_cm = normalize_mask_to_image(container_mask, container_bbox, img_cv.shape)
        zone = cv2.bitwise_and(zone, global_cm)
        if not np.any(zone):
            if debug_metrics is not None:
                debug_metrics["halo_rejected_reason"] = "outside_container"
            return None

    exclude = cv2.dilate(
        (text_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    if global_cm is not None:
        sample = (global_cm > 0) & ~exclude
    else:
        sample = ~exclude

    pixels = img_cv[sample]
    if pixels.shape[0] < 8:
        if debug_metrics is not None:
            debug_metrics["halo_rejected_reason"] = "background_sample_lt_8"
        return None
    bg_bgr = np.median(pixels.reshape(-1, 3).astype(np.float32), axis=0)
    diff = np.sqrt(np.sum((img_cv.astype(np.float32) - bg_bgr[None, None, :]) ** 2, axis=2))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    bg_gray = float(0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2])
    bg_sat = float(np.median(hsv[:, :, 1][sample]))
    sample_gray = gray[sample]
    flat_sample = float(np.std(sample_gray)) < 14.0
    diff_thresh = 14.0 if flat_sample else 20.0
    gray_thresh = 6.0 if flat_sample else 10.0
    gray_edge_thresh = 5.0 if flat_sample else 8.0
    dark_strokes = ((text_mask > 0) & (gray < bg_gray - 12.0)).astype(np.uint8) * 255
    light_strokes = ((text_mask > 0) & (gray > bg_gray + 12.0)).astype(np.uint8) * 255
    any_strokes = (text_mask > 0).astype(np.uint8) * 255
    near_text_stroke = cv2.dilate(
        cv2.bitwise_or(any_strokes, cv2.bitwise_or(dark_strokes, light_strokes)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 3, radius * 2 + 3)),
        iterations=1,
    ) > 0
    near_dark_stroke = cv2.dilate(
        dark_strokes,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 3, radius * 2 + 3)),
        iterations=1,
    ) > 0
    near_light_stroke = cv2.dilate(
        light_strokes,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 3, radius * 2 + 3)),
        iterations=1,
    ) > 0
    gray_edge = (
        (hsv[:, :, 1] < 42)
        & (np.abs(gray - bg_gray) > gray_edge_thresh)
        & (near_dark_stroke | near_light_stroke | near_text_stroke)
    )
    color_glow = (
        (hsv[:, :, 1].astype(np.float32) > bg_sat + 18.0)
        & (diff > diff_thresh)
        & near_text_stroke
    )
    halo = np.where(
        (zone > 0)
        & (
            ((diff > diff_thresh) & (np.abs(gray - bg_gray) > gray_thresh))
            | gray_edge
            | color_glow
        ),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(halo):
        if debug_metrics is not None:
            debug_metrics["halo_rejected_reason"] = "no_contrast_edge_pixels"
        return None
    text_px = int(np.count_nonzero(text_mask))
    halo_px = int(np.count_nonzero(halo))
    ratio = float(halo_px) / max(1, text_px)
    if debug_metrics is not None:
        debug_metrics["halo_added_px"] = halo_px
        debug_metrics["halo_ratio_to_text_mask"] = round(ratio, 4)
    if region_bbox is not None:
        quality = _compute_mask_quality_metrics(halo, global_cm, region_bbox)
        rectangularity = float(quality.get("rectangularity", 0.0) or 0.0)
        mask_region_ratio = float(quality.get("mask_region_ratio", 0.0) or 0.0)
        if (
            ratio > 1.75
            or mask_region_ratio > 0.18
            or (rectangularity > 0.72 and halo_px > text_px * 0.55)
        ):
            if debug_metrics is not None:
                debug_metrics["halo_added_px"] = 0
                debug_metrics["halo_ratio_to_text_mask"] = 0.0
                debug_metrics["halo_rejected_reason"] = (
                    f"unsafe_halo_shape(ratio={ratio:.2f},rect={rectangularity:.2f})"
                )
            return None
    return halo


# ──────────────────────────────────────────────────────────────────────────────
# Container mask builder
# ──────────────────────────────────────────────────────────────────────────────

def build_container_mask_from_block(
    img_cv: np.ndarray,
    block: Any,
    text_mask: Optional[np.ndarray] = None,  # FIX-4: thread text_mask in
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float, str]:
    """
    Build or validate container_mask from an OCRBlock's existing bubble detection.

    Returns (local_mask, bbox, confidence, reason).
    local_mask is relative to container_bbox, not the full image.

    Priority:
      1. block.bubble_mask / block.bubble_bbox if valid and non-rectangular
      2. Color-similarity flood fill from text-area surroundings
      3. None (no reliable container)
    """
    h_img, w_img = img_cv.shape[:2]
    bub_mask = getattr(block, "bubble_mask", None)
    bub_bbox = getattr(block, "bubble_bbox", None)

    if bub_mask is not None and bub_bbox is not None:
        bx, by, bw, bh = bub_bbox
        bx1, by1 = max(0, bx), max(0, by)
        bx2, by2 = min(w_img, bx + bw), min(h_img, by + bh)
        if bx2 > bx1 and by2 > by1:
            expected_h = by2 - by1
            expected_w = bx2 - bx1
            local = bub_mask
            if local.shape != (expected_h, expected_w):
                try:
                    local = cv2.resize(
                        local, (expected_w, expected_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception:
                    local = np.full(
                        (expected_h, expected_w), 255, dtype=np.uint8
                    )

            fill_ratio = float(np.count_nonzero(local)) / max(
                1, expected_w * expected_h
            )
            if fill_ratio > 0.92:
                reason = f"rectangular(fill={fill_ratio:.2f})"
                # Try color-based refinement
                refined, refined_bbox, ref_conf, ref_reason = _container_color_fill(
                    img_cv, block,
                    (bx1, by1, bx2 - bx1, by2 - by1),
                    text_mask=text_mask,  # FIX-4
                )
                if refined is not None and ref_conf > 0.45:
                    return refined, refined_bbox, ref_conf, ref_reason
                return local, (bx1, by1, expected_w, expected_h), 0.20, reason

            return local, (bx1, by1, expected_w, expected_h), 0.72, "bubble_detection"

    # No bubble geometry: try color fill
    return _container_color_fill(img_cv, block, None, text_mask=text_mask)  # FIX-4


def _container_color_fill(
    img_cv: np.ndarray,
    block: Any,
    hint_bbox: Optional[Tuple[int, int, int, int]],
    text_mask: Optional[np.ndarray] = None,   # FIX-4: exclude glyph pixels
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float, str]:
    """
    Attempt color-similarity flood fill to find bubble interior.

    Seeds from estimated background pixels near but outside the text area,
    grows into a connected region of similar Lab color.

    FIX-4: sample points that land on text_mask pixels are skipped so the
    seed color reflects the bubble interior, not a glyph stroke.
    """
    h_img, w_img = img_cv.shape[:2]
    region_bbox = block.bbox()
    rx, ry, rw, rh = region_bbox

    if hint_bbox is not None:
        x1, y1 = max(0, hint_bbox[0]), max(0, hint_bbox[1])
        x2      = min(w_img, hint_bbox[0] + hint_bbox[2])
        y2      = min(h_img, hint_bbox[1] + hint_bbox[3])
    else:
        pad = max(20, int(min(rw, rh) * 0.5))
        x1  = max(0, rx - pad);     y1 = max(0, ry - pad)
        x2  = min(w_img, rx + rw + pad); y2 = min(h_img, ry + rh + pad)

    if x2 <= x1 or y2 <= y1:
        return None, None, 0.0, "empty"

    roi          = img_cv[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    lab_roi      = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)

    text_local_x  = max(0, rx - x1)
    text_local_y  = max(0, ry - y1)
    text_local_x2 = min(roi_w, rx - x1 + rw)
    text_local_y2 = min(roi_h, ry - y1 + rh)

    sample_band     = max(4, min(12, int(min(rw, rh) * 0.15)))
    interior_samples = []

    for dy, dx in [
        (-sample_band, 0), (sample_band, 0),
        (0, -sample_band), (0, sample_band),
    ]:
        sy = int(np.clip(text_local_y + rh // 2 + dy, 0, roi_h - 1))
        sx = int(np.clip(text_local_x + rw // 2 + dx, 0, roi_w - 1))

        # FIX-4: skip sample points that sit on a known glyph pixel.
        if text_mask is not None:
            gsy = sy + y1
            gsx = sx + x1
            if (
                0 <= gsy < text_mask.shape[0]
                and 0 <= gsx < text_mask.shape[1]
                and text_mask[gsy, gsx] > 0
            ):
                continue

        interior_samples.append(lab_roi[sy, sx])

    if not interior_samples:
        # All samples were on glyphs; fall back to using all four anyway
        for dy, dx in [
            (-sample_band, 0), (sample_band, 0),
            (0, -sample_band), (0, sample_band),
        ]:
            sy = int(np.clip(text_local_y + rh // 2 + dy, 0, roi_h - 1))
            sx = int(np.clip(text_local_x + rw // 2 + dx, 0, roi_w - 1))
            interior_samples.append(lab_roi[sy, sx])

    if not interior_samples:
        return None, None, 0.0, "no_samples"

    seed_lab = np.mean(interior_samples, axis=0)
    thresh   = 30.0
    diff     = np.sqrt(
        np.sum((lab_roi - seed_lab[None, None, :]) ** 2, axis=2)
    )
    similar  = np.where(diff < thresh, 255, 0).astype(np.uint8)

    seed_y = int(np.clip(text_local_y + rh // 2, 0, roi_h - 1))
    seed_x = int(np.clip(text_local_x + rw // 2, 0, roi_w - 1))

    kern5          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    similar_closed = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, kern5, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        similar_closed, connectivity=8
    )

    seed_label = int(labels[seed_y, seed_x])
    if seed_label == 0:
        if n > 1:
            areas      = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
            areas.sort(reverse=True)
            seed_label = areas[0][1]
        else:
            return None, None, 0.0, "no_component"

    container_local = np.where(labels == seed_label, 255, 0).astype(np.uint8)
    fill_ratio      = float(np.count_nonzero(container_local)) / max(
        1, roi_w * roi_h
    )

    text_sub   = container_local[
        text_local_y:text_local_y2, text_local_x:text_local_x2
    ]
    text_area  = max(
        1,
        (text_local_x2 - text_local_x) * (text_local_y2 - text_local_y),
    )
    text_cover = float(np.count_nonzero(text_sub)) / text_area

    if text_cover < 0.3:
        return None, None, 0.25, "container_doesnt_cover_text"
    if fill_ratio > 0.92:
        return None, None, 0.30, "container_is_full_rect"

    confidence = float(np.clip(0.45 + text_cover * 0.3, 0.0, 0.75))
    return (
        container_local,
        (x1, y1, roi_w, roi_h),
        confidence,
        f"color_fill(text_cov={text_cover:.2f})",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gradient reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def gradient_reconstruct_idw(
    img_cv: np.ndarray,
    cleanup_mask: np.ndarray,
    support_mask: np.ndarray,
    max_support_px: int = 1500,
    search_radius:  int = 80,
) -> np.ndarray:
    """
    Fill cleanup_mask pixels using inverse-distance-weighted interpolation
    from support_mask pixels in Lab color space.

    Produces a smooth gradient continuation without solid patches.
    Falls back to TELEA inpaint if the setup is degenerate or the pixel
    count exceeds the safety cap.

    FIX-8: if fill pixel count > 8 000 fall back to cv2.inpaint (TELEA)
    rather than looping per-pixel, which would be unacceptably slow on
    large outlined SFX or accidentally-expanded masks.
    """
    ys_fill, xs_fill = np.where(cleanup_mask > 0)
    ys_sup,  xs_sup  = np.where(support_mask > 0)

    if len(ys_fill) == 0 or len(ys_sup) == 0:
        return cv2.inpaint(img_cv, cleanup_mask, 5, cv2.INPAINT_TELEA)

    # FIX-8: pixel cap — large masks fall back to TELEA.
    if len(ys_fill) > 8000:
        debug_print(
            f"gradient_reconstruct_idw: fill_px={len(ys_fill)} > 8000, "
            "falling back to TELEA"
        )
        return cv2.inpaint(img_cv, cleanup_mask, 5, cv2.INPAINT_TELEA)

    result = img_cv.copy()
    lab    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2Lab).astype(np.float32)
    out_lab = lab.copy()

    # Subsample support pixels for speed
    if len(ys_sup) > max_support_px:
        idx    = np.random.choice(len(ys_sup), max_support_px, replace=False)
        ys_sup = ys_sup[idx]
        xs_sup = xs_sup[idx]

    sup_colors = lab[ys_sup, xs_sup]                           # (N, 3)
    sup_yx     = np.stack(
        [ys_sup.astype(np.float32), xs_sup.astype(np.float32)], axis=1
    )                                                          # (N, 2)

    for fy, fx in zip(ys_fill.tolist(), xs_fill.tolist()):
        dy  = sup_yx[:, 0] - fy
        dx  = sup_yx[:, 1] - fx
        d2  = dy * dy + dx * dx
        nearby = d2 < (search_radius * search_radius)

        if np.any(nearby):
            d2_use     = d2[nearby]
            colors_use = sup_colors[nearby]
        else:
            d2_use     = d2
            colors_use = sup_colors

        d2_use  = np.maximum(d2_use, 0.25)
        w       = 1.0 / d2_use
        w      /= w.sum()
        out_lab[fy, fx] = (w[:, None] * colors_use).sum(axis=0)

    out_lab    = np.clip(out_lab, 0, 255).astype(np.uint8)
    result_lab = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)

    # Blend boundary softly: 3-px feather
    alpha     = cv2.GaussianBlur(
        cleanup_mask.astype(np.float32), (5, 5), 0
    ) / 255.0
    result_f   = result.astype(np.float32)
    result_l_f = result_lab.astype(np.float32)
    alpha3     = alpha[:, :, None]
    blended    = (
        result_l_f * alpha3 + result_f * (1.0 - alpha3)
    ).clip(0, 255).astype(np.uint8)

    result[cleanup_mask > 0] = blended[cleanup_mask > 0]
    return result.astype(np.uint8)


def _container_mask_to_full_image(plan: CleanupPlan, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if plan.container_mask is None or plan.container_bbox is None:
        return None
    return normalize_mask_to_image(plan.container_mask, plan.container_bbox, image_shape)


def _estimate_plain_bg_color(
    img_cv: np.ndarray,
    container_mask: Optional[np.ndarray],
    text_mask: np.ndarray,
    region_bbox: Tuple[int, int, int, int],
    allow_dark: bool = False,
) -> Tuple[np.ndarray, float]:
    """Sample a robust plain bubble/caption background color in BGR."""
    h, w = img_cv.shape[:2]
    rx, ry, rw, rh = region_bbox
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return np.array([255, 255, 255], dtype=np.uint8), 0.0

    exclude = cv2.dilate(
        (text_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0

    if container_mask is not None and np.any(container_mask):
        safe = cv2.erode(
            (container_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        sample = safe & ~exclude
        confidence = 0.82
    else:
        text_roi = text_mask[y1:y2, x1:x2] > 0
        ring = cv2.dilate(
            text_roi.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        ) > 0
        ring &= ~cv2.dilate(
            text_roi.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ).astype(bool)
        sample = np.zeros(img_cv.shape[:2], dtype=bool)
        sample[y1:y2, x1:x2] = ring
        sample &= ~exclude
        confidence = 0.45

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    if not allow_dark:
        sample &= gray > 90
    if int(np.count_nonzero(sample)) < 12:
        sample = np.zeros(img_cv.shape[:2], dtype=bool)
        sample[y1:y2, x1:x2] = True
        sample &= ~exclude
        if not allow_dark:
            sample &= gray > 90
        confidence = min(confidence, 0.35)

    pixels = img_cv[sample]
    if pixels.shape[0] < 8:
        return np.array([255, 255, 255], dtype=np.uint8), 0.0

    pixels_f = pixels.reshape(-1, 3).astype(np.float32)
    lum = (
        pixels_f[:, 2] * 0.299
        + pixels_f[:, 1] * 0.587
        + pixels_f[:, 0] * 0.114
    )
    lo, hi = np.percentile(lum, [10, 90])
    keep = (lum >= lo) & (lum <= hi)
    if int(np.count_nonzero(keep)) >= 8:
        pixels_f = pixels_f[keep]

    spread = float(np.max(np.std(pixels_f, axis=0)))
    if spread > 30.0:
        confidence *= 0.55
    bg_bgr = np.median(pixels_f, axis=0).clip(0, 255).astype(np.uint8)
    return bg_bgr, float(np.clip(confidence, 0.0, 0.95))


def _sample_container_bg_metrics(
    img_cv: np.ndarray,
    plan: CleanupPlan,
    cleanup_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Sample solid-bubble background from eroded container interior."""
    if plan.container_mask is None or plan.container_bbox is None:
        return None, {"reason": "no_container"}
    global_cm = normalize_mask_to_image(plan.container_mask, plan.container_bbox, img_cv.shape)
    if not np.any(global_cm):
        return None, {"reason": "empty_container"}

    safe = cv2.erode(
        (global_cm > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    exclude = (cleanup_mask > 0)
    if plan.text_mask is not None:
        exclude |= plan.text_mask > 0
    if plan.outline_shadow_mask is not None:
        exclude |= plan.outline_shadow_mask > 0
    if plan.halo_mask is not None:
        exclude |= plan.halo_mask > 0
    exclude = cv2.dilate(
        exclude.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    sample = safe & ~exclude
    sample_px = int(np.count_nonzero(sample))
    if sample_px < 24:
        return None, {"reason": "sample_lt_24", "sample_px": sample_px}

    pixels = img_cv[sample].reshape(-1, 3).astype(np.float32)
    bg_bgr = np.median(pixels, axis=0).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2Lab).astype(np.float32)
    edges = cv2.Canny(img_cv, 40, 120)
    sample_gray = gray[sample]
    sample_lab = lab[sample]
    metrics: Dict[str, Any] = {
        "sample_px": sample_px,
        "bg_bgr": [int(v) for v in bg_bgr.tolist()],
        "mean_brightness": round(float(np.mean(sample_gray)), 2),
        "gray_std": round(float(np.std(sample_gray)), 2),
        "channel_std_max": round(float(np.max(np.std(pixels, axis=0))), 2),
        "sat_std": round(float(np.std(hsv[:, :, 1][sample])), 2),
        "edge_density": round(float(edges[sample].sum()) / max(1, sample_px * 255), 5),
        "lab_a_std": round(float(np.std(sample_lab[:, 1])), 2),
        "lab_b_std": round(float(np.std(sample_lab[:, 2])), 2),
    }
    return bg_bgr, metrics


def _expand_large_glyph_components(
    img_cv: np.ndarray,
    plan: CleanupPlan,
    policy: CleanupPolicy,
) -> None:
    """Conservatively add missed high-contrast glyph components inside bubbles."""
    plan.debug_metrics.setdefault("large_component_kept_count", 0)
    plan.debug_metrics.setdefault("large_component_rejected_count", 0)
    plan.debug_metrics.setdefault("component_reject_reason", "")
    if (
        plan.region_class not in ("speech_bubble", "caption_box")
        or plan.text_mask is None
        or not np.any(plan.text_mask)
        or plan.container_mask is None
        or plan.container_bbox is None
        or float(plan.container_confidence or 0.0) < 0.45
    ):
        return
    selected_reason = str(plan.debug_metrics.get("selected_text_mask_candidate", "") or "")
    if not selected_reason.startswith((
        "region_cv_no_bbox",
        "container_first_glyph",
        "edge_components",
        "existing_mask",
        "ocr_contrast",
        "multichannel",
    )):
        return

    global_cm = normalize_mask_to_image(plan.container_mask, plan.container_bbox, img_cv.shape)
    if not np.any(global_cm):
        return

    rx, ry, rw, rh = plan.region_bbox
    h_img, w_img = img_cv.shape[:2]
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return

    region = np.zeros(img_cv.shape[:2], dtype=bool)
    region[y1:y2, x1:x2] = True
    text = plan.text_mask > 0
    bg_bgr, sample_metrics = _sample_container_bg_metrics(img_cv, plan, plan.text_mask)
    if bg_bgr is None:
        bg_bgr, _conf = _estimate_plain_bg_color(
            img_cv,
            global_cm,
            plan.text_mask,
            plan.region_bbox,
            allow_dark=(plan.background_model == "dark_bubble"),
        )
        sample_metrics = {}

    img_f = img_cv.astype(np.float32)
    diff = np.sqrt(np.sum((img_f - bg_bgr.astype(np.float32)[None, None, :]) ** 2, axis=2))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = float(0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2])
    if plan.text_bbox is not None:
        _tx, _ty, _tw, _th = plan.text_bbox
        near_px = max(11, min(31, int(round(max(_tw, _th) * 0.42)) | 1))
    else:
        near_px = 11
    near_text = cv2.dilate(
        text.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_px, near_px)),
        iterations=1,
    ) > 0
    safe = (global_cm > 0) & region & ~text & near_text
    flat_bg = (
        float(sample_metrics.get("gray_std", 999.0) or 999.0) <= 18.0
        and float(sample_metrics.get("channel_std_max", 999.0) or 999.0) <= 24.0
    )
    diff_threshold = 24.0 if flat_bg else 30.0
    gray_threshold = 16.0 if flat_bg else 24.0
    raw = np.where(
        safe
        & (
            (diff > diff_threshold)
            | (np.abs(gray - bg_gray) > gray_threshold)
        ),
        255,
        0,
    ).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    if not np.any(raw):
        return

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    region_area = max(1, int(rw * rh))
    container_area = max(1, int(np.count_nonzero(global_cm)))
    text_px = max(1, int(np.count_nonzero(plan.text_mask)))
    max_component_area = max(96, min(int(container_area * 0.18), int(region_area * 0.16), 24000))
    max_total_added = max(64, min(int(container_area * 0.16), int(max(text_px * 2.5, text_px + 6000))))
    kept_count = 0
    rejected_count = 0
    reject_reasons: Dict[str, int] = {}

    def _reject(reason: str) -> None:
        nonlocal rejected_count
        rejected_count += 1
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp = labels == label
        comp_diff = float(np.median(diff[comp])) if np.any(comp) else 0.0
        comp_gray_delta = float(np.median(np.abs(gray[comp] - bg_gray))) if np.any(comp) else 0.0
        tiny_high_contrast = area >= 2 and area <= 18 and (comp_diff >= 32.0 or comp_gray_delta >= 24.0)
        if area < max(12, int(region_area * 0.00015)) and not tiny_high_contrast:
            _reject("too_small")
            continue
        if area > max_component_area:
            _reject("too_large")
            continue
        if bw > rw * 0.78 or bh > rh * 0.78:
            _reject("spans_region")
            continue
        density = area / max(1, bw * bh)
        if density > 0.88 and area > region_area * 0.025:
            _reject("rectangular_dense")
            continue
        if comp_diff < 24.0 and comp_gray_delta < 18.0:
            _reject("low_contrast")
            continue
        kept[comp] = 255
        kept_count += 1

    added = cv2.bitwise_and(kept, cv2.bitwise_not((plan.text_mask > 0).astype(np.uint8) * 255))
    added_px = int(np.count_nonzero(added))
    if added_px <= 0:
        plan.debug_metrics["large_component_rejected_count"] = rejected_count
        if reject_reasons:
            plan.debug_metrics["component_reject_reason"] = ";".join(
                f"{k}:{v}" for k, v in sorted(reject_reasons.items())
            )
        return
    if added_px > max_total_added:
        rejected_count += kept_count
        reject_reasons["total_added_too_large"] = reject_reasons.get("total_added_too_large", 0) + 1
        plan.debug_metrics["large_component_rejected_count"] = rejected_count
        plan.debug_metrics["component_reject_reason"] = ";".join(
            f"{k}:{v}" for k, v in sorted(reject_reasons.items())
        )
        return

    candidate = cv2.bitwise_or(plan.text_mask, added)
    quality = _compute_mask_quality_metrics(
        candidate,
        global_cm,
        plan.region_bbox,
        plan.text_bbox,
        safety_bbox=_select_safety_bbox(plan),
    )
    if (
        float(quality.get("mask_container_ratio", 0.0) or 0.0)
        > max(0.22, min(0.45, policy.cleanup_max_mask_container_ratio))
        or (
            float(quality.get("rectangularity", 0.0) or 0.0) > 0.62
            and float(quality.get("mask_region_ratio", 0.0) or 0.0) > 0.16
        )
    ):
        plan.debug_metrics["large_component_rejected_count"] = rejected_count + kept_count
        plan.debug_metrics["component_reject_reason"] = "unsafe_combined_mask"
        return

    plan.text_mask = candidate
    plan.debug_metrics["large_component_kept_count"] = kept_count
    plan.debug_metrics["large_component_rejected_count"] = rejected_count
    if reject_reasons:
        plan.debug_metrics["component_reject_reason"] = ";".join(
            f"{k}:{v}" for k, v in sorted(reject_reasons.items())
        )


def _try_force_solid_bubble_flat_fill(
    plan: CleanupPlan,
    img_cv: np.ndarray,
    policy: CleanupPolicy,
) -> bool:
    if not policy.cleanup_solid_bubble_fill_enabled:
        plan.debug_metrics["solid_bubble_override"] = False
        return False
    if plan.region_class not in ("speech_bubble", "caption_box"):
        plan.debug_metrics["solid_bubble_override"] = False
        return False
    if float(plan.text_mask_confidence or 0.0) < 0.20:
        plan.debug_metrics["solid_bubble_override"] = False
        plan.debug_metrics["solid_bubble_override_rejected_reason"] = (
            f"text_mask_confidence_low({float(plan.text_mask_confidence or 0.0):.2f})"
        )
        return False
    if (
        plan.container_bbox is None
        or plan.container_mask is None
        or float(plan.container_confidence or 0.0)
        < policy.cleanup_solid_bubble_min_container_confidence
        or plan.cleanup_mask is None
        or not np.any(plan.cleanup_mask)
    ):
        plan.debug_metrics["solid_bubble_override"] = False
        return False

    quality = plan.debug_metrics.get("quality", {}) or {}
    mask_container_ratio = float(quality.get("mask_container_ratio", 0.0) or 0.0)
    rectangularity = float(quality.get("rectangularity", 0.0) or 0.0)
    border_touch_ratio = float(quality.get("border_touch_ratio", 0.0) or 0.0)
    easy_large_mask = _easy_cleanup_large_mask_allowed(plan, policy, quality)
    if (
        mask_container_ratio <= 0.0
        or (
            mask_container_ratio > policy.cleanup_solid_bubble_max_mask_container_ratio
            and not easy_large_mask
        )
        or (
            rectangularity > policy.cleanup_solid_bubble_max_rectangularity
            and not (easy_large_mask and mask_container_ratio <= 0.35)
        )
        or border_touch_ratio > policy.t2_max_border_touch
    ):
        plan.debug_metrics["solid_bubble_override"] = False
        return False

    bg_bgr, metrics = _sample_container_bg_metrics(img_cv, plan, plan.cleanup_mask)
    plan.debug_metrics["sampled_bg_metrics"] = metrics
    if bg_bgr is None:
        plan.debug_metrics["solid_bubble_override"] = False
        return False
    plan.debug_metrics["sampled_bg_bgr"] = [int(v) for v in bg_bgr.tolist()]

    mean_brightness = float(metrics.get("mean_brightness", 0.0))
    gray_std = float(metrics.get("gray_std", 999.0))
    channel_std = float(metrics.get("channel_std_max", 999.0))
    sat_std = float(metrics.get("sat_std", 999.0))
    edge_density = float(metrics.get("edge_density", 1.0))
    lab_a_std = float(metrics.get("lab_a_std", 999.0))
    lab_b_std = float(metrics.get("lab_b_std", 999.0))
    bg_channel_range = int(np.max(bg_bgr)) - int(np.min(bg_bgr))

    near_white = (
        mean_brightness >= 185.0
        and bg_channel_range <= 18
        and gray_std <= 20.0
        and channel_std <= 24.0
        and sat_std <= 22.0
        and edge_density <= 0.055
        and lab_a_std <= 12.0
        and lab_b_std <= 12.0
    )
    solid_colored = (
        70.0 <= mean_brightness < 245.0
        and gray_std <= 24.0
        and channel_std <= 26.0
        and sat_std <= 30.0
        and edge_density <= 0.050
    )
    dark_caption = (
        mean_brightness < 85.0
        and gray_std <= 18.0
        and channel_std <= 22.0
        and edge_density <= 0.045
    )
    if not (near_white or solid_colored or dark_caption):
        plan.debug_metrics["solid_bubble_override"] = False
        return False

    if dark_caption:
        plan.background_model = "dark_bubble"
    elif near_white:
        plan.background_model = "flat_light"
    else:
        plan.background_model = "flat_colored"
    plan.cleanup_strategy = "flat_fill"
    plan.inpaint_method = "local_sample"
    plan.skip_reason = ""
    plan.debug_metrics["solid_bubble_override"] = True
    debug_print(
        f"[CLEANUP_OVERRIDE] solid_bubble_flat_fill region={plan.region_id} "
        f"bg={bg_bgr.tolist()} metrics={metrics}"
    )
    return True


def _flat_fill_ladder_support_mask(
    img_cv: np.ndarray,
    plan: CleanupPlan,
    cleanup: np.ndarray,
) -> np.ndarray:
    h_img, w_img = img_cv.shape[:2]
    support = np.zeros((h_img, w_img), dtype=np.uint8)
    if plan.container_mask is not None and plan.container_bbox is not None and plan.container_confidence >= 0.35:
        global_cm = normalize_mask_to_image(plan.container_mask, plan.container_bbox, img_cv.shape)
        preserves, retained_ratio = _mask_preserves_cleanup(cleanup, global_cm)
        plan.debug_metrics["flat_fill_ladder_container_retained_ratio"] = round(retained_ratio, 4)
        if preserves and np.any(global_cm):
            support = global_cm
    if not np.any(support) and plan.region_bbox is not None:
        x, y, w, h = plan.region_bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 > x1 and y2 > y1:
            support[y1:y2, x1:x2] = 255
    return support


def _flat_fill_ladder_ink_guard(img_cv: np.ndarray, plan: CleanupPlan, support: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    if plan.background_model == "dark_bubble":
        return (gray > 180).astype(np.uint8) * 255
    if plan.background_model == "flat_colored":
        if np.any(support):
            med = float(np.median(gray[support > 0]))
        else:
            med = float(np.median(gray))
        return (np.abs(gray.astype(np.float32) - med) > 42.0).astype(np.uint8) * 255
    return (gray < 205).astype(np.uint8) * 255


def _score_flat_fill_growth_candidate(
    img_cv: np.ndarray,
    plan: CleanupPlan,
    candidate: np.ndarray,
    base_mask: np.ndarray,
    support: np.ndarray,
    growth_px: int,
    policy: CleanupPolicy,
) -> Optional[Dict[str, Any]]:
    if not np.any(candidate):
        return None
    ring_px = max(1, int(policy.cleanup_flat_fill_ring_px or 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    outer = cv2.dilate((candidate > 0).astype(np.uint8) * 255, kernel, iterations=1)
    ring = (outer > 0) & ~(candidate > 0)
    if np.any(support):
        ring &= support > 0
    if plan.text_mask is not None:
        ring &= ~(plan.text_mask > 0)
    if plan.outline_shadow_mask is not None:
        ring &= ~(plan.outline_shadow_mask > 0)
    if plan.halo_mask is not None:
        ring &= ~(plan.halo_mask > 0)
    ink_guard = _flat_fill_ladder_ink_guard(img_cv, plan, support)
    ring &= ~(ink_guard > 0)
    ring_px_count = int(np.count_nonzero(ring))
    if ring_px_count < 12:
        return None
    ring_pixels = img_cv[ring]
    gray = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1).astype(np.float32)
    lab = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    gray_std = float(np.std(gray))
    chroma_std = float(max(np.std(lab[:, 1]), np.std(lab[:, 2])))
    full_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(full_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(full_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_density = float(np.count_nonzero(edge_mag[ring] > 35.0)) / max(1, ring_px_count)
    quality = _compute_mask_quality_metrics(
        candidate,
        support if np.any(support) else None,
        plan.region_bbox,
        plan.text_bbox,
        safety_bbox=_select_safety_bbox(plan),
    )
    mask_region_ratio = float(quality.get("mask_region_ratio", 0.0) or 0.0)
    mask_container_ratio = float(quality.get("mask_container_ratio", 0.0) or 0.0)
    border_touch_ratio = float(quality.get("border_touch_ratio", 0.0) or 0.0)
    rectangularity = float(quality.get("rectangularity", 0.0) or 0.0)
    base_px = max(1, int(np.count_nonzero(base_mask)))
    growth_ratio = int(np.count_nonzero(candidate)) / base_px
    max_gray = float(policy.cleanup_flat_fill_max_ring_gray_std or 14.0)
    max_chroma = float(policy.cleanup_flat_fill_max_ring_chroma_std or 12.0)
    max_edge = float(policy.cleanup_flat_fill_max_ring_edge_density or 0.08)
    hard_reject = (
        gray_std > max_gray * 1.75
        or chroma_std > max_chroma * 1.75
        or edge_density > max_edge * 2.0
        or border_touch_ratio > max(0.65, policy.cleanup_max_border_touch_ratio + 0.20)
        or mask_container_ratio > max(0.45, policy.cleanup_max_mask_container_ratio)
        or mask_region_ratio > max(0.38, policy.cleanup_max_mask_region_ratio * 1.5)
        or (rectangularity > 0.78 and mask_region_ratio > 0.20)
        or growth_ratio > 4.0
    )
    score = (
        (gray_std / max(1.0, max_gray))
        + (chroma_std / max(1.0, max_chroma))
        + (edge_density / max(0.005, max_edge))
        + border_touch_ratio * 1.2
        + max(0.0, growth_ratio - 1.0) * 0.08
        + rectangularity * 0.10
    )
    if growth_px <= 1:
        score += 0.08
    return {
        "growth_px": int(growth_px),
        "mask_px": int(np.count_nonzero(candidate)),
        "ring_px": ring_px_count,
        "median_bgr": [int(v) for v in np.median(ring_pixels.reshape(-1, 3), axis=0).astype(np.uint8).tolist()],
        "gray_std": round(gray_std, 3),
        "chroma_std": round(chroma_std, 3),
        "edge_density": round(edge_density, 4),
        "mask_region_ratio": round(mask_region_ratio, 4),
        "mask_container_ratio": round(mask_container_ratio, 4),
        "border_touch_ratio": round(border_touch_ratio, 4),
        "rectangularity": round(rectangularity, 4),
        "growth_ratio": round(float(growth_ratio), 3),
        "score": round(float(score), 4),
        "rejected": bool(hard_reject),
    }


def _optimize_flat_fill_cleanup_mask(
    img_cv: np.ndarray,
    plan: CleanupPlan,
    cleanup: np.ndarray,
    policy: CleanupPolicy,
    extra_growth_px: int = 0,
) -> np.ndarray:
    plan.debug_metrics["flat_fill_ladder_enabled"] = bool(policy.cleanup_flat_fill_ladder_enabled)
    if (
        not policy.cleanup_flat_fill_ladder_enabled
        or plan.cleanup_strategy != "flat_fill"
        or plan.region_class not in ("speech_bubble", "caption_box")
        or plan.background_model not in {"flat_light", "flat_colored", "dark_bubble"}
        or cleanup is None
        or not np.any(cleanup)
    ):
        return cleanup
    support = _flat_fill_ladder_support_mask(img_cv, plan, cleanup)
    if not np.any(support):
        plan.debug_metrics["flat_fill_ladder_rejection_reason"] = "no_support"
        return cleanup
    base = (cleanup > 0).astype(np.uint8) * 255
    if np.any(support):
        confined = cv2.bitwise_and(base, support)
        preserves, _retained = _mask_preserves_cleanup(base, support)
        if np.any(confined) and preserves:
            base = confined
    max_growth = max(0, int(policy.cleanup_flat_fill_max_growth_px or 10) + int(extra_growth_px or 0))
    candidates: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    candidate_metrics: List[Dict[str, Any]] = []
    for growth in range(0, max_growth + 1):
        if growth <= 0:
            candidate = base.copy()
        else:
            k = growth * 2 + 1
            candidate = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
        candidate = cv2.bitwise_and(candidate, support)
        metrics = _score_flat_fill_growth_candidate(img_cv, plan, candidate, base, support, growth, policy)
        if metrics is None:
            continue
        candidate_metrics.append(metrics)
        if not metrics.get("rejected"):
            candidates.append((candidate, metrics))
    plan.debug_metrics["flat_fill_ladder_candidates"] = candidate_metrics
    if not candidates:
        plan.debug_metrics["flat_fill_ladder_rejection_reason"] = "no_uniform_candidate"
        return cleanup
    selected_mask, selected = min(candidates, key=lambda item: float(item[1].get("score", 999.0)))
    plan.debug_metrics["flat_fill_ladder_selected_growth_px"] = int(selected["growth_px"])
    plan.debug_metrics["flat_fill_ladder_selected_px"] = int(selected["mask_px"])
    plan.debug_metrics["flat_fill_ladder_uniformity_score"] = float(selected["score"])
    plan.debug_metrics["flat_fill_ladder_fill_bgr"] = selected["median_bgr"]
    plan.debug_metrics["flat_fill_ladder_rejection_reason"] = ""
    return selected_mask


def _score_cleanup_residual(
    img_cv: np.ndarray,
    result: np.ndarray,
    plan: CleanupPlan,
    mask: np.ndarray,
) -> Dict[str, Any]:
    bg_bgr, metrics = _sample_container_bg_metrics(img_cv, plan, mask)
    if bg_bgr is None:
        bg_bgr, _conf = _estimate_plain_bg_color(
            img_cv,
            None,
            mask,
            plan.region_bbox,
            allow_dark=(plan.background_model == "dark_bubble"),
        )
    active = mask > 0
    px = result[active].astype(np.float32)
    if px.shape[0] == 0:
        return {"bad": False, "reason": "empty_mask"}
    diff = np.sqrt(np.sum((px - bg_bgr.astype(np.float32)[None, :]) ** 2, axis=1))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    bg_gray = float(0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2])
    mask_gray = gray[active].astype(np.float32)
    residual_dark = int(np.count_nonzero((diff > 42.0) & (np.abs(mask_gray - bg_gray) > 24.0)))
    far_ratio = float(residual_dark) / max(1, int(px.shape[0]))
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt(sobel_x[active] ** 2 + sobel_y[active] ** 2)))
    boundary = cv2.dilate(
        active.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    boundary &= ~active
    seam_score = 0.0
    if np.any(boundary):
        seam_score = float(abs(float(np.mean(gray[boundary])) - float(np.mean(mask_gray))))
    score = {
        "sampled_bg_bgr": [int(v) for v in bg_bgr.tolist()],
        "residual_dark_pixels": residual_dark,
        "residual_far_ratio": round(far_ratio, 4),
        "residual_edge_energy": round(edge_energy, 3),
        "color_distance_to_bg": round(float(np.mean(diff)), 3),
        "color_distance_p90": round(float(np.percentile(diff, 90)), 3),
        "seam_score": round(seam_score, 3),
    }
    score["bad"] = bool(
        far_ratio > 0.08
        or float(score["color_distance_to_bg"]) > 34.0
        or float(score["color_distance_p90"]) > 58.0
        or edge_energy > 24.0
    )
    score["sampled_bg_metrics"] = metrics
    return score


def _build_residual_guided_expansion(
    img_cv: np.ndarray,
    result: np.ndarray,
    plan: CleanupPlan,
    mask: np.ndarray,
    residual: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], int]:
    """Add only connected leftover pixels immediately next to the cleanup mask."""
    plan.debug_metrics["residual_expansion_px"] = 0
    if (
        plan.cleanup_strategy != "flat_fill"
        or plan.region_class not in ("speech_bubble", "caption_box")
        or plan.container_mask is None
        or plan.container_bbox is None
        or mask is None
        or not np.any(mask)
    ):
        return None, 0

    global_cm = normalize_mask_to_image(plan.container_mask, plan.container_bbox, img_cv.shape)
    if not np.any(global_cm):
        return None, 0
    active = mask > 0
    ring = cv2.dilate(
        active.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    ring &= ~active
    ring &= global_cm > 0
    if not np.any(ring):
        return None, 0

    bg_bgr = residual.get("sampled_bg_bgr")
    if isinstance(bg_bgr, list) and len(bg_bgr) == 3:
        bg = np.array(bg_bgr, dtype=np.float32)
    else:
        sampled, _metrics = _sample_container_bg_metrics(img_cv, plan, mask)
        if sampled is None:
            sampled, _conf = _estimate_plain_bg_color(
                img_cv,
                global_cm,
                mask,
                plan.region_bbox,
                allow_dark=(plan.background_model == "dark_bubble"),
            )
        bg = sampled.astype(np.float32)

    result_f = result.astype(np.float32)
    diff = np.sqrt(np.sum((result_f - bg[None, None, :]) ** 2, axis=2))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = float(0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2])
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    flat_light_target = plan.background_model in {"flat_light", "flat_colored", "dark_bubble"}
    diff_gate = 28.0 if flat_light_target else 36.0
    gray_gate = 14.0 if flat_light_target else 18.0
    edge_gate = 36.0 if flat_light_target else 48.0
    raw = np.where(
        ring
        & (
            ((diff > diff_gate) & (np.abs(gray - bg_gray) > gray_gate))
            | (edge > edge_gate)
        ),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(raw):
        return None, 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    mask_px = max(1, int(np.count_nonzero(mask)))
    max_total = max(32, min(int(mask_px * (0.55 if flat_light_target else 0.35)), 3200 if flat_light_target else 1800))
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > max(192 if flat_light_target else 128, int(mask_px * (0.26 if flat_light_target else 0.18))):
            continue
        comp = labels == label
        touches = cv2.dilate(
            comp.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ) > 0
        if not np.any(touches & active):
            continue
        kept[comp] = 255

    added_px = int(np.count_nonzero(kept))
    if added_px <= 0 or added_px > max_total:
        return None, 0
    expanded = cv2.bitwise_or(mask, kept)
    quality = _compute_mask_quality_metrics(
        expanded,
        global_cm,
        plan.region_bbox,
        plan.text_bbox,
        safety_bbox=_select_safety_bbox(plan),
    )
    if (
        float(quality.get("mask_container_ratio", 0.0) or 0.0) > (0.55 if flat_light_target else 0.45)
        or float(quality.get("mask_region_ratio", 0.0) or 0.0) > (0.44 if flat_light_target else 0.36)
        or (
            float(quality.get("rectangularity", 0.0) or 0.0) > (0.82 if flat_light_target else 0.70)
            and float(quality.get("mask_region_ratio", 0.0) or 0.0) > (0.24 if flat_light_target else 0.18)
        )
    ):
        return None, 0
    plan.debug_metrics["residual_expansion_px"] = added_px
    return expanded, added_px


def _mark_residual_review(plan: CleanupPlan) -> None:
    plan.skip_reason = "cleanup_residual_text_remains"
    plan.debug_metrics["review_required_after_cleanup"] = True


def _group_cleanup_masks_by_container(regions: List[CleanupPlan]) -> List[List[CleanupPlan]]:
    """Foundation stub for future grouped bubble cleanup; disabled by default."""
    # Pass C2 intentionally keeps execution per-region. A future pass can call
    # this when cleanup_allow_grouped_inpaint is enabled and the caller has a
    # page-level list of plans with shared high-confidence containers.
    return [[plan] for plan in regions]


def policy_cleanup_residual_retry_enabled(plan: CleanupPlan) -> bool:
    return bool(plan.debug_metrics.get("cleanup_residual_retry_enabled", True))


# ──────────────────────────────────────────────────────────────────────────────
# Strategy selector
# ──────────────────────────────────────────────────────────────────────────────

def select_strategy(
    region_class: str,
    background_model: str,
    text_mask_confidence: float,
    container_confidence: float,
    auto_clean_sfx: bool = False,
    policy: Optional[CleanupPolicy] = None,
) -> Tuple[str, str]:
    """
    Return (cleanup_strategy, inpaint_method) based on classification.

    cleanup_strategy : flat_fill | gradient_fill | texture_clone |
                       mask_inpaint | skip | review
    inpaint_method   : telea | ns | idw_lab | local_sample | skip
    """
    policy = policy or CleanupPolicy(auto_clean_sfx=auto_clean_sfx)
    policy._apply_mode_thresholds()

    if policy.cleanup_manual_review_only:
        return "review", "skip"

    if text_mask_confidence < 0.12 and region_class != "caption_box":
        return "skip", "skip"

    if region_class == "caption_box":
        if text_mask_confidence < 0.25:
            return "skip", "skip"
        if background_model in ("flat_light", "flat_colored", "dark_bubble"):
            return "flat_fill", "local_sample"
        if background_model in ("smooth_gradient", "translucent_gradient"):
            if background_model == "translucent_gradient" and not policy.cleanup_allow_translucent_caption:
                return "skip", "skip"
            return "gradient_fill", "idw_lab"
        if background_model == "halftone_texture":
            if not policy.cleanup_allow_texture_inpaint:
                return "skip", "skip"
            if policy.cleanup_fallback_backend == "iopaint" or policy.cleanup_prefer_iopaint_for_texture:
                return "texture_clone", "telea"
            return "review", "skip"
        return "skip", "skip"

    if region_class == "sfx":
        if (
            policy.auto_clean_sfx
            and text_mask_confidence >= policy.t2_text_conf
        ):
            return "mask_inpaint", "telea"
        return "skip", "skip"

    if region_class == "text_on_art":
        if policy.auto_clean_text_over_art and text_mask_confidence >= policy.t2_text_conf:
            if background_model in ("flat_light", "flat_colored", "dark_bubble"):
                return "flat_fill", "local_sample"
            return "mask_inpaint", "telea"
        return "skip", "skip"

    # speech_bubble, thought_bubble, unknown
    if background_model in ("flat_light", "flat_colored", "dark_bubble"):
        if text_mask_confidence >= policy.t2_text_conf:
            return "flat_fill", "local_sample"
        return "skip", "skip"

    if background_model == "smooth_gradient":
        if not policy.allow_gradient_fill:
            return "skip", "skip"
        if text_mask_confidence >= 0.20 and container_confidence >= 0.35:
            return "gradient_fill", "idw_lab"
        if text_mask_confidence >= 0.20:
            return "gradient_fill", "telea"
        return "skip", "skip"

    if background_model == "translucent_gradient":
        if not policy.allow_gradient_fill or not policy.cleanup_allow_translucent_caption:
            return "skip", "skip"
        if text_mask_confidence >= 0.20 and container_confidence >= 0.35:
            return "gradient_fill", "idw_lab"
        if text_mask_confidence >= 0.30 and policy.allow_texture_inpaint:
            return "mask_inpaint", "telea"
        return "skip", "skip"

    if background_model == "halftone_texture":
        if not policy.cleanup_allow_texture_inpaint:
            return "skip", "skip"
        if text_mask_confidence >= 0.25:
            if policy.cleanup_fallback_backend == "iopaint" or policy.cleanup_prefer_iopaint_for_texture:
                return "texture_clone", "telea"
            return "review", "skip"
        return "skip", "skip"

    if background_model == "busy_art":
        if not policy.allow_texture_inpaint:
            return "skip", "skip"
        if (
            policy.auto_clean_busy_background
            and text_mask_confidence >= policy.t2_text_conf
            and policy.busy_background_cleanup_mode in ("tight_mask", "telea")
        ):
            return "mask_inpaint", "telea"
        return "skip", "skip"

    # unknown
    if not policy.allow_texture_inpaint:
        return "review", "skip"
    if text_mask_confidence >= 0.35:
        return "mask_inpaint", "telea"
    return "review", "skip"


def _hybrid_cleanup_route(
    plan: CleanupPlan,
    policy: CleanupPolicy,
    model_config: Optional[Any],
) -> None:
    """Label and, when configured, route hard backgrounds to external inpaint."""
    bg = str(plan.background_model or "")
    region_class = str(plan.region_class or "")
    if region_class in {"sfx", "text_on_art"}:
        plan.debug_metrics["cleanup_route"] = "protected_region"
        return

    if bg in {"flat_light", "flat_colored", "dark_bubble"}:
        plan.debug_metrics["cleanup_route"] = "solid_bubble_cv"
        return
    if bg == "smooth_gradient":
        plan.debug_metrics["cleanup_route"] = "gradient_cv"
        return

    hard_bg = bg in {"halftone_texture", "busy_art", "unknown", "translucent_gradient"}
    if not hard_bg:
        plan.debug_metrics["cleanup_route"] = "opencv_fallback"
        return

    prefers_iopaint = (
        str(getattr(model_config, "cleanup_backend", "") or "").strip().lower() == "iopaint"
        or policy.cleanup_fallback_backend == "iopaint"
        or (bg == "halftone_texture" and policy.cleanup_prefer_iopaint_for_texture)
        or (bg == "translucent_gradient" and policy.cleanup_prefer_iopaint_for_translucent)
    )
    if not prefers_iopaint:
        plan.debug_metrics["cleanup_route"] = "hard_background_review_or_opencv"
        return

    url = str(getattr(model_config, "iopaint_url", "") or "").strip() if model_config is not None else ""
    plan.debug_metrics["cleanup_route"] = "model_inpaint"
    plan.debug_metrics["model_inpaint_required"] = True
    plan.cleanup_backend = "iopaint"
    plan.iopaint_url = url
    if plan.text_mask_confidence < 0.18:
        plan.cleanup_strategy = "review"
        plan.inpaint_method = "skip"
        plan.skip_reason = "model_inpaint_text_mask_confidence_low"
        return
    if not url:
        plan.cleanup_strategy = "review"
        plan.inpaint_method = "skip"
        plan.skip_reason = "iopaint_required_unavailable:no_url"
        plan.debug_metrics["cleanup_backend_fallback"] = "iopaint_required_unavailable:no_url"
        return
    if plan.cleanup_strategy in {"skip", "review"}:
        plan.cleanup_strategy = "texture_clone" if bg in {"halftone_texture", "translucent_gradient"} else "mask_inpaint"
        plan.inpaint_method = "telea"
        plan.skip_reason = ""


# ──────────────────────────────────────────────────────────────────────────────
# CleanupPlan builder
# ──────────────────────────────────────────────────────────────────────────────

def _is_yolo_caption_like(block: Any, img_cv: Optional[np.ndarray] = None) -> Tuple[bool, str]:
    detector_source = getattr(block, "detector_source", "") or ""
    if detector_source != "yolo":
        return False, "not_yolo"

    role = str(getattr(block, "bubble_role", "") or "").strip().lower()
    yolo_kind = str(getattr(block, "yolo_kind", "") or "").strip().lower()
    if role in {"caption", "narration"}:
        return True, f"role:{role}"
    if yolo_kind in {"caption", "caption_box", "narration", "narration_box", "text_box"}:
        return True, f"yolo_kind:{yolo_kind}"
    if role == "sfx" or yolo_kind == "sfx":
        return False, "sfx"

    try:
        x, y, w, h = block.bbox()
    except Exception:
        return False, "no_bbox"
    aspect = w / max(1, h)
    if aspect < 2.6 or h > 140:
        return False, f"geometry_not_caption(aspect={aspect:.2f},h={h})"

    flat_score = 0.0
    if img_cv is not None:
        h_img, w_img = img_cv.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 > x1 and y2 > y1:
            roi = img_cv[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            flat_score = float(np.std(gray))
            if flat_score > 22.0:
                return False, f"not_flat(std={flat_score:.1f})"

    bubble_mask = getattr(block, "bubble_mask", None)
    if bubble_mask is not None:
        try:
            fill = float(np.count_nonzero(bubble_mask)) / max(1, int(bubble_mask.size))
            if fill < 0.82:
                return False, f"non_rect_mask(fill={fill:.2f})"
        except Exception:
            pass

    return True, f"geometry_flat(aspect={aspect:.2f},h={h},std={flat_score:.1f})"


def _region_class_from_block(block: Any, img_cv: Optional[np.ndarray] = None) -> str:
    """Map block fields to canonical region_class string."""
    role      = getattr(block, "bubble_role", "dialog") or "dialog"
    kind      = getattr(block, "region_kind", None)
    kind_name = getattr(kind, "name", "") if kind is not None else ""
    if not kind_name and kind is not None:
        kind_name = str(kind)
    yolo_class = getattr(block, "yolo_kind", "") or ""
    detector_source = getattr(block, "detector_source", "") or ""

    if kind_name == "CAPTION_BOX":
        return "caption_box"
    if kind_name == "SFX_OVER_ART" or role == "sfx" or yolo_class == "sfx":
        return "sfx"
    if kind_name == "DIALOGUE_OVER_ART":
        return "text_on_art"
    if detector_source == "yolo":
        is_caption, _reason = _is_yolo_caption_like(block, img_cv)
        if is_caption:
            return "caption_box"
    if role in ("dialog", "thought", "bold") or kind_name in (
        "PLAIN_BUBBLE", "GRADIENT_BUBBLE", "TEXTURED_BUBBLE"
    ):
        return "speech_bubble"
    if kind_name == "UNKNOWN":
        return "unknown"
    return "unknown"


def build_cleanup_plan(
    img_cv: np.ndarray,
    block: Any,
    page_index: int = 0,
    region_id: str = "",
    cleanup_debug_artifacts: bool = False,
    cleanup_debug_dir: str = "",
    auto_clean_sfx: bool = False,
    cleanup_policy: Optional[CleanupPolicy] = None,
    model_config: Optional[Any] = None,
) -> CleanupPlan:
    """
    Build a full CleanupPlan for one OCRBlock.

    Runs all mask candidates, selects best, classifies background, determines
    strategy, and assembles the final cleanup_mask.

    Does NOT execute the plan; call execute_cleanup_plan() for that.
    """
    from backend.core.regions import RegionKind  # lazy import – avoids circular

    policy = cleanup_policy or (
        CleanupPolicy.from_config(model_config) if model_config is not None else CleanupPolicy(auto_clean_sfx=auto_clean_sfx)
    )
    policy._apply_mode_thresholds()
    policy = _policy_with_region_override(policy, block)
    override_mode = _cleanup_override_mode(block)

    plan = CleanupPlan(
        page_index      = page_index,
        region_id       = region_id or block.bbox().__str__(),
        detector_source = getattr(block, "detector_source", "ocr") or "ocr",
        yolo_class      = getattr(block, "yolo_kind", "") or "",
        region_class    = _region_class_from_block(block, img_cv),
        region_bbox     = block.bbox(),
        ocr_boxes       = list(getattr(block, "boxes", []) or []),
        ocr_confidence  = float(getattr(block, "confidence", 0.0) or 0.0),
        cleanup_debug_artifacts = cleanup_debug_artifacts,
        cleanup_debug_dir = cleanup_debug_dir,
    )
    override = getattr(block, "override", None)
    override_region_class = str(getattr(override, "cleanup_region_class", "") or "").strip()
    if override_region_class in {"speech_bubble", "caption_box", "text_on_art", "sfx"}:
        plan.region_class = override_region_class
    plan.debug_metrics["cleanup_residual_retry_enabled"] = bool(
        policy.cleanup_residual_retry_enabled
    )
    plan.debug_metrics["region_role"] = str(getattr(block, "bubble_role", "") or "")
    plan.debug_metrics["detector_source"] = str(getattr(block, "detector_source", "") or "")
    plan.debug_metrics["yolo_class"] = str(getattr(block, "yolo_kind", "") or "")
    plan.debug_metrics["yolo_class_id"] = getattr(block, "yolo_class_id", None)
    plan.debug_metrics["region_kind"] = str(getattr(getattr(block, "region_kind", None), "name", "") or "")
    plan.debug_metrics["background_kind"] = str(getattr(getattr(block, "background_kind", None), "name", "") or "")
    plan.debug_metrics["safe_rect"] = _bbox_list(getattr(block, "safe_rect", None))
    plan.debug_metrics["cleanup_safe_rect_existing"] = _bbox_list(getattr(block, "cleanup_safe_rect", None))
    plan.debug_metrics["cleanup_safe_rect_existing_confidence"] = float(getattr(block, "cleanup_safe_rect_confidence", 0.0) or 0.0)
    plan.debug_metrics["cleanup_container_bbox_existing"] = _bbox_list(getattr(block, "cleanup_container_bbox", None))
    plan.debug_metrics["cleanup_residual_retry_dilate_px"] = int(
        policy.cleanup_residual_retry_dilate_px
    )
    plan.debug_metrics["cleanup_allow_grouped_inpaint"] = bool(
        policy.cleanup_allow_grouped_inpaint
    )
    plan.debug_metrics["cleanup_fallback_backend"] = policy.cleanup_fallback_backend
    plan.debug_metrics["cleanup_override_mode"] = override_mode
    plan.debug_metrics["cleanup_allow_low_confidence"] = bool(
        getattr(override, "cleanup_allow_low_confidence", False)
    ) if override is not None else False
    plan.debug_metrics.setdefault("text_mask_candidate_scores", [])
    plan.debug_metrics.setdefault("selected_text_mask_candidate", "")
    plan.debug_metrics.setdefault("large_component_kept_count", 0)
    plan.debug_metrics.setdefault("large_component_rejected_count", 0)
    plan.debug_metrics.setdefault("component_reject_reason", "")
    plan.debug_metrics.setdefault("halo_added_px", 0)
    plan.debug_metrics.setdefault("halo_ratio_to_text_mask", 0.0)
    plan.debug_metrics.setdefault("halo_rejected_reason", "")
    plan.debug_metrics.setdefault("residual_expansion_px", 0)
    plan.debug_metrics.setdefault("final_cleanup_mask_px", 0)
    if plan.detector_source == "yolo":
        _caption_like, caption_reason = _is_yolo_caption_like(block, img_cv)
        plan.debug_metrics["caption_reason"] = caption_reason

    # ── text_bbox: union of OCR boxes; never fall back to region_bbox ────────
    plan.text_bbox = _text_bbox_from_boxes(plan.ocr_boxes, img_cv.shape)
    has_text_signal = bool(str(getattr(block, "text", "") or "").strip())
    if plan.text_bbox is None and not has_text_signal and plan.region_class != "sfx":
        plan.cleanup_strategy = "review"
        plan.inpaint_method = "skip"
        plan.skip_reason = (
            "no_ocr_text_for_cleanup"
            if plan.detector_source == "yolo"
            else "no_text_for_cleanup"
        )
        plan.debug_metrics["mask"] = {
            "region_area": int(max(1, plan.region_bbox[2] * plan.region_bbox[3])),
            "text_bbox_area": 0,
            "mask_area": 0,
            "mask_region_ratio": 0.0,
            "mask_text_ratio": 0.0,
            "reason": plan.skip_reason,
        }
        plan.debug_metrics["quality"] = _compute_mask_quality_metrics(
            None, None, plan.region_bbox, plan.text_bbox
        )
        plan.debug_metrics["selected_text_mask_candidate"] = "none"
        plan.debug_metrics["final_cleanup_mask_px"] = 0
        debug_print(
            "cleanup_quality "
            f"page={plan.page_index} region={plan.region_id} "
            f"class={plan.region_class!r} detector={plan.detector_source!r} "
            f"strategy={plan.cleanup_strategy!r} text_conf=0.00 "
            "container_conf=0.00 mask_region_ratio=0.0000 "
            "mask_container_ratio=0.0000 border_touch_ratio=0.00 "
            f"rectangularity=0.000 candidate='none' skip={plan.skip_reason!r}"
        )
        plan.log()
        return plan

    # ── Text mask candidates ──────────────────────────────────────────────────
    existing = getattr(block, "text_mask", None)
    # Discard existing mask if it's a full-bbox fill (the YOLO regression bug).
    if existing is not None:
        rx, ry, rw, rh  = plan.region_bbox
        h_img, w_img    = img_cv.shape[:2]
        x1, y1          = max(0, rx), max(0, ry)
        x2, y2          = min(w_img, rx + rw), min(h_img, ry + rh)
        region_area     = max(1, (x2 - x1) * (y2 - y1))
        existing_px     = int(np.count_nonzero(existing[y1:y2, x1:x2]))
        if existing_px / region_area > 0.85:
            plan.debug_metrics["legacy_block_text_mask_rejected_reason"] = "bbox_like_or_full_rectangle"
            plan.debug_metrics["legacy_block_text_mask_rejected_px"] = existing_px
            plan.debug_metrics["legacy_block_text_mask_rejected_coverage"] = round(existing_px / region_area, 4)
            existing = None

    candidate_debug: Dict[str, Any] = {}
    candidates = build_text_mask_candidates(
        img_cv,
        _expand_bbox(plan.text_bbox, 8, img_cv.shape) if plan.text_bbox is not None else plan.region_bbox,
        plan.ocr_boxes,
        existing_mask=existing,
        text_bbox=plan.text_bbox,
        block=block,
        region_class=plan.region_class,
        debug_metrics=candidate_debug,
    )
    candidate_rows = candidate_debug.get("text_mask_candidate_scores", [])
    if plan.debug_metrics.get("legacy_block_text_mask_rejected_reason"):
        candidate_rows = [
            {
                "reason": "existing_mask(pre_rejected)",
                "source": "legacy_block_text_mask",
                "confidence": 0.0,
                "mask_px": int(plan.debug_metrics.get("legacy_block_text_mask_rejected_px", 0) or 0),
                "accepted": False,
                "selected": False,
                "rejection_reason": str(plan.debug_metrics.get("legacy_block_text_mask_rejected_reason", "")),
                "coverage": plan.debug_metrics.get("legacy_block_text_mask_rejected_coverage"),
            },
            *candidate_rows,
        ]
    plan.debug_metrics["text_mask_candidate_scores"] = candidate_rows
    if candidates:
        best_mask, best_conf, best_reason = candidates[0]
        plan.text_mask            = best_mask
        plan.text_mask_confidence = best_conf
        plan.text_mask_reason     = best_reason
        selected_source = _candidate_source_from_reason(best_reason)
        plan.debug_metrics["selected_candidate"] = best_reason
        plan.debug_metrics["selected_text_mask_candidate"] = best_reason
        plan.debug_metrics["selected_text_mask_candidate_source"] = selected_source
        for row in plan.debug_metrics.get("text_mask_candidate_scores", []) or []:
            if isinstance(row, dict):
                row["selected"] = bool(row.get("reason") == best_reason and row.get("accepted", False))
        if plan.text_bbox is None:
            plan.text_bbox = _mask_bbox(best_mask)
            plan.debug_metrics["text_bbox_source"] = "cv_mask"
    else:
        plan.text_mask            = None
        plan.text_mask_confidence = 0.0
        plan.text_mask_reason     = "no_candidates"
        plan.debug_metrics["selected_candidate"] = "none"
        plan.debug_metrics["selected_text_mask_candidate"] = "none"
        plan.debug_metrics["selected_text_mask_candidate_source"] = "none"

    # ── Container mask ────────────────────────────────────────────────────────
    # FIX-5: SFX and text_on_art have no meaningful bubble container.
    # Building one produces a fake region that can confine or corrupt the
    # cleanup mask, so skip it entirely unless YOLO explicitly detected
    # a full bubble/container for this block.
    if plan.region_class in ("sfx", "text_on_art"):
        plan.container_mask       = None
        plan.container_bbox       = None
        plan.container_confidence = 0.0
        plan.container_reason     = "disabled_for_art_text"
    else:
        try:
            cont_local, cont_bbox, cont_conf, cont_reason = (
                build_container_mask_from_block(
                    img_cv, block,
                    text_mask=plan.text_mask,  # FIX-4
                )
            )
            plan.container_mask       = cont_local
            plan.container_bbox       = cont_bbox
            plan.container_confidence = cont_conf
            plan.container_reason     = cont_reason
        except Exception as exc:
            debug_print(f"build_cleanup_plan: container_mask failed: {exc}")
            plan.container_mask       = None
            plan.container_bbox       = None
            plan.container_confidence = 0.0
            plan.container_reason     = f"error:{exc}"

    _expand_large_glyph_components(img_cv, plan, policy)

    # ── Outline / shadow expansion ────────────────────────────────────────────
    if plan.text_mask is not None and plan.text_mask_confidence >= 0.30:
        plan.outline_shadow_mask = build_outline_shadow_mask(
            img_cv,
            plan.text_mask,
            container_mask=plan.container_mask,
            container_bbox=plan.container_bbox,
        )

    if (
        policy.cleanup_halo_mask_enabled
        and plan.text_mask is not None
        and plan.text_mask_confidence >= 0.20
    ):
        plan.halo_mask = build_text_halo_mask(
            img_cv,
            plan.text_mask,
            container_mask=plan.container_mask,
            container_bbox=plan.container_bbox,
            max_px=policy.cleanup_halo_max_px,
            region_bbox=plan.region_bbox,
            debug_metrics=plan.debug_metrics,
        )
        halo_px = int(np.count_nonzero(plan.halo_mask)) if plan.halo_mask is not None else 0
        plan.debug_metrics["halo_mask"] = {
            "enabled": True,
            "halo_px": halo_px,
            "max_px": int(policy.cleanup_halo_max_px),
        }
        plan.debug_metrics["halo_px"] = halo_px
    else:
        plan.debug_metrics["halo_mask"] = {"enabled": False, "halo_px": 0}
        plan.debug_metrics["halo_px"] = 0
        plan.debug_metrics.setdefault("halo_added_px", 0)
        plan.debug_metrics.setdefault("halo_ratio_to_text_mask", 0.0)
        plan.debug_metrics.setdefault("halo_rejected_reason", "disabled_or_low_confidence")

    # ── Background model ──────────────────────────────────────────────────────
    try:
        bg_exclude = plan.text_mask
        if plan.outline_shadow_mask is not None:
            bg_exclude = (
                plan.outline_shadow_mask
                if bg_exclude is None
                else cv2.bitwise_or(bg_exclude, plan.outline_shadow_mask)
            )
        if plan.halo_mask is not None:
            bg_exclude = (
                plan.halo_mask
                if bg_exclude is None
                else cv2.bitwise_or(bg_exclude, plan.halo_mask)
            )
        bg_model, bg_metrics = classify_background_model(
            img_cv,
            plan.region_bbox,
            container_mask=plan.container_mask,
            container_bbox=plan.container_bbox,
            container_confidence=plan.container_confidence,
            exclude_mask=bg_exclude,
            text_bbox=plan.text_bbox,
        )
        plan.background_model      = bg_model
        plan.debug_metrics["bg"]   = bg_metrics
        if (
            plan.region_class in ("speech_bubble", "caption_box")
            and plan.background_model in {"halftone_texture", "busy_art", "unknown"}
            and plan.text_mask is not None
            and plan.container_mask is not None
            and plan.container_bbox is not None
            and float(plan.container_confidence or 0.0) >= 0.55
        ):
            solid_bg_bgr, solid_bg_metrics = _sample_container_bg_metrics(
                img_cv, plan, plan.text_mask
            )
            plan.debug_metrics["solid_bg_recheck"] = solid_bg_metrics
            if solid_bg_bgr is not None:
                mean_brightness = float(solid_bg_metrics.get("mean_brightness", 0.0))
                gray_std = float(solid_bg_metrics.get("gray_std", 999.0))
                channel_std = float(solid_bg_metrics.get("channel_std_max", 999.0))
                sat_std = float(solid_bg_metrics.get("sat_std", 999.0))
                edge_density = float(solid_bg_metrics.get("edge_density", 1.0))
                if gray_std <= 16.0 and channel_std <= 20.0 and sat_std <= 24.0 and edge_density <= 0.035:
                    channel_range = int(np.max(solid_bg_bgr)) - int(np.min(solid_bg_bgr))
                    if mean_brightness < 85.0:
                        plan.background_model = "dark_bubble"
                    elif mean_brightness >= 185.0 and channel_range <= 18:
                        plan.background_model = "flat_light"
                    else:
                        plan.background_model = "flat_colored"
                    plan.debug_metrics["background_override"] = "solid_container_recheck"
        _try_adopt_existing_safe_rect_container(img_cv, plan)
        if plan.background_model == "translucent_gradient":
            detail_score = _translucent_detail_score(bg_metrics)
            plan.debug_metrics["translucent_detail_score"] = detail_score
            plan.debug_metrics["translucent_detail_class"] = (
                "hard_art" if detail_score >= 1.25 else "mild"
            )
        if (
            plan.background_model == "halftone_texture"
            and (
                policy.cleanup_fallback_backend == "iopaint"
                or policy.cleanup_prefer_iopaint_for_texture
            )
        ):
            plan.debug_metrics["cleanup_fallback_backend"] = "iopaint"
            plan.cleanup_backend = "iopaint"
            if model_config is not None:
                plan.iopaint_url = str(getattr(model_config, "iopaint_url", "") or "")
    except Exception as exc:
        debug_print(f"build_cleanup_plan: background_model failed: {exc}")
        plan.background_model = "unknown"

    # ── Strategy selection ────────────────────────────────────────────────────
    plan.cleanup_strategy, plan.inpaint_method = select_strategy(
        plan.region_class,
        plan.background_model,
        plan.text_mask_confidence,
        plan.container_confidence,
        auto_clean_sfx=auto_clean_sfx,
        policy=policy,
    )
    if plan.background_model == "translucent_gradient":
        detail_score = float(plan.debug_metrics.get("translucent_detail_score", 0.0) or 0.0)
        if detail_score >= 1.25:
            if (
                policy.cleanup_prefer_iopaint_for_translucent
                and str(getattr(model_config, "iopaint_url", "") or "").strip()
            ):
                plan.cleanup_strategy = "texture_clone"
                plan.inpaint_method = "telea"
                plan.cleanup_backend = "iopaint"
                plan.iopaint_url = str(getattr(model_config, "iopaint_url", "") or "")
                plan.debug_metrics["cleanup_fallback_backend"] = "iopaint"
            else:
                plan.cleanup_strategy = "review"
                plan.inpaint_method = "skip"
                plan.skip_reason = "translucent_art_requires_iopaint_or_review"
        elif (
            policy.cleanup_allow_translucent_caption
            and policy.allow_gradient_fill
            and plan.text_mask_confidence >= 0.20
            and plan.container_confidence >= 0.35
        ):
            plan.cleanup_strategy = "gradient_fill"
            plan.inpaint_method = "idw_lab"
    _hybrid_cleanup_route(plan, policy, model_config)
    if override_mode == "skip":
        plan.cleanup_strategy = "skip"
        plan.inpaint_method = "skip"
        plan.skip_reason = "cleanup_override_skip"
    elif override_mode == "review":
        plan.cleanup_strategy = "review"
        plan.inpaint_method = "skip"
        plan.skip_reason = "cleanup_override_review"
    elif override_mode == "force_solid":
        plan.cleanup_strategy = "flat_fill"
        plan.inpaint_method = "local_sample"
        plan.skip_reason = ""
    elif override_mode == "force_telea":
        plan.cleanup_strategy = "mask_inpaint"
        plan.inpaint_method = "telea"
        plan.skip_reason = ""
    elif override_mode == "force_ns":
        plan.cleanup_strategy = "mask_inpaint"
        plan.inpaint_method = "ns"
        plan.skip_reason = ""
    elif override_mode == "force_iopaint":
        plan.cleanup_strategy = "mask_inpaint"
        plan.inpaint_method = "telea"
        plan.cleanup_backend = "iopaint"
        plan.skip_reason = ""
    elif override_mode == "force_allow":
        if plan.cleanup_strategy in ("skip", "review"):
            plan.cleanup_strategy = "mask_inpaint"
            plan.inpaint_method = "telea"
            plan.skip_reason = ""
    elif override_mode == "force_review":
        if plan.cleanup_strategy in ("skip", "review"):
            plan.cleanup_strategy = "mask_inpaint"
            plan.inpaint_method = "telea"
        plan.skip_reason = "cleanup_override_force_review"
        plan.debug_metrics["review_required_after_cleanup"] = True

    if (
        plan.cleanup_strategy not in ("skip", "review")
        and policy.cleanup_min_container_confidence > 0.0
        and plan.region_class in ("speech_bubble", "caption_box")
        and float(plan.container_confidence or 0.0) < policy.cleanup_min_container_confidence
        and override_mode != "force_allow"
        and not bool(plan.debug_metrics.get("cleanup_allow_low_confidence", False))
    ):
        reason = f"cleanup_container_confidence_low({plan.container_confidence:.2f})"
        if policy.cleanup_risky_action == "attempt":
            plan.debug_metrics["safety_override"] = f"attempt:{reason}"
        elif policy.cleanup_risky_action == "review" or override_mode == "force_review":
            plan.skip_reason = reason
            plan.debug_metrics["review_required_after_cleanup"] = True
        else:
            plan.cleanup_strategy = "skip"
            plan.inpaint_method = "skip"
            plan.skip_reason = reason
    if plan.cleanup_strategy == "skip" and not plan.skip_reason:
        if plan.region_class == "sfx":
            plan.skip_reason = "skipped_sfx_default_policy"
        elif plan.region_class == "text_on_art":
            plan.skip_reason = "skipped_text_over_art_default_policy"
        elif plan.background_model == "busy_art":
            plan.skip_reason = "skipped_busy_background_default_policy"
        elif plan.background_model == "halftone_texture" and not policy.cleanup_allow_texture_inpaint:
            plan.skip_reason = "skipped_texture_inpaint_disabled"
        elif plan.background_model in ("translucent_gradient", "busy_art") and not policy.allow_texture_inpaint:
            plan.skip_reason = "skipped_texture_inpaint_disabled"
        elif plan.background_model in ("smooth_gradient", "translucent_gradient") and not policy.allow_gradient_fill:
            plan.skip_reason = "skipped_gradient_fill_disabled"

    # ── Assemble cleanup_mask ─────────────────────────────────────────────────
    original_strategy = plan.cleanup_strategy
    original_method = plan.inpaint_method
    original_skip_reason = plan.skip_reason
    plan.debug_metrics["strategy_before_mask_assembly"] = original_strategy
    plan.debug_metrics["pre_strategy_mask_px"] = (
        int(np.count_nonzero(plan.text_mask)) if plan.text_mask is not None else 0
    )
    solid_override_candidate = (
        policy.cleanup_solid_bubble_fill_enabled
        and override_mode not in {"skip", "review"}
        and plan.region_class in ("speech_bubble", "caption_box")
        and plan.container_bbox is not None
        and plan.container_mask is not None
        and float(plan.container_confidence or 0.0)
        >= policy.cleanup_solid_bubble_min_container_confidence
    )
    if (
        (plan.cleanup_strategy not in ("skip", "review") or solid_override_candidate)
        and plan.text_mask is not None
    ):
        cleanup = plan.text_mask.copy()
        if plan.outline_shadow_mask is not None:
            cleanup = cv2.bitwise_or(cleanup, plan.outline_shadow_mask)
        if plan.halo_mask is not None:
            cleanup = cv2.bitwise_or(cleanup, plan.halo_mask)

        # Confine to container_mask if available and high-confidence.
        # FIX-6: use normalize_mask_to_image instead of manual canvas build.
        if (
            plan.container_mask is not None
            and plan.container_bbox is not None
            and plan.container_confidence >= 0.45
        ):
            global_cm = normalize_mask_to_image(
                plan.container_mask, plan.container_bbox, img_cv.shape
            )
            confined = cv2.bitwise_and(cleanup, global_cm)
            preserves, retained_ratio = _mask_preserves_cleanup(cleanup, global_cm)
            plan.debug_metrics["container_confine_retained_ratio"] = round(retained_ratio, 4)
            # Only use confinement if the detected container preserves the
            # glyph mask. On dark narration art the container detector can lock
            # onto only half a word, which used to erase only that half.
            if np.any(confined) and preserves:
                cleanup = confined
            elif np.any(confined):
                plan.debug_metrics["container_confine_ignored"] = "partial_container_mask"

        cleanup = _optimize_flat_fill_cleanup_mask(img_cv, plan, cleanup, policy)
        reject_reason = _reject_unsafe_cleanup_mask(plan, cleanup, policy=policy)
        plan.debug_metrics["cleanup_mask_rejected"] = bool(reject_reason)
        plan.debug_metrics["cleanup_mask_rejection_reason"] = str(reject_reason or "")
        if reject_reason:
            plan.cleanup_strategy        = "skip"
            plan.inpaint_method          = "skip"
            plan.skip_reason             = reject_reason
            plan.cleanup_mask            = None
            plan.cleanup_mask_confidence = 0.0
        else:
            plan.cleanup_mask            = cleanup
            plan.cleanup_mask_confidence = min(
                plan.text_mask_confidence,
                0.95 if np.any(cleanup) else 0.0,
            )
            forced_solid = _try_force_solid_bubble_flat_fill(plan, img_cv, policy)
            safe_policy_upgrade = False
            if (
                not forced_solid
                and original_strategy in ("skip", "review")
                and plan.region_class in ("speech_bubble", "caption_box")
                and plan.background_model in {"flat_light", "flat_colored", "dark_bubble"}
                and float(plan.text_mask_confidence or 0.0) >= 0.20
                and np.any(cleanup)
            ):
                plan.cleanup_strategy = "flat_fill"
                plan.inpaint_method = "local_sample"
                plan.skip_reason = ""
                plan.debug_metrics["strategy_policy_upgrade"] = "safe_mask_available"
                safe_policy_upgrade = True
            if not forced_solid and original_strategy in ("skip", "review") and not safe_policy_upgrade:
                plan.cleanup_strategy        = original_strategy
                plan.inpaint_method          = original_method
                plan.skip_reason             = original_skip_reason
                plan.cleanup_mask            = None
                plan.cleanup_mask_confidence = 0.0
                plan.debug_metrics["mask_dropped_reason"] = "original_strategy_skip_or_review"
    else:
        plan.cleanup_mask            = None
        plan.cleanup_mask_confidence = 0.0
        plan.debug_metrics.setdefault(
            "mask",
            _cleanup_mask_metrics(None, plan.region_bbox, plan.text_bbox),
        )
        if not plan.skip_reason and plan.text_mask is None:
            plan.skip_reason = plan.text_mask_reason or "no_text_mask"
        plan.debug_metrics.setdefault("cleanup_mask_rejected", False)
        plan.debug_metrics.setdefault("cleanup_mask_rejection_reason", "")

    plan.debug_metrics["final_cleanup_mask_px"] = (
        int(np.count_nonzero(plan.cleanup_mask)) if plan.cleanup_mask is not None else 0
    )
    plan.debug_metrics["post_strategy_mask_px"] = int(plan.debug_metrics["final_cleanup_mask_px"])
    plan.debug_metrics.setdefault("residual_expansion_px", 0)
    _refresh_cleanup_quality(plan, img_cv)
    plan.debug_metrics["final_cleanup_decision"] = {
        "strategy": plan.cleanup_strategy,
        "inpaint_method": plan.inpaint_method,
        "skip_reason": plan.skip_reason,
    }
    q = plan.debug_metrics["quality"]
    debug_print(
        "cleanup_quality "
        f"page={plan.page_index} region={plan.region_id} "
        f"class={plan.region_class!r} detector={plan.detector_source!r} "
        f"strategy={plan.cleanup_strategy!r} "
        f"text_conf={plan.text_mask_confidence:.2f} "
        f"container_conf={plan.container_confidence:.2f} "
        f"mask_region_ratio={float(q.get('mask_region_ratio', 0.0)):.4f} "
        f"mask_container_ratio={float(q.get('mask_container_ratio', 0.0)):.4f} "
        f"border_touch_ratio={float(q.get('border_touch_ratio', 0.0)):.2f} "
        f"rectangularity={float(q.get('rectangularity', 0.0)):.3f} "
        f"candidate={plan.text_mask_reason!r} "
        f"skip={plan.skip_reason!r}"
    )

    plan.log()
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# Plan executor
# ──────────────────────────────────────────────────────────────────────────────

def execute_cleanup_plan(
    img_cv: np.ndarray,
    result: np.ndarray,
    plan: CleanupPlan,
) -> None:
    """
    Apply a CleanupPlan to `result` (a copy of img_cv) in-place.

    img_cv is read-only (used only for sampling / gradient reconstruction).
    result is mutated.
    """
    if plan.cleanup_strategy in ("skip", "review") or plan.cleanup_mask is None:
        debug_print(
            f"execute_cleanup_plan: skip region={plan.region_id} "
            f"strategy={plan.cleanup_strategy!r} reason={plan.skip_reason!r}"
        )
        _save_cleanup_debug_artifacts(img_cv, result, plan)
        return

    mask = plan.cleanup_mask
    if not np.any(mask):
        _save_cleanup_debug_artifacts(img_cv, result, plan)
        return

    strategy = plan.cleanup_strategy
    method   = plan.inpaint_method

    debug_print(
        f"execute_cleanup_plan: region={plan.region_id} "
        f"class={plan.region_class!r} bg={plan.background_model!r} "
        f"strategy={strategy!r} method={method!r} "
        f"mask_px={int(np.count_nonzero(mask))}"
    )

    if strategy == "flat_fill":
        _execute_flat_fill(img_cv, result, mask, plan)
        _guard_flat_bubble_smear(img_cv, result, mask, plan)
        residual = _score_cleanup_residual(img_cv, result, plan, mask)
        plan.debug_metrics["residual_score"] = residual
        retry_used = False
        if (
            bool(residual.get("bad", False))
            and policy_cleanup_residual_retry_enabled(plan)
            and plan.region_class in ("speech_bubble", "caption_box")
            and plan.container_mask is not None
            and plan.container_bbox is not None
        ):
            extra_growth = int(plan.debug_metrics.get("cleanup_flat_fill_retry_extra_growth_px", 2) or 2)
            ladder_retry = _optimize_flat_fill_cleanup_mask(
                img_cv,
                plan,
                mask,
                CleanupPolicy(
                    cleanup_flat_fill_ladder_enabled=True,
                    cleanup_flat_fill_max_growth_px=int(plan.debug_metrics.get("cleanup_flat_fill_max_growth_px", 10) or 10),
                    cleanup_flat_fill_retry_extra_growth_px=extra_growth,
                    cleanup_flat_fill_ring_px=int(plan.debug_metrics.get("cleanup_flat_fill_ring_px", 3) or 3),
                    cleanup_flat_fill_max_ring_gray_std=float(plan.debug_metrics.get("cleanup_flat_fill_max_ring_gray_std", 14.0) or 14.0),
                    cleanup_flat_fill_max_ring_chroma_std=float(plan.debug_metrics.get("cleanup_flat_fill_max_ring_chroma_std", 12.0) or 12.0),
                    cleanup_flat_fill_max_ring_edge_density=float(plan.debug_metrics.get("cleanup_flat_fill_max_ring_edge_density", 0.08) or 0.08),
                ),
                extra_growth_px=extra_growth,
            )
            if ladder_retry is not None and np.any(ladder_retry) and int(np.count_nonzero(ladder_retry)) > int(np.count_nonzero(mask)):
                retry_used = True
                plan.cleanup_mask = ladder_retry
                _execute_flat_fill(img_cv, result, ladder_retry, plan)
                residual = _score_cleanup_residual(img_cv, result, plan, ladder_retry)
                plan.debug_metrics["residual_score"] = residual
                mask = ladder_retry
            retry_mask, expansion_px = _build_residual_guided_expansion(
                img_cv, result, plan, mask, residual
            )
            if retry_mask is not None and expansion_px > 0 and np.any(retry_mask):
                retry_used = True
                plan.cleanup_mask = retry_mask
                _execute_flat_fill(img_cv, result, retry_mask, plan)
                residual = _score_cleanup_residual(img_cv, result, plan, retry_mask)
                plan.debug_metrics["residual_score"] = residual
        plan.debug_metrics["retry_used"] = retry_used
        residual_components = _detect_cleanup_residual_components(img_cv, result, plan, mask)
        component_retry = residual_components.pop("residual_retry_mask", None)
        if int(residual_components.get("residual_component_count", 0) or 0) > 0:
            plan.debug_metrics.update(residual_components)
            authoritative = int(residual_components.get("residual_component_authoritative_count", 0) or 0) > 0
            if bool(residual_components.get("residual_retry_safe", False)) and component_retry is not None and np.any(component_retry):
                retry_used = True
                union_retry = cv2.bitwise_or((mask > 0).astype(np.uint8) * 255, component_retry)
                plan.cleanup_mask = union_retry
                _execute_flat_fill(img_cv, result, union_retry, plan)
                residual = _score_cleanup_residual(img_cv, result, plan, union_retry)
                plan.debug_metrics["residual_score"] = residual
                after_components = _detect_cleanup_residual_components(img_cv, result, plan, union_retry)
                after_components.pop("residual_retry_mask", None)
                after_components["residual_component_retry_used"] = True
                plan.debug_metrics.update(after_components)
                authoritative = int(after_components.get("residual_component_authoritative_count", 0) or 0) > 0
                mask = union_retry
            if authoritative:
                _mark_residual_review(plan)
            else:
                plan.debug_metrics["cleanup_warning_reason"] = "residual_component_low_confidence"
        else:
            plan.debug_metrics.update(residual_components)
        plan.debug_metrics["retry_used"] = retry_used
        if bool(residual.get("bad", False)):
            _mark_residual_review(plan)
        plan.debug_metrics["final_cleanup_decision"] = {
            "strategy": plan.cleanup_strategy,
            "inpaint_method": plan.inpaint_method,
            "skip_reason": plan.skip_reason,
            "retry_used": retry_used,
        }
        plan.debug_metrics["final_cleanup_mask_px"] = int(np.count_nonzero(plan.cleanup_mask)) if plan.cleanup_mask is not None else 0
        _save_cleanup_debug_artifacts(img_cv, result, plan)
        return

    if strategy == "gradient_fill":
        if method == "idw_lab" and plan.container_mask is not None:
            if (
                plan.container_confidence >= 0.35
                and _execute_color_plane_fill(img_cv, result, mask, plan)
            ):
                pass
            else:
                _execute_gradient_idw(img_cv, result, mask, plan)
        else:
            if _try_external_inpaint_backend(img_cv, result, mask, plan):
                _guard_flat_bubble_smear(img_cv, result, mask, plan)
                _save_cleanup_debug_artifacts(img_cv, result, plan)
                return
            if _external_inpaint_blocked(plan):
                _save_cleanup_debug_artifacts(img_cv, result, plan)
                return
            _execute_ns(result, mask) if method == "ns" else _execute_telea(result, mask)
        _guard_flat_bubble_smear(img_cv, result, mask, plan)
        residual = _score_cleanup_residual(img_cv, result, plan, mask)
        plan.debug_metrics["residual_score"] = residual
        plan.debug_metrics["retry_used"] = False
        if method == "telea" and bool(residual.get("bad", False)):
            _mark_residual_review(plan)
        plan.debug_metrics["final_cleanup_decision"] = {
            "strategy": plan.cleanup_strategy,
            "inpaint_method": plan.inpaint_method,
            "skip_reason": plan.skip_reason,
            "retry_used": False,
        }
        _save_cleanup_debug_artifacts(img_cv, result, plan)
        return

    if strategy == "texture_clone":
        if _try_external_inpaint_backend(img_cv, result, mask, plan):
            _guard_flat_bubble_smear(img_cv, result, mask, plan)
            _save_cleanup_debug_artifacts(img_cv, result, plan)
            return
        if _external_inpaint_blocked(plan):
            _save_cleanup_debug_artifacts(img_cv, result, plan)
            return
        _execute_ns(result, mask) if method == "ns" else _execute_telea(result, mask)
        _guard_flat_bubble_smear(img_cv, result, mask, plan)
        residual = _score_cleanup_residual(img_cv, result, plan, mask)
        plan.debug_metrics["residual_score"] = residual
        plan.debug_metrics["retry_used"] = False
        if bool(residual.get("bad", False)):
            _mark_residual_review(plan)
        plan.debug_metrics["final_cleanup_decision"] = {
            "strategy": plan.cleanup_strategy,
            "inpaint_method": plan.inpaint_method,
            "skip_reason": plan.skip_reason,
            "retry_used": False,
        }
        _save_cleanup_debug_artifacts(img_cv, result, plan)
        return

    if strategy == "mask_inpaint":
        if _try_external_inpaint_backend(img_cv, result, mask, plan):
            _guard_flat_bubble_smear(img_cv, result, mask, plan)
            _save_cleanup_debug_artifacts(img_cv, result, plan)
            return
        if _external_inpaint_blocked(plan):
            _save_cleanup_debug_artifacts(img_cv, result, plan)
            return
        _execute_ns(result, mask) if method == "ns" else _execute_telea(result, mask)
        _guard_flat_bubble_smear(img_cv, result, mask, plan)
        residual = _score_cleanup_residual(img_cv, result, plan, mask)
        plan.debug_metrics["residual_score"] = residual
        plan.debug_metrics["retry_used"] = False
        if bool(residual.get("bad", False)):
            _mark_residual_review(plan)
        plan.debug_metrics["final_cleanup_decision"] = {
            "strategy": plan.cleanup_strategy,
            "inpaint_method": plan.inpaint_method,
            "skip_reason": plan.skip_reason,
            "retry_used": False,
        }
        _save_cleanup_debug_artifacts(img_cv, result, plan)
        return

    _save_cleanup_debug_artifacts(img_cv, result, plan)


def _crop_from_bbox(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    h, w = arr.shape[:2]
    x, y, bw, bh = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return arr[y1:y2, x1:x2]


def _safe_region_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _union_bboxes(
    image_shape: Tuple[int, ...],
    *bboxes: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    h_img, w_img = image_shape[:2]
    xs1: List[int] = []
    ys1: List[int] = []
    xs2: List[int] = []
    ys2: List[int] = []
    for bbox in bboxes:
        if bbox is None:
            continue
        x, y, w, h = [int(v) for v in bbox]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    if not xs1:
        return None
    x1, y1 = min(xs1), min(ys1)
    x2, y2 = max(xs2), max(ys2)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _write_mask_crop(path: str, mask: Optional[np.ndarray], crop_bbox: Tuple[int, int, int, int]) -> bool:
    if mask is None:
        return False
    crop = _crop_from_bbox(mask, crop_bbox)
    if crop is None:
        return False
    return bool(cv2.imwrite(path, crop))


def _overlay_cleanup_debug(
    raw_crop: np.ndarray,
    crop_bbox: Tuple[int, int, int, int],
    text_mask: Optional[np.ndarray],
    outline_shadow_mask: Optional[np.ndarray],
    halo_mask: Optional[np.ndarray],
    container_mask: Optional[np.ndarray],
    cleanup_mask: Optional[np.ndarray],
) -> np.ndarray:
    overlay = raw_crop.copy()
    color_layers = [
        (container_mask, np.array([80, 80, 255], dtype=np.uint8), 0.25),
        (text_mask, np.array([0, 255, 0], dtype=np.uint8), 0.55),
        (outline_shadow_mask, np.array([255, 0, 255], dtype=np.uint8), 0.55),
        (halo_mask, np.array([0, 255, 255], dtype=np.uint8), 0.55),
        (cleanup_mask, np.array([0, 0, 255], dtype=np.uint8), 0.35),
    ]
    for mask, color, alpha in color_layers:
        if mask is None:
            continue
        crop = _crop_from_bbox(mask, crop_bbox)
        if crop is None or not np.any(crop):
            continue
        active = crop > 0
        overlay[active] = (
            overlay[active].astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)
    return overlay


def _region_index_from_id(region_id: str) -> Optional[int]:
    text = str(region_id or "")
    if text.upper().startswith("R-"):
        try:
            return max(0, int(text.split("-", 1)[1]) - 1)
        except Exception:
            return None
    return None


def _cleanup_debug_meta(
    plan: CleanupPlan,
    crop_bbox: Tuple[int, int, int, int],
    files: Dict[str, bool],
) -> Dict[str, Any]:
    quality = plan.debug_metrics.get("quality", {}) or {}
    selected_reason = str(plan.debug_metrics.get("selected_text_mask_candidate", plan.text_mask_reason) or "")
    candidate_rows = plan.debug_metrics.get("text_mask_candidate_scores", []) or []
    cleanup_backend = str(plan.cleanup_backend or "opencv")
    backend_label = "skipped"
    if bool(plan.debug_metrics.get("manual_mask_used", False)):
        backend_label = "manual_mask"
    elif plan.cleanup_strategy not in ("skip", "review") and plan.cleanup_mask is not None:
        backend_label = "future-LaMa placeholder" if cleanup_backend == "lama" else "CV-only"
    meta = {
        "page_index": int(plan.page_index),
        "region_id": str(plan.region_id or ""),
        "region_index": _region_index_from_id(str(plan.region_id or "")),
        "role": plan.debug_metrics.get("region_role", ""),
        "detector_source": plan.detector_source,
        "yolo_class": plan.yolo_class,
        "yolo_class_id": plan.debug_metrics.get("yolo_class_id"),
        "region_class": plan.region_class,
        "region_kind": plan.debug_metrics.get("region_kind", ""),
        "background_kind": plan.debug_metrics.get("background_kind", ""),
        "bbox": _bbox_list(plan.region_bbox),
        "debug_crop_bbox": _bbox_list(crop_bbox),
        "text_bbox": _bbox_list(plan.text_bbox),
        "cleanup_container_bbox": _bbox_list(plan.container_bbox),
        "safe_rect": plan.debug_metrics.get("safe_rect"),
        "cleanup_safe_rect": (
            _bbox_list(plan.debug_metrics.get("cleanup_safe_rect"))
            if isinstance(plan.debug_metrics.get("cleanup_safe_rect"), tuple)
            else plan.debug_metrics.get("cleanup_safe_rect") or plan.debug_metrics.get("cleanup_safe_rect_existing")
        ),
        "chosen_cleanup_strategy": plan.cleanup_strategy,
        "chosen_inpaint_method": plan.inpaint_method,
        "chosen_cleanup_tier": plan.debug_metrics.get("cleanup_tier"),
        "cleanup_status": plan.debug_metrics.get("cleanup_status"),
        "cleanup_reason": plan.debug_metrics.get("cleanup_reason"),
        "skip_reason": plan.skip_reason or None,
        "selected_text_mask_candidate": selected_reason or None,
        "selected_text_mask_candidate_source": plan.debug_metrics.get(
            "selected_text_mask_candidate_source",
            _candidate_source_from_reason(selected_reason),
        ),
        "text_mask_candidates": candidate_rows,
        "text_confidence": round(float(plan.text_mask_confidence or 0.0), 4),
        "container_confidence": round(float(plan.container_confidence or 0.0), 4),
        "border_collision_score": round(float(quality.get("border_touch_ratio", 0.0) or 0.0), 4),
        "mask_container_ratio": round(float(quality.get("mask_container_ratio", 0.0) or 0.0), 4),
        "mask_region_ratio": round(float(quality.get("mask_region_ratio", 0.0) or 0.0), 4),
        "rectangularity": round(float(quality.get("rectangularity", 0.0) or 0.0), 4),
        "border_collision_bbox_source": quality.get("safety_bbox_source"),
        "cleanup_mask_rejected": bool(plan.debug_metrics.get("cleanup_mask_rejected", False)),
        "rejection_reason": str(plan.debug_metrics.get("cleanup_mask_rejection_reason", "") or "") or None,
        "cleanup_execution": backend_label,
        "cleanup_backend": cleanup_backend,
        "manual_mask_used": bool(plan.debug_metrics.get("manual_mask_used", False)),
        "diagnostic_only": bool(plan.debug_metrics.get("diagnostic_only", False)),
        "diagnostic_cleanup_ran": bool(plan.debug_metrics.get("diagnostic_cleanup_ran", False)),
        "destructive_cleanup_executed": bool(plan.debug_metrics.get("destructive_cleanup_executed", False)),
        "production_patch_accepted": bool(plan.debug_metrics.get("production_patch_accepted", False)),
        "proposal_valid": bool(plan.debug_metrics.get("proposal_valid", False)),
        "proposal_failure_reason": str(plan.debug_metrics.get("proposal_failure_reason", "") or "") or None,
        "cleanup_failure_reason": str(plan.debug_metrics.get("cleanup_failure_reason", "") or "") or None,
        "gate_violation": bool(plan.debug_metrics.get("gate_violation", False)),
        "residual_text_visible": bool(plan.debug_metrics.get("residual_text_visible", False)),
        "visual_quality_ok": bool(plan.debug_metrics.get("visual_quality_ok", True)),
        "fill_patch_visible": bool(plan.debug_metrics.get("fill_patch_visible", False)),
        "cleanup_effective": bool(plan.debug_metrics.get("cleanup_effective", False)),
        "bbox_like_or_full_rectangle_rejected": bool(
            "bbox_matches" in str(plan.skip_reason or "")
            or "rectangular" in str(plan.skip_reason or "")
            or any(
                isinstance(row, dict)
                and "bbox_like_or_full_rectangle" in str(row.get("rejection_reason", ""))
                for row in candidate_rows
            )
        ),
        "mask_pixels": {
            "text_mask": _mask_px(plan.text_mask),
            "outline_shadow_mask": _mask_px(plan.outline_shadow_mask),
            "halo_mask": _mask_px(plan.halo_mask),
            "container_mask": _mask_px(plan.container_mask),
            "cleanup_mask": _mask_px(plan.cleanup_mask),
        },
        "artifacts": {name: (bool(ok) if ok else None) for name, ok in files.items()},
        "debug_metrics": _json_safe(plan.debug_metrics),
    }
    return _json_safe(meta)


def _save_cleanup_debug_artifacts(
    img_cv: np.ndarray,
    result: np.ndarray,
    plan: CleanupPlan,
) -> None:
    if not plan.cleanup_debug_artifacts or plan.region_bbox is None:
        return
    try:
        out_dir = (plan.cleanup_debug_dir or "").strip()
        if not out_dir:
            out_dir = os.path.join(os.getcwd(), "debug_cleanup")
        page_dir = os.path.join(out_dir, f"page_{int(plan.page_index):03d}")
        os.makedirs(page_dir, exist_ok=True)
        rid = _safe_region_filename_part(str(plan.region_id or "region"))
        prefix = os.path.join(page_dir, rid)

        container_full = (
            normalize_mask_to_image(plan.container_mask, plan.container_bbox, img_cv.shape)
            if plan.container_mask is not None and plan.container_bbox is not None
            else None
        )
        crop_bbox = _union_bboxes(
            img_cv.shape,
            plan.region_bbox,
            plan.text_bbox,
            plan.container_bbox,
            _mask_bbox(plan.text_mask) if plan.text_mask is not None else None,
            _mask_bbox(plan.outline_shadow_mask) if plan.outline_shadow_mask is not None else None,
            _mask_bbox(plan.halo_mask) if plan.halo_mask is not None else None,
            _mask_bbox(plan.cleanup_mask) if plan.cleanup_mask is not None else None,
        ) or plan.region_bbox

        files: Dict[str, bool] = {
            "raw": False,
            "text_mask": False,
            "outline_shadow_mask": False,
            "halo_mask": False,
            "container_mask": False,
            "cleanup_mask": False,
            "overlay": False,
            "cleaned": False,
            "meta": False,
        }

        raw_crop = _crop_from_bbox(img_cv, crop_bbox)
        cleaned_crop = _crop_from_bbox(result, crop_bbox)
        if raw_crop is not None:
            files["raw"] = bool(cv2.imwrite(f"{prefix}_raw.png", raw_crop))
        if cleaned_crop is not None:
            files["cleaned"] = bool(cv2.imwrite(f"{prefix}_cleaned.png", cleaned_crop))

        files["text_mask"] = _write_mask_crop(f"{prefix}_text_mask.png", plan.text_mask, crop_bbox)
        files["outline_shadow_mask"] = _write_mask_crop(f"{prefix}_outline_shadow_mask.png", plan.outline_shadow_mask, crop_bbox)
        files["halo_mask"] = _write_mask_crop(f"{prefix}_halo_mask.png", plan.halo_mask, crop_bbox)
        files["container_mask"] = _write_mask_crop(f"{prefix}_container_mask.png", container_full, crop_bbox)
        files["cleanup_mask"] = _write_mask_crop(f"{prefix}_cleanup_mask.png", plan.cleanup_mask, crop_bbox)

        if raw_crop is not None:
            overlay = _overlay_cleanup_debug(
                raw_crop,
                crop_bbox,
                plan.text_mask,
                plan.outline_shadow_mask,
                plan.halo_mask,
                container_full,
                plan.cleanup_mask,
            )
            files["overlay"] = bool(cv2.imwrite(f"{prefix}_overlay.png", overlay))

        meta = _cleanup_debug_meta(plan, crop_bbox, files)
        with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        files["meta"] = True
        if files["meta"]:
            meta["artifacts"]["meta"] = True
            with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        debug_print(
            f"cleanup_debug_artifacts: wrote page={plan.page_index} "
            f"region={plan.region_id} dir={page_dir!r} "
            f"mask_px={_mask_px(plan.cleanup_mask)} "
            f"candidate={plan.text_mask_reason!r}"
        )
    except Exception as exc:
        debug_print(
            f"cleanup_debug_artifacts: failed region={plan.region_id} reason={exc}"
        )


def _execute_flat_fill(
    img_cv: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    plan: CleanupPlan,
) -> None:
    """
    Fill mask pixels with a locally-sampled background colour.

    Uses the bubble/caption interior when available, samples a robust BGR
    background colour, then blends a small feather clipped to the safe area.
    """
    rx, ry, rw, rh = plan.region_bbox
    h, w           = img_cv.shape[:2]
    x1, y1         = max(0, rx), max(0, ry)
    x2, y2         = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return

    global_cm = _container_mask_to_full_image(plan, img_cv.shape)
    use_container = False
    if global_cm is not None and plan.container_confidence >= 0.35:
        use_container, retained_ratio = _mask_preserves_cleanup(mask, global_cm)
        plan.debug_metrics["flat_fill_container_retained_ratio"] = round(retained_ratio, 4)
        if not use_container:
            plan.debug_metrics["flat_fill_container_ignored"] = "partial_container_mask"
    ladder_fill = plan.debug_metrics.get("flat_fill_ladder_fill_bgr")
    if isinstance(ladder_fill, list) and len(ladder_fill) == 3:
        bg_bgr = np.array(ladder_fill, dtype=np.uint8)
        bg_conf = 0.95
        plan.debug_metrics["flat_fill_color_source"] = "ladder_border_median"
    else:
        bg_bgr, bg_conf = _estimate_plain_bg_color(
            img_cv,
            global_cm if use_container else None,
            mask,
            plan.region_bbox,
            allow_dark=(plan.background_model == "dark_bubble"),
        )
        plan.debug_metrics["flat_fill_color_source"] = "estimated_plain_bg"
    if bg_conf < 0.25:
        debug_print(
            f"_execute_flat_fill: low bg sample confidence {bg_conf:.2f}, "
            f"falling back to telea for region={plan.region_id}"
        )
        _execute_telea(result, mask)
        return

    alpha_core = (mask > 0).astype(np.float32)
    feather_support = cv2.dilate(
        (mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    alpha = cv2.GaussianBlur(alpha_core, (5, 5), 0)
    alpha = np.maximum(alpha, alpha_core)
    alpha = np.clip(alpha, 0.0, 1.0)

    safe = np.zeros(img_cv.shape[:2], dtype=bool)
    safe[y1:y2, x1:x2] = True
    if global_cm is not None and use_container:
        safe &= cv2.erode(
            (global_cm > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0

    blend_area = (feather_support > 0) & safe & (alpha > 0.01)
    write_area = blend_area & (mask > 0)
    if not np.any(blend_area):
        write_area = (mask > 0) & safe
    if not np.any(write_area):
        _execute_telea(result, mask)
        return

    fill = np.zeros_like(result, dtype=np.float32)
    fill[:, :] = bg_bgr.astype(np.float32)
    result_f = result.astype(np.float32)
    alpha3 = alpha[:, :, None]
    blended = (fill * alpha3 + result_f * (1.0 - alpha3)).clip(0, 255).astype(np.uint8)
    result[write_area] = blended[write_area]
    debug_print(
        f"_execute_flat_fill: region={plan.region_id} "
        f"bg_bgr={bg_bgr.tolist()} bg_conf={bg_conf:.2f} "
        f"mask_px={int(np.count_nonzero(mask))} "
        f"blend_px={int(np.count_nonzero(blend_area))} "
        f"write_px={int(np.count_nonzero(write_area))}"
    )


def _guard_flat_bubble_smear(
    img_cv: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    plan: CleanupPlan,
) -> None:
    """Retry flat bubble cleanup with sampled fill if inpaint dirties the bubble."""
    if (
        plan.region_class not in ("speech_bubble", "caption_box")
        or plan.background_model not in ("flat_light", "flat_colored")
        or mask is None
        or not np.any(mask)
    ):
        return

    rx, ry, rw, rh = plan.region_bbox
    h, w = img_cv.shape[:2]
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img_cv[y1:y2, x1:x2]
    m_roi = mask[y1:y2, x1:x2] > 0
    bg_mask = ~m_roi

    if (
        plan.container_mask is not None
        and plan.container_bbox is not None
        and plan.container_confidence >= 0.40
    ):
        bx, by, bw, bh = plan.container_bbox
        cm_x1 = max(0, bx - x1); cm_y1 = max(0, by - y1)
        cm_x2 = min(x2 - x1, bx - x1 + bw)
        cm_y2 = min(y2 - y1, by - y1 + bh)
        lm_x1 = max(0, x1 - bx); lm_y1 = max(0, y1 - by)
        lm_x2 = lm_x1 + (cm_x2 - cm_x1)
        lm_y2 = lm_y1 + (cm_y2 - cm_y1)
        local_bg = np.zeros_like(bg_mask)
        if cm_x2 > cm_x1 and cm_y2 > cm_y1:
            try:
                cm_slice = plan.container_mask[lm_y1:lm_y2, lm_x1:lm_x2] > 0
                local_bg[cm_y1:cm_y2, cm_x1:cm_x2] = cm_slice
                bg_mask = local_bg & ~m_roi
            except Exception:
                pass

    bg_pixels = roi[bg_mask]
    if bg_pixels.shape[0] < 8:
        return

    bg_bgr = np.median(bg_pixels.reshape(-1, 3).astype(np.float32), axis=0)
    cleaned_pixels = result[y1:y2, x1:x2][m_roi].astype(np.float32)
    if cleaned_pixels.shape[0] == 0:
        return
    diff = np.sqrt(np.sum((cleaned_pixels - bg_bgr[None, :]) ** 2, axis=1))
    gray_std = float(np.std(cv2.cvtColor(
        cleaned_pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    )))
    mean_diff = float(np.mean(diff))
    if mean_diff > 34.0 or gray_std > 24.0:
        result[mask > 0] = bg_bgr.astype(np.uint8)
        debug_print(
            f"flat_bubble_smear_guard: retry_local_fill region={plan.region_id} "
            f"mean_diff={mean_diff:.1f} gray_std={gray_std:.1f} "
            f"bg_bgr={bg_bgr.astype(np.uint8).tolist()}"
        )


def _fit_color_plane(
    support_pixels_yx: np.ndarray,
    support_colors_bgr: np.ndarray,
    roi_shape: Tuple[int, int],
) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Fit a linear color plane per channel: intensity = a*x_norm + b*y_norm + d.
    """
    roi_h, roi_w = roi_shape
    yx = support_pixels_yx.astype(np.float32)
    colors = support_colors_bgr.astype(np.float32)
    x_norm = yx[:, 1] / max(1, roi_w - 1)
    y_norm = yx[:, 0] / max(1, roi_h - 1)
    a_mat = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)
    coeffs: List[Tuple[float, float, float]] = []
    residuals: List[float] = []
    for ch in range(3):
        b_vec = colors[:, ch]
        result_lstsq, _res, _rank, _sing = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        coeffs.append(tuple(float(v) for v in result_lstsq.tolist()))
        predicted = a_mat @ result_lstsq
        residuals.append(float(np.mean(np.abs(predicted - b_vec))))
    return coeffs, float(np.mean(residuals))


def _execute_color_plane_fill(
    img_cv: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    plan: CleanupPlan,
) -> bool:
    """
    Fill cleanup mask pixels using a linear color plane fitted to container support.

    Returns True only when the color-plane fill wrote into result. Existing
    gradient fallback logic remains in execute_cleanup_plan().
    """
    if plan.container_mask is None or plan.container_bbox is None:
        return False

    bx, by, bw, bh = plan.container_bbox
    h_img, w_img = img_cv.shape[:2]
    x1, y1 = max(0, bx), max(0, by)
    x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
    if x2 <= x1 or y2 <= y1:
        return False

    global_cm = normalize_mask_to_image(plan.container_mask, plan.container_bbox, img_cv.shape)
    safe_container = cv2.erode(
        (global_cm > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0

    exclude = cv2.dilate(
        (mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    if plan.outline_shadow_mask is not None:
        exclude |= plan.outline_shadow_mask > 0

    support = safe_container & ~exclude
    ys_sup, xs_sup = np.where(support)
    if len(ys_sup) < 30:
        plan.debug_metrics["gradient_color_plane"] = "fallback:support_lt_30"
        return False

    if len(ys_sup) > 5000:
        idx = np.random.choice(len(ys_sup), 5000, replace=False)
        ys_sup = ys_sup[idx]
        xs_sup = xs_sup[idx]

    support_yx_local = np.stack(
        [(ys_sup - y1).astype(np.float32), (xs_sup - x1).astype(np.float32)],
        axis=1,
    )
    support_bgr = img_cv[ys_sup, xs_sup].astype(np.float32)
    coeffs, fit_error = _fit_color_plane(support_yx_local, support_bgr, (y2 - y1, x2 - x1))
    plan.debug_metrics["gradient_fit_error"] = round(float(fit_error), 3)
    if fit_error > 18.0:
        plan.debug_metrics["gradient_color_plane"] = f"fallback:fit_error({fit_error:.2f})"
        return False

    alpha_core = (mask > 0).astype(np.float32)
    feather_support = cv2.dilate(
        (mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    alpha = cv2.GaussianBlur(alpha_core, (5, 5), 0)
    alpha = np.maximum(alpha, alpha_core)
    alpha = np.clip(alpha, 0.0, 1.0)
    blend_area = (feather_support > 0) & safe_container & (alpha > (4.0 / 255.0))
    write_area = blend_area & (mask > 0)
    if not np.any(write_area):
        plan.debug_metrics["gradient_color_plane"] = "fallback:no_blend_area"
        return False

    ys_fill, xs_fill = np.where(write_area)
    local_y = (ys_fill - y1).astype(np.float32)
    local_x = (xs_fill - x1).astype(np.float32)
    x_norm = local_x / max(1, (x2 - x1) - 1)
    y_norm = local_y / max(1, (y2 - y1) - 1)
    pred = np.empty((len(ys_fill), 3), dtype=np.float32)
    for ch, (a, b, d) in enumerate(coeffs):
        pred[:, ch] = a * x_norm + b * y_norm + d
    pred = np.clip(pred, 0, 255)

    alpha_vals = alpha[ys_fill, xs_fill][:, None]
    original = result[ys_fill, xs_fill].astype(np.float32)
    blended = (pred * alpha_vals + original * (1.0 - alpha_vals)).clip(0, 255).astype(np.uint8)
    result[ys_fill, xs_fill] = blended
    plan.debug_metrics["gradient_color_plane"] = "ok"
    debug_print(
        f"_execute_color_plane_fill: region={plan.region_id} "
        f"fit_error={fit_error:.2f} support_px={len(support_yx_local)} "
        f"blend_px={int(np.count_nonzero(blend_area))} "
        f"write_px={len(ys_fill)}"
    )
    return True


def _execute_gradient_idw(
    img_cv: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    plan: CleanupPlan,
) -> None:
    """Gradient-aware fill using Lab IDW reconstruction."""
    if plan.container_mask is None or plan.container_bbox is None:
        _execute_telea(result, mask)
        return

    # FIX-6: use normalize_mask_to_image instead of manual canvas build.
    global_cm = normalize_mask_to_image(
        plan.container_mask, plan.container_bbox, img_cv.shape
    )
    support = np.where((global_cm > 0) & (mask == 0), 255, 0).astype(np.uint8)

    if not np.any(support):
        _execute_telea(result, mask)
        return

    reconstructed        = gradient_reconstruct_idw(img_cv, mask, support)
    result[mask > 0]     = reconstructed[mask > 0]
    debug_print(
        f"_execute_gradient_idw: region={plan.region_id} "
        f"mask_px={int(mask.sum() // 255)} "
        f"support_px={int(np.count_nonzero(support))}"
    )


def _execute_telea(
    result: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
) -> None:
    """TELEA inpaint over mask; write-back only within mask."""
    inpainted        = cv2.inpaint(result, mask, radius, cv2.INPAINT_TELEA)
    result[mask > 0] = inpainted[mask > 0]


def _execute_ns(
    result: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
) -> None:
    """OpenCV Navier-Stokes inpaint over mask; write-back only within mask."""
    inpainted        = cv2.inpaint(result, mask, radius, cv2.INPAINT_NS)
    result[mask > 0] = inpainted[mask > 0]


def _try_external_inpaint_backend(
    img_cv: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    plan: CleanupPlan,
) -> bool:
    backend = (plan.cleanup_backend or "opencv").strip().lower()
    if backend not in {"iopaint", "lama"}:
        return False
    if backend == "iopaint" and not (plan.iopaint_url or "").strip():
        plan.debug_metrics["cleanup_backend_fallback"] = "iopaint_fallback:no_url"
        if bool(plan.debug_metrics.get("model_inpaint_required", False)):
            plan.debug_metrics["external_inpaint_blocked"] = True
            plan.cleanup_strategy = "review"
            plan.inpaint_method = "skip"
            plan.skip_reason = "iopaint_required_unavailable:no_url"
        return False
    if backend == "lama":
        plan.debug_metrics["cleanup_backend_fallback"] = "lama_fallback:not_configured"
        return False
    try:
        ok_img, img_buf = cv2.imencode(".png", img_cv)
        ok_mask, mask_buf = cv2.imencode(".png", mask)
        if not ok_img or not ok_mask:
            raise RuntimeError("encode_failed")
        resp = requests.post(
            plan.iopaint_url,
            files={
                "image": ("image.png", img_buf.tobytes(), "image/png"),
                "mask": ("mask.png", mask_buf.tobytes(), "image/png"),
            },
            timeout=8,
        )
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2] != result.shape[:2]:
            raise RuntimeError("invalid_output")
        result[mask > 0] = decoded[mask > 0]
        debug_print(
            f"external_inpaint_backend: backend={backend} region={plan.region_id} "
            f"mask_px={int(np.count_nonzero(mask))}"
        )
        return True
    except Exception as exc:
        plan.debug_metrics["cleanup_backend_fallback"] = f"{backend}_fallback:{exc}"
        if bool(plan.debug_metrics.get("model_inpaint_required", False)):
            plan.debug_metrics["external_inpaint_blocked"] = True
            plan.cleanup_strategy = "review"
            plan.inpaint_method = "skip"
            plan.skip_reason = f"{backend}_required_unavailable:{exc}"
        debug_print(
            f"external_inpaint_backend: fallback_to_opencv region={plan.region_id} "
            f"backend={backend} reason={exc}"
        )
        return False


def _external_inpaint_blocked(plan: CleanupPlan) -> bool:
    return bool(plan.debug_metrics.get("external_inpaint_blocked", False))


# ──────────────────────────────────────────────────────────────────────────────
# Top-level: erase one page using CleanupPlan
# ──────────────────────────────────────────────────────────────────────────────

def erase_text_region_planned(
    img_cv: np.ndarray,
    bubbles: List[Any],
    page_index: int = 0,
    cleanup_backend: str = "opencv",
    iopaint_url: str = "",
    cleanup_debug_artifacts: bool = False,
    cleanup_debug_dir: str = "",
    auto_clean_sfx: bool = False,
    cleanup_mode: str = "balanced",
    cleanup_policy: Optional[CleanupPolicy] = None,
    model_config: Optional[Any] = None,
) -> np.ndarray:
    """
    Manhwa-aware erase pipeline using CleanupPlan for each region.

    Replaces the old bbox/bg_color-based pipeline.
    Returns a copy of img_cv with original text removed.

    Integration note for engine.py
    ───────────────────────────────
    cleanup_current_page() should call this function, NOT the old
    erase_text_region().  Example:

        from backend.core.cleanup_plan import erase_text_region_planned
        page.cleaned_cv = erase_text_region_planned(
            self._raw_cv, self._regions, page_index=self._page_idx
        )
    """
    result = img_cv.copy()
    policy = cleanup_policy or (
        CleanupPolicy.from_config(model_config) if model_config is not None else CleanupPolicy(
            cleanup_mode=cleanup_mode,
            auto_clean_sfx=auto_clean_sfx,
        )
    )
    policy._apply_mode_thresholds()

    for idx, block in enumerate(bubbles):
        if not getattr(block, "visible", True):
            continue

        region_id = f"R-{idx + 1:02d}"
        debug_print(
            f'cleanup_route="planned" page={page_index} '
            f"region={idx} region_id={region_id}"
        )

        # ── Honour only explicit manual strategy overrides (Phase 3) ─────────
        override_strategy = None
        override = getattr(block, "override", None)
        if override is not None:
            eff = getattr(override, "cleanup_strategy", None)
            if eff not in ("auto", None, ""):
                override_strategy = eff

        plan = build_cleanup_plan(
            img_cv,
            block,
            page_index=page_index,
            region_id=region_id,
            cleanup_debug_artifacts=cleanup_debug_artifacts,
            cleanup_debug_dir=cleanup_debug_dir,
            auto_clean_sfx=auto_clean_sfx,
            cleanup_policy=policy,
            model_config=model_config,
        )
        plan.cleanup_backend = cleanup_backend or "opencv"
        plan.iopaint_url = iopaint_url or ""
        if (
            _cleanup_override_mode(block) == "force_iopaint"
            or plan.debug_metrics.get("cleanup_fallback_backend") == "iopaint"
        ):
            plan.cleanup_backend = "iopaint"

        # Apply explicit override if set, but still route through the plan
        # so all mask-safety invariants are honoured.
        if override_strategy is not None and override_strategy not in (
            "review", "skip"
        ):
            old_strategy = plan.cleanup_strategy
            old_method = plan.inpaint_method
            plan.cleanup_strategy = override_strategy
            if override_strategy == "flat_fill":
                plan.inpaint_method = "local_sample"
            elif override_strategy == "gradient_fill":
                plan.inpaint_method = (
                    "idw_lab" if plan.container_confidence >= 0.45 else "telea"
                )
            elif override_strategy == "texture_clone":
                plan.inpaint_method = "telea"
            elif override_strategy in ("mask_only_inpaint", "mask_inpaint"):
                plan.cleanup_strategy = "mask_inpaint"
                plan.inpaint_method   = "telea"
            debug_print(
                "cleanup_strategy_override source=manual "
                f"page={page_index} region={idx} old={old_strategy!r}/"
                f"{old_method!r} new={plan.cleanup_strategy!r}/"
                f"{plan.inpaint_method!r}"
            )

            # Rebuild cleanup_mask from plan.text_mask under the new strategy.
            if plan.text_mask is not None and np.any(plan.text_mask):
                cleanup = plan.text_mask.copy()
                if plan.outline_shadow_mask is not None:
                    cleanup = cv2.bitwise_or(cleanup, plan.outline_shadow_mask)
                if plan.halo_mask is not None:
                    cleanup = cv2.bitwise_or(cleanup, plan.halo_mask)
                reject_reason = _reject_unsafe_cleanup_mask(plan, cleanup, policy=policy)
                if reject_reason:
                    plan.cleanup_strategy        = "skip"
                    plan.inpaint_method          = "skip"
                    plan.skip_reason             = reject_reason
                    plan.cleanup_mask            = None
                    plan.cleanup_mask_confidence = 0.0
                else:
                    plan.cleanup_mask            = cleanup
                    plan.cleanup_mask_confidence = plan.text_mask_confidence

        execute_cleanup_plan(img_cv, result, plan)
        _write_cleanup_metadata_to_block(block, plan, img_cv, policy=policy)
        if (
            plan.cleanup_strategy in ("skip", "review")
            and plan.skip_reason
            and hasattr(block, "flag")
            and not bool(getattr(block, "is_flagged", False))
        ):
            block.flag(
                plan.skip_reason,
                {
                    "cleanup_route": "planned",
                    "page_index": page_index,
                    "region_index": idx,
                    "region_id": region_id,
                },
            )
        fallback = str(plan.debug_metrics.get("cleanup_backend_fallback", ""))
        if fallback and hasattr(block, "flag") and not bool(getattr(block, "is_flagged", False)):
            block.flag("iopaint_fallback", {"reason": fallback})

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Debug-only summary helper (for tools/eval_cleanup.py)
# ──────────────────────────────────────────────────────────────────────────────

def summarize_cleanup_plan(plan: "CleanupPlan") -> Dict[str, Any]:
    """Return a JSON-safe scalar summary of a CleanupPlan for eval tooling.

    Does NOT change plan state.  Never expose through frontend/API.
    """
    quality: Dict[str, Any] = plan.debug_metrics.get("quality", {}) or {}
    safe_rect = plan.debug_metrics.get("cleanup_safe_rect")
    safe_conf = float(plan.debug_metrics.get("cleanup_safe_rect_confidence", 0.0) or 0.0)
    safe_reason = str(plan.debug_metrics.get("cleanup_safe_rect_reason", "") or "")
    return {
        "region_class":          plan.region_class,
        "background_model":      plan.background_model,
        "cleanup_strategy":      plan.cleanup_strategy,
        "inpaint_method":        plan.inpaint_method,
        "text_mask_confidence":  round(float(plan.text_mask_confidence), 4),
        "container_confidence":  round(float(plan.container_confidence), 4),
        "cleanup_tier":          None,
        "cleanup_status":        None,
        "cleanup_reason":        None,
        "skip_reason":           plan.skip_reason,
        "mask_region_ratio":     round(float(quality.get("mask_region_ratio", 0.0) or 0.0), 4),
        "border_touch_ratio":    round(float(quality.get("border_touch_ratio", 0.0) or 0.0), 4),
        "rectangularity":        round(float(quality.get("rectangularity", 0.0) or 0.0), 4),
        "gradient_fit_error":    plan.debug_metrics.get("gradient_fit_error", ""),
        "cleanup_safe_rect":     [int(v) for v in safe_rect] if safe_rect is not None else None,
        "cleanup_safe_rect_confidence": round(safe_conf, 4),
        "cleanup_safe_rect_reason":     safe_reason,
        "diagnostic_only":       bool(plan.debug_metrics.get("diagnostic_only", False)),
        "diagnostic_cleanup_ran": bool(plan.debug_metrics.get("diagnostic_cleanup_ran", False)),
        "destructive_cleanup_executed": bool(plan.debug_metrics.get("destructive_cleanup_executed", False)),
        "production_patch_accepted": bool(plan.debug_metrics.get("production_patch_accepted", False)),
        "proposal_valid":       bool(plan.debug_metrics.get("proposal_valid", False)),
        "proposal_failure_reason": str(plan.debug_metrics.get("proposal_failure_reason", "") or ""),
        "cleanup_failure_reason": str(plan.debug_metrics.get("cleanup_failure_reason", "") or ""),
        "gate_violation":       bool(plan.debug_metrics.get("gate_violation", False)),
        "residual_text_visible": bool(plan.debug_metrics.get("residual_text_visible", False)),
        "visual_quality_ok":    bool(plan.debug_metrics.get("visual_quality_ok", True)),
        "fill_patch_visible":   bool(plan.debug_metrics.get("fill_patch_visible", False)),
        "cleanup_effective":    bool(plan.debug_metrics.get("cleanup_effective", False)),
        "region_bbox":           list(plan.region_bbox) if plan.region_bbox else None,
        "container_bbox":        list(plan.container_bbox) if plan.container_bbox else None,
    }
