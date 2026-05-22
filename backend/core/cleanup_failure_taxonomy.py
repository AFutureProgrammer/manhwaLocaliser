from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping


CLEANUP_FAILURE_CLASSES = (
    "bad_text_mask",
    "bad_container_mask",
    "unsafe_mask_rejection",
    "bad_cleanup_routing",
    "bad_flat_fill",
    "bad_inpaint_backend",
    "halo_residual",
    "leftover_glyphs",
    "art_damage",
    "overcleanup",
    "undercleanup",
    "needs_manual_mask",
)

_ORDER = {name: idx for idx, name in enumerate(CLEANUP_FAILURE_CLASSES)}


def normalize_cleanup_failure_classes(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        name = str(value or "").strip()
        if name in _ORDER and name not in out:
            out.append(name)
    out.sort(key=lambda item: _ORDER[item])
    return out


def primary_cleanup_failure_class(values: Iterable[Any]) -> str:
    classes = normalize_cleanup_failure_classes(values)
    return classes[0] if classes else ""


def classify_cleanup_failure(data: Mapping[str, Any]) -> List[str]:
    """Map cleanup planner/report signals to canonical QA failure classes."""

    classes: List[str] = []
    debug = _mapping(data.get("debug_metrics"))
    quality = _mapping(debug.get("quality"))
    effectiveness = _mapping(data.get("effectiveness"))
    text_candidate_rows = data.get("text_mask_candidates") or debug.get("text_mask_candidate_scores") or []

    text_reason = _lower(
        data.get("text_mask_reason"),
        debug.get("selected_text_mask_candidate"),
        data.get("selected_text_mask_candidate"),
    )
    selected_source = _lower(
        data.get("selected_text_mask_candidate_source"),
        debug.get("selected_text_mask_candidate_source"),
    )
    container_reason = _lower(data.get("container_reason"), debug.get("container_reason"))
    skip_reason = _lower(data.get("skip_reason"), debug.get("cleanup_mask_rejection_reason"))
    failure_reason = _lower(
        data.get("cleanup_failure_reason"),
        effectiveness.get("cleanup_failure_reason"),
        data.get("proposal_failure_reason"),
        debug.get("proposal_failure_reason"),
    )
    strategy = _lower(data.get("strategy"), data.get("cleanup_strategy"), debug.get("cleanup_strategy"))
    method = _lower(data.get("inpaint_method"), data.get("method"))
    backend = _lower(data.get("cleanup_backend"), debug.get("cleanup_backend_used"), debug.get("cleanup_backend"))

    if selected_source in {"none", ""} or _has(text_reason, "no_candidates", "no_text", "force_cleanup_bbox_mask"):
        classes.append("bad_text_mask")
    if selected_source == "fallback_cv_no_bbox" or _has(
        _candidate_rejection_text(text_candidate_rows),
        "fragmented_broad_fallback",
        "bbox_like_or_full_rectangle",
        "spans_region",
        "rectangular_dense",
    ):
        classes.append("bad_text_mask")

    if _has(container_reason, "error", "rejected", "missing", "partial_container", "safe_rect_rejected"):
        classes.append("bad_container_mask")

    if data.get("cleanup_mask_rejected") or debug.get("cleanup_mask_rejected") or _has(
        skip_reason,
        "cleanup_mask_",
        "too_large",
        "border_collision",
        "full_rect",
        "bbox_matches",
        "unsafe",
        "protected_sfx",
    ):
        classes.append("unsafe_mask_rejection")

    if _has(skip_reason, "protected_sfx", "manual", "review") or strategy in {"review", "skip"}:
        classes.append("needs_manual_mask")

    if _has(failure_reason, "residual", "leftover") or int(_number(data.get("residual_component_count", debug.get("residual_component_count")))) > 0:
        classes.append("leftover_glyphs")
    if _has(failure_reason, "halo") or int(_number(debug.get("halo_added_px"))) > 0 and _has(failure_reason, "residual"):
        classes.append("halo_residual")

    if _has(failure_reason, "fill_patch", "visual_quality") or _has(
        _lower(debug.get("fill_patch_reason"), debug.get("boundary_mask_constraint_rejection_reason")),
        "boundary",
        "fill_patch",
    ):
        classes.append("bad_flat_fill")

    if backend in {"iopaint", "lama", "lama_pt", "lama_onnx"} and _has(_lower(skip_reason, failure_reason), "error", "timeout", "unavailable"):
        classes.append("bad_inpaint_backend")

    if _has(_lower(debug.get("cleanup_route")), "solid_bubble_cv") and backend in {"iopaint", "lama", "lama_pt", "lama_onnx"} and strategy == "flat_fill":
        classes.append("bad_cleanup_routing")

    mask_region_ratio = _number(data.get("mask_region_ratio", quality.get("mask_region_ratio")))
    mask_container_ratio = _number(data.get("mask_container_ratio", quality.get("mask_container_ratio")))
    border_touch_ratio = _number(data.get("border_touch_ratio", quality.get("border_touch_ratio")))
    if mask_region_ratio >= 0.45 or mask_container_ratio >= 0.70 or (border_touch_ratio >= 0.50 and mask_region_ratio >= 0.25):
        classes.extend(["art_damage", "overcleanup"])
    elif mask_region_ratio > 0.0 and mask_region_ratio < 0.01:
        classes.append("undercleanup")

    if not classes and bool(data.get("cleanup_effective", debug.get("cleanup_effective", False))) is False:
        classes.append("undercleanup")

    return normalize_cleanup_failure_classes(classes)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _lower(*values: Any) -> str:
    return " ".join(str(value or "").lower() for value in values if value not in (None, ""))


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _candidate_rejection_text(rows: Any) -> str:
    if not isinstance(rows, list):
        return ""
    parts: List[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            parts.append(str(row.get("rejection_reason") or ""))
            parts.append(str(row.get("reason") or ""))
    return re.sub(r"\s+", " ", " ".join(parts).lower())
