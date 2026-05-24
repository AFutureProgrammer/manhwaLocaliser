"""
bubble_smudge_cleaner.py
════════════════════════════════════════════════════════════════════════════════
Generalised algorithm to remove residual text smudges from speech bubble
interiors after a primary cleanup pass (LaMa / TELEA / flat-fill).

Works with the artifact set produced by the existing pipeline:
    <R>_raw.png            – original crop  (optional, used for reference)
    <R>_cleaned.png        – post-primary inpaint (may still have smudges)
    <R>_container_mask.png – white = region bounding box  ← hard boundary
    <R>_text_mask.png      – detected glyph strokes
    <R>_cleanup_mask.png   – combined mask used by primary pass
    <R>_halo_mask.png      – glyph outline / glow dilation
    <R>_meta.json          – pipeline metrics  (optional, improves routing)

Hard invariants — NEVER violated:
    • No pixel OUTSIDE container_mask is ever modified.
    • Only the "clean bubble interior" (the actual white/colored oval, not the
      surrounding manga artwork that may sit inside the bbox) is touched.
    • Raw / cleaned images are never mutated; all writes go to a copy.

Algorithm (tiered):
    ┌─ STEP 1  Find clean bubble interior
    │   • Flood fill from brightest-seed pixels inside container
    │   • Isolates the white oval from dark manga art at the bbox edges
    │
    ├─ STEP 2  Classify background
    │   • Sample bg from clean interior excluding text/cleanup mask pixels
    │   • Compute median, std, classify:
    │       near-white (median ≥ 240)    → "white"       → flat_fill
    │       flat  (std < 15)             → "flat"         → flat_fill
    │       semi_flat (15 ≤ std < 42)   → "semi_flat"    → TELEA
    │       textured  (std ≥ 42)        → "textured"     → NS inpaint
    │
    ├─ STEP 3  Detect residual smudge pixels
    │   • L2 colour distance from bg inside clean interior
    │   • Adaptive threshold per bg class
    │   • Merge text_mask to catch missed glyphs
    │   • Remove sub-pixel noise specks
    │
    ├─ STEP 4  Build cleanup mask
    │   • Horizontal + isotropic morphological close (bridges CJK strokes)
    │   • Dilation (recovers anti-aliased stroke edges)
    │   • Per-component hole fill (no inter-row bridging)
    │   • Hard AND with clean interior
    │
    ├─ STEP 5  Inpaint
    │   • white/flat  → direct colour fill (no blur, no bleed)
    │   • semi_flat   → TELEA  (radius proportional to mask area)
    │   • textured    → Navier-Stokes
    │
    └─ STEP 6  Validate + optional retry
        • Score remaining residuals
        • Re-run with wider mask / different method if still bad

CLI:
    python bubble_smudge_cleaner.py --region R-02 --dir /path/to/artifacts

Library:
    from bubble_smudge_cleaner import clean_bubble_smudges
    result_img, metrics = clean_bubble_smudges(cleaned_img, container_mask, ...)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BgModel:
    """Sampled background statistics for one bubble region."""
    median_bgr:  np.ndarray    # (3,) float32 – target fill colour
    mean_bgr:    np.ndarray
    std:         float         # overall gray std within clean interior
    channel_std: np.ndarray    # per-channel (3,)
    sample_px:   int
    kind:        str = "unknown"   # "white" | "flat" | "semi_flat" | "textured"

    def __post_init__(self):
        # Near-white bubbles: median is very bright in all channels.
        # This check comes FIRST so that a white bubble with high std
        # (caused by scattered dark art pixels in the container bbox)
        # is correctly classified as "white", not "textured".
        if float(self.median_bgr.min()) >= 235.0:
            self.kind = "white"
        elif self.std < 15:
            self.kind = "flat"
        elif self.std < 42:
            self.kind = "semi_flat"
        else:
            self.kind = "textured"


@dataclass
class SmudgeMetrics:
    """Diagnostic output from clean_bubble_smudges()."""
    residual_px_before: int   = 0
    residual_px_after:  int   = 0
    smudge_mask_px:     int   = 0
    bg_kind:            str   = ""
    bg_median_bgr:      tuple = ()
    inpaint_method:     str   = ""
    retry_used:         bool  = False
    clean_ok:           bool  = False
    container_px:       int   = 0
    clean_interior_px:  int   = 0
    extra:              Dict  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – detect the actual white/colored oval (clean bubble interior)
# ─────────────────────────────────────────────────────────────────────────────

def find_clean_bubble_interior(
    img: np.ndarray,
    container_mask: np.ndarray,
    *,
    brightness_pct: float = 85.0,
    fill_tolerance: int   = 60,
    min_seed_px:    int   = 15,
) -> np.ndarray:
    """
    Locate the actual light-colored oval interior of a speech bubble by
    flood-filling from the brightest pixels inside the container mask.

    The container_mask bounding box typically includes dark manga artwork at
    the corners / edges.  This function separates the clean bubble interior
    from that artwork so the inpaint stays inside the actual bubble.

    Returns uint8 mask (255 = safe-to-clean interior).
    """
    h, w = img.shape[:2]
    cont_bool = container_mask > 127
    gray_max  = img.max(axis=2).astype(np.float32)

    inside_vals = gray_max[cont_bool]
    if inside_vals.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    seed_thresh = max(200.0, float(np.percentile(inside_vals, brightness_pct)))
    seeds = cont_bool & (gray_max >= seed_thresh)
    if int(np.count_nonzero(seeds)) < min_seed_px:
        # Darker bubble: lower the bar
        seed_thresh = max(140.0, float(np.percentile(inside_vals, 55.0)))
        seeds = cont_bool & (gray_max >= seed_thresh)

    result = np.zeros((h, w), dtype=np.uint8)
    seed_u8 = seeds.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed_u8, 4)

    for ci in range(1, n):
        area = int(stats[ci, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        cx = int(stats[ci, cv2.CC_STAT_LEFT] + stats[ci, cv2.CC_STAT_WIDTH]  / 2)
        cy = int(stats[ci, cv2.CC_STAT_TOP]  + stats[ci, cv2.CC_STAT_HEIGHT] / 2)
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        canvas    = img.copy()
        flood_msk = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(
            canvas, flood_msk, (cx, cy),
            newVal  = (0, 128, 0),
            loDiff  = (fill_tolerance,) * 3,
            upDiff  = (fill_tolerance,) * 3,
            flags   = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
        )
        flooded = flood_msk[1:-1, 1:-1] > 0
        result[flooded & cont_bool] = 255

    if not np.any(result):
        # Absolute fallback: treat whole container as interior
        result = (container_mask > 127).astype(np.uint8) * 255

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – background sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_bubble_background(
    img: np.ndarray,
    clean_interior: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    *,
    min_sample_px: int = 80,
) -> BgModel:
    """
    Sample the background colour from clean areas of the bubble interior,
    excluding known glyph / cleanup pixels to avoid contaminating the estimate.
    """
    interior_bool = clean_interior > 127
    sample_bool   = interior_bool.copy()

    if exclude_mask is not None:
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        excl_d = cv2.dilate((exclude_mask > 0).astype(np.uint8), k).astype(bool)
        sample_bool = sample_bool & ~excl_d

    n_sample = int(np.count_nonzero(sample_bool))
    if n_sample < min_sample_px:
        sample_bool = interior_bool.copy()
        if exclude_mask is not None:
            sample_bool = sample_bool & ~(exclude_mask > 0)
        n_sample = int(np.count_nonzero(sample_bool))
    if n_sample < 20:
        sample_bool = interior_bool.copy()
        n_sample    = int(np.count_nonzero(sample_bool))

    pixels = img[sample_bool].astype(np.float32)
    gray   = 0.114 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.299 * pixels[:, 2]

    return BgModel(
        median_bgr  = np.median(pixels, axis=0),
        mean_bgr    = np.mean  (pixels, axis=0),
        std         = float(np.std(gray)),
        channel_std = np.std  (pixels, axis=0),
        sample_px   = n_sample,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – smudge detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_smudge_pixels(
    img: np.ndarray,
    clean_interior: np.ndarray,
    bg: BgModel,
    *,
    base_threshold: float = 22.0,
    min_speck_px:   int   = 3,
) -> np.ndarray:
    """
    Return uint8 mask (255 = smudge) for pixels inside the clean bubble
    interior that deviate from the sampled background by > threshold.

    Adaptive scale factor per bg classification:
        white     → 1.0  (bg is pure white; any deviation is ink)
        flat      → 1.0
        semi_flat → 1.6  (allow some texture variation)
        textured  → 2.4  (bg itself varies; need headroom)

    Tiny isolated components (< min_speck_px) are dropped as sensor noise.
    """
    scale = {"white": 1.0, "flat": 1.0, "semi_flat": 1.6, "textured": 2.4}.get(bg.kind, 1.6)
    thresh = base_threshold * scale

    interior_bool = clean_interior > 127
    dist = np.sqrt(np.sum(
        (img.astype(np.float32) - bg.median_bgr.astype(np.float32)[None, None, :]) ** 2,
        axis=2,
    ))
    smudge_u8 = (interior_bool & (dist > thresh)).astype(np.uint8) * 255

    if min_speck_px > 1 and np.any(smudge_u8):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(smudge_u8, 8)
        clean = np.zeros_like(smudge_u8)
        for ci in range(1, n):
            if int(stats[ci, cv2.CC_STAT_AREA]) >= min_speck_px:
                clean[labels == ci] = 255
        smudge_u8 = clean

    return smudge_u8


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – cleanup mask assembly
# ─────────────────────────────────────────────────────────────────────────────

def _fill_holes_per_component(mask: np.ndarray) -> np.ndarray:
    """Per-component hole fill — avoids bridging separate text rows."""
    if not np.any(mask):
        return mask
    result = mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for ci in range(1, n):
        x  = int(stats[ci, cv2.CC_STAT_LEFT]);   y  = int(stats[ci, cv2.CC_STAT_TOP])
        bw = int(stats[ci, cv2.CC_STAT_WIDTH]);   bh = int(stats[ci, cv2.CC_STAT_HEIGHT])
        x1, y1 = max(0, x-1),              max(0, y-1)
        x2, y2 = min(mask.shape[1],x+bw+1), min(mask.shape[0],y+bh+1)
        crop   = (labels[y1:y2, x1:x2] == ci).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood  = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 128)
        holes  = (flood == 0).astype(np.uint8) * 255
        result[y1:y2, x1:x2] = cv2.bitwise_or(
            result[y1:y2, x1:x2],
            cv2.bitwise_or(padded, holes)[1:-1, 1:-1],
        )
    return result


def build_cleanup_mask(
    smudge_raw: np.ndarray,
    clean_interior: np.ndarray,
    text_mask: Optional[np.ndarray] = None,
    *,
    close_px:  int = 3,
    dilate_px: int = 2,
) -> np.ndarray:
    """
    Refine the raw smudge detection mask:
    1. Merge text_mask (recovers glyphs missed by colour detection).
    2. Horizontal + elliptical morphological close (bridges CJK strokes).
    3. Dilation (catches anti-aliased stroke edges).
    4. Per-component hole fill (no inter-row bridging).
    5. Hard AND with clean_interior.
    """
    mask = smudge_raw.copy()
    if text_mask is not None and np.any(text_mask):
        # FIX-BBOX-ARTIFACT: The halo/outline expansion of the text_mask can
        # leave hard 90-degree corners at the OCR bounding-box edges.  These
        # look like comic panel borders to inpainting algorithms, which then
        # connect the box edges and produce large blocky smudges.
        # A 3×3 ellipse morphological opening removes isolated straight-line
        # artifacts (< 9 px area) without touching real glyph blobs.
        _open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        _tm_clean = cv2.morphologyEx(
            (text_mask > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, _open_k
        )
        mask = cv2.bitwise_or(mask, _tm_clean)

    if close_px > 0:
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT,    (max(3, close_px*4+1), 3))
        k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px*2+1, close_px*2+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_h)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_e)

    if dilate_px > 0:
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        mask = cv2.dilate(mask, k_d)

    mask = _fill_holes_per_component(mask)

    return cv2.bitwise_and(mask, (clean_interior > 127).astype(np.uint8) * 255)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – tiered inpainting
# ─────────────────────────────────────────────────────────────────────────────

def apply_inpaint(
    img: np.ndarray,
    cleanup_mask: np.ndarray,
    clean_interior: np.ndarray,
    bg: BgModel,
    *,
    inpaint_radius: Optional[int] = None,
    force_method: Optional[str]   = None,
) -> Tuple[np.ndarray, str]:
    """
    Inpaint cleanup_mask pixels strictly within clean_interior.

    Routing:
        white / flat → flat_fill  – direct fill, no bleed from dark edges
        semi_flat    → TELEA      – radius tuned to mask area
        textured     → NS         – better texture continuity

    TELEA/NS are avoided for near-white bubbles because they can pull in dark
    colours from manga art pixels at the container edges.
    """
    result   = img.copy()
    interior = clean_interior > 127
    mask_bin = (cleanup_mask > 127) & interior
    if not np.any(mask_bin):
        return result, "no_op"

    mask_px  = int(np.count_nonzero(mask_bin))
    mask_u8  = mask_bin.astype(np.uint8) * 255
    method   = force_method or bg.kind

    # ── Flat fill ─────────────────────────────────────────────────────────────
    if method in ("white", "flat", "flat_fill"):
        fill = np.clip(np.round(bg.median_bgr), 0, 255).astype(np.uint8)
        result[mask_bin] = fill
        return result, "flat_fill"

    # ── Adaptive inpaint radius ───────────────────────────────────────────────
    if inpaint_radius is None:
        inpaint_radius = max(3, min(9, int(np.sqrt(mask_px) * 0.07)))

    # ── TELEA ─────────────────────────────────────────────────────────────────
    if method in ("semi_flat", "telea"):
        inpainted = cv2.inpaint(img, mask_u8, inpaint_radius, cv2.INPAINT_TELEA)
        result[mask_bin] = inpainted[mask_bin]
        return result, f"telea_r{inpaint_radius}"

    # ── Navier-Stokes ─────────────────────────────────────────────────────────
    if method in ("textured", "ns"):
        inpainted = cv2.inpaint(img, mask_u8, inpaint_radius, cv2.INPAINT_NS)
        result[mask_bin] = inpainted[mask_bin]
        return result, f"ns_r{inpaint_radius}"

    # Fallback
    inpainted = cv2.inpaint(img, mask_u8, max(3, inpaint_radius or 3), cv2.INPAINT_TELEA)
    result[mask_bin] = inpainted[mask_bin]
    return result, f"telea_fallback_r{inpaint_radius}"


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – residual scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_residuals(
    img: np.ndarray,
    clean_interior: np.ndarray,
    bg: BgModel,
) -> Tuple[int, bool]:
    """
    Count real (≥ 3 px) non-background components inside the clean interior.
    Returns (residual_px, is_bad).
    is_bad = True when residual_px > max(30, 0.5 % of interior area).
    """
    interior = clean_interior > 127
    area     = int(np.count_nonzero(interior))
    scale    = {"white": 1.0, "flat": 1.0, "semi_flat": 1.8, "textured": 2.7}.get(bg.kind, 1.8)
    thresh   = 22.0 * scale

    dist  = np.sqrt(np.sum(
        (img.astype(np.float32) - bg.median_bgr.astype(np.float32)[None,None,:]) ** 2,
        axis=2,
    ))
    raw   = (interior & (dist > thresh)).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    real  = sum(int(stats[ci, cv2.CC_STAT_AREA]) for ci in range(1, n)
                if int(stats[ci, cv2.CC_STAT_AREA]) >= 3)
    bad   = real > max(30, int(area * 0.005))
    return real, bad


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def clean_bubble_smudges(
    cleaned_img: np.ndarray,
    container_mask: np.ndarray,
    *,
    text_mask:      Optional[np.ndarray] = None,
    cleanup_mask:   Optional[np.ndarray] = None,
    halo_mask:      Optional[np.ndarray] = None,
    meta:           Optional[Dict]       = None,
    base_threshold: float = 22.0,
    close_px:       int   = 3,
    dilate_px:      int   = 2,
    max_retries:    int   = 1,
    force_method:   Optional[str] = None,
    verbose:        bool  = False,
) -> Tuple[np.ndarray, SmudgeMetrics]:
    """
    Remove residual text smudges from a speech bubble interior.

    Parameters
    ----------
    cleaned_img    – BGR image after the primary cleanup pass.
    container_mask – white = region bounding box.  Hard boundary, never crossed.
    text_mask      – detected glyph strokes (merged into cleanup mask, boosts recall).
    cleanup_mask   – combined mask from the primary pass.
    halo_mask      – glyph outline / glow dilation.
    meta           – dict from *_meta.json (routing hints for background_kind).
    base_threshold – base L2 smudge distance; scaled adaptively by bg.kind.
    close_px       – morph-close radius in build_cleanup_mask.
    dilate_px      – dilation radius in build_cleanup_mask.
    max_retries    – additional passes if first pass leaves residuals.
    force_method   – override inpaint routing: "flat_fill" | "telea" | "ns".
    verbose        – print per-step diagnostics.

    Returns
    -------
    (result_img, SmudgeMetrics)
        result_img – cleaned BGR array, same shape/dtype as cleaned_img.
    """
    m = SmudgeMetrics()

    def _log(msg: str):
        if verbose: print(f"  [smudge] {msg}")

    # Build combined exclusion mask (text + cleanup + halo)
    excl_parts = [x for x in (text_mask, cleanup_mask, halo_mask) if x is not None]
    exclude = None
    if excl_parts:
        exclude = np.zeros(cleaned_img.shape[:2], dtype=np.uint8)
        for p in excl_parts:
            exclude = cv2.bitwise_or(exclude, (p > 0).astype(np.uint8) * 255)

    # ── Step 1 ───────────────────────────────────────────────────────────────
    clean_interior = find_clean_bubble_interior(cleaned_img, container_mask)
    m.container_px      = int(np.count_nonzero(container_mask > 127))
    m.clean_interior_px = int(np.count_nonzero(clean_interior))
    _log(f"container={m.container_px}px  clean_interior={m.clean_interior_px}px")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    bg = sample_bubble_background(cleaned_img, clean_interior, exclude_mask=exclude)
    m.bg_kind       = bg.kind
    m.bg_median_bgr = tuple(int(v) for v in bg.median_bgr.tolist())
    _log(f"bg kind={bg.kind}  median={m.bg_median_bgr}  std={bg.std:.1f}  sample_px={bg.sample_px}")

    # Override from meta background_kind field
    if force_method is None and meta:
        bk = str(meta.get("background_kind", "")).lower()
        if bk in ("clean",):      force_method = "flat_fill"
        elif bk in ("textured",): force_method = "telea"

    # ── Initial residual count ────────────────────────────────────────────────
    res_before, _ = score_residuals(cleaned_img, clean_interior, bg)
    m.residual_px_before = res_before
    _log(f"residuals before: {res_before}px")

    if res_before == 0:
        m.clean_ok = True;  m.inpaint_method = "no_op"
        return cleaned_img.copy(), m

    # ── Steps 3–6 loop ───────────────────────────────────────────────────────
    result      = cleaned_img.copy()
    method_used = "none"

    for attempt in range(max_retries + 1):
        # Step 3: detect
        smudge_raw = detect_smudge_pixels(
            result, clean_interior, bg, base_threshold=base_threshold
        )
        if not np.any(smudge_raw):
            _log(f"attempt {attempt}: no smudge pixels — stopping")
            break

        # Step 4: build mask
        cleanup = build_cleanup_mask(
            smudge_raw, clean_interior,
            text_mask = text_mask if attempt == 0 else None,
            close_px  = close_px,
            dilate_px = dilate_px,
        )
        mask_px = int(np.count_nonzero(cleanup))
        m.smudge_mask_px = mask_px
        _log(f"attempt {attempt}: smudge_mask={mask_px}px  method={force_method or bg.kind}")

        if mask_px == 0:
            break

        # Step 5: inpaint
        result, method_used = apply_inpaint(
            result, cleanup, clean_interior, bg, force_method=force_method
        )

        # Step 6: validate
        res_now, still_bad = score_residuals(result, clean_interior, bg)
        _log(f"attempt {attempt}: residual_after={res_now}  still_bad={still_bad}")
        if not still_bad:
            break

        if attempt < max_retries:
            m.retry_used   = True
            base_threshold = base_threshold * 0.80      # widen detection window
            # On retry, only escalate if method was not already flat_fill
            if force_method not in ("flat_fill", "white"):
                prev = force_method or bg.kind
                force_method = "telea" if prev in ("semi_flat", "ns") else force_method
            _log(f"retry {attempt+1}: threshold→{base_threshold:.1f}  method={force_method or bg.kind}")

    # Final scoring
    res_after, is_bad = score_residuals(result, clean_interior, bg)
    m.residual_px_after = res_after
    m.inpaint_method    = method_used
    m.clean_ok          = not is_bad

    pct = 100.0 * (m.residual_px_before - res_after) / max(1, m.residual_px_before)
    _log(f"done: residual={res_after}  clean_ok={m.clean_ok}  reduction={pct:.0f}%")
    return result, m


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic overlay
# ─────────────────────────────────────────────────────────────────────────────

def make_diagnostic_overlay(
    raw: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    container_mask: np.ndarray,
    clean_interior: np.ndarray,
    metrics: SmudgeMetrics,
) -> np.ndarray:
    """
    Four-panel diagnostic strip:
        raw (with boundaries) | before | after | diff heat-map
    """
    cont  = container_mask > 127
    inter = clean_interior  > 127

    raw_v = raw.copy()
    # Yellow-green = container bbox boundary
    border = cv2.morphologyEx(
        (cont.astype(np.uint8)*255), cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    raw_v[border > 0] = [0, 200, 200]
    # Bright green = clean interior boundary
    inner_bdr = cv2.morphologyEx(
        (inter.astype(np.uint8)*255), cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    raw_v[inner_bdr > 0] = [0, 255, 0]

    diff = np.max(np.abs(before.astype(float) - after.astype(float)), axis=2)
    heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heat[~inter] = [20, 20, 20]

    strip = np.concatenate([raw_v, before, after, heat], axis=1)
    bar   = np.full((30, strip.shape[1], 3), 30, dtype=np.uint8)
    txt   = (
        f"bg={metrics.bg_kind}  "
        f"residual {metrics.residual_px_before}→{metrics.residual_px_after}px  "
        f"method={metrics.inpaint_method}  clean_ok={metrics.clean_ok}"
    )
    cv2.putText(bar, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 200), 1, cv2.LINE_AA)
    return np.concatenate([strip, bar], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# File-level runner
# ─────────────────────────────────────────────────────────────────────────────

def _load(region_id: str, artifact_dir: str):
    d = Path(artifact_dir)
    def img(s, flags=cv2.IMREAD_COLOR):
        p = d / f"{region_id}_{s}.png"
        return cv2.imread(str(p), flags) if p.exists() else None
    def msk(s): return img(s, cv2.IMREAD_GRAYSCALE)
    meta = None
    mp = d / f"{region_id}_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return (img("raw"), img("cleaned"), msk("container_mask"),
            msk("text_mask"), msk("cleanup_mask"), msk("halo_mask"), meta)


def process_region(
    region_id:    str,
    artifact_dir: str,
    output_dir:   str,
    *,
    verbose:       bool = True,
    write_overlay: bool = True,
    **kwargs,
) -> SmudgeMetrics:
    """
    Load artifacts for one region, run the smudge cleaner, write outputs.

    Outputs:
        <out>/<region_id>_smudge_cleaned.png  – cleaned image
        <out>/<region_id>_smudge_overlay.png  – diagnostic strip (if write_overlay)
        <out>/<region_id>_smudge_metrics.json – metrics JSON
    """
    raw, cleaned, container, text_m, cleanup_m, halo_m, meta = _load(region_id, artifact_dir)
    if cleaned   is None: raise FileNotFoundError(f"No cleaned image for {region_id}")
    if container is None: raise FileNotFoundError(f"No container_mask for {region_id}")

    result, m = clean_bubble_smudges(
        cleaned, container,
        text_mask=text_m, cleanup_mask=cleanup_m, halo_mask=halo_m,
        meta=meta, verbose=verbose, **kwargs,
    )

    os.makedirs(output_dir, exist_ok=True)
    base = str(Path(output_dir) / region_id)
    cv2.imwrite(base + "_smudge_cleaned.png", result)

    if write_overlay and raw is not None:
        ci = find_clean_bubble_interior(cleaned, container)
        ov = make_diagnostic_overlay(raw, cleaned, result, container, ci, m)
        cv2.imwrite(base + "_smudge_overlay.png", ov)

    info = {
        "region_id":            region_id,
        "residual_px_before":   m.residual_px_before,
        "residual_px_after":    m.residual_px_after,
        "smudge_mask_px":       m.smudge_mask_px,
        "bg_kind":              m.bg_kind,
        "bg_median_bgr":        list(m.bg_median_bgr),
        "inpaint_method":       m.inpaint_method,
        "retry_used":           m.retry_used,
        "clean_ok":             m.clean_ok,
        "container_px":         m.container_px,
        "clean_interior_px":    m.clean_interior_px,
        "reduction_pct": round(
            100.0 * (m.residual_px_before - m.residual_px_after)
            / max(1, m.residual_px_before), 1
        ),
    }
    Path(base + "_smudge_metrics.json").write_text(json.dumps(info, indent=2))

    if verbose:
        print(f"\n[{region_id}] Result:")
        for k, v in info.items(): print(f"  {k}: {v}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Remove residual text smudges from manga speech bubble interiors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-route (recommended):
  python bubble_smudge_cleaner.py -r R-02 -d /path/to/artifacts

  # Force flat fill (always safe on white bubbles):
  python bubble_smudge_cleaner.py -r R-02 -d /path/to/artifacts --method flat_fill

  # Stricter detection (lower threshold catches lighter residuals):
  python bubble_smudge_cleaner.py -r R-02 -d /path/to/artifacts --threshold 15
        """,
    )
    ap.add_argument("--region",    "-r", required=True, help="Region ID (e.g. R-01)")
    ap.add_argument("--dir",       "-d", default=".",   help="Artifact directory")
    ap.add_argument("--out",       "-o", default=None,  help="Output directory (default = --dir)")
    ap.add_argument("--method",          default=None,  choices=["flat_fill", "telea", "ns"],
                    help="Force inpaint method (default: auto)")
    ap.add_argument("--threshold", type=float, default=22.0,
                    help="Base L2 smudge threshold (default 22; scale ×1 flat / ×1.6 semi / ×2.4 textured)")
    ap.add_argument("--retries",   type=int,   default=1,
                    help="Additional cleanup passes if residuals remain (default 1)")
    ap.add_argument("--no-overlay", action="store_true", help="Skip diagnostic overlay")
    ap.add_argument("--quiet",     "-q", action="store_true")
    args = ap.parse_args()
    process_region(
        region_id     = args.region,
        artifact_dir  = args.dir,
        output_dir    = args.out or args.dir,
        verbose       = not args.quiet,
        write_overlay = not args.no_overlay,
        base_threshold = args.threshold,
        max_retries   = args.retries,
        force_method  = args.method,
    )