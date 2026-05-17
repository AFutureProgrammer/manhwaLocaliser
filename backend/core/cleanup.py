from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from backend.core.constants import debug_print
from backend.core.regions import BackgroundKind, OCRBlock, RegionKind

def build_text_mask_for_block(
    shape: Tuple[int, int, int],
    block: "OCRBlock",
    pad: int = 2,
) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in block.boxes:
        pts = np.array(poly, dtype=np.int32)
        if pts.size == 0:
            continue
        cv2.fillPoly(mask, [pts], 255)
    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def build_ellipse_mask(width: int, height: int, inset: int = 4) -> np.ndarray:
    mask = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
    cx = width // 2
    cy = height // 2
    ax = max(1, (width - inset * 2) // 2)
    ay = max(1, (height - inset * 2) // 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask

def _boxes_cover_block_bbox(block: "OCRBlock", tol: int = 2) -> bool:
    if len(getattr(block, "boxes", []) or []) != 1:
        return False
    x, y, w, h = block.bbox()
    pts = np.array(block.boxes[0], dtype=np.float32)
    if pts.size == 0:
        return False
    px1, py1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    px2, py2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    return (
        abs(px1 - x) <= tol and abs(py1 - y) <= tol
        and abs(px2 - (x + w)) <= tol and abs(py2 - (y + h)) <= tol
    )

def hydrate_restored_cleanup_masks(img_cv: np.ndarray, block: "OCRBlock") -> bool:
    """
    Restored projects do not persist large text/bubble masks.  If the only
    saved OCR polygon is the edited bbox, derive masks from pixels so cleanup
    does not erase the whole region rectangle.
    """
    if not _boxes_cover_block_bbox(block):
        return False
    x, y, w, h = block.bbox()
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_cv.shape[1], x + w), min(img_cv.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = img_cv[y1:y2, x1:x2]
    bg_rgb = tuple(int(v) for v in (getattr(block, "bg_color", None) or estimate_initial_bg_color(img_cv, block.bbox()))[:3])
    bg_bgr = np.array([bg_rgb[2], bg_rgb[1], bg_rgb[0]], dtype=np.int16)
    diff = np.abs(crop.astype(np.int16) - bg_bgr[None, None, :]).sum(axis=2)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Approximate the visible bubble/caption surface first, then select only
    # foreground strokes inside it.  This avoids treating black page art outside
    # a white bubble as text.
    if sum(bg_rgb) >= 540:
        surface = np.where((diff < 95) | (gray > 185), 255, 0).astype(np.uint8)
    else:
        surface = np.where(diff < 110, 255, 0).astype(np.uint8)
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        bubble_local = np.zeros(surface.shape, dtype=np.uint8)
        cv2.drawContours(bubble_local, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    else:
        bubble_local = np.full(surface.shape, 255, dtype=np.uint8)

    if sum(bg_rgb) >= 540:
        text_local = np.where(((diff > 85) | (gray < 165)) & (bubble_local > 0), 255, 0).astype(np.uint8)
    else:
        text_local = np.where((diff > 75) & (bubble_local > 0), 255, 0).astype(np.uint8)
    text_local = cv2.morphologyEx(text_local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    if not np.any(text_local):
        return False

    max_reasonable = max(64, int(w * h * 0.45))
    if int(np.count_nonzero(text_local)) > max_reasonable:
        return False

    block.text_mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    block.text_mask[y1:y2, x1:x2] = text_local
    if block.bubble_bbox is not None:
        block.bubble_mask = bubble_local
        block.bubble_bbox = (x1, y1, x2 - x1, y2 - y1)
    debug_print(
        f"hydrate_restored_cleanup_masks: bbox={block.bbox()} "
        f"text_pixels={int(np.count_nonzero(text_local))}"
    )
    return True

def estimate_initial_bg_color(
    img_cv: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int]:
    x, y, w, h = bbox
    x1 = max(0, x - max(4, w // 8))
    y1 = max(0, y - max(4, h // 8))
    x2 = min(img_cv.shape[1], x + w + max(4, w // 8))
    y2 = min(img_cv.shape[0], y + h + max(4, h // 8))
    crop = img_cv[y1:y2, x1:x2]
    if crop.size == 0:
        return (255, 255, 255)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright = gray >= np.percentile(gray, 80)
    if bright.sum() < 10:
        bright = np.ones_like(gray, dtype=bool)
    b, g, r = cv2.split(crop)
    return (
        int(np.median(r[bright])),
        int(np.median(g[bright])),
        int(np.median(b[bright])),
    )

def detect_bubble_region(
    img_cv: np.ndarray,
    text_bbox: Tuple[int, int, int, int],
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    x, y, w, h = text_bbox
    h_img, w_img = img_cv.shape[:2]

    pad_x = max(14, int(w * 0.9))
    pad_y = max(14, int(h * 0.9))
    rx1 = max(0, x - pad_x)
    ry1 = max(0, y - pad_y)
    rx2 = min(w_img, x + w + pad_x)
    ry2 = min(h_img, y + h + pad_y)
    roi = img_cv[ry1:ry2, rx1:rx2]

    if roi.size == 0:
        fallback_bbox = (max(0, x - 10), max(0, y - 10), w + 20, h + 20)
        return fallback_bbox, build_ellipse_mask(fallback_bbox[2], fallback_bbox[3], inset=4)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    candidate_masks = [otsu_light, otsu_dark]
    if bg_color_rgb is not None:
        bg_bgr = np.array([bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0]], dtype=np.int16)
        diff = np.abs(roi.astype(np.int16) - bg_bgr[None, None, :]).sum(axis=2)
        sim_mask = np.where(diff < 85, 255, 0).astype(np.uint8)
        candidate_masks.insert(0, sim_mask)

    text_cx = int(np.clip((x + w // 2) - rx1, 0, roi.shape[1] - 1))
    text_cy = int(np.clip((y + h // 2) - ry1, 0, roi.shape[0] - 1))
    text_local = np.zeros(roi.shape[:2], dtype=np.uint8)
    tx1 = max(0, x - rx1)
    ty1 = max(0, y - ry1)
    tx2 = min(roi.shape[1], x - rx1 + w)
    ty2 = min(roi.shape[0], y - ry1 + h)
    text_local[ty1:ty2, tx1:tx2] = 255

    best = None
    best_score = -1
    for cand in candidate_masks:
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for label in range(1, num_labels):
            bx = int(stats[label, cv2.CC_STAT_LEFT])
            by = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if bw < w or bh < h:
                continue
            if bw > roi.shape[1] * 0.98 or bh > roi.shape[0] * 0.98:
                continue
            comp = np.where(labels == label, 255, 0).astype(np.uint8)
            overlap = int(np.count_nonzero(cv2.bitwise_and(comp, text_local)))
            contains_center = comp[text_cy, text_cx] > 0
            if overlap <= 0 and not contains_center:
                continue
            fill_ratio = area / max(1, bw * bh)
            score = overlap + area * 0.02 + fill_ratio * 100
            if score > best_score:
                best_score = score
                best = (bx, by, bw, bh, comp)

    if best is not None:
        bx, by, bw, bh, comp = best

        # FIX-2: clip the winning component to a vertical band centred on
        # the OCR text box.  Without this, two chained/stacked balloons
        # that share a single white connected component get returned as one
        # giant region, causing the fitter to work with the wrong geometry.
        v_expand = max(int(h * 1.4), 45)           # generous: 140 % text height above/below
        text_y1_roi = y - ry1                       # text top in ROI coordinates
        text_y2_roi = y - ry1 + h                   # text bottom in ROI coordinates
        clip_top = max(0, text_y1_roi - v_expand)
        clip_bot = min(roi.shape[0], text_y2_roi + v_expand)

        comp_c = comp.copy()
        if clip_top > 0:
            comp_c[:clip_top] = 0
        if clip_bot < roi.shape[0]:
            comp_c[clip_bot:] = 0

        nz = np.nonzero(comp_c)
        if len(nz[0]) > 0:
            cy1, cy2 = int(nz[0].min()), int(nz[0].max())
            cx1, cx2 = int(nz[1].min()), int(nz[1].max())
            new_bw = cx2 - cx1 + 1
            new_bh = cy2 - cy1 + 1
            local = comp_c[cy1:cy2 + 1, cx1:cx2 + 1]
            local = cv2.morphologyEx(
                local,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            global_bbox = (rx1 + cx1, ry1 + cy1, new_bw, new_bh)
            debug_print(
                f"detect_bubble_region: clipped component "
                f"orig=({rx1+bx},{ry1+by},{bw},{bh}) → "
                f"clipped=({global_bbox})"
            )
            return global_bbox, local

        # Clipping removed everything (shouldn't happen); fall back to
        # the unclipped component.
        local = comp[by:by + bh, bx:bx + bw]
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        global_bbox = (rx1 + bx, ry1 + by, bw, bh)
        return global_bbox, local

    fallback_bbox = (max(0, x - max(10, w // 3)), max(0, y - max(10, h // 2)),
                     min(w_img - max(0, x - max(10, w // 3)), w + max(20, int(w * 0.66))),
                     min(h_img - max(0, y - max(10, h // 2)), h + max(20, int(h * 1.0))))
    return fallback_bbox, build_ellipse_mask(fallback_bbox[2], fallback_bbox[3], inset=4)

def _measure_local_variance(roi_gray: np.ndarray) -> float:
    """Mean-subtracted std — low for uniform fills, high for texture/art."""
    if roi_gray.size == 0:
        return 0.0
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    diff = roi_gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(diff))

def _measure_bg_business(img_cv: np.ndarray, block: "OCRBlock") -> float:
    """
    Score 0.0–1.0: how visually complex is the region around the text.
    Combines edge density and local variance.
    High score → text is over art / complex background.
    """
    x, y, w, h = block.bbox()
    pad = max(10, int(min(w, h) * 0.4))
    x1 = max(0, x - pad);        y1 = max(0, y - pad)
    x2 = min(img_cv.shape[1], x + w + pad)
    y2 = min(img_cv.shape[0], y + h + pad)
    roi = img_cv[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_density  = float(edges.sum()) / max(1, gray.size * 255)
    local_var     = _measure_local_variance(gray)
    score = min(1.0, edge_density * 4000.0 + local_var / 40.0)
    return score

def _detect_halftone(gray: np.ndarray, mask: np.ndarray) -> "tuple[bool, float]":
    """
    Detect halftone / screentone texture in a bubble interior.

    Halftone = high Laplacian (fine-scale) energy relative to coarse-scale
    variation.  Plain bubbles: both low.  Gradients: low fine, high coarse.
    Halftone: high fine, low-to-moderate coarse.

    Returns (is_halftone, confidence_score 0-1).
    """
    if gray.size < 100:
        return False, 0.0

    active = mask > 0 if mask.shape == gray.shape else np.ones(gray.shape, dtype=bool)
    if active.sum() < 50:
        return False, 0.0

    # Fine-scale energy via Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    fine_std = float(np.std(lap[active]))

    # Coarse-scale: large blur removes fine detail, measure residual variation
    coarse = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)
    coarse_std = float(np.std(coarse[active]))

    # Halftone: fine_std dominates; score = fine relative to combined energy
    denom = coarse_std + fine_std + 1e-6
    score = float(np.clip(fine_std / denom * 2.0, 0.0, 1.0))
    is_halftone = fine_std > 10.0 and score > 0.48

    debug_print(
        f"_detect_halftone: fine_std={fine_std:.1f} coarse_std={coarse_std:.1f} "
        f"score={score:.2f} is_halftone={is_halftone}"
    )
    return is_halftone, score

def _detect_gradient(gray: np.ndarray, mask: np.ndarray) -> "tuple[bool, float]":
    """
    Detect a smooth luminance gradient across a bubble interior.

    A gradient bubble has monotonically changing mean brightness across row or
    column bands.  Returns (is_gradient, confidence_score 0-1).
    """
    if gray.size < 100:
        return False, 0.0

    active = mask > 0 if mask.shape == gray.shape else np.ones(gray.shape, dtype=bool)
    if active.sum() < 50:
        return False, 0.0

    h, w = gray.shape

    def _band_means(n_bands: int, axis: int) -> List[float]:
        """Mean pixel value in n_bands evenly-spaced slices along axis."""
        size = h if axis == 0 else w
        band_size = max(1, size // n_bands)
        means: List[float] = []
        for start in range(0, size, band_size):
            end = min(size, start + band_size)
            if axis == 0:
                m = active[start:end, :]
                g = gray[start:end, :]
            else:
                m = active[:, start:end]
                g = gray[:, start:end]
            px = g[m].astype(np.float32)
            if len(px) >= 3:
                means.append(float(np.mean(px)))
        return means

    def _monotone_score(vals: List[float]) -> float:
        if len(vals) < 3:
            return 0.0
        d = np.diff(vals)
        pos = int((d > 1.5).sum())
        neg = int((d < -1.5).sum())
        return max(pos, neg) / max(1, len(d))

    n_bands = min(8, max(3, h // 8))
    row_means = _band_means(n_bands, axis=0)
    col_means = _band_means(n_bands, axis=1)

    row_mono  = _monotone_score(row_means)
    col_mono  = _monotone_score(col_means)
    mono      = max(row_mono, col_mono)

    all_means = row_means + col_means
    spread    = (max(all_means) - min(all_means)) if len(all_means) >= 2 else 0.0

    is_gradient = mono > 0.58 and spread > 12.0
    score       = float(np.clip(mono * spread / 50.0, 0.0, 1.0))

    debug_print(
        f"_detect_gradient: mono={mono:.2f} spread={spread:.1f} "
        f"score={score:.2f} is_gradient={is_gradient}"
    )
    return is_gradient, score

def _detect_caption_box(block: "OCRBlock") -> "tuple[bool, float]":
    """
    Detect whether a bubble is actually a rectangular caption / narration box.

    Caption boxes: high mask fill-ratio (nearly rectangular) and wide aspect
    ratio.  Speech bubbles: lower fill ratio due to curved / oval shape.

    Returns (is_caption, confidence_score 0-1).
    """
    if block.bubble_bbox is None or block.bubble_mask is None:
        return False, 0.0
    bx, by, bw, bh = block.bubble_bbox
    if bw < 4 or bh < 4:
        return False, 0.0

    fill_ratio = float(block.bubble_mask.sum()) / max(1, int(bw) * int(bh) * 255)
    aspect     = bw / max(1, bh)

    # Rectangular caption: fill > 82 % and aspect > 1.2
    is_caption  = fill_ratio > 0.82 and aspect > 1.2
    score       = float(np.clip(fill_ratio * min(1.0, aspect / 2.5), 0.0, 1.0))

    debug_print(
        f"_detect_caption_box: fill={fill_ratio:.2f} aspect={aspect:.2f} "
        f"score={score:.2f} is_caption={is_caption}"
    )
    return is_caption, score

def _make_gradient_strip(
    width:     int,
    height:    int,
    start_rgb: Tuple[int, int, int],
    end_rgb:   Tuple[int, int, int],
    angle_deg: int = 90,
) -> Image.Image:
    """
    Return a width×height RGB PIL image filled with a linear gradient.

    angle_deg: 0 = left→right, 90 = top→bottom, 180 = right→left, 270 = bottom→top.
    The gradient is computed by projecting each pixel's position onto the direction
    vector, then linearly interpolating between start_rgb and end_rgb.
    Result is always fully deterministic for the same inputs.
    """
    w = max(1, width)
    h = max(1, height)
    if w == 1 and h == 1:
        return Image.new("RGB", (1, 1), start_rgb)

    angle_rad = float(np.deg2rad(angle_deg % 360))
    dx = float(np.cos(angle_rad))
    dy = float(np.sin(angle_rad))

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    proj    = (xs - cx) * dx + (ys - cy) * dy
    p_min, p_max = float(proj.min()), float(proj.max())
    if abs(p_max - p_min) < 1e-6:
        t = np.zeros((h, w), dtype=np.float32)
    else:
        t = ((proj - p_min) / (p_max - p_min)).astype(np.float32)

    arr = np.empty((h, w, 3), dtype=np.float32)
    for ch, (s, e) in enumerate(zip(start_rgb, end_rgb)):
        arr[:, :, ch] = s + (e - s) * t

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

def _render_plate(
    pil_img:  Image.Image,
    x1: int, y1: int, x2: int, y2: int,
    color:    Tuple[int, int, int],
    opacity:  float,
) -> None:
    """
    Composite a semi-transparent filled rectangle onto pil_img in-place.
    Coordinates are clamped to image bounds automatically.
    """
    if opacity <= 0.0 or x1 >= x2 or y1 >= y2:
        return
    iw, ih = pil_img.size
    cx1 = max(0, x1); cy1 = max(0, y1)
    cx2 = min(iw, x2); cy2 = min(ih, y2)
    if cx1 >= cx2 or cy1 >= cy2:
        return
    alpha = int(round(float(np.clip(opacity, 0.0, 1.0)) * 255))
    plate = Image.new("RGBA", (cx2 - cx1, cy2 - cy1),
                      (int(color[0]), int(color[1]), int(color[2]), alpha))
    pil_img.paste(plate, (cx1, cy1), plate)

def _render_shadow(
    pil_img: Image.Image,
    draw:    ImageDraw.Draw,
    text:    str,
    x_pos:   int,
    y_pos:   int,
    font:    ImageFont.ImageFont,
    color:   Tuple[int, int, int],
    offset:  Tuple[int, int],
    opacity: float,
    blur:    float = 0.0,
) -> None:
    """
    Composite a drop shadow behind text using a temporary RGBA layer.
    Uses actual glyph alpha so the shadow shape matches the text exactly.
    Falls back to an opaque direct draw on any error.
    """
    if not text or opacity <= 0.0:
        return
    sx, sy = x_pos + int(offset[0]), y_pos + int(offset[1])
    try:
        bb = draw.textbbox((sx, sy), text, font=font)
        margin = max(2, int(round(max(0.0, float(blur)) * 3)) + 2)
        bx1 = int(bb[0]) - margin; by1 = int(bb[1]) - margin
        bx2 = int(bb[2]) + margin; by2 = int(bb[3]) + margin
        bw = max(1, bx2 - bx1); bh = max(1, by2 - by1)

        shd = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        shd_draw = ImageDraw.Draw(shd)
        shd_draw.text(
            (sx - bx1, sy - by1), text, font=font,
            fill=(int(color[0]), int(color[1]), int(color[2]),
                  int(round(float(np.clip(opacity, 0.0, 1.0)) * 255))),
        )
        if blur and float(blur) > 0.0:
            alpha = shd.getchannel("A").filter(ImageFilter.GaussianBlur(radius=max(0.1, float(blur))))
            blurred = Image.new("RGBA", (bw, bh), (int(color[0]), int(color[1]), int(color[2]), 0))
            blurred.putalpha(alpha)
            shd = blurred
        iw, ih = pil_img.size
        cx1 = max(0, bx1); cy1 = max(0, by1)
        cx2 = min(iw, bx2); cy2 = min(ih, by2)
        if cx1 < cx2 and cy1 < cy2:
            region = shd.crop((cx1 - bx1, cy1 - by1, cx2 - bx1, cy2 - by1))
            pil_img.paste(region, (cx1, cy1), region)
    except Exception:
        draw.text((sx, sy), text, font=font,
                  fill=(int(color[0]), int(color[1]), int(color[2])))

def _render_glow(
    pil_img: Image.Image,
    draw: ImageDraw.Draw,
    text: str,
    x_pos: int,
    y_pos: int,
    font: ImageFont.ImageFont,
    color: Tuple[int, int, int],
    radius: int,
    intensity: float,
) -> None:
    """Composite a soft halo from the glyph alpha before the main text draw."""
    if not text or intensity <= 0.0 or radius <= 0:
        return
    try:
        bb = draw.textbbox((x_pos, y_pos), text, font=font)
        margin = max(2, int(radius) * 4)
        bx1 = int(bb[0]) - margin; by1 = int(bb[1]) - margin
        bx2 = int(bb[2]) + margin; by2 = int(bb[3]) + margin
        bw = max(1, bx2 - bx1); bh = max(1, by2 - by1)
        alpha = Image.new("L", (bw, bh), 0)
        alpha_draw = ImageDraw.Draw(alpha)
        alpha_draw.text(
            (x_pos - bx1, y_pos - by1), text, font=font,
            fill=int(round(float(np.clip(intensity, 0.0, 1.0)) * 255)),
        )
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=max(1, int(radius))))
        glow = Image.new("RGBA", (bw, bh), (int(color[0]), int(color[1]), int(color[2]), 0))
        glow.putalpha(alpha)
        iw, ih = pil_img.size
        cx1 = max(0, bx1); cy1 = max(0, by1)
        cx2 = min(iw, bx2); cy2 = min(ih, by2)
        if cx1 < cx2 and cy1 < cy2:
            region = glow.crop((cx1 - bx1, cy1 - by1, cx2 - bx1, cy2 - by1))
            pil_img.paste(region, (cx1, cy1), region)
    except Exception:
        return

def _render_gradient_text(
    pil_img:  Image.Image,
    draw:     ImageDraw.Draw,
    text:     str,
    x_pos:    int,
    y_pos:    int,
    font:     ImageFont.ImageFont,
    style:    "TextStyle",
    stroke_w: int,
) -> None:
    """
    Render gradient-filled text by masking a gradient through glyph alpha.

    Pipeline:
      1. Measure glyph bounding box in image coordinates.
      2. Render text in white on transparent (L channel only) → alpha mask.
      3. Build a gradient strip matching the bbox dimensions.
      4. Apply alpha mask → composite gradient through glyph shape onto pil_img.

    Falls back to solid gradient_start color if the bbox is degenerate or any
    step fails.  Always stable / deterministic.
    """
    if not text:
        return
    fallback_color = style.gradient_start
    try:
        bb = draw.textbbox((x_pos, y_pos), text, font=font,
                           stroke_width=max(0, stroke_w))
        bx1, by1, bx2, by2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
    except Exception:
        draw.text((x_pos, y_pos), text, font=font, fill=fallback_color)
        return

    if bw < 2 or bh < 2:
        draw.text((x_pos, y_pos), text, font=font, fill=fallback_color)
        return

    # Glyph alpha mask (no stroke — stroke is drawn separately as outline)
    mask_img  = Image.new("L", (bw, bh), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.text((x_pos - bx1, y_pos - by1), text, font=font, fill=255)

    # Gradient strip in image direction
    grad = _make_gradient_strip(bw, bh, style.gradient_start, style.gradient_end,
                                style.gradient_angle)

    # Clip target rect to image bounds
    iw, ih = pil_img.size
    cx1 = max(0, bx1); cy1 = max(0, by1)
    cx2 = min(iw, bx2); cy2 = min(ih, by2)
    if cx1 >= cx2 or cy1 >= cy2:
        return

    gx1, gy1 = cx1 - bx1, cy1 - by1
    gx2, gy2 = gx1 + (cx2 - cx1), gy1 + (cy2 - cy1)

    grad_clip = grad.crop((gx1, gy1, gx2, gy2)).convert("RGBA")
    mask_clip = mask_img.crop((gx1, gy1, gx2, gy2))
    grad_clip.putalpha(mask_clip)
    pil_img.paste(grad_clip, (cx1, cy1), mask_clip)

def _check_text_contrast(
    fg:      Tuple[int, int, int],
    pil_img: Image.Image,
    x: int, y: int, w: int, h: int,
    min_luma_diff: float = 70.0,
) -> Tuple[int, int, int]:
    """
    Sample the local background under the text region and verify contrast.

    Returns fg unchanged when contrast ≥ min_luma_diff.
    Returns black or white (whichever contrasts better) when contrast is too low
    AND the style source is "auto" (never overrides a deliberate manual choice).

    Contrast is measured as |fg_luma − median_bg_luma| using BT.601 weights.
    """
    try:
        crop = pil_img.crop((max(0, x), max(0, y),
                             min(pil_img.width,  x + w),
                             min(pil_img.height, y + h)))
        if crop.size[0] == 0 or crop.size[1] == 0:
            return fg
        arr    = np.array(crop.convert("RGB"), dtype=np.float32)
        bg_luma = float(
            0.299 * np.median(arr[:, :, 0])
            + 0.587 * np.median(arr[:, :, 1])
            + 0.114 * np.median(arr[:, :, 2])
        )
        fg_luma = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
        if abs(fg_luma - bg_luma) < min_luma_diff:
            return (255, 255, 255) if bg_luma < 128 else (0, 0, 0)
    except Exception:
        pass
    return fg

def _draw_line_with_style(
    pil_img:   Image.Image,
    draw:      ImageDraw.Draw,
    text:      str,
    x_pos:     int,
    y_pos:     int,
    font:      ImageFont.ImageFont,
    style:     "TextStyle",
    outline_w: int,
) -> None:
    """
    Full single-line draw pipeline:
      1. Glow / halo (if enabled)   — lowest layer
      2. Drop shadow (if enabled)
      3. Outline / stroke            — 8-direction offset draws
      4. Glyph fill                  — solid or gradient
    """
    # 1. Glow
    if getattr(style, "glow_on", False):
        _render_glow(
            pil_img, draw, text, x_pos, y_pos, font,
            getattr(style, "glow_color", (255, 255, 255)),
            getattr(style, "glow_radius", 4),
            getattr(style, "glow_intensity", 0.45),
        )

    # 2. Shadow
    if style.shadow_on:
        _render_shadow(
            pil_img, draw, text, x_pos, y_pos, font,
            style.shadow_color, style.shadow_offset, style.shadow_opacity,
            getattr(style, "shadow_blur", 0.0),
        )

    # 3. Outline (8-direction; always before fill so outline goes under glyph)
    if outline_w > 0:
        oc = style.outline_color
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
            draw.text(
                (x_pos + dx * outline_w, y_pos + dy * outline_w),
                text, font=font, fill=oc,
            )

    # 4. Glyph fill
    if style.gradient_on:
        _render_gradient_text(pil_img, draw, text, x_pos, y_pos, font,
                              style, outline_w)
    else:
        draw.text((x_pos, y_pos), text, font=font, fill=style.fg_color)

def classify_region(img_cv: np.ndarray, block: "OCRBlock") -> None:
    """
    Phase 2 classifier.  Populates block.region_kind, background_kind,
    region_confidence, and block.text_mask in-place.

    Phase 3: if block.override carries region_kind / background_kind values,
    those win unconditionally and auto-classification is skipped.

    Call after detect_bubble_region() and pick_role().
    """
    # ── Phase 3: honour explicit override ────────────────────────────────────
    # Always build the text mask — it is needed by every cleanup strategy.
    block.text_mask = build_text_mask_for_block(img_cv.shape, block, pad=2)
    if block.override is not None:
        changed = False
        if block.override.region_kind is not None:
            try:
                block.region_kind = RegionKind[block.override.region_kind]
                changed = True
            except KeyError:
                pass
        if block.override.background_kind is not None:
            try:
                block.background_kind = BackgroundKind[block.override.background_kind]
                changed = True
            except KeyError:
                pass
        if changed:
            block.region_confidence = 1.0   # override = maximum confidence
            debug_print(
                f"classify_region: override → kind={block.region_kind} "
                f"bg={block.background_kind}"
            )
            return   # skip auto-classification entirely

    has_bubble  = (block.bubble_mask is not None and block.bubble_bbox is not None)
    bg_business = _measure_bg_business(img_cv, block)
    bg_busy     = bg_business > 0.35

    debug_print(
        f"classify_region: bbox={block.bbox()} role={block.bubble_role!r} "
        f"has_bubble={has_bubble} bg_business={bg_business:.2f}"
    )

    if not has_bubble:
        # ── No enclosing bubble/caption found ────────────────────────────────
        if block.bubble_role == "sfx":
            block.region_kind       = RegionKind.SFX_OVER_ART
            block.background_kind   = BackgroundKind.ART
            block.region_confidence = 0.70
        elif bg_busy:
            block.region_kind       = RegionKind.DIALOGUE_OVER_ART
            block.background_kind   = BackgroundKind.ART
            block.region_confidence = 0.60
        else:
            block.region_kind       = RegionKind.UNKNOWN
            block.background_kind   = BackgroundKind.UNKNOWN
            block.region_confidence = 0.30
        return

    # ── Phase 2: caption box detection (before interior analysis) ────────────
    is_caption, caption_score = _detect_caption_box(block)
    if getattr(block, "detector_source", "") == "yolo":
        is_caption = False
    if is_caption:
        block.region_kind       = RegionKind.CAPTION_BOX
        block.background_kind   = BackgroundKind.CLEAN
        block.region_confidence = float(np.clip(0.65 + caption_score * 0.25, 0.0, 0.90))
        debug_print(f"classify_region: CAPTION_BOX score={caption_score:.2f}")
        return

    # ── Bubble interior analysis ──────────────────────────────────────────────
    bx, by, bw, bh = block.bubble_bbox
    bubble_roi = img_cv[by:by + bh, bx:bx + bw]
    if bubble_roi.size == 0:
        block.region_kind       = RegionKind.UNKNOWN
        block.background_kind   = BackgroundKind.UNKNOWN
        block.region_confidence = 0.10
        return

    gray_full  = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
    interior   = gray_full.copy()
    mask_local = block.bubble_mask
    # Neutral-out pixels outside bubble so texture signals aren't polluted by art
    if mask_local.shape == interior.shape:
        interior[mask_local == 0] = 128

    active_mask = (mask_local if mask_local.shape == interior.shape
                   else np.ones(interior.shape, dtype=np.uint8) * 255)

    local_var = _measure_local_variance(interior)
    edges_roi = cv2.Canny(interior, 50, 150)
    edge_dens = float(edges_roi.sum()) / max(1, interior.size * 255)

    debug_print(
        f"classify_region: bubble interior local_var={local_var:.1f} "
        f"edge_dens={edge_dens:.4f}"
    )

    # ── Plain bubble: low local variance AND low edge density ─────────────────
    if local_var < 12.0 and edge_dens < 0.02:
        block.region_kind       = RegionKind.PLAIN_BUBBLE
        block.background_kind   = BackgroundKind.CLEAN
        block.region_confidence = 0.85
        return

    # ── Complex interior: run Phase 2 sub-tests ───────────────────────────────
    is_halftone, ht_score = _detect_halftone(interior, active_mask)
    is_gradient, gr_score = _detect_gradient(interior, active_mask)

    if local_var > 20.0 or edge_dens > 0.03:
        # Clearly complex — commit to a specific kind
        if is_gradient and not is_halftone:
            block.region_kind       = RegionKind.GRADIENT_BUBBLE
            block.background_kind   = BackgroundKind.GRADIENT
            block.region_confidence = float(np.clip(0.55 + gr_score * 0.27, 0.0, 0.82))
        elif is_halftone:
            # Halftone wins even if gradient also fires (both often co-occur)
            block.region_kind       = RegionKind.TEXTURED_BUBBLE
            block.background_kind   = BackgroundKind.TEXTURED
            block.region_confidence = float(np.clip(0.55 + ht_score * 0.27, 0.0, 0.82))
        else:
            # Complex but unidentified — conservative TEXTURED_BUBBLE rather
            # than UNKNOWN so cleanup still runs (auto legacy path)
            block.region_kind       = RegionKind.TEXTURED_BUBBLE
            block.background_kind   = BackgroundKind.TEXTURED
            block.region_confidence = 0.55
        return

    # ── Middle zone: use sub-tests to escape UNKNOWN ──────────────────────────
    if is_gradient and gr_score > 0.50:
        block.region_kind       = RegionKind.GRADIENT_BUBBLE
        block.background_kind   = BackgroundKind.GRADIENT
        block.region_confidence = 0.60
    elif is_halftone and ht_score > 0.50:
        block.region_kind       = RegionKind.TEXTURED_BUBBLE
        block.background_kind   = BackgroundKind.TEXTURED
        block.region_confidence = 0.58
    else:
        # Truly ambiguous — leave for review
        block.region_kind       = RegionKind.UNKNOWN
        block.background_kind   = BackgroundKind.UNKNOWN
        block.region_confidence = 0.45

def decide_cleanup_strategy(block: "OCRBlock") -> None:
    """
    Phase 2 strategy assignment.  Populates block.cleanup_strategy and
    block.placement_strategy in-place.

    Phase 3: explicit overrides in block.override take priority and short-circuit
    the auto logic.  Call this after classify_region().
    """
    # ── Phase 3: honour explicit override ────────────────────────────────────
    if block.override is not None:
        cs = block.override.cleanup_strategy
        ps = block.override.placement_strategy
        if cs is not None or ps is not None:
            if cs is not None:
                block.cleanup_strategy   = cs
            if ps is not None:
                block.placement_strategy = ps
            debug_print(
                f"decide_cleanup_strategy: override → "
                f"cleanup={block.cleanup_strategy!r} "
                f"placement={block.placement_strategy!r}"
            )
            return   # skip auto strategy selection entirely
    kind = block.region_kind
    conf = block.region_confidence

    if kind is None or conf < 0.30:
        block.cleanup_strategy   = "auto"
        block.placement_strategy = "bubble_center"
        debug_print(
            f"decide_cleanup_strategy: no classification (kind={kind}) → auto"
        )
        return

    if kind == RegionKind.PLAIN_BUBBLE:
        block.cleanup_strategy   = "flat_fill"
        block.placement_strategy = "bubble_center"

    elif kind == RegionKind.TEXTURED_BUBBLE:
        # Phase 2: use local TELEA on the bubble ROI for halftone/screentone
        block.cleanup_strategy   = "texture_clone"
        block.placement_strategy = "bubble_center"
        if conf < 0.65:
            block.flag("textured_bubble_low_conf", {
                "conf": round(conf, 2),
                "note": "Halftone/screentone bubble — verify texture_clone result",
            })

    elif kind == RegionKind.GRADIENT_BUBBLE:
        # Phase 2: gradient-preserving local TELEA
        block.cleanup_strategy   = "gradient_fill"
        block.placement_strategy = "bubble_center"

    elif kind == RegionKind.CAPTION_BOX:
        block.cleanup_strategy   = "flat_fill"
        block.placement_strategy = "caption"

    elif kind == RegionKind.SFX_OVER_ART:
        # Conservative — Phase 2 does not add deep inpaint yet
        block.cleanup_strategy   = "mask_only_inpaint"
        block.placement_strategy = "bubble_center"
        block.flag("sfx_over_art", {
            "conf": round(conf, 2),
            "note": "SFX over art — verify inpaint result manually",
        })

    elif kind == RegionKind.DIALOGUE_OVER_ART:
        # Conservative — same as Phase 1
        block.cleanup_strategy   = "mask_only_inpaint"
        block.placement_strategy = "bubble_center"
        block.flag("dialogue_over_art", {
            "conf": round(conf, 2),
            "note": "Dialogue over art — verify inpaint result manually",
        })

    elif kind == RegionKind.UNKNOWN:
        block.cleanup_strategy   = "review"
        block.placement_strategy = "bubble_center"
        block.flag("unknown_region", {
            "conf": round(conf, 2),
            "note": "Region type ambiguous — classify manually before cleanup",
        })

    else:
        # Fallthrough for future enum members
        block.cleanup_strategy   = "auto"
        block.placement_strategy = "bubble_center"

    debug_print(
        f"decide_cleanup_strategy: kind={kind.name} conf={conf:.2f} "
        f"→ strategy={block.cleanup_strategy!r}"
    )

def flat_fill_region(
    img_cv: np.ndarray,
    block:  "OCRBlock",
    result: np.ndarray,
) -> None:
    """
    Erase text by filling with the block's detected background colour.
    Uses the bubble mask to confine the fill — never draws outside the bubble.

    Phase 1 uses block.bg_color which is already sampled by extract_block_colors().
    Phase 2 can improve this by re-sampling from the bubble interior only.
    """
    # Use cached text_mask if available, otherwise build it
    if block.text_mask is not None:
        text_mask = block.text_mask
    else:
        text_mask = build_text_mask_for_block(img_cv.shape, block, pad=4)

    if not np.any(text_mask):
        return

    # Confine fill to bubble interior (avoids white rectangles on art)
    if block.bubble_mask is not None and block.bubble_bbox is not None:
        bx, by, bw, bh = block.bubble_bbox
        bubble_global = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        bubble_global[by:by + bh, bx:bx + bw] = block.bubble_mask
        fill_mask = cv2.bitwise_and(text_mask, bubble_global)
        if not np.any(fill_mask):
            # text_mask fell entirely outside bubble — fall back to text_mask
            fill_mask = text_mask
    else:
        fill_mask = text_mask

    kind_name = getattr(getattr(block, "region_kind", None), "name", "")
    if kind_name == "CAPTION_BOX":
        bg_rgb = tuple(int(v) for v in (getattr(block, "bg_color", None) or (255, 255, 255))[:3])
    else:
        bg_rgb = tuple(int(v) for v in estimate_initial_bg_color(img_cv, block.bbox()))
        saved_bg = tuple(int(v) for v in (getattr(block, "bg_color", None) or bg_rgb)[:3])
        if max(saved_bg) - min(saved_bg) > 120:
            debug_print(
                "style_migration ignored_bg_for_cleanup "
                f"region={block.bbox()} bg={saved_bg} reason=normal_bubble"
            )

    bg_bgr = np.array([bg_rgb[2], bg_rgb[1], bg_rgb[0]], dtype=np.uint8)
    result[fill_mask > 0] = bg_bgr
    debug_print(
        f"flat_fill_region: bbox={block.bbox()} "
        f"bg={bg_rgb} pixels={int(fill_mask.sum() // 255)}"
    )

def mask_only_inpaint_region(
    img_cv: np.ndarray,
    block:  "OCRBlock",
    result: np.ndarray,
) -> None:
    """
    Tight inpaint: erase only the actual text stroke pixels (OCR polygon shapes
    with minimal 2-pixel dilation), then run TELEA inpaint.

    Deliberately does NOT use the bubble mask — the goal is minimum footprint
    on underlying art.  Used for SFX_OVER_ART and DIALOGUE_OVER_ART.
    """
    # Use cached tight mask (pad=2, not 4) for minimum art damage
    if block.text_mask is not None:
        text_mask = block.text_mask.copy()
    else:
        text_mask = build_text_mask_for_block(img_cv.shape, block, pad=2)

    if not np.any(text_mask):
        return

    # Small dilation to catch anti-aliasing fringe, but no more
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    inpainted = cv2.inpaint(result, text_mask, 5, cv2.INPAINT_TELEA)
    # Write only within the mask — don't touch the rest of result
    result[text_mask > 0] = inpainted[text_mask > 0]
    debug_print(
        f"mask_only_inpaint_region: bbox={block.bbox()} "
        f"pixels={int(text_mask.sum() // 255)}"
    )

def texture_clone_region(
    img_cv: np.ndarray,
    block:  "OCRBlock",
    result: np.ndarray,
) -> None:
    """
    Erase halftone / screentone text by running TELEA inpaint locally within
    the bubble ROI.

    Why better than global TELEA ('auto'):
      - The inpainting context is drawn entirely from the bubble interior —
        no pollution from adjacent art pixels at the image boundary.
      - The bubble mask confines the write-back so the inpainted texture never
        bleeds outside the bubble outline.

    Falls back to flat_fill_region() if no bubble geometry is available.
    """
    if block.bubble_bbox is None or block.bubble_mask is None:
        flat_fill_region(img_cv, block, result)
        return

    bx, by, bw, bh = block.bubble_bbox

    # ── Build local masks ────────────────────────────────────────────────────
    text_global = (block.text_mask if block.text_mask is not None
                   else build_text_mask_for_block(img_cv.shape, block, pad=2))

    local_text   = text_global[by:by + bh, bx:bx + bw].copy()
    local_bubble = block.bubble_mask

    if local_bubble.shape != local_text.shape:
        local_bubble = cv2.resize(
            local_bubble, (bw, bh), interpolation=cv2.INTER_NEAREST
        )

    # Inpaint mask = text pixels intersected with bubble interior
    inpaint_mask = cv2.bitwise_and(local_text, local_bubble)
    if not np.any(inpaint_mask):
        inpaint_mask = local_text

    # Small dilation to cover anti-aliasing fringe
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(inpaint_mask, kern, iterations=1)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, local_bubble)

    if not np.any(inpaint_mask):
        flat_fill_region(img_cv, block, result)
        return

    # ── TELEA on the bubble ROI only ─────────────────────────────────────────
    roi          = result[by:by + bh, bx:bx + bw].copy()
    roi_inpainted = cv2.inpaint(roi, inpaint_mask, 5, cv2.INPAINT_TELEA)

    # Write back only where we inpainted AND inside the bubble
    write_mask = (inpaint_mask > 0) & (local_bubble > 0)
    roi[write_mask] = roi_inpainted[write_mask]
    result[by:by + bh, bx:bx + bw] = roi

    debug_print(
        f"texture_clone_region: bbox={block.bbox()} "
        f"bubble=({bx},{by},{bw},{bh}) pixels={int(inpaint_mask.sum() // 255)}"
    )

def gradient_fill_region(
    img_cv: np.ndarray,
    block:  "OCRBlock",
    result: np.ndarray,
) -> None:
    """
    Erase gradient-background text by propagating the gradient into text pixels.

    Runs TELEA inpaint on the bubble ROI with a larger radius (7 vs 5) so the
    smooth luminance ramp propagates across the gap left by erased text.
    The bubble mask confines all writes to the bubble interior.

    Falls back to flat_fill_region() if no bubble geometry is available.
    """
    if block.bubble_bbox is None or block.bubble_mask is None:
        flat_fill_region(img_cv, block, result)
        return

    bx, by, bw, bh = block.bubble_bbox

    # ── Build local masks ────────────────────────────────────────────────────
    text_global = (block.text_mask if block.text_mask is not None
                   else build_text_mask_for_block(img_cv.shape, block, pad=2))

    local_text   = text_global[by:by + bh, bx:bx + bw].copy()
    local_bubble = block.bubble_mask

    if local_bubble.shape != local_text.shape:
        local_bubble = cv2.resize(
            local_bubble, (bw, bh), interpolation=cv2.INTER_NEAREST
        )

    inpaint_mask = cv2.bitwise_and(local_text, local_bubble)
    if not np.any(inpaint_mask):
        inpaint_mask = local_text

    # Slightly larger dilation for gradient (covers more of the text stroke)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_mask = cv2.dilate(inpaint_mask, kern, iterations=1)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, local_bubble)

    if not np.any(inpaint_mask):
        flat_fill_region(img_cv, block, result)
        return

    # ── TELEA with radius=7 for gradient propagation ─────────────────────────
    roi           = result[by:by + bh, bx:bx + bw].copy()
    roi_inpainted = cv2.inpaint(roi, inpaint_mask, 7, cv2.INPAINT_TELEA)

    write_mask = (inpaint_mask > 0) & (local_bubble > 0)
    roi[write_mask] = roi_inpainted[write_mask]
    result[by:by + bh, bx:bx + bw] = roi

    debug_print(
        f"gradient_fill_region: bbox={block.bbox()} "
        f"bubble=({bx},{by},{bw},{bh}) pixels={int(inpaint_mask.sum() // 255)}"
    )

def _auto_erase_block(
    img_cv:         np.ndarray,
    block:          "OCRBlock",
    result:         np.ndarray,
    global_inpaint: np.ndarray,
) -> None:
    """
    The original per-block erase logic from v14, extracted verbatim so it can
    be used as the 'auto' fallback in the new dispatcher.
    Mutates result and global_inpaint in-place.
    """
    text_mask = build_text_mask_for_block(img_cv.shape, block, pad=4)
    ys, xs = np.where(text_mask > 0)
    if len(xs) == 0:
        return

    if block.bubble_bbox is not None and block.bubble_mask is not None:
        bx, by, bw, bh = block.bubble_bbox
        bubble_mask_global = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        bubble_mask_global[by:by + bh, bx:bx + bw] = block.bubble_mask
    else:
        bubble_mask_global = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        x, y, w, h = block.bbox()
        bubble_mask_global[y:y + h, x:x + w] = 255

    fill_mask = cv2.bitwise_and(text_mask, bubble_mask_global)
    if not np.any(fill_mask):
        fill_mask = text_mask

    x1 = max(0, xs.min() - 8)
    y1 = max(0, ys.min() - 8)
    x2 = min(img_cv.shape[1], xs.max() + 9)
    y2 = min(img_cv.shape[0], ys.max() + 9)

    local_fill = fill_mask[y1:y2, x1:x2]
    local_img  = result[y1:y2, x1:x2]
    if local_img.size == 0 or not np.any(local_fill):
        return

    bubble_local = bubble_mask_global[y1:y2, x1:x2] > 0
    bg_pixels    = local_img[np.logical_and(bubble_local, local_fill == 0)]
    if bg_pixels.size == 0:
        border    = np.concatenate([
            local_img[0, :, :], local_img[-1, :, :],
            local_img[:, 0, :], local_img[:, -1, :],
        ], axis=0)
        bg_pixels = border

    bg_std = float(np.std(bg_pixels.astype(np.float32))) if bg_pixels.size else 999.0

    if bg_std < 28.0:
        kind_name = getattr(getattr(block, "region_kind", None), "name", "")
        if kind_name == "CAPTION_BOX":
            bg_rgb = tuple(int(v) for v in (getattr(block, "bg_color", None) or (255, 255, 255))[:3])
            bg_bgr = np.array([bg_rgb[2], bg_rgb[1], bg_rgb[0]], dtype=np.uint8)
        else:
            bg_bgr = np.median(bg_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
            saved_bg = tuple(int(v) for v in (getattr(block, "bg_color", None) or (255, 255, 255))[:3])
            if max(saved_bg) - min(saved_bg) > 120:
                debug_print(
                    "style_migration ignored_bg_for_cleanup "
                    f"region={block.bbox()} bg={saved_bg} reason=normal_bubble"
                )
        local_img[local_fill > 0] = bg_bgr
        result[y1:y2, x1:x2] = local_img
        debug_print(
            f"_auto_erase_block: flat fill  bbox={block.bbox()} bg_std={bg_std:.1f}"
        )
    else:
        global_inpaint[y1:y2, x1:x2] = cv2.bitwise_or(
            global_inpaint[y1:y2, x1:x2], local_fill
        )
        debug_print(
            f"_auto_erase_block: inpaint    bbox={block.bbox()} bg_std={bg_std:.1f}"
        )

def erase_text_region(
    img_cv:  np.ndarray,
    bubbles: List["OCRBlock"],
) -> np.ndarray:
    """
    Phase 2 dispatcher.  Routes each block to the correct erase strategy.

    Strategies (Phase 1 + Phase 2):
      'flat_fill'          → fill text pixels with bg_color (plain bubbles)
      'texture_clone'      → local TELEA on bubble ROI (halftone / screentone)
      'gradient_fill'      → local TELEA with larger radius (gradient bubbles)
      'mask_only_inpaint'  → tight polygon mask + TELEA (SFX / over-art)
      'review'             → skip entirely (unknown / ambiguous)
      'auto'               → original bg_std legacy fallback

    Blocks on 'auto' accumulate into global_inpaint; TELEA fires once at the end.
    """
    result         = img_cv.copy()
    global_inpaint = np.zeros(img_cv.shape[:2], dtype=np.uint8)

    for block in bubbles:
        if not getattr(block, 'visible', True):
            continue
        hydrate_restored_cleanup_masks(img_cv, block)

        # Phase 3: use effective_cleanup_strategy() to honour manual overrides
        strategy = (block.effective_cleanup_strategy()
                    if hasattr(block, 'effective_cleanup_strategy')
                    else (getattr(block, 'cleanup_strategy', 'auto') or 'auto'))

        if strategy == 'review':
            flag_reason = getattr(block.review, 'flag_reason', '') if block.review else ''
            debug_print(
                f"erase_text_region: SKIP (review) bbox={block.bbox()} "
                f"reason={flag_reason!r}"
            )
            continue

        if strategy == 'flat_fill':
            flat_fill_region(img_cv, block, result)
            continue

        if strategy == 'texture_clone':
            texture_clone_region(img_cv, block, result)
            continue

        if strategy == 'gradient_fill':
            gradient_fill_region(img_cv, block, result)
            continue

        if strategy == 'mask_only_inpaint':
            mask_only_inpaint_region(img_cv, block, result)
            continue

        # 'auto' or any unrecognised value → original bg_std logic
        _auto_erase_block(img_cv, block, result, global_inpaint)

    # Final TELEA pass for all 'auto'-strategy blocks accumulated above
    if np.any(global_inpaint):
        result = cv2.inpaint(result, global_inpaint, 5, cv2.INPAINT_TELEA)

    return result

def extract_block_colors(
    img_cv: np.ndarray,
    block:  "OCRBlock",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    x, y, w, h = block.bbox()
    crop = img_cv[max(0, y): y + h, max(0, x): x + w]
    if crop.size == 0:
        return (255, 255, 255), (0, 0, 0)

    full_text_mask = build_text_mask_for_block(img_cv.shape, block, pad=1)
    text_mask = full_text_mask[max(0, y): y + h, max(0, x): x + w] > 0

    if block.bubble_bbox is not None and block.bubble_mask is not None:
        bx, by, bw, bh = block.bubble_bbox
        bubble_mask_global = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        bubble_mask_global[by:by+bh, bx:bx+bw] = block.bubble_mask
        bubble_crop = bubble_mask_global[max(0, y): y + h, max(0, x): x + w] > 0
    else:
        bubble_crop = np.ones(crop.shape[:2], dtype=bool)

    bg_mask = np.logical_and(bubble_crop, np.logical_not(text_mask))
    if bg_mask.sum() < 20:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bg_mask = gray >= np.percentile(gray, 75)

    fg_mask = text_mask
    if fg_mask.sum() < 10:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        fg_mask = gray <= np.percentile(gray, 20)

    b_ch, g_ch, r_ch = cv2.split(crop)

    def robust_rgb(mask: np.ndarray, prefer_dark: bool = False) -> Tuple[int, int, int]:
        if mask.sum() == 0:
            return (0, 0, 0) if prefer_dark else (255, 255, 255)
        vals_r = r_ch[mask]
        vals_g = g_ch[mask]
        vals_b = b_ch[mask]
        if prefer_dark:
            lum = 0.299 * vals_r + 0.587 * vals_g + 0.114 * vals_b
            cutoff = np.percentile(lum, 35)
            keep = lum <= cutoff
            if np.count_nonzero(keep) >= 3:
                vals_r, vals_g, vals_b = vals_r[keep], vals_g[keep], vals_b[keep]
        return (int(np.median(vals_r)), int(np.median(vals_g)), int(np.median(vals_b)))

    bg_color = robust_rgb(bg_mask, prefer_dark=False)
    fg_color = robust_rgb(fg_mask, prefer_dark=True)

    bg_luma = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    fg_luma = 0.299 * fg_color[0] + 0.587 * fg_color[1] + 0.114 * fg_color[2]
    fg_chroma = max(fg_color) - min(fg_color)
    role = getattr(block, "bubble_role", "dialog")

    if role == "dialog":
        if bg_luma > 160 and fg_luma > 125 and fg_chroma < 40:
            fg_color = (0, 0, 0)
        if abs(bg_luma - fg_luma) < 60:
            fg_color = (0, 0, 0) if bg_luma > 127 else (255, 255, 255)
    else:
        if abs(bg_luma - fg_luma) < 40:
            fg_color = (0, 0, 0) if bg_luma > 127 else (255, 255, 255)

    return bg_color, fg_color

def compute_placement(img_cv: np.ndarray, block: "OCRBlock") -> None:
    """
    Phase 2 placement computation.  Populates block.safe_rect and
    block.safe_center by eroding the bubble mask to find a conservative
    text-safe interior rectangle.

    Call after decide_cleanup_strategy() in the enrichment loop.
    _get_typeset_box() will prefer safe_rect over the raw bubble_bbox when it
    is large enough to contain the OCR text area.

    If no bubble geometry is available, falls back to the raw OCR bbox with a
    small flat margin.
    """
    # ── No bubble: use raw OCR bbox with a flat margin ────────────────────────
    if block.bubble_bbox is None or block.bubble_mask is None:
        rx, ry, rw, rh = block.bbox()
        pad = 4
        block.safe_center = (rx + rw // 2, ry + rh // 2)
        block.safe_rect   = (
            max(0, rx - pad), max(0, ry - pad),
            rw + pad * 2,     rh + pad * 2,
        )
        debug_print(
            f"compute_placement: no bubble bbox={block.bbox()} "
            f"safe_rect={block.safe_rect}"
        )
        return

    bx, by, bw, bh = block.bubble_bbox
    mask = block.bubble_mask  # local to (bx, by, bw, bh)

    # Erode the bubble mask by ~9 % of the shorter dimension to get an inset
    # region that is clear of the bubble border / outline stroke.
    inset  = max(6, int(min(bw, bh) * 0.09))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
    )
    eroded = cv2.erode(mask, kernel, iterations=1)

    nz = np.nonzero(eroded)
    if len(nz[0]) == 0:
        # Erosion removed everything (tiny bubble) — fall back to mask centroid
        nz = np.nonzero(mask)
        if len(nz[0]) == 0:
            rx, ry, rw, rh = block.bbox()
            block.safe_center = (rx + rw // 2, ry + rh // 2)
            block.safe_rect   = block.bubble_bbox
            return

    y_min, y_max = int(nz[0].min()), int(nz[0].max())
    x_min, x_max = int(nz[1].min()), int(nz[1].max())

    safe_w = max(1, x_max - x_min + 1)
    safe_h = max(1, y_max - y_min + 1)
    safe_x = bx + x_min
    safe_y = by + y_min

    block.safe_center = (safe_x + safe_w // 2, safe_y + safe_h // 2)
    block.safe_rect   = (safe_x, safe_y, safe_w, safe_h)

    debug_print(
        f"compute_placement: bbox={block.bbox()} bubble={block.bubble_bbox} "
        f"inset={inset} safe_rect={block.safe_rect} "
        f"safe_center={block.safe_center}"
    )

def derive_yolo_text_mask(img_cv, block, min_confidence: float = 0.12):
    """
    Compatibility wrapper for engine.py.

    YOLO gives a region/container bbox, not a text mask. This derives a tight
    glyph mask from OCR boxes inside that region using the new cleanup planner.
    Returns a full-image uint8 mask, or None if confidence is too low.
    """
    try:
        from backend.core.cleanup_plan import build_text_mask_candidates

        region_bbox = block.bbox()
        boxes = list(getattr(block, "boxes", []) or [])

        if not boxes:
            return None

        existing_mask = getattr(block, "text_mask", None)

        candidates = build_text_mask_candidates(
            img_cv,
            region_bbox,
            boxes,
            existing_mask=existing_mask,
        )

        if not candidates:
            return None

        mask, confidence, reason = candidates[0]

        if confidence < min_confidence:
            return None

        return mask

    except Exception as exc:
        try:
            from backend.core.constants import debug_print
            debug_print(f"derive_yolo_text_mask: failed: {exc}")
        except Exception:
            pass
        return None
