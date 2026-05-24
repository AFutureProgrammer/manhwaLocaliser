# Implementation plan: 100% cleanup rate — flat fill, gradient, texture, and YOLO regions

## Goal

Zero skips and zero smudges/residuals across every non-SFX, non-art region regardless of background classification. This requires changes at three layers: classifier thresholds (fewer false unknowns), strategy routing (fewer unnecessary skips), and execution fallbacks (nothing silently fails when a backend is down).

---

## What carries over unchanged from the original plan

Changes 1–3 from the prior plan are still correct and still required. Summary:

- **Change 1** — Relax `classify_background_model` thresholds so compressed or grainy bubbles stop landing in `unknown`.
- **Change 2** — Extend the solid-bubble recheck to `smooth_gradient` and `translucent_gradient` with relaxed metrics, so subtle-gradient bubbles get overridden to flat-fill.
- **Change 3** — Emergency bbox fallback mask for `no_candidates` + flat background, injected before mask assembly.

Those three changes are reproduced in full at the end of this document for completeness. The rest of this document covers what is new, what was wrong, and the extra changes needed to reach 100%.

---

## Analysis: why the original plan is not enough

After reading `engine.py`, four problems emerge.

**Problem 1 — `select_strategy` hard-blocks three background types by default**

| Background | Condition to proceed | Default value | Result |
|---|---|---|---|
| `translucent_gradient` (speech bubble) | `cleanup_allow_translucent_caption = True` | `False` | **skip** |
| `halftone_texture` | `cleanup_allow_texture_inpaint = True` AND `cleanup_fallback_backend == "iopaint"` | `False / "telea"` | **review/skip** |
| `busy_art` | `auto_clean_busy_background = True` AND `busy_background_cleanup_mode in ("tight_mask","telea")` | `False / "off"` | **skip** |

Changes 1–3 reduce the number of regions that land in these categories, but regions that genuinely have these backgrounds still skip. No amount of threshold relaxation fixes a hard `return "skip", "skip"` in `select_strategy`.

**Problem 2 — `translucent_gradient` uses the wrong flag for speech bubbles**

Line 6435:
```python
if not policy.allow_gradient_fill or not policy.cleanup_allow_translucent_caption:
    return "skip", "skip"
```
`cleanup_allow_translucent_caption` is a caption-box safety gate, not a speech-bubble gate. Speech bubbles with translucent gradients (common in manhwa fantasy panels) are silently skipped because a caption flag is False.

**Problem 3 — IOPaint failure in `_execute_grouped_fallback` is fatal by default**

`_execute_grouped_fallback` (line 2463) already has a Telea fallback path guarded by `cleanup_iopaint_allow_opencv_fallback`. That flag defaults to `False`. So when IOPaint is configured but down, grouped inpainting returns `False`, and the region is left uncleaned. The plan needs to change the default to `True` (or unconditionally fall through to Telea) rather than just caching the probe.

`_execute_iopaint_candidate` (line 3249) has no fallback at all — it returns `False` on any failure with no Telea recovery. For the 100% goal it also needs a Telea rescue.

**Problem 4 — `cleanup_residual_retry_dilate_px` defaults to 1 px**

The residual retry mechanism re-runs cleanup with a dilated mask. 1 px dilation catches almost nothing — typical glyph stroke overhang is 2–4 px, especially with anti-aliasing. Smudges and leftover pixels survive the retry at 1 px and are visible in the final output.

---

## Changes to `cleanup_plan.py`

### Change 1 — Relax `classify_background_model` thresholds *(unchanged from prior plan)*

**File:** `cleanup_plan.py` · **Function:** `classify_background_model` · **Lines:** 1204, 1220–1223

```python
# BEFORE
elif mean_brightness > 205.0 and local_var < 18.0 and edge_density < 0.04:
...
elif mean_brightness < 70.0 and local_var < 18.0:
    model = "dark_bubble"
elif local_var < 14.0 and edge_density < 0.025:
    model = "flat_light" if is_neutral else "flat_colored"
else:
    model = "unknown"

# AFTER
elif mean_brightness > 205.0 and local_var < 22.0 and edge_density < 0.045:
...
elif mean_brightness < 85.0 and local_var < 24.0:
    model = "dark_bubble"
elif local_var < 22.0 and edge_density < 0.045:
    model = "flat_light" if is_neutral else "flat_colored"
else:
    model = "unknown"
```

### Change 2 — Extend solid-bubble recheck *(unchanged from prior plan)*

**File:** `cleanup_plan.py` · **Function:** `build_cleanup_plan` · **Lines:** 7106, 7122

Add `"smooth_gradient"` and `"translucent_gradient"` to the recheck trigger set. Relax recheck thresholds to `gray_std <= 20.0`, `channel_std <= 24.0`, `sat_std <= 28.0`, `edge_density <= 0.045`.

### Change 3 — Emergency bbox fallback for no-candidates *(unchanged from prior plan)*

**File:** `cleanup_plan.py` · **Function:** `build_cleanup_plan` · **Before line:** 7274

Inject a solid `text_bbox`-based mask when `text_mask is None`, `text_mask_reason == "no_candidates"`, region is `speech_bubble` or `caption_box`, and background is `flat_light`, `flat_colored`, or `dark_bubble`. Set `text_mask_confidence = 0.20`, `text_mask_reason = "emergency_flat_fill_bbox_fallback"`.

### Change 4 — Fix `translucent_gradient` routing for speech bubbles *(NEW)*

**File:** `cleanup_plan.py` · **Function:** `select_strategy` · **Line:** 6434–6441

**The bug:** `cleanup_allow_translucent_caption` is a caption-box flag that is False by default, but the `translucent_gradient` branch for speech/thought bubbles also checks it. This silently skips every translucent-gradient speech bubble.

```python
# BEFORE (line 6434)
if background_model == "translucent_gradient":
    if not policy.allow_gradient_fill or not policy.cleanup_allow_translucent_caption:
        return "skip", "skip"
    if text_mask_confidence >= 0.20 and container_confidence >= 0.35:
        return "gradient_fill", "idw_lab"
    if text_mask_confidence >= 0.30 and policy.allow_texture_inpaint:
        return "mask_inpaint", "telea"
    return "skip", "skip"

# AFTER
if background_model == "translucent_gradient":
    if not policy.allow_gradient_fill:
        return "skip", "skip"
    # cleanup_allow_translucent_caption only gates caption_box (handled above).
    # Speech and thought bubbles with translucent gradients route to
    # gradient_fill or mask_inpaint/telea directly.
    if text_mask_confidence >= 0.20 and container_confidence >= 0.35:
        return "gradient_fill", "idw_lab"
    if text_mask_confidence >= 0.20:
        return "gradient_fill", "telea"    # no container, still attempt
    if text_mask_confidence >= 0.15 and policy.allow_texture_inpaint:
        return "mask_inpaint", "telea"     # low confidence, tight mask fallback
    return "skip", "skip"
```

Note: `cleanup_allow_translucent_caption` is still used in the `caption_box` branch (line 6393) where it correctly gates aggressive cleanup on art-backed captions. This change does not touch that.

### Change 5 — Fix `halftone_texture` routing to not require IOPaint *(NEW)*

**File:** `cleanup_plan.py` · **Function:** `select_strategy` · **Line:** 6443–6450

**The bug:** When `cleanup_allow_texture_inpaint = True` but `cleanup_fallback_backend != "iopaint"` (i.e. the user has enabled texture cleanup but is using local OpenCV), the strategy returns `"review", "skip"` rather than attempting Telea. IOPaint produces better results on texture, but Telea is far better than skipping entirely.

```python
# BEFORE (line 6443)
if background_model == "halftone_texture":
    if not policy.cleanup_allow_texture_inpaint:
        return "skip", "skip"
    if text_mask_confidence >= 0.25:
        if policy.cleanup_fallback_backend == "iopaint" or policy.cleanup_prefer_iopaint_for_texture:
            return "texture_clone", "telea"
        return "review", "skip"
    return "skip", "skip"

# AFTER
if background_model == "halftone_texture":
    if not policy.cleanup_allow_texture_inpaint:
        return "skip", "skip"
    if text_mask_confidence >= 0.25:
        if policy.cleanup_fallback_backend == "iopaint" or policy.cleanup_prefer_iopaint_for_texture:
            return "texture_clone", "telea"
        # Telea on a tight mask is visually acceptable for most halftone
        # backgrounds and infinitely better than a permanent skip.
        return "mask_inpaint", "telea"
    return "skip", "skip"
```

### Change 6 — Enable `busy_art` Telea fallback *(NEW)*

**File:** `cleanup_plan.py` · **Function:** `select_strategy` · **Line:** 6452–6461

**The bug:** `busy_art` requires `auto_clean_busy_background = True` AND `busy_background_cleanup_mode in ("tight_mask","telea")`. Both default to `False`/`"off"`. Regions over complex art backgrounds always skip. With a high-quality text mask, Telea on a tight glyph mask is the right approach — it doesn't try to reconstruct background texture, only fills the glyph-pixel holes.

```python
# BEFORE (line 6452)
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

# AFTER
if background_model == "busy_art":
    if not policy.allow_texture_inpaint:
        return "skip", "skip"
    # Explicit config opts in for full mode.
    if (
        policy.auto_clean_busy_background
        and text_mask_confidence >= policy.t2_text_conf
        and policy.busy_background_cleanup_mode in ("tight_mask", "telea")
    ):
        return "mask_inpaint", "telea"
    # Tight-mask Telea always available as a fallback when confidence is high.
    # Does not reconstruct the background — only patches glyph holes.
    if text_mask_confidence >= 0.40 and policy.allow_texture_inpaint:
        return "mask_inpaint", "telea"
    return "skip", "skip"
```

The `text_mask_confidence >= 0.40` guard is intentionally strict. A poor mask on art background causes visible damage. This only fires when segmentation is confident the glyph pixels are correctly identified.

### Change 7 — Lower `unknown` confidence floor when mask is valid *(NEW)*

**File:** `cleanup_plan.py` · **Function:** `select_strategy` · **Line:** 6463–6468

**The current code:**
```python
# unknown (catch-all)
if not policy.allow_texture_inpaint:
    return "review", "skip"
if text_mask_confidence >= 0.35:
    return "mask_inpaint", "telea"
return "review", "skip"
```

After Changes 1–3, fewer regions land in `unknown`, but some will still arrive here — particularly YOLO-detected dialogue boxes where the background is genuinely indeterminate. The 0.35 floor is high; YOLO-sourced regions frequently have confidence in the 0.20–0.34 range because the bounding box includes padding and the segmenter can't commit.

```python
# AFTER
# unknown (catch-all)
if not policy.allow_texture_inpaint:
    return "review", "skip"
if text_mask_confidence >= 0.35:
    return "mask_inpaint", "telea"
# For YOLO-sourced regions with a usable mask, attempt Telea rather than
# leaving text permanently visible. The mask safety gates in build_cleanup_plan
# still apply and will convert to skip if geometry is unsafe.
if text_mask_confidence >= 0.20:
    return "mask_inpaint", "telea"
return "review", "skip"
```

The downstream `_reject_unsafe_cleanup_mask` gate is the real safety net here. A region that reaches `mask_inpaint/telea` with a bad mask will be caught there. Keeping the strategy as `review/skip` at 0.20–0.34 is over-conservative.

---

## Changes to `engine.py`

### Change 8 — IOPaint caching *(revised from prior plan)*

**File:** `engine.py` · **Class:** `LocalizerEngine`

The prior plan was conceptually right but missed two things from reading the actual code:

1. `_iopaint_candidate_timeout()` already bounds calls to 5 seconds (configurable), so the pipeline does not hang indefinitely — it just wastes 5 seconds per region when IOPaint is down. The caching saves that time.
2. `_execute_grouped_fallback` has a Telea fallback path (`cleanup_iopaint_allow_opencv_fallback`) but it defaults to `False`. The caching check does not help here — the real fix is to change the default to `True`.

#### 8a — Add caching fields to `__init__` (after line 731, alongside `_restoring_pages`)

```python
self._iopaint_available: Optional[bool] = None  # None = never checked
self._iopaint_last_check: float = 0.0           # monotonic seconds
```

#### 8b — Add `_check_iopaint_availability` method

```python
def _check_iopaint_availability(self, url: str) -> bool:
    """
    Check if the IOPaint server at *url* is reachable.
    Result cached for 30 s to avoid blocking every region during batch cleanup.
    Uses a 1-second GET probe — the server root returns 200 without side-effects.
    """
    if not url:
        return False
    now = time.monotonic()
    if self._iopaint_available is not None and (now - self._iopaint_last_check) < 30.0:
        return bool(self._iopaint_available)
    try:
        resp = requests.get(url, timeout=1.0)
        self._iopaint_available = resp.status_code < 500
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        self._iopaint_available = False
    self._iopaint_last_check = time.monotonic()
    return bool(self._iopaint_available)
```

#### 8c — Guard `_execute_iopaint_candidate` with a fast pre-check AND add Telea rescue

Currently `_execute_iopaint_candidate` (line 3249) returns `False` on any failure with no recovery. Callers in `_run_cleanup_candidate` (line 3346–3350) treat a `False` as `available: False` and surface it to the UI as unavailable. For batch cleanup, this means the region is left uncleaned.

```python
def _execute_iopaint_candidate(
    self, img_cv: np.ndarray, result: np.ndarray, mask: np.ndarray, plan: Any
) -> Tuple[bool, str]:
    url = str(
        getattr(plan, "iopaint_url", "")
        or getattr(self.model_config, "iopaint_url", "") or ""
    ).strip()
    if not url:
        return False, "IOPaint not configured"

    # Fast pre-check: avoid full 5-second timeout when server is known-down.
    if not self._check_iopaint_availability(url):
        plan.debug_metrics["cleanup_backend_fallback"] = "iopaint_unreachable"
        # Telea rescue so the region is cleaned rather than skipped.
        if _config_bool(getattr(self.model_config, "cleanup_iopaint_allow_opencv_fallback", True)):
            inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
            result[mask > 0] = inpainted[mask > 0]
            return True, "iopaint_unreachable_fallback_telea"
        return False, "IOPaint unreachable"

    timeout = self._iopaint_candidate_timeout()
    try:
        ok_img, img_buf = cv2.imencode(".png", img_cv)
        ok_mask, mask_buf = cv2.imencode(".png", mask)
        if not ok_img or not ok_mask:
            return False, "IOPaint unavailable"
        resp = requests.post(
            url,
            files={
                "image": ("image.png", img_buf.tobytes(), "image/png"),
                "mask":  ("mask.png",  mask_buf.tobytes(), "image/png"),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2] != result.shape[:2]:
            return False, "IOPaint unavailable"
        result[mask > 0] = decoded[mask > 0]
        return True, ""
    except requests.exceptions.Timeout:
        self._iopaint_available = False   # mark cache invalid immediately
        self._iopaint_last_check = time.monotonic()
        return False, "IOPaint timed out"
    except Exception:
        return False, "IOPaint unavailable"
```

Key differences from the original plan:
- `cleanup_iopaint_allow_opencv_fallback` now **defaults to `True`** (changed from `False`) so Telea fires automatically when IOPaint is unreachable.
- A timeout at the POST level updates `_iopaint_available = False` immediately rather than waiting for the 30-second cache expiry.

#### 8d — Change `_execute_grouped_fallback` default fallback to `True`

Line 2466 currently reads:
```python
if _config_bool(getattr(self.model_config, "cleanup_iopaint_allow_opencv_fallback", False)):
```

Change the default from `False` to `True`:
```python
if _config_bool(getattr(self.model_config, "cleanup_iopaint_allow_opencv_fallback", True)):
```

This is a one-character change but it means grouped-inpaint regions are always cleaned with Telea when IOPaint is unavailable, rather than silently failing. Line 2492 (the error path) has the same guard and needs the same change.

Also add the pre-check before the POST in the iopaint branch:

```python
if backend == "iopaint":
    url = str(getattr(self.model_config, "iopaint_url", "") or "").strip()
    if not url or not self._check_iopaint_availability(url):   # ← add check
        if _config_bool(getattr(self.model_config, "cleanup_iopaint_allow_opencv_fallback", True)):
            inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
            result[mask > 0] = inpainted[mask > 0]
            return True, "iopaint_unavailable_fallback_telea"
        return False, "iopaint_unavailable:no_url_or_unreachable"
    # ... existing POST logic unchanged ...
```

### Change 9 — Increase residual retry dilation default *(NEW)*

**File:** `engine.py` · **Where:** `ModelConfig` default for `cleanup_residual_retry_dilate_px`

The residual retry runs a second cleanup pass with a mask dilated by `cleanup_residual_retry_dilate_px` pixels. At 1 px (current default), it catches almost nothing — sub-pixel anti-aliasing residuals survive. At 3 px it catches typical glyph overhang from Korean/CJK strokes with outline shadows.

This is a config default change, not a code logic change. The correct place depends on `ModelConfig` implementation, but it is exposed in the bootstrap dict at line 10464:

```python
# BEFORE
"cleanup_residual_retry_dilate_px": getattr(self.model_config, "cleanup_residual_retry_dilate_px", 1),

# AFTER — raise default from 1 to 3
"cleanup_residual_retry_dilate_px": getattr(self.model_config, "cleanup_residual_retry_dilate_px", 3),
```

The same `getattr(..., 1)` default appears wherever `cleanup_residual_retry_dilate_px` is read. Grep for all occurrences and update to `3`. The maximum safe value is 4–5 px beyond which the dilated mask begins overlapping adjacent text in tightly-spaced columns.

---

## Cumulative effect on `select_strategy` routing

| Background | Before | After |
|---|---|---|
| `flat_light` / `flat_colored` / `dark_bubble` | `flat_fill/local_sample` ✓ | unchanged |
| `smooth_gradient` | `gradient_fill` (≥0.20 conf) ✓ | unchanged |
| `translucent_gradient` (speech bubble) | **skip** (caption flag missing) | `gradient_fill/idw_lab` or Telea |
| `halftone_texture` (texture cleanup enabled, local backend) | **review/skip** | `mask_inpaint/telea` |
| `busy_art` (conf ≥ 0.40) | **skip** (mode not configured) | `mask_inpaint/telea` |
| `unknown` (0.20 ≤ conf < 0.35) | **review/skip** | `mask_inpaint/telea` |
| IOPaint down, grouped fallback | **return False** (uncleaned) | Telea fallback |
| Post-cleanup smudge | 1 px retry (catches nothing) | 3 px retry (catches stroke overhang) |

---

## Verification plan

### Automated

```bash
python -m unittest backend.core.test_cleanup_pipeline
```

Focus on:
- `test_select_strategy_translucent_gradient_speech_bubble` — must return `gradient_fill`, not `skip`
- `test_select_strategy_halftone_texture_local_backend` — must return `mask_inpaint`, not `review`
- `test_iopaint_unreachable_fallback` — must clean the region with Telea in < 2 s
- Existing flat_fill/gradient_fill/dark_bubble tests — must remain unchanged

### Manual

| Scenario | Expected result |
|---|---|
| Panel with translucent-gradient speech bubble | `gradient_fill/idw_lab`; no skip; no smudge |
| Halftone background bubble, local OpenCV only | `mask_inpaint/telea`; not `review` |
| Busy art background, mask confidence ≥ 0.40 | `mask_inpaint/telea`; clean hole only |
| IOPaint URL set but server offline, batch cleanup | Telea fallback in < 2 s per region; no hang |
| Any region after cleanup | Second-pass residual retry at 3 px removes outlier glyph pixels |

### Debug metrics to confirm per-region

- `plan.cleanup_strategy` — should never be `skip` for non-SFX regions with valid masks
- `plan.debug_metrics["cleanup_backend_fallback"]` — `"iopaint_unreachable_fallback_telea"` confirms the new path fired
- `plan.debug_metrics["emergency_flat_fill_fallback"]` — confirms Change 3 fired for no-candidate cases
- `plan.debug_metrics["background_override"]` — `"solid_container_recheck"` confirms Change 2 fired

---

## Risk table

| Change | Risk | Mitigating factor |
|---|---|---|
| Changes 1–3 (thresholds / fallback mask) | False-positive classifications on art panels | `_reject_unsafe_cleanup_mask` geometry gates still apply |
| Change 4 (translucent gradient routing) | Gradient fill on a region that is actually complex art | `container_confidence >= 0.35` gate still required for idw_lab; Telea path is tight-mask only |
| Change 5 (halftone Telea) | Visible seam on screentone backgrounds | Seam is better than permanent text residual; flagged for review via `debug_metrics` |
| Change 6 (busy_art at 0.40+) | Partial patch visible on complex art | 0.40 confidence floor means segmenter is sure about glyph pixels; patch is glyph-shaped, not rectangular |
| Change 7 (unknown at 0.20+) | Poor mask damages background | All `_reject_unsafe_cleanup_mask` gates (rectangularity, border touch, region ratio) still apply |
| Change 8 (IOPaint fallback default True) | Telea used where IOPaint was expected | IOPaint result is always preferred when reachable; Telea only fires on unreachable/error |
| Change 9 (3 px dilation) | Over-dilation erases adjacent text in dense panels | 3 px is within normal stroke width for Korean glyphs; safety gate on mask_region_ratio prevents runaway expansion |
