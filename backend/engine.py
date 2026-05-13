"""
backend/engine.py
─────────────────
LocalizerEngine: all pipeline logic with zero Tkinter dependency.

This module owns the headless backend logic directly.  The legacy monolith
is not imported during normal execution; only Tkinter-free data models, OCR,
cleanup, translation, and typesetting logic needed by the pywebview layer
live here.

State ownership
───────────────
The engine owns all mutable state.  The React UI only ever holds a snapshot
(the bootstrap dict).  Mutations flow:  UI action → api.py → engine method →
returns new bootstrap dict → React re-renders.

Progress / status
─────────────────
The engine does NOT call root.after().  Instead it calls self._notify() with
a status string.  The api layer converts that into a JS CustomEvent push via
window.evaluate_js() so the React UI can show live progress without polling.

Memory integration (Phases 1–6)
────────────────────────────────
Three scoped stores are initialised on import_chapter():

    _global_glossary / _global_names / _global_blocked
        Shared across ALL series.  Empty until manually populated.

    _glossary / _name_mem / _blocked
        Per-series.  Always starts empty for new series.
        migrate_legacy_series() is the only path that seeds from NAME_MAP /
        GLOSSARY_ANCHORS — never called automatically.

    _chapter_tm
        Chapter-local TM.  Machine entries stored after every translation.
        Write-only for prompts until explicit future review/promotion work.

Phase 4  — retrieve_batch() builds bounded exact name/glossary constraints.
Phase 5  — approve / reject / promote API; all require explicit calls.
Phase 6  — blocked mappings checked post-translation; rejections auto-block.
"""

from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum, auto as _auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

EDITOR_DEBUG = os.environ.get("MANHWA_RENDER_DEBUG", "1").strip().lower() not in {"0", "false", "off", "no"}


def render_debug(event: str, **payload: Any) -> None:
    if EDITOR_DEBUG:
        print(f"[renderDebug] {event} {json.dumps(payload, default=str, ensure_ascii=False)}", flush=True)


from backend.core.cleanup import (
    _check_text_contrast,
    _draw_line_with_style,
    _render_plate,
    build_ellipse_mask,
    build_text_mask_for_block,
    classify_region,
    compute_placement,
    decide_cleanup_strategy,
    estimate_initial_bg_color,
    detect_bubble_region,
    extract_block_colors,
)
from backend.core.cleanup_plan import (
    CleanupPolicy,
    _write_cleanup_metadata_to_block,
    _try_force_solid_bubble_flat_fill,
    build_cleanup_plan,
    erase_text_region_planned,
    execute_cleanup_plan,
    summarize_cleanup_plan,
)
from backend.core.config import ModelConfig
from backend.core import sam2_mask
from backend.core.constants import (
    COMIC_FONTS_DIR,
    GLOSSARY_ANCHORS,
    NAME_MAP,
    NLLB_MODEL_DIR,
    SFX_MAP,
    debug_print,
)
from backend.core.ocr import (
    OCRProcessor,
    OCRRegionDetector,
    YoloV6RegionDetector,
    build_mask,
    group_ocr_blocks,
    image_to_base64,
)
from backend.core.project import ChapterManager, ChapterPage, SeriesDB
from backend.core.project import _count_images, _has_ml_state
from backend.core.regions import (
    BackgroundKind,
    CharacterMemory,
    OCRBlock,
    ROLE_DEFAULT_PRESET,
    RegionKind,
    RegionOverride,
    RegionReview,
    STYLE_PRESETS,
    TextStyle,
    _FitResult,
    _apply_block_dict,
    _block_to_dict,
)
from backend.core.text_utils import (
    apply_glossary_anchors,
    clean_translation_text,
    contains_hangul,
    heuristic_localize_line,
    is_likely_garbage_literal,
    localize_name,
    normalize_ocr_korean,
    sanitize_final_translation,
)
from backend.core.translation import (
    POLISHER_SCHEMA,
    QA_SCHEMA,
    QWEN_OCR_SCHEMA,
    TRANSLATOR_SCHEMA,
    VISION_SCHEMA,
    NLLBTranslator,
    OllamaClient,
    _HAS_CTRANSLATE,
)
from backend.core.typesetting import _text_width, _wrap_text
from backend.core.sources import get_provider, list_providers


# ── Project paths ──────────────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BACKEND_DIR  = pathlib.Path(__file__).resolve().parent
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)



def clamp_int(n: float, floor: int, ceil: int) -> int:
    """Clamps a number between floor and ceil and returns it as int."""
    return max(floor, min(int(n), ceil))

def choose_system_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Last-resort system-font loader used by ComicFontLibrary when no comic
    font is available.  Walks a priority list of common system fonts.
    """
    candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            return font
        except (IOError, OSError):
            continue
    return ImageFont.load_default()

class ComicFontLibrary:
    """
    Scans the comic_fonts/ directory at startup, ranks every font by how
    well its filename matches a set of scanlation roles, then serves the
    best available font for each bubble at render time.
    """

    _ROLE_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
        "dialog": [
            ("animeace", 18), ("anime ace", 18), ("animace", 16), ("wildwords", 12), ("wild", 10), ("anime", 9),
            ("ccwild", 12), ("blambot", 8), ("comic", 6), ("manga", 6),
            ("speech", 8), ("balloon", 7), ("dialog", 8), ("dialogue", 8),
            ("letter", 5), ("regular", 3), ("roman", 3), ("caption", 6),
        ],
        "bold": [
            ("bold", 10), ("heavy", 9), ("black", 8), ("ultra", 8),
            ("extrabold", 11), ("semibold", 6), ("shout", 10), ("yell", 9),
            ("wide", 4), ("strong", 5), ("impact", 7), ("condensed", 4),
        ],
        "thought": [
            ("light", 9), ("thin", 8), ("italic", 7), ("oblique", 6),
            ("thought", 12), ("whisper", 11), ("soft", 7), ("medium", 4),
        ],
        "sfx": [
            ("sfx", 12), ("effect", 9), ("action", 7), ("bang", 9),
            ("pow", 8), ("burst", 7), ("brush", 6), ("grunge", 6),
            ("display", 5), ("inline", 5), ("outline", 5),
        ],
    }

    def __init__(self, font_dir: str = COMIC_FONTS_DIR) -> None:
        self.font_dir = font_dir
        self._ranked: Dict[str, List[Tuple[int, str]]] = {r: [] for r in self._ROLE_KEYWORDS}
        self._cache:  Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        # stem → path mapping for get_by_name()
        self._stem_to_path: Dict[str, str] = {}
        self._scan()

    def _scan(self) -> None:
        if not os.path.isdir(self.font_dir):
            debug_print(f"ComicFontLibrary: {self.font_dir!r} not found — will use system fonts")
            return

        paths: List[str] = []
        for root, _dirs, files in os.walk(self.font_dir):
            for fn in files:
                if fn.lower().endswith((".ttf", ".otf")):
                    paths.append(os.path.join(root, fn))

        debug_print(f"ComicFontLibrary: {len(paths)} font file(s) found in {self.font_dir!r}")

        for path in paths:
            stem = (os.path.splitext(os.path.basename(path))[0]
                    .lower()
                    .replace("-", "").replace("_", "").replace(" ", ""))
            self._stem_to_path[stem] = path   # for get_by_name()
            for role, kw_list in self._ROLE_KEYWORDS.items():
                score = sum(w for kw, w in kw_list if kw in stem)
                if score > 0:
                    self._ranked[role].append((score, path))

        for role in self._ranked:
            self._ranked[role].sort(key=lambda t: -t[0])
            if self._ranked[role]:
                best = os.path.basename(self._ranked[role][0][1])
                debug_print(f"ComicFontLibrary: best '{role}' → {best!r} (score={self._ranked[role][0][0]})")

    def get(self, role: str, size: int) -> ImageFont.FreeTypeFont:
        requested_role = role
        if str(role or "").lower() != "sfx":
            role = "dialog"
        return self.get_native(role, size, requested_role=requested_role)

    def get_native(self, role: str, size: int, requested_role: Optional[str] = None) -> ImageFont.FreeTypeFont:
        requested_role = requested_role or role
        key = (role, size)
        if key in self._cache:
            return self._cache[key]

        fallback_order = [role, "dialog"] + [r for r in self._ROLE_KEYWORDS if r not in (role, "dialog")]
        for r in fallback_order:
            for _score, path in self._ranked.get(r, []):
                try:
                    font = ImageFont.truetype(path, size)
                    debug_print(f"ComicFontLibrary.get: role={requested_role!r} effective_role={role!r} size={size} → {os.path.basename(path)!r}")
                    self._cache[key] = font
                    return font
                except (IOError, OSError):
                    continue

        font = choose_system_font(size)
        self._cache[key] = font
        return font

    # ── Public API ────────────────────────────────────────────────────────────

    def list_fonts(self) -> List[str]:
        """Return a sorted, deduplicated list of font basenames (no extension).
        Use this instead of reading ._ranked directly."""
        seen: set = set()
        names: List[str] = []
        for role_list in self._ranked.values():
            for _score, path in role_list:
                stem = os.path.splitext(os.path.basename(path))[0]
                if stem not in seen:
                    seen.add(stem)
                    names.append(stem)
        return sorted(names)

    def get_by_name(self, stem: str, size: int) -> ImageFont.FreeTypeFont:
        """Load a font by its bare filename stem (no extension).
        Falls back to the best dialog font if the name is not found."""
        key = (f"__named__{stem}", size)
        if key in self._cache:
            return self._cache[key]
        # normalise to the same form used by _scan()
        norm = stem.lower().replace("-", "").replace("_", "").replace(" ", "")
        path = self._stem_to_path.get(norm)
        if path:
            try:
                font = ImageFont.truetype(path, size)
                self._cache[key] = font
                debug_print(f"ComicFontLibrary.get_by_name: {stem!r} size={size} → {os.path.basename(path)!r}")
                return font
            except (IOError, OSError):
                pass
        debug_print(f"ComicFontLibrary.get_by_name: {stem!r} not found, falling back to dialog")
        return self.get("dialog", size)

    def pick_role(self, img_cv: np.ndarray, bbox: Tuple[int, int, int, int], text: str = "") -> str:
        x, y, w, h = bbox
        crop = img_cv[max(0, y): y + h, max(0, x): x + w]
        compact = re.sub(r"\s+", "", text or "")
        # Exclude ':' so speaker-name labels like '한스:' are not tagged as SFX.
        # SFX fonts have very wide glyphs; a name label rendered in one overflows.
        if compact in SFX_MAP or (len(compact) <= 4 and not re.search(r"[,.!?:]", compact) and h <= 110 and w <= 180):
            return "sfx"
        if crop.size == 0 or h < 12:
            return "dialog"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < 80:
            return "bold"

        if (w / max(h, 1)) > 4.0 and h < 45:
            return "sfx"

        dark_thresh  = float(np.percentile(gray, 12))
        dark_pixels  = gray[gray <= dark_thresh]
        if dark_pixels.size > 0:
            avg_dark     = float(np.mean(dark_pixels))
            dark_fraction = dark_pixels.size / max(gray.size, 1)
            if avg_dark < 35 and dark_fraction > 0.10:
                return "bold"

        return "dialog"

# ── Memory package (optional — engine degrades gracefully if absent) ──────────
try:
    from memory import (
        GlossaryStore,
        NameMemory,
        ChapterTM,
        BlockedMappingStore,
        GlossaryEntry,
        NameEntry,
        retrieve_batch,
        check_name_drift,
        check_glossary_drift,
        check_blocked_output,
        approve_entry,
        reject_entry,
        mark_reviewed,
        promote_entry_to_series,
        promote_entry_to_global,
    )
    from memory.models import make_id, now_iso
    _HAS_MEMORY = True
    _MEMORY_IMPORT_ERROR = ""
except ImportError as _mem_err:
    _MEMORY_IMPORT_ERROR = str(_mem_err)
    print(f"[engine] memory package unavailable: {_mem_err} — running without memory")
    traceback.print_exc()
    _HAS_MEMORY = False

_MEMORY_ROOT = str(_PROJECT_ROOT / "series_memory")


def _memory_slug(value: str) -> str:
    try:
        from memory.storage import series_slug
        return series_slug(value)
    except Exception:
        slug = str(value or "").lower().replace(":", "_")
        slug = re.sub(r"[^\w\s-]", "", slug)
        return re.sub(r"[\s-]+", "_", slug).strip("_") or "unnamed_series"


def _hex_color(rgb: Any, fallback: str = "#111111") -> str:
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        return fallback
    try:
        r, g, b = [max(0, min(255, int(v))) for v in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return fallback


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_SFX_LIKE_ROLES = {"sfx", "sound", "sound_effect", "impact"}
_SFX_LIKE_KINDS = {"SFX_OVER_ART", "DIALOGUE_OVER_ART"}

# Pass 4: a narrower SFX-class set used by the pipeline master toggle
# `process_sfx_regions`. We deliberately exclude `bold` /
# `DIALOGUE_OVER_ART` here because those can legitimately be dialogue that
# the user still wants translated. YOLO class ids 2=sfx and 3=shout are
# treated as pipeline SFX when the master toggle is off.
_PIPELINE_SFX_ROLES = {"sfx", "sound", "sound_effect", "impact", "shout"}
_PIPELINE_SFX_KINDS = {"SFX_OVER_ART"}


def _is_pipeline_sfx(block: Any) -> bool:
    """Return True for SFX-class regions that should be skipped by the pipeline
    when ``ModelConfig.process_sfx_regions`` is False. Also treats explicit
    YOLO class ids 2 (sfx) and 3 (shout) as SFX regardless of role string."""
    role = str(getattr(block, "bubble_role", None) or "").lower()
    yolo_kind = str(getattr(block, "yolo_kind", None) or "").lower()
    kind_name = getattr(getattr(block, "region_kind", None), "name", "")
    yolo_class_id = getattr(block, "yolo_class_id", None)
    try:
        yolo_class_id = int(yolo_class_id) if yolo_class_id is not None else None
    except Exception:
        yolo_class_id = None
    return (
        role in _PIPELINE_SFX_ROLES
        or yolo_kind in _PIPELINE_SFX_ROLES
        or kind_name in _PIPELINE_SFX_KINDS
        or yolo_class_id in {2, 3}
    )


def _cleanup_override_allows_pipeline_sfx(block: Any) -> bool:
    override = getattr(block, "override", None)
    mode = str(getattr(override, "cleanup_override_mode", "") or "").strip().lower()
    region_class = str(getattr(override, "cleanup_region_class", "") or "").strip().lower()
    return mode in {
        "force_allow", "force_solid", "force_telea", "force_ns", "force_iopaint"
    } or region_class == "sfx"


_CLEANUP_FORCE_ALLOW_MODES = {"force_allow", "force_solid", "force_telea", "force_ns", "force_iopaint"}


def _cleanup_override_force_allows_destructive(block: Any) -> bool:
    override = getattr(block, "override", None)
    mode = str(getattr(override, "cleanup_override_mode", "") or "").strip().lower()
    return mode in _CLEANUP_FORCE_ALLOW_MODES


def _destructive_protected_region_type(block: Any, region_class: Optional[str] = None) -> str:
    region_class = str(region_class or "").strip().lower()
    role = str(getattr(block, "bubble_role", None) or "").strip().lower()
    yolo_kind = str(getattr(block, "yolo_kind", None) or "").strip().lower()
    kind_name = getattr(getattr(block, "region_kind", None), "name", "")
    yolo_class_id = getattr(block, "yolo_class_id", None)
    try:
        yolo_class_id = int(yolo_class_id) if yolo_class_id is not None else None
    except Exception:
        yolo_class_id = None
    if (
        region_class == "sfx"
        or role in {"sfx", "sound", "sound_effect", "impact", "shout"}
        or yolo_kind in {"sfx", "sound", "sound_effect", "impact", "shout"}
        or kind_name == "SFX_OVER_ART"
        or yolo_class_id in {2, 3}
    ):
        return "sfx"
    if (
        region_class in {"text_on_art", "dialogue_over_art", "text_over_art"}
        or role in {"text_on_art", "dialogue_over_art", "text_over_art"}
        or yolo_kind in {"text_on_art", "dialogue_over_art", "text_over_art"}
        or kind_name == "DIALOGUE_OVER_ART"
    ):
        return "text_over_art"
    return ""


def _can_destructively_clean_region(
    block: Any,
    region_class: Optional[str],
    config: Any,
    override: Optional[Any] = None,
    operation: str = "cleanup",
) -> Tuple[bool, str]:
    protected = _destructive_protected_region_type(block, region_class)
    if not protected:
        return True, "allowed"
    if protected == "sfx":
        if not _config_bool(getattr(config, "cleanup_allow_sfx_cleanup", False)):
            return False, f"{operation}:protected_sfx_cleanup_disabled"
    elif protected == "text_over_art":
        if not _config_bool(getattr(config, "cleanup_allow_text_over_art", False)):
            return False, f"{operation}:protected_text_over_art_cleanup_disabled"
    if override is not None:
        block_override = getattr(block, "override", None)
        try:
            block.override = override
            force_allowed = _cleanup_override_force_allows_destructive(block)
        finally:
            block.override = block_override
    else:
        force_allowed = _cleanup_override_force_allows_destructive(block)
    if not force_allowed:
        return False, f"{operation}:protected_{protected}_requires_region_force"
    return True, f"{operation}:protected_{protected}_explicitly_allowed"


def _is_visual_sfx_like(block: Any) -> bool:
    """Return True for regions that should never be auto-typesetted over raw art.

    Checks bubble_role, yolo_kind, and region_kind.name for SFX-like values.
    Does NOT inspect bbox, manually_adjusted, or any geometry.
    """
    role = str(getattr(block, "bubble_role", None) or "").lower()
    yolo_kind = str(getattr(block, "yolo_kind", None) or "").lower()
    kind_name = getattr(getattr(block, "region_kind", None), "name", "")
    return (
        role in _SFX_LIKE_ROLES
        or yolo_kind in _SFX_LIKE_ROLES
        or kind_name in _SFX_LIKE_KINDS
    )


def _has_explicit_typeset_override(block: Any) -> bool:
    """Return True only when the user has explicitly overridden typeset styling.

    Deliberately excluded (must NOT count as override):
        • block.manually_adjusted  — set by update_region_bbox (bbox move only)
        • bbox_override            — detector or manual bbox geometry, not style
        • safe_rect / cleanup_container_bbox / computed geometry — auto values
        • automatic role / font / style derived from OCR output

    Counts as override:
        • typeset_override explicitly set by a manual text/translation edit
        • RegionOverride object that is not empty
    """
    override = getattr(block, "override", None)
    override_is_empty = True
    if override is not None:
        try:
            override_is_empty = bool(override.is_empty())
        except Exception:
            override_is_empty = False
    return (
        bool(getattr(block, "typeset_override", False))
        or not override_is_empty
    )


def _has_manual_typeset_override(block: Any) -> bool:
    """Backward-compatible alias for _has_explicit_typeset_override.

    NOTE: unlike the old implementation, this no longer treats
    block.manually_adjusted alone as an override.  Bbox moves set
    manually_adjusted but must not bypass the visual-SFX skip gate.
    """
    return _has_explicit_typeset_override(block)


def _should_skip_auto_typeset_for_cleanup(block: Any) -> bool:
    if _has_explicit_typeset_override(block):
        return False
    if _is_visual_sfx_like(block):
        debug_print(
            f"[TYPESET_SKIP] role={getattr(block, 'bubble_role', None)!r} "
            f"yolo_kind={getattr(block, 'yolo_kind', None)!r} "
            f"kind={getattr(getattr(block, 'region_kind', None), 'name', '')!r} "
            f"reason=visual_sfx_no_explicit_override"
        )
        return True
    # cleanup_tier==3 no longer silently suppresses typeset.
    # Tier-3 blocks (border-collision skip, busy-art, etc.) still get typeset
    # rendered; the cleanup simply did not erase the source art beneath them.
    return False


def _typeset_cleanup_skip_reason(block: Any) -> str:
    if _is_visual_sfx_like(block):
        return "typeset_skipped_sfx_like"
    return "typeset_skipped_cleanup_gate"


def _empty_preview_response(reason: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {"b64": None, "x": 0, "y": 0, "w": 0, "h": 0}
    if reason:
        out.update({"skipped": True, "reason": reason})
    return out


# ────────────────────────────────────────────────────────────────────────────
class LocalizerEngine:
    """
    The single source of truth for all pipeline state.

    Public interface (called by api.py):

    Pipeline
        import_chapter(folder)      → bootstrap dict
        go_to_page(idx)             → bootstrap dict
        detect_current_page()       → bootstrap dict
        ocr_current_page()          → bootstrap dict
        translate_current_page()    → bootstrap dict
        cleanup_current_page()      → bootstrap dict
        typeset_current_page()      → bootstrap dict
        run_all_steps()             → bootstrap dict
        export_chapter(export_dir)  → export_dir str
        get_page_image_b64(idx)     → base64 PNG string
        get_bootstrap()             → bootstrap dict
        update_region_field(idx, field, value) → bootstrap dict

    Phase 5 — TM approval
        approve_tm_entry(entry_id)
        reject_tm_entry(entry_id, reason)
        mark_tm_reviewed(entry_id)
        promote_tm_entry(entry_id, scope, as_name, kr_override, en_override)
        edit_tm_entry(entry_id, new_en)

    Phase 6 — Blocked mappings
        add_blocked_mapping(source_kr, blocked_en, reason, global_scope)
        remove_blocked_mapping(entry_id, global_scope)

    Memory introspection
        get_memory_stats()    → dict
        get_pending_review()  → list[dict]

    Legacy migration (opt-in, one series only)
        migrate_legacy_series(series_title) → dict
    """

    def __init__(self,
                 on_progress: Optional[Callable[..., None]] = None) -> None:
        """
        on_progress(message, current, total) is called whenever a step makes
        progress.  The api layer turns this into a JS event push.
        """
        self._on_progress = on_progress or (lambda m, c, t, payload=None: None)

        self.model_config  = ModelConfig.load()
        self.chapter_mgr   = ChapterManager()
        self.series_db     = SeriesDB()
        self.client        = OllamaClient()

        self._ocr_proc: Optional[OCRProcessor] = None
        self._paddle_ocr: Any = None
        self._paddle_ocr_lang: str = ""
        self._ocr_cache: Dict[str, Dict[str, Any]] = {}
        self._nllb: Any = None
        self.font_lib: Optional[ComicFontLibrary] = None
        self._yolo_detector: Any = None
        self._yolo_model_path: str = ""
        self._yolo_training_proc: Optional[subprocess.Popen] = None
        raw_yolo_path = getattr(self.model_config, "yolo_model_path", "") or ""
        resolved_yolo_path = self._resolve_yolo_model_path(raw_yolo_path)
        debug_print(
            "yolo_model_path_raw="
            f"{raw_yolo_path!r} yolo_model_path_resolved={resolved_yolo_path!r} "
            f"exists={os.path.isfile(resolved_yolo_path)}"
        )

        # Per-page working state  (always mirrors chapter_mgr.current_page)
        self._raw_cv: Optional[np.ndarray] = None
        self._regions: List[Any] = []          # List[OCRBlock]
        self._translations: List[str] = []
        self._undo_stack: List[Dict[str, Any]] = []
        self._region_mutation_version: int = 0

        self.busy   = False
        self.status = "Ready"
        self._progress_ctx: Dict[str, Any] = {
            "running": False,
            "job": "",
            "stage": "",
            "page_idx": 0,
            "page_total": 0,
            "region_idx": None,
            "region_total": None,
            "updated_pages": [],
        }

        self._lock = threading.Lock()

        # ── Active operation depth guard (PATCH B) ──────────────────────────────
        # Depth-based so nested calls never accidentally clear the flag.
        self._active_op_depth: int = 0
        self._active_op_names: List[str] = []
        # Per-page restore guard: prevents duplicate startup_restore runs
        self._restoring_pages: set = set()

        # ── Memory stores (None until _init_memory is called on import_chapter)
        self._memory_root:     str = _MEMORY_ROOT
        self._series_title:    str = ""
        self._display_series_title: str = ""
        self._chapter_id:      str = ""
        self._memory_aliases:  List[str] = []

        self._global_glossary: Any = None   # GlossaryStore | None
        self._global_names:    Any = None   # NameMemory    | None
        self._global_blocked:  Any = None   # BlockedMappingStore | None
        self._glossary:        Any = None   # GlossaryStore | None
        self._name_mem:        Any = None   # NameMemory    | None
        self._blocked:         Any = None   # BlockedMappingStore | None
        self._chapter_tm:      Any = None   # ChapterTM     | None
        self._alias_glossaries: List[Any] = []
        self._alias_names:      List[Any] = []
        self._alias_blocked:    List[Any] = []

        # Consistency warnings from the most-recent translate_current_page() call.
        # Cleared at the start of each call and appended to issues in get_bootstrap().
        self._consistency_warnings: List[Dict[str, Any]] = []
        self._memory_hits: Dict[int, List[Dict[str, Any]]] = {}
        self._last_batch_ctx: Any = None

        # Kick model initialisation off the main thread
        threading.Thread(target=self._init_models, daemon=True).start()

    # ── Model init ──────────────────────────────────────────────────────────

    def _init_models(self) -> None:
        detector_backend = str(getattr(self.model_config, "detector_backend", "ocr") or "ocr").lower()
        ocr_backend = str(getattr(self.model_config, "ocr_backend", "cascade") or "cascade").strip().lower()
        easyocr_fallback = str(
            getattr(self.model_config, "easyocr_fallback_enabled", False)
        ).strip().lower() in {"1", "true", "yes", "on"}
        needs_easyocr = detector_backend == "ocr" or ocr_backend == "easyocr" or easyocr_fallback
        if needs_easyocr:
            self._notify("Loading EasyOCR compatibility OCR...", 0, 2)
            try:
                self._ocr_proc = OCRProcessor()
                self._notify("EasyOCR compatibility OCR ready", 1, 2)
            except Exception as exc:
                self._notify(f"EasyOCR compatibility OCR failed: {exc}", 1, 2)
        else:
            self._ocr_proc = None
            debug_print(
                "EasyOCR disabled by config; using configured OCR backend "
                f"ocr_backend={getattr(self.model_config, 'ocr_backend', '')!r}"
            )
            self._notify(f"OCR backend {ocr_backend}; EasyOCR disabled", 1, 2)

        if _HAS_CTRANSLATE and NLLBTranslator and NLLB_MODEL_DIR:
            try:
                self._nllb = NLLBTranslator(NLLB_MODEL_DIR)
                self._notify("NLLB ready", 2, 2)
            except Exception:
                self._nllb = None

        try:
            self.font_lib = ComicFontLibrary(COMIC_FONTS_DIR)
        except Exception:
            self.font_lib = ComicFontLibrary("")

        self._notify("Ready", 2, 2)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _set_progress(self, **updates: Any) -> None:
        self._progress_ctx.update(updates)

    def _begin_active_operation(self, name: str) -> None:
        self._active_op_depth += 1
        self._active_op_names.append(name)

    def _end_active_operation(self, name: str) -> None:
        self._active_op_depth = max(0, self._active_op_depth - 1)
        try:
            self._active_op_names.remove(name)
        except ValueError:
            pass

    def _is_active_operation(self) -> bool:
        return self._active_op_depth > 0

    def _bump_region_mutation_version(self) -> None:
        self._region_mutation_version = int(getattr(self, "_region_mutation_version", 0) or 0) + 1

    def _notify(self, message: str, current: int = 0, total: int = 0, **extra: Any) -> None:
        self.status = message
        payload = {
            **self._progress_ctx,
            **extra,
            "message": message,
            "current": current,
            "total": total,
        }
        page_total = int(payload.get("page_total") or self.chapter_mgr.total_pages() or 0)
        if not payload.get("page_total"):
            payload["page_total"] = page_total
        if total:
            payload["percent"] = max(0.0, min(100.0, (float(current) / float(total)) * 100.0))
        elif page_total:
            payload["percent"] = max(0.0, min(100.0, ((int(payload.get("page_idx") or 0) + 1) / page_total) * 100.0))
        else:
            payload["percent"] = 0.0
        try:
            self._on_progress(message, current, total, payload)
        except TypeError:
            self._on_progress(message, current, total)

    def _debug_page_state(self) -> Dict[str, Any]:
        page = self.chapter_mgr.current_page
        return {
            "page": int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
            "dirty": bool(getattr(page, "render_dirty", False)) if page is not None else False,
            "has_cleaned": bool(getattr(page, "cleaned_cv", None) is not None) if page is not None else False,
            "has_typeset": bool(getattr(page, "typeset_pil", None) is not None) if page is not None else False,
            "render_version": int(getattr(page, "render_version", 0) or 0) if page is not None else 0,
        }

    def _debug_region_style(self, block: Any, bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        style = block.effective_style() if hasattr(block, "effective_style") else None
        return {
            "bbox": bbox or (block.bbox() if hasattr(block, "bbox") else None),
            "font": getattr(block, "font_name", "") or getattr(block, "bubble_role", "auto") or "auto",
            "size": int(getattr(block, "font_size", 0) or 0),
            "fg": _hex_color(getattr(style, "fg_color", None) if style else getattr(block, "fg_color", None), "#111111"),
            "bg": _hex_color(getattr(block, "bg_color", None), "#ffffff"),
            "outline": _hex_color(getattr(style, "outline_color", None) if style else getattr(block, "outline_color", None), "#ffffff"),
            "outline_width": int((getattr(style, "outline_width", None) if style else None) or getattr(block, "outline_width", 1) or 1),
            "shadow": _hex_color(getattr(style, "shadow_color", None), "#000000"),
            "shadow_on": bool(getattr(style, "shadow_on", False)),
            "align": getattr(block, "align", "center") or "center",
            "detector_source": getattr(block, "detector_source", "ocr") or "ocr",
        }

    def _load_page_into_working_state(self) -> None:
        """Sync self._raw_cv/_regions/_translations from chapter_mgr.current_page."""
        page = self.chapter_mgr.current_page
        if page is None:
            self._raw_cv       = None
            self._regions      = []
            self._translations = []
            return
        try:
            self._raw_cv = cv2.imread(page.image_path)
        except Exception:
            self._raw_cv = None
        self._regions      = page.regions
        self._translations = page.translations

    def _flush_working_state_to_page(self) -> None:
        """Write working state back to the current ChapterPage."""
        page = self.chapter_mgr.current_page
        if page is None:
            return
        page.regions      = self._regions
        page.translations = self._translations

    def _current_page_detected(self) -> bool:
        page = self.chapter_mgr.current_page
        return bool(page is not None and getattr(page, "detected", False))

    def _edge_candidate_reason(
        self,
        block: Any,
        page_idx: int,
        edge: str,
        image_shape: Tuple[int, ...],
        threshold: int = 48,
    ) -> str:
        ih, iw = image_shape[:2]
        reasons: List[str] = []

        def _near(label: str, bbox: Optional[Tuple[int, int, int, int]]) -> None:
            if not bbox:
                return
            x, y, w, h = [int(v) for v in bbox]
            if w <= 0 or h <= 0:
                return
            if edge == "top" and y <= threshold:
                reasons.append(f"{label}_near_top({y}px)")
            elif edge == "bottom" and y + h >= ih - threshold:
                reasons.append(f"{label}_near_bottom({ih - (y + h)}px)")

        _near("region_bbox", block.bbox() if hasattr(block, "bbox") else None)
        _near("ocr_text_bbox", getattr(block, "detector_text_bbox", None))
        _near("container_bbox", getattr(block, "bubble_bbox", None))
        _near("cleanup_container_bbox", getattr(block, "cleanup_container_bbox", None))

        for label in ("text_mask", "safe_text_mask", "bubble_mask"):
            mask = getattr(block, label, None)
            if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
                continue
            try:
                strip = mask[:threshold, :] if edge == "top" else mask[max(0, mask.shape[0] - threshold):, :]
                if np.count_nonzero(strip) > 0:
                    reasons.append(f"{label}_touches_{edge}")
            except Exception:
                pass

        raw = None
        if page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0):
            raw = self._raw_cv
        else:
            pages = getattr(self.chapter_mgr, "pages", []) or []
            if 0 <= page_idx < len(pages):
                raw = cv2.imread(pages[page_idx].image_path)
        if reasons and raw is not None and raw.size > 0:
            try:
                strip = raw[:threshold, :] if edge == "top" else raw[max(0, raw.shape[0] - threshold):, :]
                gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
                dark_ratio = float(np.count_nonzero(gray < 96)) / max(1, int(gray.size))
                if dark_ratio >= 0.015:
                    reasons.append(f"visual_dark_pixels_{edge}({dark_ratio:.3f})")
            except Exception:
                pass

        return ",".join(reasons)

    def _log_cross_page_candidates(
        self,
        page_idx: int,
        regions: Optional[List[Any]] = None,
        raw_cv: Optional[np.ndarray] = None,
    ) -> Dict[str, List[Tuple[int, Any, str]]]:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        if regions is None:
            if page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0):
                regions = self._regions
            elif 0 <= page_idx < len(pages):
                regions = getattr(pages[page_idx], "regions", []) or []
            else:
                regions = []
        if raw_cv is None:
            if page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0):
                raw_cv = self._raw_cv
            elif 0 <= page_idx < len(pages):
                raw_cv = cv2.imread(pages[page_idx].image_path)
        if raw_cv is None:
            return {"top": [], "bottom": []}

        out: Dict[str, List[Tuple[int, Any, str]]] = {"top": [], "bottom": []}
        for idx, block in enumerate(regions or []):
            for edge in ("top", "bottom"):
                reason = self._edge_candidate_reason(block, page_idx, edge, raw_cv.shape)
                if not reason:
                    continue
                rid = f"r{idx + 1}"
                debug_print(f"[CROSS_PAGE] candidate page={page_idx} edge={edge} region={rid} reason={reason}")
                out[edge].append((idx, block, reason))
        return out

    def _bbox_x_overlap_ratio(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, _, aw, _ = a
        bx1, _, bw, _ = b
        ax2 = ax1 + aw
        bx2 = bx1 + bw
        overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        return float(overlap) / max(1.0, float(min(aw, bw)))

    def _cross_page_roles_compatible(self, a: Any, b: Any) -> bool:
        if _is_pipeline_sfx(a) or _is_pipeline_sfx(b):
            return _config_bool(getattr(self.model_config, "cross_page_merge_sfx", False))
        roles = {
            str(getattr(a, "bubble_role", "") or "").lower(),
            str(getattr(b, "bubble_role", "") or "").lower(),
        }
        return not roles.isdisjoint({"", "dialog", "caption", "narration", "thought", "bold", "shout"})

    def _build_stitched_context(self, page_indices: List[int]) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        loaded: List[Tuple[int, np.ndarray]] = []
        for page_idx in sorted(set(page_indices)):
            if not (0 <= page_idx < len(pages)):
                continue
            img = self._raw_cv if page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0) else cv2.imread(pages[page_idx].image_path)
            if img is not None and img.size > 0:
                loaded.append((page_idx, img))
        if not loaded:
            return None, {}
        width = max(int(img.shape[1]) for _, img in loaded)
        height = sum(int(img.shape[0]) for _, img in loaded)
        composite = np.full((height, width, 3), 255, dtype=np.uint8)
        offsets: Dict[int, int] = {}
        y = 0
        for page_idx, img in loaded:
            offsets[page_idx] = y
            h, w = img.shape[:2]
            composite[y:y + h, 0:w] = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            y += h
        return composite, offsets

    def _page_render_base_cv(self, page_idx: int, current_override: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        if not (0 <= page_idx < len(pages)):
            return None
        if current_override is not None and page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0):
            return current_override.copy()
        page = pages[page_idx]
        if getattr(page, "typeset_pil", None) is not None and not bool(getattr(page, "render_dirty", False)):
            try:
                return cv2.cvtColor(np.array(page.typeset_pil), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        if getattr(page, "cleaned_cv", None) is not None:
            return page.cleaned_cv.copy()
        raw = self._raw_cv if page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0) else cv2.imread(page.image_path)
        return raw.copy() if raw is not None else None

    def _build_stitched_render_base(
        self,
        page_indices: List[int],
        current_override: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
        loaded: List[Tuple[int, np.ndarray]] = []
        for page_idx in sorted(set(page_indices)):
            img = self._page_render_base_cv(page_idx, current_override=current_override)
            if img is not None and img.size > 0:
                loaded.append((page_idx, img))
        if not loaded:
            return None, {}
        width = max(int(img.shape[1]) for _, img in loaded)
        height = sum(int(img.shape[0]) for _, img in loaded)
        composite = np.full((height, width, 3), 255, dtype=np.uint8)
        offsets: Dict[int, int] = {}
        y = 0
        for page_idx, img in loaded:
            offsets[page_idx] = y
            h, w = img.shape[:2]
            composite[y:y + h, 0:w] = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            y += h
        return composite, offsets

    def _cross_page_context_for_bbox(
        self,
        page_idx: int,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Dict[int, int], List[int], Tuple[int, int, int, int], Dict[int, Tuple[int, int, int, int]]]:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        if self._raw_cv is None or not (0 <= page_idx < len(pages)):
            return None, {}, [], bbox, {}
        x, y, w, h = [int(v) for v in bbox]
        page_h, page_w = self._raw_cv.shape[:2]
        needed = [page_idx]
        if y < 0 and page_idx > 0:
            needed.insert(0, page_idx - 1)
        if y + h > page_h and page_idx + 1 < len(pages):
            needed.append(page_idx + 1)
        composite, offsets = self._build_stitched_context(needed)
        if composite is None or page_idx not in offsets:
            return None, {}, [], bbox, {}
        comp_bbox = (x, y + offsets[page_idx], w, h)
        local: Dict[int, Tuple[int, int, int, int]] = {}
        for idx in needed:
            page_img = self._raw_cv if idx == page_idx else cv2.imread(pages[idx].image_path)
            if page_img is None:
                continue
            ph, pw = page_img.shape[:2]
            oy = offsets[idx]
            x1 = max(0, x)
            x2 = min(pw, x + w)
            cy1 = max(oy, comp_bbox[1])
            cy2 = min(oy + ph, comp_bbox[1] + h)
            if x2 > x1 and cy2 > cy1:
                local[idx] = (int(x1), int(cy1 - oy), int(x2 - x1), int(cy2 - cy1))
        return composite, offsets, needed, comp_bbox, local

    def _cross_page_context_for_block(
        self,
        page_idx: int,
        block: Any,
    ) -> Tuple[Optional[np.ndarray], Dict[int, int], List[int], Tuple[int, int, int, int], Dict[int, Tuple[int, int, int, int]]]:
        local_meta = getattr(block, "page_local_bboxes", {}) or {}
        pages_meta = [int(v) for v in (getattr(block, "cross_page_pages", []) or [])]
        if isinstance(local_meta, dict) and len(local_meta) >= 2:
            pages = sorted(set(pages_meta or [int(k) for k in local_meta.keys()]))
            composite, offsets = self._build_stitched_context(pages)
            if composite is None or page_idx not in offsets:
                return None, {}, [], tuple(int(v) for v in block.bbox()), {}
            comp_boxes: Dict[int, Tuple[int, int, int, int]] = {}
            for raw_key, raw_box in local_meta.items():
                try:
                    pidx = int(raw_key)
                    if pidx not in offsets:
                        continue
                    x, y, w, h = [int(v) for v in raw_box]
                    comp_boxes[pidx] = (x, y + int(offsets[pidx]), w, h)
                except Exception:
                    continue
            if len(comp_boxes) >= 2:
                x1 = min(x for x, _y, _w, _h in comp_boxes.values())
                y1 = min(y for _x, y, _w, _h in comp_boxes.values())
                x2 = max(x + w for x, _y, w, _h in comp_boxes.values())
                y2 = max(y + h for _x, y, _w, h in comp_boxes.values())
                pad = 24
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(int(composite.shape[1]), x2 + pad)
                y2 = min(int(composite.shape[0]), y2 + pad)
                local = {int(k): tuple(int(v) for v in val) for k, val in local_meta.items()}
                return composite, offsets, pages, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), local
        return self._cross_page_context_for_bbox(page_idx, tuple(int(v) for v in block.bbox()))

    def _is_cross_page_secondary(self, block: Any) -> bool:
        for attr in ("cleanup_meta", "typeset_meta"):
            meta = getattr(block, attr, {}) or {}
            if isinstance(meta, dict) and bool(meta.get("cross_page_secondary", False)):
                return True
        return False

    def _is_cross_page_bbox(self, page_idx: int, bbox: Tuple[int, int, int, int]) -> bool:
        if self._raw_cv is None:
            return False
        _x, y, _w, h = [int(v) for v in bbox]
        page_h = int(self._raw_cv.shape[0])
        return y < 0 or y + h > page_h

    def _update_cross_page_metadata(self, block: Any, page_idx: int, bbox: Tuple[int, int, int, int]) -> None:
        composite, _offsets, pages, comp_bbox, local = self._cross_page_context_for_bbox(page_idx, bbox)
        if composite is None or len(pages) <= 1:
            block.cross_page = False
            block.cross_page_group_id = None
            block.cross_page_pages = []
            block.composite_bbox = None
            block.page_local_bboxes = {}
            return
        block.cross_page = True
        if not getattr(block, "cross_page_group_id", None):
            block.cross_page_group_id = f"cp-manual-{page_idx}-{int(time.time() * 1000)}"
        block.cross_page_pages = [int(v) for v in pages]
        block.composite_bbox = tuple(int(v) for v in comp_bbox)
        block.page_local_bboxes = {int(k): tuple(int(v) for v in val) for k, val in local.items()}
        if not isinstance(getattr(block, "cleanup_meta", None), dict):
            block.cleanup_meta = {}
        if not isinstance(getattr(block, "typeset_meta", None), dict):
            block.typeset_meta = {}

    def _ocr_composite_crop(self, crop: np.ndarray, group_id: str) -> str:
        if crop.size == 0:
            return ""
        backend = self._normalise_ocr_backend()
        if backend in {"cascade", "paddleocr"}:
            cached = self._read_ocr_cache(crop, f"{backend}:cross_page", None)
            if cached is not None:
                debug_print(f"[CROSS_PAGE_OCR] group={group_id} backend={backend} cache_hit=True has_text={bool(cached.strip())}")
                return cached
            paddle = self._run_paddleocr_on_crop(crop)
            if paddle.get("ok"):
                text = normalize_ocr_korean(str(paddle.get("text", "") or ""))
                confidence = float(paddle.get("confidence", 0.0) or 0.0)
                if backend == "paddleocr" or (text.strip() and confidence >= float(getattr(self.model_config, "ocr_vlm_fallback_confidence", 0.70) or 0.70)):
                    self._write_ocr_cache(crop, f"{backend}:cross_page", None, text)
                    debug_print(f"[CROSS_PAGE_OCR] group={group_id} backend=paddleocr confidence={confidence:.3f} has_text={bool(text)}")
                    return text
                debug_print(f"[CROSS_PAGE_OCR] group={group_id} backend=paddleocr fallback_to_qwen confidence={confidence:.3f} has_text={bool(text)}")
            else:
                debug_print(f"[CROSS_PAGE_OCR] group={group_id} backend=paddleocr failed={paddle.get('error', '')}")
                if backend == "paddleocr":
                    return ""

        cached_qwen = self._read_ocr_cache(crop, "qwen_vl:cross_page", None)
        if cached_qwen is not None:
            debug_print(f"[CROSS_PAGE_OCR] group={group_id} backend=qwen_vl cache_hit=True has_text={bool(cached_qwen.strip())}")
            return cached_qwen
        _, buf = cv2.imencode(".png", crop)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        resp = self.client.chat_json(
            model=(
                getattr(self.model_config, "qwen_ocr_model", "")
                or getattr(self.model_config, "ocr_model", "")
                or getattr(self.model_config, "vision_model", "")
            ),
            prompt=(
                "You are OCR/localization for a stitched crop from adjacent manhwa pages. "
                "The crop may contain one text block split across a page seam. "
                "Return strict JSON only. Read the full source text in order. "
                "Do not output pixel coordinates, bounding boxes, polygons, masks, or placement geometry."
            ),
            schema=QWEN_OCR_SCHEMA,
            image_b64=b64,
            keep_alive=self.model_config.keep_alive,
        )
        raw_blocks = resp.get("text_blocks") if isinstance(resp, dict) else []
        if not isinstance(raw_blocks, list):
            raw_blocks = []
        items = []
        for raw_block in raw_blocks:
            if not isinstance(raw_block, dict):
                continue
            try:
                order = int(raw_block.get("reading_order", len(items) + 1))
            except Exception:
                order = len(items) + 1
            source_text = normalize_ocr_korean(str(raw_block.get("source_text") or "").strip())
            if source_text:
                items.append((order, source_text))
        items.sort(key=lambda item: item[0])
        text = normalize_ocr_korean(" ".join(item[1] for item in items))
        self._write_ocr_cache(crop, "qwen_vl:cross_page", None, text)
        debug_print(f"[CROSS_PAGE_OCR] group={group_id} backend=qwen_vl text_blocks={len(items)} has_text={bool(text)}")
        return text

    def _try_cross_page_ocr_groups(self, page_idx: int) -> None:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        if self._raw_cv is None or not (0 <= page_idx < len(pages)):
            return
        current = self._log_cross_page_candidates(page_idx, self._regions, self._raw_cv)
        if page_idx > 0 and current["top"]:
            prev_page = pages[page_idx - 1]
            prev_raw = cv2.imread(prev_page.image_path)
            if prev_raw is not None:
                prev = self._log_cross_page_candidates(page_idx - 1, getattr(prev_page, "regions", []) or [], prev_raw)
                for curr_idx, curr_block, _ in current["top"]:
                    curr_bbox = curr_block.bbox()
                    curr_center = curr_bbox[0] + curr_bbox[2] / 2.0
                    best_prev: Optional[Tuple[int, Any, Tuple[int, int, int, int], float]] = None
                    for prev_idx, prev_block, _ in prev["bottom"]:
                        prev_bbox = prev_block.bbox()
                        prev_center = prev_bbox[0] + prev_bbox[2] / 2.0
                        overlap = self._bbox_x_overlap_ratio(prev_bbox, curr_bbox)
                        center_close = abs(prev_center - curr_center) <= max(prev_bbox[2], curr_bbox[2], 80) * 0.65
                        if (overlap < 0.35 and not center_close) or not self._cross_page_roles_compatible(prev_block, curr_block):
                            continue
                        score = overlap + (0.5 if center_close else 0.0)
                        if best_prev is None or score > best_prev[3]:
                            best_prev = (prev_idx, prev_block, prev_bbox, score)
                    if best_prev is None:
                        continue
                    prev_idx, _prev_block, prev_bbox, _score = best_prev
                    group_id = f"cp-{page_idx - 1}-{prev_idx}-{page_idx}-{curr_idx}"
                    if str(getattr(curr_block, "text", "") or "").strip():
                        debug_print(
                            f"[CROSS_PAGE_MERGE_SKIP] pages={[page_idx - 1, page_idx]} "
                            f"regions={[f'r{prev_idx + 1}', f'r{curr_idx + 1}']} "
                            "reason=current_region_has_ocr_text"
                        )
                        curr_block.cross_page = False
                        curr_block.cross_page_group_id = None
                        curr_block.cross_page_pages = []
                        curr_block.composite_bbox = None
                        curr_block.page_local_bboxes = {}
                        for attr in ("cleanup_meta", "typeset_meta"):
                            meta = getattr(curr_block, attr, {}) or {}
                            if isinstance(meta, dict):
                                meta.pop("cross_page_secondary", None)
                                meta.pop("cross_page_cleanup_limited", None)
                                meta.pop("cross_page_typeset_limited", None)
                        continue
                    debug_print(
                        f"[CROSS_PAGE_MERGE] pages={[page_idx - 1, page_idx]} "
                        f"regions={[f'r{prev_idx + 1}', f'r{curr_idx + 1}']} secondary_only=True"
                    )
                    curr_block.cross_page = True
                    curr_block.cross_page_group_id = group_id
                    curr_block.cross_page_pages = [page_idx - 1, page_idx]
                    curr_block.page_local_bboxes = {
                        page_idx - 1: tuple(int(v) for v in prev_bbox),
                        page_idx: tuple(int(v) for v in curr_bbox),
                    }
                    if not isinstance(getattr(curr_block, "cleanup_meta", None), dict):
                        curr_block.cleanup_meta = {}
                    if not isinstance(getattr(curr_block, "typeset_meta", None), dict):
                        curr_block.typeset_meta = {}
                    curr_block.cleanup_meta["cross_page_secondary"] = True
                    curr_block.cleanup_meta["cross_page_cleanup_limited"] = True
                    curr_block.typeset_meta["cross_page_secondary"] = True
                    curr_block.typeset_meta["cross_page_typeset_limited"] = True
                    curr_block.text = ""
                    if curr_idx < len(self._translations):
                        self._translations[curr_idx] = ""
        if page_idx >= len(pages) - 1:
            return
        next_page = pages[page_idx + 1]
        next_raw = cv2.imread(next_page.image_path)
        if next_raw is None:
            return
        nxt = self._log_cross_page_candidates(page_idx + 1, getattr(next_page, "regions", []) or [], next_raw)
        if not current["bottom"] or not nxt["top"]:
            return

        for curr_idx, curr_block, _ in current["bottom"]:
            curr_bbox = curr_block.bbox()
            curr_center = curr_bbox[0] + curr_bbox[2] / 2.0
            best: Optional[Tuple[int, Any, Tuple[int, int, int, int], float]] = None
            for next_idx, next_block, _ in nxt["top"]:
                next_bbox = next_block.bbox()
                next_center = next_bbox[0] + next_bbox[2] / 2.0
                overlap = self._bbox_x_overlap_ratio(curr_bbox, next_bbox)
                center_close = abs(curr_center - next_center) <= max(curr_bbox[2], next_bbox[2], 80) * 0.65
                if (overlap < 0.35 and not center_close) or not self._cross_page_roles_compatible(curr_block, next_block):
                    continue
                score = overlap + (0.5 if center_close else 0.0)
                if best is None or score > best[3]:
                    best = (next_idx, next_block, next_bbox, score)
            if best is None:
                continue

            next_idx, _next_block, next_bbox, _score = best
            composite, offsets = self._build_stitched_context([page_idx, page_idx + 1])
            if composite is None or page_idx not in offsets or page_idx + 1 not in offsets:
                continue
            c_y = curr_bbox[1] + offsets[page_idx]
            n_y = next_bbox[1] + offsets[page_idx + 1]
            x1 = max(0, min(curr_bbox[0], next_bbox[0]) - 24)
            y1 = max(0, min(c_y, n_y) - 24)
            x2 = min(composite.shape[1], max(curr_bbox[0] + curr_bbox[2], next_bbox[0] + next_bbox[2]) + 24)
            y2 = min(composite.shape[0], max(c_y + curr_bbox[3], n_y + next_bbox[3]) + 24)
            if x2 <= x1 or y2 <= y1:
                continue
            group_id = f"cp-{page_idx}-{curr_idx}-{page_idx + 1}-{next_idx}"
            debug_print(f"[CROSS_PAGE_MERGE] pages={[page_idx, page_idx + 1]} regions={[f'r{curr_idx + 1}', f'r{next_idx + 1}']}")
            try:
                text = self._ocr_composite_crop(composite[y1:y2, x1:x2], group_id)
                if text:
                    curr_block.text = text
                    curr_block.ocr_confidence = max(float(getattr(curr_block, "ocr_confidence", 0.0) or 0.0), 0.5)
            except Exception as exc:
                debug_print(f"[CROSS_PAGE_OCR] group={group_id} failed={exc}")
            curr_block.cross_page = True
            curr_block.cross_page_group_id = group_id
            curr_block.cross_page_pages = [page_idx, page_idx + 1]
            curr_block.composite_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            curr_block.page_local_bboxes = {
                page_idx: tuple(int(v) for v in curr_bbox),
                page_idx + 1: tuple(int(v) for v in next_bbox),
            }
            curr_block.cleanup_meta["cross_page_cleanup_limited"] = True
            curr_block.typeset_meta["cross_page_typeset_limited"] = True
            if not isinstance(getattr(_next_block, "cleanup_meta", None), dict):
                _next_block.cleanup_meta = {}
            if not isinstance(getattr(_next_block, "typeset_meta", None), dict):
                _next_block.typeset_meta = {}
            _next_block.cross_page = True
            _next_block.cross_page_group_id = group_id
            _next_block.cross_page_pages = [page_idx, page_idx + 1]
            _next_block.composite_bbox = curr_block.composite_bbox
            _next_block.page_local_bboxes = dict(curr_block.page_local_bboxes)
            _next_block.cleanup_meta["cross_page_secondary"] = True
            _next_block.cleanup_meta["cross_page_cleanup_limited"] = True
            _next_block.typeset_meta["cross_page_secondary"] = True
            _next_block.typeset_meta["cross_page_typeset_limited"] = True
            _next_block.text = ""
            if next_idx < len(getattr(next_page, "translations", []) or []):
                next_page.translations[next_idx] = ""

    def _push_undo_snapshot(self) -> None:
        """Store a page-local editor snapshot before a mutating edit."""
        page = self.chapter_mgr.current_page
        if page is None:
            return
        self._flush_working_state_to_page()
        snapshot = {
            "page_idx": int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
            "regions": copy.deepcopy(page.regions),
            "translations": list(page.translations),
            "cleaned_cv": page.cleaned_cv.copy() if page.cleaned_cv is not None else None,
            "typeset_pil": page.typeset_pil.copy() if page.typeset_pil is not None else None,
            "cleanup_patches": copy.deepcopy(getattr(page, "cleanup_patches", []) or []),
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 25:
            self._undo_stack = self._undo_stack[-25:]

    def undo_last_edit(self) -> dict:
        if not self._undo_stack:
            self._notify("Nothing to undo.")
            return self.get_bootstrap()
        self._flush_working_state_to_page()
        snapshot = self._undo_stack.pop()
        page_idx = int(snapshot.get("page_idx", 0) or 0)
        self.chapter_mgr.go_to(page_idx)
        page = self.chapter_mgr.current_page
        if page is None:
            return self.get_bootstrap()
        page.regions = snapshot["regions"]
        page.translations = snapshot["translations"]
        page.cleaned_cv = snapshot["cleaned_cv"]
        page.typeset_pil = snapshot["typeset_pil"]
        page.cleanup_patches = snapshot.get("cleanup_patches", [])
        self._load_page_into_working_state()
        self.chapter_mgr.save_state()
        self._notify("Undo applied.")
        return self.get_bootstrap()

    def _region_id_for_cleanup_patch(self, region_idx: int) -> str:
        return f"r{int(region_idx) + 1}"

    def _cleanup_patch_summary(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v for k, v in patch.items()
            if k not in {"patch_png_b64", "mask_png_b64"}
        }

    def _json_safe(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, default=str, ensure_ascii=False))
        except Exception:
            return {}

    def _cleanup_patch_for_region(self, page: Any, region_idx: int) -> Optional[Dict[str, Any]]:
        region_id = self._region_id_for_cleanup_patch(region_idx)
        for patch in reversed(getattr(page, "cleanup_patches", []) or []):
            if patch.get("region_id") == region_id or int(patch.get("region_idx", -1) or -1) == int(region_idx):
                return patch
        return None

    def _encode_cv_png_b64(self, img_cv: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", img_cv)
        if not ok:
            raise RuntimeError("Could not encode cleanup preview.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _decode_cv_png_b64(self, b64: str) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(base64.b64decode(str(b64 or "")), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _decode_mask_png_b64(self, b64: str) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(base64.b64decode(str(b64 or "")), dtype=np.uint8)
            mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            return ((mask > 127).astype(np.uint8) * 255)
        except Exception:
            return None

    def _encode_mask_crop(self, mask: Optional[np.ndarray], fallback: Tuple[int, int, int, int]) -> Dict[str, Any]:
        bbox = self._mask_bbox(mask, fallback)
        x, y, w, h = bbox
        crop = (
            mask[y:y + h, x:x + w].copy()
            if mask is not None else np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
        )
        return {
            "b64": self._encode_cv_png_b64(crop),
            "bbox": [int(x), int(y), int(w), int(h)],
            "available": bool(mask is not None and np.any(mask)),
        }

    def _mask_crop_for_bbox(
        self,
        mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        if mask is None or not np.any(mask):
            return None
        x, y, w, h = [int(v) for v in bbox]
        ih, iw = image_shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        src_x1, src_y1 = x1 - x, y1 - y
        crop = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
        mask_crop = mask[y1:y2, x1:x2]
        crop[src_y1:src_y1 + mask_crop.shape[0], src_x1:src_x1 + mask_crop.shape[1]] = mask_crop
        return ((crop > 0).astype(np.uint8) * 255)

    def _attach_cleanup_patch_mask_crop(self, patch: Dict[str, Any], mask_crop: Optional[np.ndarray]) -> bool:
        if mask_crop is None or not np.any(mask_crop):
            return False
        patch["patch_version"] = 2
        patch["mask_png_b64"] = self._encode_cv_png_b64((mask_crop > 0).astype(np.uint8) * 255)
        patch["mask_bbox"] = [int(v) for v in (patch.get("bbox") or [])]
        return True

    def _attach_cleanup_patch_mask(
        self,
        patch: Dict[str, Any],
        mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...],
    ) -> bool:
        return self._attach_cleanup_patch_mask_crop(
            patch,
            self._mask_crop_for_bbox(mask, bbox, image_shape),
        )

    def _mask_bbox(self, mask: Optional[np.ndarray], fallback: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if mask is not None and np.any(mask):
            ys, xs = np.where(mask > 0)
            if xs.size and ys.size:
                pad = 4
                h, w = mask.shape[:2]
                x1 = max(0, int(xs.min()) - pad)
                y1 = max(0, int(ys.min()) - pad)
                x2 = min(w, int(xs.max()) + 1 + pad)
                y2 = min(h, int(ys.max()) + 1 + pad)
                return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        return self._changed_bbox(
            np.zeros((*mask.shape[:2], 3), dtype=np.uint8) if mask is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((*mask.shape[:2], 3), dtype=np.uint8) if mask is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            fallback,
        ) if mask is not None else fallback

    def _manual_mask_full(self, manual_mask: Optional[Dict[str, Any]], image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        if not isinstance(manual_mask, dict):
            return None
        b64 = str(manual_mask.get("b64", "") or "")
        bbox = manual_mask.get("bbox") or []
        mask_crop = self._decode_mask_png_b64(b64)
        if mask_crop is None or len(bbox) != 4:
            return None
        x, y, w, h = [int(v) for v in bbox]
        ih, iw = image_shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        full = np.zeros((ih, iw), dtype=np.uint8)
        full[y1:y2, x1:x2] = mask_crop[: y2 - y1, : x2 - x1]
        return full

    def _apply_manual_cleanup_mask(
        self,
        plan: Any,
        manual_mask: Optional[Dict[str, Any]],
        image_shape: Tuple[int, ...],
        block: Optional[Any] = None,
    ) -> bool:
        full = self._manual_mask_full(manual_mask, image_shape)
        if full is None:
            return False
        if block is not None:
            allowed, reason = _can_destructively_clean_region(
                block,
                getattr(plan, "region_class", ""),
                self.model_config,
                operation="manual_mask",
            )
            if not allowed:
                plan.debug_metrics["manual_mask_blocked_reason"] = reason
                return False
        if not np.any(full):
            plan.cleanup_mask = full
            plan.cleanup_strategy = "skip"
            plan.skip_reason = "manual_cleanup_mask_empty"
        else:
            plan.cleanup_mask = full
        plan.debug_metrics["manual_mask_used"] = True
        plan.debug_metrics["manual_mask_bbox"] = [int(v) for v in self._mask_bbox(full, (0, 0, image_shape[1], image_shape[0]))]
        plan.debug_metrics["manual_mask_px"] = int(np.count_nonzero(full))
        return True

    def _cleanup_group_backend(self, selected_plan: Any) -> str:
        backend = str(
            getattr(self.model_config, "cleanup_easy_fallback_backend", "")
            or getattr(self.model_config, "cleanup_fallback_backend", "")
            or "telea"
        ).strip().lower()
        if backend in {"ns", "opencv_ns"}:
            backend = "opencv_ns"
        if backend not in {"telea", "opencv_ns", "iopaint"}:
            backend = "telea"
        if (
            selected_plan.background_model == "halftone_texture"
            and _config_bool(getattr(self.model_config, "cleanup_prefer_iopaint_for_texture", False))
        ):
            backend = "iopaint"
        if (
            selected_plan.background_model == "translucent_gradient"
            and _config_bool(getattr(self.model_config, "cleanup_prefer_iopaint_for_translucent", False))
        ):
            backend = "iopaint"
        return backend

    def _cleanup_grouping_enabled(self) -> bool:
        return (
            _config_bool(getattr(self.model_config, "cleanup_allow_grouped_inpaint", False))
            or _config_bool(getattr(self.model_config, "cleanup_easy_fallback_enabled", False))
        )

    def _cleanup_force_skip(self, block: Any) -> bool:
        override = getattr(block, "override", None)
        mode = str(getattr(override, "cleanup_override_mode", "") or "").strip().lower()
        return mode == "skip"

    def _cleanup_force_fallback(self, block: Any) -> bool:
        override = getattr(block, "override", None)
        mode = str(getattr(override, "cleanup_override_mode", "") or "").strip().lower()
        return mode in {"force_telea", "force_ns", "force_iopaint"}

    def _cleanup_group_class_allowed(self, plan: Any, block: Any, selected_block: Any) -> bool:
        if self._cleanup_force_skip(block):
            return False
        allowed, reason = _can_destructively_clean_region(
            block,
            getattr(plan, "region_class", ""),
            self.model_config,
            operation="grouped",
        )
        if not allowed:
            debug_print(
                f"[CLEANUP_GROUP_SKIP] incompatible protected region "
                f"region={getattr(plan, 'region_id', '')} reason={reason}"
            )
            return False
        scope = str(getattr(self.model_config, "cleanup_easy_fallback_scope", "bubbles") or "bubbles").strip().lower()
        region_class = str(getattr(plan, "region_class", "") or "")
        if region_class in {"speech_bubble", "caption_box"}:
            return True
        if region_class == "sfx":
            return (
                scope in {"all", "sfx", "art"}
                and _config_bool(getattr(self.model_config, "cleanup_allow_sfx_cleanup", False))
                and (_cleanup_override_allows_pipeline_sfx(block) or _cleanup_override_allows_pipeline_sfx(selected_block))
            )
        if region_class == "text_on_art":
            return (
                scope in {"all", "art", "text_over_art"}
                and _config_bool(getattr(self.model_config, "cleanup_allow_text_over_art", False))
                and (self._cleanup_force_fallback(block) or self._cleanup_force_fallback(selected_block))
            )
        return scope == "all"

    def _cleanup_plan_bbox_for_group(self, plan: Any) -> Tuple[int, int, int, int]:
        bbox = getattr(plan, "container_bbox", None) or getattr(plan, "region_bbox", None) or (0, 0, 1, 1)
        return tuple(int(v) for v in bbox)

    def _bbox_iou_and_gap(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix = max(0, min(ax2, bx2) - max(ax, bx))
        iy = max(0, min(ay2, by2) - max(ay, by))
        inter = ix * iy
        union = max(1, aw * ah + bw * bh - inter)
        gap_x = max(0, max(ax, bx) - min(ax2, bx2))
        gap_y = max(0, max(ay, by) - min(ay2, by2))
        return float(inter) / float(union), float((gap_x * gap_x + gap_y * gap_y) ** 0.5)

    def _cleanup_group_compatible(self, selected_plan: Any, other_plan: Any) -> bool:
        a = self._cleanup_plan_bbox_for_group(selected_plan)
        b = self._cleanup_plan_bbox_for_group(other_plan)
        iou, gap = self._bbox_iou_and_gap(a, b)
        max_dim = max(a[2], a[3], b[2], b[3], 1)
        return iou >= 0.25 or gap <= max(18.0, max_dim * 0.18)

    def _should_use_grouped_fallback(self, selected_plan: Any, selected_block: Any, group_size: int) -> Tuple[bool, str]:
        if not self._cleanup_grouping_enabled():
            return False, "disabled"
        if self._cleanup_force_skip(selected_block):
            return False, "force_skip"
        if self._cleanup_force_fallback(selected_block):
            return True, "forced_fallback"
        bg_model = str(getattr(selected_plan, "background_model", "") or "")
        if bg_model == "halftone_texture" and not (
            _config_bool(getattr(self.model_config, "cleanup_allow_texture_telea", False))
            or _config_bool(getattr(self.model_config, "cleanup_allow_texture_inpaint", False))
        ):
            return False, "texture_fallback_disabled"
        if bg_model == "translucent_gradient" and not _config_bool(
            getattr(self.model_config, "cleanup_allow_translucent_caption", False)
        ):
            return False, "translucent_fallback_disabled"
        hard_bg = bg_model in {"halftone_texture", "translucent_gradient", "busy_art", "unknown"}
        risky_strategy = str(getattr(selected_plan, "cleanup_strategy", "") or "") in {"skip", "review", "texture_clone", "mask_inpaint"}
        deterministic = (
            getattr(selected_plan, "cleanup_strategy", "") == "flat_fill"
            and getattr(selected_plan, "inpaint_method", "") == "local_sample"
            and not hard_bg
        )
        if deterministic:
            return False, "deterministic_flat_fill"
        if group_size > 1:
            return True, "shared_container"
        if hard_bg or risky_strategy:
            return True, "hard_background"
        return False, "not_needed"

    def _iopaint_candidate_timeout(self) -> float:
        try:
            return max(0.25, float(getattr(self.model_config, "cleanup_iopaint_candidate_timeout_sec", 5) or 5))
        except (TypeError, ValueError):
            return 5.0

    def _execute_grouped_fallback(self, img_cv: np.ndarray, result: np.ndarray, mask: np.ndarray, backend: str, selected_plan: Any) -> Tuple[bool, str]:
        if backend == "opencv_ns":
            inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_NS)
            result[mask > 0] = inpainted[mask > 0]
            return True, ""
        if backend == "telea":
            inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
            result[mask > 0] = inpainted[mask > 0]
            return True, ""
        if backend == "iopaint":
            url = str(getattr(self.model_config, "iopaint_url", "") or "").strip()
            if not url:
                if _config_bool(getattr(self.model_config, "cleanup_iopaint_allow_opencv_fallback", False)):
                    inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
                    result[mask > 0] = inpainted[mask > 0]
                    return True, "iopaint_unavailable_fallback_telea"
                return False, "iopaint_unavailable:no_url"
            try:
                ok_img, img_buf = cv2.imencode(".png", img_cv)
                ok_mask, mask_buf = cv2.imencode(".png", mask)
                if not ok_img or not ok_mask:
                    raise RuntimeError("encode_failed")
                resp = requests.post(
                    url,
                    files={
                        "image": ("image.png", img_buf.tobytes(), "image/png"),
                        "mask": ("mask.png", mask_buf.tobytes(), "image/png"),
                    },
                    timeout=self._iopaint_candidate_timeout(),
                )
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[:2] != result.shape[:2]:
                    raise RuntimeError("invalid_output")
                result[mask > 0] = decoded[mask > 0]
                return True, ""
            except Exception as exc:
                if _config_bool(getattr(self.model_config, "cleanup_iopaint_allow_opencv_fallback", False)):
                    inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
                    result[mask > 0] = inpainted[mask > 0]
                    return True, f"iopaint_error_fallback_telea:{exc}"
                return False, f"iopaint_error:{exc}"
        return False, f"unknown_backend:{backend}"

    def _try_grouped_fallback(
        self,
        region_idx: int,
        selected_plan: Any,
        base: np.ndarray,
        result: np.ndarray,
        manual_mask_used: bool,
    ) -> Dict[str, Any]:
        selected_block = self._regions[region_idx]
        selected_allowed, selected_reason = _can_destructively_clean_region(
            selected_block,
            getattr(selected_plan, "region_class", ""),
            self.model_config,
            operation="grouped",
        )
        if not selected_allowed:
            return {"used": False, "reason": selected_reason, "plans": [selected_plan], "indices": [region_idx]}
        if not self._cleanup_grouping_enabled():
            return {"used": False, "reason": "disabled", "plans": [selected_plan], "indices": [region_idx]}
        plans: List[Any] = [selected_plan]
        indices: List[int] = [region_idx]
        for idx, block in enumerate(self._regions):
            if idx == region_idx:
                continue
            try:
                plan = build_cleanup_plan(
                    self._raw_cv,
                    block,
                    page_index=int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
                    region_id=f"R-{idx + 1:02d}",
                    cleanup_debug_artifacts=_config_bool(getattr(self.model_config, "cleanup_debug_artifacts", False)),
                    cleanup_debug_dir=getattr(self.model_config, "cleanup_debug_dir", ""),
                    auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
                    model_config=self.model_config,
                )
            except Exception as exc:
                debug_print(f"[CLEANUP_GROUP] plan_failed region=R-{idx + 1:02d} reason={exc}")
                continue
            if plan.cleanup_mask is None or not np.any(plan.cleanup_mask):
                continue
            if not self._cleanup_group_class_allowed(plan, block, selected_block):
                continue
            if not self._cleanup_group_compatible(selected_plan, plan):
                continue
            plans.append(plan)
            indices.append(idx)

        should_use, reason = self._should_use_grouped_fallback(selected_plan, selected_block, len(plans))
        if not should_use:
            return {"used": False, "reason": reason, "plans": [selected_plan], "indices": [region_idx]}

        combined = np.zeros(self._raw_cv.shape[:2], dtype=np.uint8)
        for plan in plans:
            if plan.cleanup_mask is not None:
                combined = cv2.bitwise_or(combined, plan.cleanup_mask)
        if not np.any(combined):
            return {"used": False, "reason": "empty_group_mask", "plans": [selected_plan], "indices": [region_idx]}

        backend = self._cleanup_group_backend(selected_plan)
        group_id = f"p{int(getattr(self.chapter_mgr, 'current_idx', 0) or 0)}-g{region_idx + 1}"
        ok, error = self._execute_grouped_fallback(self._raw_cv, result, combined, backend, selected_plan)
        region_ids = [f"r{i + 1}" for i in indices]
        debug_print(
            f"[CLEANUP_GROUP] group_id={group_id} region_ids={region_ids} "
            f"backend={backend} mask_px={int(np.count_nonzero(combined))}"
        )
        for plan in plans:
            plan.debug_metrics["grouped_inpaint"] = True
            plan.debug_metrics["group_id"] = group_id
            plan.debug_metrics["group_region_ids"] = region_ids
            plan.debug_metrics["group_backend"] = backend
            plan.debug_metrics["group_reason"] = reason
            plan.debug_metrics["fallback_used"] = bool(ok)
            if manual_mask_used and plan is selected_plan:
                plan.debug_metrics["manual_mask_used"] = True
            if error:
                plan.debug_metrics["fallback_error"] = error
            if ok:
                plan.cleanup_strategy = "mask_inpaint"
                plan.inpaint_method = backend
            else:
                plan.cleanup_strategy = "review"
                plan.skip_reason = error or "grouped_fallback_failed"
        return {
            "used": True,
            "ok": ok,
            "reason": reason,
            "backend": backend,
            "error": error,
            "plans": plans,
            "indices": indices,
            "combined_mask": combined,
            "group_id": group_id,
            "region_ids": region_ids,
        }

    def _rebuild_cleaned_from_cleanup_patches(
        self,
        page: ChapterPage,
        raw_cv: np.ndarray,
        exclude_region_idx: Optional[int] = None,
    ) -> np.ndarray:
        result = raw_cv.copy()
        exclude_id = (
            self._region_id_for_cleanup_patch(exclude_region_idx)
            if exclude_region_idx is not None else None
        )
        for patch in getattr(page, "cleanup_patches", []) or []:
            if exclude_id and (patch.get("region_id") == exclude_id or int(patch.get("region_idx", -1) or -1) == int(exclude_region_idx)):
                continue
            crop = self._decode_cv_png_b64(str(patch.get("patch_png_b64", "") or ""))
            bbox = patch.get("bbox") or []
            if crop is None or len(bbox) != 4:
                continue
            x, y, w, h = [int(v) for v in bbox]
            ih, iw = result.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            mask_crop = self._decode_mask_png_b64(str(patch.get("mask_png_b64", "") or ""))
            if mask_crop is None or not np.any(mask_crop):
                debug_print(
                    f"[CLEANUP_PATCH] skipped_unmasked_patch "
                    f"region_id={patch.get('region_id', '')} bbox={[x, y, w, h]}"
                )
                continue
            src_x1, src_y1 = x1 - x, y1 - y
            src_x2, src_y2 = src_x1 + (x2 - x1), src_y1 + (y2 - y1)
            crop_roi = crop[src_y1:src_y2, src_x1:src_x2]
            mask_roi = mask_crop[src_y1:src_y2, src_x1:src_x2]
            hh = min(crop_roi.shape[0], mask_roi.shape[0], y2 - y1)
            ww = min(crop_roi.shape[1], mask_roi.shape[1], x2 - x1)
            if hh <= 0 or ww <= 0:
                continue
            active = mask_roi[:hh, :ww] > 0
            if not np.any(active):
                continue
            target = result[y1:y1 + hh, x1:x1 + ww]
            target[active] = crop_roi[:hh, :ww][active]
        return result

    def _rebuild_page_cleaned(self, page_idx: int) -> None:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        if not (0 <= page_idx < len(pages)):
            return
        page = pages[page_idx]
        raw = self._raw_cv if page_idx == int(getattr(self.chapter_mgr, "current_idx", 0) or 0) else cv2.imread(page.image_path)
        if raw is None:
            return
        page.cleaned_cv = self._rebuild_cleaned_from_cleanup_patches(page, raw) if getattr(page, "cleanup_patches", None) else None
        page.typeset_pil = None
        page.render_dirty = True
        page.bump_render_version()

    def _append_cleanup_patch_to_page(self, page_idx: int, patch: Dict[str, Any]) -> None:
        pages = getattr(self.chapter_mgr, "pages", []) or []
        if not (0 <= page_idx < len(pages)):
            return
        page = pages[page_idx]
        patches = list(getattr(page, "cleanup_patches", []) or [])
        page.cleanup_patches = patches
        page.cleanup_patches.append(patch)

    def _remove_cleanup_patches_for_region(self, page_indices: List[int], region_idx: int) -> None:
        region_id = self._region_id_for_cleanup_patch(region_idx)
        pages = getattr(self.chapter_mgr, "pages", []) or []
        for page_idx in sorted(set(int(v) for v in page_indices)):
            if not (0 <= page_idx < len(pages)):
                continue
            page = pages[page_idx]
            page.cleanup_patches = [
                p for p in (getattr(page, "cleanup_patches", []) or [])
                if not (
                    p.get("region_id") == region_id
                    or int(p.get("region_idx", -1) or -1) == int(region_idx)
                    or p.get("cross_page_group_id") == region_id
                )
            ]

    def _changed_bbox(self, before: np.ndarray, after: np.ndarray, fallback: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        changed = np.any(before != after, axis=2) if before.ndim == 3 and after.ndim == 3 else before != after
        ys, xs = np.where(changed)
        if xs.size and ys.size:
            pad = 2
            h, w = after.shape[:2]
            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(w, int(xs.max()) + 1 + pad)
            y2 = min(h, int(ys.max()) + 1 + pad)
            return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        x, y, bw, bh = [int(v) for v in fallback]
        h, w = after.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def _run_selected_region_cleanup(
        self,
        region_idx: int,
        *,
        mutate_block: bool,
        manual_mask: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")

        block = self._regions[region_idx]
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        if bool(getattr(block, "cross_page", False)):
            bbox = tuple(int(v) for v in block.bbox())
            composite, offsets, pages, comp_bbox, local = self._cross_page_context_for_block(page_idx, block)
            if composite is None or page_idx not in offsets or len(pages) <= 1:
                self._update_cross_page_metadata(block, page_idx, bbox)
            else:
                comp_block = copy.deepcopy(block)
                cx, cy, cw, ch = comp_bbox
                comp_block.bbox_override = (cx, cy, cw, ch)
                comp_block.bubble_bbox = (cx, cy, cw, ch)
                comp_block.page_local_bboxes = local
                comp_block.composite_bbox = comp_bbox
                comp_block.cross_page_pages = pages
                comp_block.cross_page = True
                base = composite.copy()
                result = base.copy()
                plan = build_cleanup_plan(
                    composite,
                    comp_block,
                    page_index=page_idx,
                    region_id=f"R-{region_idx + 1:02d}",
                    cleanup_debug_artifacts=_config_bool(getattr(self.model_config, "cleanup_debug_artifacts", False)),
                    cleanup_debug_dir=getattr(self.model_config, "cleanup_debug_dir", ""),
                    auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
                    model_config=self.model_config,
                )
                allowed, block_reason = _can_destructively_clean_region(
                    block,
                    getattr(plan, "region_class", ""),
                    self.model_config,
                    operation="apply",
                )
                if not allowed:
                    plan.cleanup_mask = None
                    plan.cleanup_strategy = "skip"
                    plan.inpaint_method = "skip"
                    plan.skip_reason = block_reason
                elif _destructive_protected_region_type(block, getattr(plan, "region_class", "")):
                    if self._ensure_forced_cleanup_mask(plan):
                        plan.cleanup_strategy = "mask_inpaint"
                        plan.inpaint_method = "telea"
                        plan.cleanup_backend = "opencv"
                manual_mask_used = False
                if manual_mask and allowed:
                    manual_mask_used = self._apply_manual_cleanup_mask(plan, manual_mask, composite.shape, comp_block)
                if allowed:
                    execute_cleanup_plan(composite, result, plan)
                tier = None
                if mutate_block:
                    tier = _write_cleanup_metadata_to_block(
                        block,
                        plan,
                        composite,
                        cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
                    )
                    block.cross_page = True
                    block.cross_page_pages = pages
                    block.composite_bbox = comp_bbox
                    block.page_local_bboxes = local
                    block.cleanup_meta["cross_page_cleanup_split"] = True
                summary = summarize_cleanup_plan(plan)
                if mutate_block:
                    summary["cleanup_tier"] = int(tier or 0)
                    summary["cleanup_status"] = str(getattr(block, "cleanup_status", "") or "")
                    summary["cleanup_reason"] = str(getattr(block, "cleanup_reason", "") or "")
                changed_bbox = self._changed_bbox(base, result, comp_bbox)
                page_results: List[Dict[str, Any]] = []
                for pidx in pages:
                    if pidx not in offsets:
                        continue
                    pimg = self._raw_cv if pidx == page_idx else cv2.imread(self.chapter_mgr.pages[pidx].image_path)
                    if pimg is None:
                        continue
                    ph, pw = pimg.shape[:2]
                    oy = offsets[pidx]
                    x1 = max(0, changed_bbox[0])
                    x2 = min(pw, changed_bbox[0] + changed_bbox[2])
                    cy1 = max(oy, changed_bbox[1])
                    cy2 = min(oy + ph, changed_bbox[1] + changed_bbox[3])
                    if x2 <= x1 or cy2 <= cy1:
                        continue
                    crop = result[cy1:cy2, x1:x2].copy()
                    mask_crop = (
                        plan.cleanup_mask[cy1:cy2, x1:x2].copy()
                        if getattr(plan, "cleanup_mask", None) is not None else None
                    )
                    page_results.append({
                        "page_idx": int(pidx),
                        "bbox": (int(x1), int(cy1 - oy), int(x2 - x1), int(cy2 - cy1)),
                        "crop": crop,
                        "mask_crop": mask_crop,
                    })
                mask_bytes = b""
                if getattr(plan, "cleanup_mask", None) is not None:
                    try:
                        mask_bytes = np.ascontiguousarray(plan.cleanup_mask).tobytes()
                    except Exception:
                        mask_bytes = b""
                x, y, w, h = changed_bbox
                return {
                    "base": base,
                    "result": result,
                    "bbox": changed_bbox,
                    "crop": result[y:y + h, x:x + w].copy(),
                    "plan": plan,
                    "plans": [plan],
                    "group": {},
                    "group_region_indices": [region_idx],
                    "summary": summary,
                    "mask_hash": hashlib.sha1(mask_bytes).hexdigest()[:16] if mask_bytes else "",
                    "manual_mask_used": manual_mask_used,
                    "cross_page": True,
                    "cross_page_pages": pages,
                    "page_results": page_results,
                }
        base = self._rebuild_cleaned_from_cleanup_patches(page, self._raw_cv, exclude_region_idx=region_idx)
        result = base.copy()
        plan = build_cleanup_plan(
            self._raw_cv,
            block,
            page_index=page_idx,
            region_id=f"R-{region_idx + 1:02d}",
            cleanup_debug_artifacts=_config_bool(getattr(self.model_config, "cleanup_debug_artifacts", False)),
            cleanup_debug_dir=getattr(self.model_config, "cleanup_debug_dir", ""),
            auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
            model_config=self.model_config,
        )
        allowed, block_reason = _can_destructively_clean_region(
            block,
            getattr(plan, "region_class", ""),
            self.model_config,
            operation="apply",
        )
        if not allowed:
            plan.cleanup_mask = None
            plan.cleanup_strategy = "skip"
            plan.inpaint_method = "skip"
            plan.skip_reason = block_reason
        elif _destructive_protected_region_type(block, getattr(plan, "region_class", "")):
            if self._ensure_forced_cleanup_mask(plan):
                plan.cleanup_strategy = "mask_inpaint"
                plan.inpaint_method = "telea"
                plan.cleanup_backend = "opencv"
        manual_mask_used = self._apply_manual_cleanup_mask(plan, manual_mask, self._raw_cv.shape, block) if allowed else False
        group = self._try_grouped_fallback(region_idx, plan, base, result, manual_mask_used) if allowed else {"used": False, "reason": block_reason, "plans": [plan], "indices": [region_idx]}
        plans = list(group.get("plans") or [plan])
        indices = list(group.get("indices") or [region_idx])
        if allowed and not bool(group.get("used", False)):
            execute_cleanup_plan(self._raw_cv, result, plan)
        tier = None
        if mutate_block:
            tier = _write_cleanup_metadata_to_block(
                block,
                plan,
                self._raw_cv,
                cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
            )
        summary = summarize_cleanup_plan(plan)
        if mutate_block:
            summary["cleanup_tier"] = int(tier or 0)
            summary["cleanup_status"] = str(getattr(block, "cleanup_status", "") or "")
            summary["cleanup_reason"] = str(getattr(block, "cleanup_reason", "") or "")

        bbox = self._changed_bbox(base, result, block.bbox())
        x, y, w, h = bbox
        crop = result[y:y + h, x:x + w].copy()
        mask_bytes = b""
        if getattr(plan, "cleanup_mask", None) is not None:
            try:
                mask_bytes = np.ascontiguousarray(plan.cleanup_mask).tobytes()
            except Exception:
                mask_bytes = b""
        mask_hash = hashlib.sha1(mask_bytes).hexdigest()[:16] if mask_bytes else ""
        return {
            "base": base,
            "result": result,
            "bbox": bbox,
            "crop": crop,
            "plan": plan,
            "plans": plans,
            "group": group,
            "group_region_indices": indices,
            "summary": summary,
            "mask_hash": mask_hash,
            "manual_mask_used": manual_mask_used,
        }

    def preview_region_cleanup(self, region_idx: int, manual_mask: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        run = self._run_selected_region_cleanup(int(region_idx), mutate_block=False, manual_mask=manual_mask)
        plan = run["plan"]
        mask = getattr(plan, "cleanup_mask", None)
        mask_bbox = self._mask_bbox(mask, self._regions[int(region_idx)].bbox())
        mx, my, mw, mh = mask_bbox
        mask_crop = (
            mask[my:my + mh, mx:mx + mw].copy()
            if mask is not None else np.zeros((max(1, mh), max(1, mw)), dtype=np.uint8)
        )
        preview_crop = run["result"][my:my + mh, mx:mx + mw].copy()
        return {
            "ok": True,
            "b64": self._encode_cv_png_b64(preview_crop),
            "bbox": [int(mx), int(my), int(mw), int(mh)],
            "mask_b64": self._encode_cv_png_b64(mask_crop),
            "mask_bbox": [int(mx), int(my), int(mw), int(mh)],
            "manual_mask_used": bool(run.get("manual_mask_used", False)),
            "grouped_inpaint": bool((run.get("group") or {}).get("used", False)),
            "group_region_ids": list((run.get("group") or {}).get("region_ids", []) or []),
            "group_backend": str((run.get("group") or {}).get("backend", "") or ""),
            "group_reason": str((run.get("group") or {}).get("reason", "") or ""),
            "fallback_error": str((run.get("group") or {}).get("error", "") or ""),
            "plan": self._json_safe(run["summary"]),
            "debug": self._json_safe(dict(getattr(plan, "debug_metrics", {}) or {})),
        }

    def _candidate_review_required(self, plan: Any) -> bool:
        return bool(
            getattr(plan, "cleanup_strategy", "") in {"skip", "review"}
            or getattr(plan, "skip_reason", "")
            or (getattr(plan, "debug_metrics", {}) or {}).get("review_required_after_cleanup", False)
        )

    def _candidate_bg_sample(self, img_cv: np.ndarray, mask: Optional[np.ndarray], plan: Any) -> np.ndarray:
        if mask is None:
            return np.array([255.0, 255.0, 255.0], dtype=np.float32)
        container = None
        if getattr(plan, "container_mask", None) is not None and getattr(plan, "container_bbox", None) is not None:
            try:
                container = np.zeros(mask.shape[:2], dtype=np.uint8)
                x, y, w, h = [int(v) for v in plan.container_bbox]
                container[y:y + h, x:x + w] = plan.container_mask[:h, :w]
            except Exception:
                container = None
        sample = (container > 0) & (mask == 0) if container is not None else (mask == 0)
        if int(np.count_nonzero(sample)) < 16:
            dilated = cv2.dilate((mask > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=1)
            sample = (dilated > 0) & (mask == 0)
        if int(np.count_nonzero(sample)) < 16:
            return np.array([255.0, 255.0, 255.0], dtype=np.float32)
        return img_cv[sample].reshape(-1, 3).mean(axis=0).astype(np.float32)

    def _score_cleanup_candidate(self, before: np.ndarray, after: np.ndarray, plan: Any, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        mask = getattr(plan, "cleanup_mask", None)
        x, y, w, h = [int(v) for v in bbox]
        roi_before = before[y:y + h, x:x + w]
        roi_after = after[y:y + h, x:x + w]
        if mask is None:
            mask_crop = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
        else:
            mask_crop = mask[y:y + h, x:x + w]
        active = mask_crop > 0
        gray_after = cv2.cvtColor(roi_after, cv2.COLOR_BGR2GRAY) if roi_after.size else np.zeros((1, 1), dtype=np.uint8)
        gray_before = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY) if roi_before.size else np.zeros((1, 1), dtype=np.uint8)
        residual_dark = int(np.count_nonzero((gray_after < 92) & active)) if active.shape == gray_after.shape else 0
        residual_light = 0
        residual_text = residual_dark
        edges = cv2.Canny(gray_after, 40, 120)
        residual_edge = float(edges[active].mean()) if active.any() and active.shape == edges.shape else 0.0
        bg = self._candidate_bg_sample(before, mask, plan)
        if active.any() and roi_after.size:
            active_px = roi_after[active].reshape(-1, 3).astype(np.float32)
            color_dist = float(np.linalg.norm(active_px.mean(axis=0) - bg))
            bg_luma = float(0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2])
            active_luma = (
                active_px[:, 2] * 0.299
                + active_px[:, 1] * 0.587
                + active_px[:, 0] * 0.114
            )
            if bg_luma < 96.0:
                residual_light = int(np.count_nonzero(active_luma > bg_luma + 42.0))
                residual_text = residual_light
        else:
            color_dist = 0.0
        seam_score = 0.0
        texture_var_loss = 0.0
        if mask_crop.size:
            ring = cv2.dilate((mask_crop > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
            ring = (ring > 0) & (mask_crop == 0)
            seam_score = float(edges[ring].mean()) if np.any(ring) else 0.0
            if np.any(ring) and ring.shape == gray_before.shape:
                before_texture = float(cv2.Laplacian(gray_before, cv2.CV_32F)[ring].var())
                after_texture = float(cv2.Laplacian(gray_after, cv2.CV_32F)[ring].var())
                texture_var_loss = max(0.0, before_texture - after_texture)
        blur_loss = max(0.0, float(gray_before.var()) - float(gray_after.var()))
        mask_area = int(np.count_nonzero(mask)) if mask is not None else 0
        container_area = 0
        if getattr(plan, "container_mask", None) is not None:
            try:
                container_area = int(np.count_nonzero(plan.container_mask))
            except Exception:
                container_area = 0
        mask_container_ratio = float(mask_area) / float(max(1, container_area)) if container_area > 0 else 0.0
        total = (
            residual_text * 1.35
            + residual_edge * 0.9
            + color_dist * 1.15
            + seam_score * 0.85
            + blur_loss * 0.035
            + texture_var_loss * 0.002
            + mask_container_ratio * 55.0
        )
        return {
            "score": round(float(total), 3),
            "residual_dark_pixels": residual_dark,
            "residual_light_pixels": residual_light,
            "residual_text_pixels": residual_text,
            "residual_edge_energy": round(float(residual_edge), 3),
            "color_distance_to_sampled_bg": round(float(color_dist), 3),
            "seam_score": round(float(seam_score), 3),
            "blur_local_variance_loss": round(float(blur_loss), 3),
            "texture_variance_loss": round(float(texture_var_loss), 3),
            "mask_area": mask_area,
            "mask_container_ratio": round(float(mask_container_ratio), 4),
        }

    def _candidate_warnings(self, candidate_id: str, plan: Any, scores: Dict[str, Any], unavailable_reason: str = "") -> List[str]:
        warnings: List[str] = []
        if unavailable_reason:
            warnings.append(unavailable_reason)
        if getattr(plan, "skip_reason", ""):
            warnings.append(str(getattr(plan, "skip_reason", "")))
        fallback = str((getattr(plan, "debug_metrics", {}) or {}).get("cleanup_backend_fallback", "") or "")
        if "iopaint" in fallback or "lama" in fallback:
            warnings.append("IOPaint unavailable")
        if int(scores.get("residual_text_pixels", scores.get("residual_dark_pixels", 0)) or 0) > 12 or float(scores.get("residual_edge_energy", 0.0) or 0.0) > 38.0:
            warnings.append("Text remnants detected")
        bg_model = str(getattr(plan, "background_model", "") or "")
        if candidate_id == "solid_fill" and bg_model not in {"flat_light", "flat_colored", "dark_bubble"}:
            warnings.append("Unavailable: texture/gradient")
        if bg_model == "halftone_texture":
            if candidate_id in {"telea", "opencv_ns", "grouped"}:
                warnings.append("Review: texture blur risk")
            elif candidate_id == "default" and str(getattr(plan, "cleanup_backend", "") or "") not in {"iopaint", "lama"}:
                warnings.append("Review: texture fallback")
            elif candidate_id == "iopaint":
                warnings.append("Review: textured background")
        if candidate_id in {"telea", "opencv_ns", "grouped", "iopaint"} and (
            float(scores.get("blur_local_variance_loss", 0.0) or 0.0) > 120.0
            or float(scores.get("texture_variance_loss", 0.0) or 0.0) > 1200.0
        ):
            warnings.append("May blur texture")
        if float(scores.get("seam_score", 0.0) or 0.0) > 42.0:
            warnings.append("Visible seam risk")
        if float(scores.get("mask_container_ratio", 0.0) or 0.0) > 0.35:
            warnings.append("Large cleanup mask")
        return list(dict.fromkeys([w for w in warnings if w]))

    def _ensure_forced_cleanup_mask(self, plan: Any) -> bool:
        if getattr(plan, "cleanup_mask", None) is not None and np.any(plan.cleanup_mask):
            return True
        text_mask = getattr(plan, "text_mask", None)
        if text_mask is None or not np.any(text_mask):
            return False
        cleanup = text_mask.copy()
        for attr in ("outline_shadow_mask", "halo_mask"):
            extra = getattr(plan, attr, None)
            if extra is not None and np.any(extra):
                cleanup = cv2.bitwise_or(cleanup, extra)
        plan.cleanup_mask = cleanup
        plan.cleanup_mask_confidence = float(getattr(plan, "text_mask_confidence", 0.0) or 0.0)
        plan.debug_metrics["candidate_mask_from_text"] = True
        return bool(np.any(cleanup))

    def _make_forced_cleanup_plan(self, base_plan: Any, candidate_id: str) -> Tuple[Optional[Any], str]:
        plan = copy.deepcopy(base_plan)
        if candidate_id == "default":
            if getattr(plan, "cleanup_mask", None) is None or not np.any(plan.cleanup_mask):
                return None, "Not eligible: no cleanup mask"
            return plan, ""
        if candidate_id == "manual_mask":
            if getattr(plan, "cleanup_mask", None) is None or not np.any(plan.cleanup_mask):
                return None, "Not eligible: no cleanup mask"
            if not bool((getattr(plan, "debug_metrics", {}) or {}).get("manual_mask_used", False)):
                return None, "Not eligible: no edited manual mask"
            plan.cleanup_strategy = "mask_inpaint"
            plan.inpaint_method = "telea"
            return plan, ""
        if candidate_id == "solid_fill":
            if getattr(plan, "container_mask", None) is None or getattr(plan, "container_bbox", None) is None:
                return None, "Not eligible: no solid container"
            if getattr(plan, "region_class", "") not in {"speech_bubble", "caption_box"}:
                return None, "Not eligible: not a bubble/caption"
            background = str(getattr(plan, "background_model", "") or "")
            if background not in {"flat_light", "flat_colored", "dark_bubble"}:
                return None, f"Not eligible: background is {background or 'unknown'}"
            if getattr(plan, "cleanup_mask", None) is None or not np.any(plan.cleanup_mask):
                return None, "Not eligible: no cleanup mask"
            policy = CleanupPolicy.from_config(self.model_config)
            if not _try_force_solid_bubble_flat_fill(plan, self._raw_cv, policy):
                return None, "Not eligible: solid fill safety gates failed"
            return plan, ""
        if candidate_id in {"telea", "opencv_ns", "iopaint"}:
            self._ensure_forced_cleanup_mask(plan)
        if getattr(plan, "cleanup_mask", None) is None or not np.any(plan.cleanup_mask):
            return None, "Not eligible: no cleanup mask"
        if candidate_id == "telea":
            plan.cleanup_strategy = "mask_inpaint"
            plan.inpaint_method = "telea"
            plan.cleanup_backend = "opencv"
            if getattr(plan, "background_model", "") == "halftone_texture":
                plan.debug_metrics["review_required_after_cleanup"] = True
                plan.debug_metrics["texture_fallback_policy"] = "opencv_review_only"
            return plan, ""
        if candidate_id == "opencv_ns":
            plan.cleanup_strategy = "mask_inpaint"
            plan.inpaint_method = "ns"
            plan.cleanup_backend = "opencv"
            if getattr(plan, "background_model", "") == "halftone_texture":
                plan.debug_metrics["review_required_after_cleanup"] = True
                plan.debug_metrics["texture_fallback_policy"] = "opencv_review_only"
            return plan, ""
        if candidate_id == "iopaint":
            url = str(getattr(self.model_config, "iopaint_url", "") or "").strip()
            if not url:
                return None, "IOPaint not configured"
            plan.cleanup_strategy = "mask_inpaint"
            plan.inpaint_method = "telea"
            plan.cleanup_backend = "iopaint"
            plan.iopaint_url = url
            return plan, ""
        return None, f"Unknown candidate: {candidate_id}"

    def _execute_iopaint_candidate(self, img_cv: np.ndarray, result: np.ndarray, mask: np.ndarray, plan: Any) -> Tuple[bool, str]:
        url = str(getattr(plan, "iopaint_url", "") or getattr(self.model_config, "iopaint_url", "") or "").strip()
        if not url:
            return False, "IOPaint not configured"
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
                    "mask": ("mask.png", mask_buf.tobytes(), "image/png"),
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
            return False, "IOPaint timed out"
        except Exception:
            return False, "IOPaint unavailable"

    def _run_cleanup_candidate(
        self,
        region_idx: int,
        candidate_id: str,
        manual_mask: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        block = self._regions[region_idx]
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        base = self._rebuild_cleaned_from_cleanup_patches(page, self._raw_cv, exclude_region_idx=region_idx)
        result = base.copy()
        plan = build_cleanup_plan(
            self._raw_cv,
            block,
            page_index=page_idx,
            region_id=f"R-{region_idx + 1:02d}",
            cleanup_debug_artifacts=False,
            cleanup_debug_dir="",
            auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
            model_config=self.model_config,
        )
        allowed, block_reason = _can_destructively_clean_region(
            block,
            getattr(plan, "region_class", ""),
            self.model_config,
            operation=f"candidate:{candidate_id}",
        )
        if not allowed:
            return {
                "available": False,
                "unavailable_reason": block_reason,
                "base": base,
                "result": result,
                "plan": plan,
                "plans": [plan],
                "group": {"used": False, "plans": [plan], "indices": [region_idx], "reason": block_reason},
                "group_region_indices": [region_idx],
                "manual_mask_used": False,
            }
        manual_mask_used = self._apply_manual_cleanup_mask(plan, manual_mask, self._raw_cv.shape, block)
        group: Dict[str, Any] = {"used": False, "plans": [plan], "indices": [region_idx], "reason": "not_grouped"}
        if candidate_id == "grouped":
            group = self._try_grouped_fallback(region_idx, plan, base, result, manual_mask_used)
            if not bool(group.get("used", False)):
                return {"available": False, "unavailable_reason": f"Grouped inpaint unavailable: {group.get('reason', 'not applicable')}", "base": base, "result": result, "plan": plan, "plans": [plan], "group": group, "group_region_indices": [region_idx], "manual_mask_used": manual_mask_used}
            plans = list(group.get("plans") or [plan])
            indices = list(group.get("indices") or [region_idx])
        else:
            forced, unavailable = self._make_forced_cleanup_plan(plan, candidate_id)
            if forced is None:
                return {"available": False, "unavailable_reason": unavailable, "base": base, "result": result, "plan": plan, "plans": [plan], "group": group, "group_region_indices": [region_idx], "manual_mask_used": manual_mask_used}
            plan = forced
            if candidate_id == "iopaint":
                if _config_bool(getattr(self.model_config, "cleanup_skip_unavailable_iopaint_candidate", True)):
                    ok, unavailable = self._execute_iopaint_candidate(self._raw_cv, result, plan.cleanup_mask, plan)
                    if not ok:
                        plan.debug_metrics["cleanup_backend_fallback"] = unavailable
                        return {"available": False, "unavailable_reason": unavailable, "base": base, "result": result, "plan": plan, "plans": [plan], "group": group, "group_region_indices": [region_idx], "manual_mask_used": manual_mask_used}
                else:
                    execute_cleanup_plan(self._raw_cv, result, plan)
            else:
                execute_cleanup_plan(self._raw_cv, result, plan)
            plans = [plan]
            indices = [region_idx]
        bbox = self._changed_bbox(base, result, block.bbox())
        return {
            "available": True,
            "unavailable_reason": "",
            "base": base,
            "result": result,
            "bbox": bbox,
            "plan": plan,
            "plans": plans,
            "group": group,
            "group_region_indices": indices,
            "manual_mask_used": manual_mask_used,
        }

    def _cleanup_candidate_entry(self, region_idx: int, candidate_id: str, label: str, manual_mask: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        block = self._regions[region_idx]
        try:
            run = self._run_cleanup_candidate(region_idx, candidate_id, manual_mask)
            plan = run.get("plan")
        except Exception as exc:
            bbox = tuple(int(v) for v in block.bbox())
            x, y, w, h = bbox
            crop = np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
            return {
                "candidate_id": candidate_id,
                "label": label,
                "backend": "",
                "strategy": "",
                "method": "",
                "b64": self._encode_cv_png_b64(crop),
                "bbox": [int(x), int(y), int(w), int(h)],
                "scores": {
                    "score": 999999.0,
                    "residual_dark_pixels": 0,
                    "residual_edge_energy": 0.0,
                    "color_distance_to_sampled_bg": 0.0,
                    "seam_score": 0.0,
                    "blur_local_variance_loss": 0.0,
                    "mask_area": 0,
                    "mask_container_ratio": 0.0,
                },
                "warnings": [f"{label} failed"],
                "reasons": [str(exc)],
                "review_required": True,
                "is_available": False,
                "unavailable_reason": f"{label} failed",
                "manual_mask_used": False,
                "grouped_inpaint": False,
            }
        available = bool(run.get("available", False))
        bbox = tuple(int(v) for v in (run.get("bbox") or block.bbox()))
        x, y, w, h = bbox
        crop = run["result"][y:y + h, x:x + w].copy() if available else np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
        scores = self._score_cleanup_candidate(run["base"], run["result"], plan, bbox) if available else {
            "score": 999999.0,
            "residual_dark_pixels": 0,
            "residual_edge_energy": 0.0,
            "color_distance_to_sampled_bg": 0.0,
            "seam_score": 0.0,
            "blur_local_variance_loss": 0.0,
            "mask_area": 0,
            "mask_container_ratio": 0.0,
        }
        unavailable_reason = str(run.get("unavailable_reason", "") or "")
        warnings = self._candidate_warnings(candidate_id, plan, scores, unavailable_reason)
        return {
            "candidate_id": candidate_id,
            "label": label,
            "backend": str((run.get("group") or {}).get("backend") or getattr(plan, "cleanup_backend", "") or getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
            "strategy": str(getattr(plan, "cleanup_strategy", "") or ""),
            "method": str(getattr(plan, "inpaint_method", "") or ""),
            "b64": self._encode_cv_png_b64(crop),
            "bbox": [int(x), int(y), int(w), int(h)],
            "scores": scores,
            "warnings": warnings,
            "reasons": [
                str(getattr(plan, "skip_reason", "") or ""),
                str((getattr(plan, "debug_metrics", {}) or {}).get("selected_candidate", "") or ""),
                str((run.get("group") or {}).get("reason", "") or ""),
            ],
            "review_required": self._candidate_review_required(plan) or bool(warnings),
            "is_available": available,
            "unavailable_reason": unavailable_reason,
            "manual_mask_used": bool(run.get("manual_mask_used", False)),
            "grouped_inpaint": bool((run.get("group") or {}).get("used", False)),
        }

    def compare_region_cleanup_candidates(
        self,
        region_idx: int,
        manual_mask: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        region_idx = int(region_idx)
        start = time.monotonic()
        try:
            timeout = max(1.0, float(getattr(self.model_config, "cleanup_candidate_timeout_sec", 8) or 8))
        except (TypeError, ValueError):
            timeout = 8.0
        candidates = [
            ("default", "Current/default plan"),
            ("solid_fill", "Safe solid fill"),
            ("telea", "OpenCV TELEA"),
            ("opencv_ns", "OpenCV NS"),
            ("grouped", "Grouped inpaint"),
            ("iopaint", "IOPaint / LaMa"),
        ]
        if isinstance(manual_mask, dict):
            candidates.append(("manual_mask", "Manual-mask TELEA"))
        entries = []
        for cid, label in candidates:
            if time.monotonic() - start > timeout:
                block = self._regions[region_idx]
                x, y, w, h = [int(v) for v in block.bbox()]
                entries.append({
                    "candidate_id": cid,
                    "label": label,
                    "backend": "",
                    "strategy": "",
                    "method": "",
                    "b64": self._encode_cv_png_b64(np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)),
                    "bbox": [x, y, w, h],
                    "scores": {
                        "score": 999999.0,
                        "residual_dark_pixels": 0,
                        "residual_edge_energy": 0.0,
                        "color_distance_to_sampled_bg": 0.0,
                        "seam_score": 0.0,
                        "blur_local_variance_loss": 0.0,
                        "mask_area": 0,
                        "mask_container_ratio": 0.0,
                    },
                    "warnings": ["Candidate comparison timed out"],
                    "reasons": ["Candidate comparison timed out"],
                    "review_required": True,
                    "is_available": False,
                    "unavailable_reason": "Candidate comparison timed out",
                    "manual_mask_used": False,
                    "grouped_inpaint": False,
                })
                continue
            entries.append(self._cleanup_candidate_entry(region_idx, cid, label, manual_mask))
        available = [c for c in entries if c.get("is_available")]
        scored = sorted(
            available,
            key=lambda c: float((c.get("scores") or {}).get("score", 999999.0) or 999999.0),
        )
        best_id = ""
        if scored:
            best = scored[0]
            best_score = float((best.get("scores") or {}).get("score", 999999.0) or 999999.0)
            next_score = (
                float((scored[1].get("scores") or {}).get("score", 999999.0) or 999999.0)
                if len(scored) > 1
                else best_score + 25.0
            )
            if not best.get("warnings") and not best.get("review_required") and next_score - best_score >= 8.0:
                best_id = str(best.get("candidate_id") or "")
        return {"ok": True, "candidates": entries, "recommended_candidate_id": best_id}

    def apply_region_cleanup_candidate(
        self,
        region_idx: int,
        candidate_id: str,
        manual_mask: Optional[Dict[str, Any]] = None,
    ) -> dict:
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")
        self._push_undo_snapshot()
        run = self._run_cleanup_candidate(int(region_idx), str(candidate_id or "default"), manual_mask)
        if not bool(run.get("available", False)):
            raise RuntimeError(str(run.get("unavailable_reason") or "Cleanup candidate unavailable."))
        if bool(run.get("cross_page", False)):
            block = self._regions[int(region_idx)]
            pages_touched = [int(v) for v in (run.get("cross_page_pages") or [int(getattr(self.chapter_mgr, "current_idx", 0) or 0)])]
            self._remove_cleanup_patches_for_region(pages_touched, int(region_idx))
            _write_cleanup_metadata_to_block(
                block,
                run["plan"],
                run["base"],
                cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
            )
            created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            for page_result in run.get("page_results", []) or []:
                pidx = int(page_result["page_idx"])
                x, y, w, h = [int(v) for v in page_result["bbox"]]
                patch = {
                    "page_idx": pidx,
                    "region_id": self._region_id_for_cleanup_patch(int(region_idx)),
                    "region_idx": int(region_idx),
                    "bbox": [x, y, w, h],
                    "strategy": str(getattr(run["plan"], "cleanup_strategy", "") or ""),
                    "backend": str(getattr(run["plan"], "cleanup_backend", "") or getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
                    "inpaint_method": str(getattr(run["plan"], "inpaint_method", "") or ""),
                    "candidate_id": str(candidate_id or "default"),
                    "mask_hash": str(run.get("mask_hash", "") or ""),
                    "manual_mask_used": bool(run.get("manual_mask_used", False)),
                    "grouped_inpaint": False,
                    "cross_page": True,
                    "cross_page_group_id": str(getattr(block, "cross_page_group_id", "") or self._region_id_for_cleanup_patch(int(region_idx))),
                    "cross_page_pages": pages_touched,
                    "composite_bbox": [int(v) for v in getattr(block, "composite_bbox", [])] if getattr(block, "composite_bbox", None) else None,
                    "page_local_bboxes": {
                        str(int(k)): [int(vv) for vv in v]
                        for k, v in (getattr(block, "page_local_bboxes", {}) or {}).items()
                    },
                    "created_at": created_at,
                    "review_required": bool((getattr(block, "cleanup_meta", {}) or {}).get("review_required", False)),
                    "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                    "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                    "rerun": False,
                    "patch_png_b64": self._encode_cv_png_b64(page_result["crop"]),
                }
                if not self._attach_cleanup_patch_mask_crop(patch, page_result.get("mask_crop")):
                    continue
                self._append_cleanup_patch_to_page(pidx, patch)
            for pidx in pages_touched:
                self._rebuild_page_cleaned(pidx)
            self._flush_working_state_to_page()
            self.chapter_mgr.save_state()
            self._notify("Cross-page cleanup candidate applied.", 1, 1, updated_pages=pages_touched)
            return self.get_bootstrap()
        group_indices = [int(v) for v in (run.get("group_region_indices") or [region_idx])]
        group_plans = list(run.get("plans") or [run["plan"]])
        replace_ids = {self._region_id_for_cleanup_patch(i) for i in group_indices}
        page.cleanup_patches = [
            p for p in (getattr(page, "cleanup_patches", []) or [])
            if not (p.get("region_id") in replace_ids or int(p.get("region_idx", -1) or -1) in group_indices)
        ]
        created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        patches: List[Dict[str, Any]] = []
        for idx, patch_plan in zip(group_indices, group_plans):
            block = self._regions[idx]
            _write_cleanup_metadata_to_block(
                block,
                patch_plan,
                self._raw_cv,
                cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
            )
            x, y, w, h = self._mask_bbox(getattr(patch_plan, "cleanup_mask", None), block.bbox())
            crop = run["result"][y:y + h, x:x + w].copy()
            mask_bytes = b""
            if getattr(patch_plan, "cleanup_mask", None) is not None:
                try:
                    mask_bytes = np.ascontiguousarray(patch_plan.cleanup_mask).tobytes()
                except Exception:
                    mask_bytes = b""
            patch = {
                "page_idx": int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
                "region_id": self._region_id_for_cleanup_patch(idx),
                "region_idx": int(idx),
                "bbox": [int(x), int(y), int(w), int(h)],
                "strategy": str(getattr(patch_plan, "cleanup_strategy", "") or ""),
                "backend": str((run.get("group") or {}).get("backend") or getattr(patch_plan, "cleanup_backend", "") or getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
                "inpaint_method": str(getattr(patch_plan, "inpaint_method", "") or ""),
                "candidate_id": str(candidate_id or "default"),
                "mask_hash": hashlib.sha1(mask_bytes).hexdigest()[:16] if mask_bytes else "",
                "manual_mask_used": bool(run.get("manual_mask_used", False) and idx == int(region_idx)),
                "grouped_inpaint": bool((run.get("group") or {}).get("used", False)),
                "group_id": str((run.get("group") or {}).get("group_id", "") or ""),
                "group_region_ids": list((run.get("group") or {}).get("region_ids", []) or []),
                "group_backend": str((run.get("group") or {}).get("backend", "") or ""),
                "group_reason": str((run.get("group") or {}).get("reason", "") or ""),
                "fallback_error": str((run.get("group") or {}).get("error", "") or ""),
                "created_at": created_at,
                "review_required": bool((getattr(block, "cleanup_meta", {}) or {}).get("review_required", False)),
                "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                "rerun": False,
                "patch_png_b64": self._encode_cv_png_b64(crop),
            }
            if not self._attach_cleanup_patch_mask(patch, getattr(patch_plan, "cleanup_mask", None), (x, y, w, h), run["result"].shape):
                continue
            patches.append(patch)
        page.cleanup_patches.extend(patches)
        page.cleaned_cv = self._rebuild_cleaned_from_cleanup_patches(page, self._raw_cv)
        page.typeset_pil = None
        page.render_dirty = True
        page.bump_render_version()
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        self._notify("Cleanup candidate applied.", 1, 1, updated_pages=[int(getattr(self.chapter_mgr, "current_idx", 0) or 0)])
        return self.get_bootstrap()

    def propose_cleanup_mask_sam2(self, region_idx: int, prompt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = prompt if isinstance(prompt, dict) else {}
        if not _config_bool(getattr(self.model_config, "sam2_enabled", False)):
            return {"ok": False, "status": "disabled", "error": "SAM2 mask assist is disabled in settings."}
        url = str(getattr(self.model_config, "sam2_backend_url", "") or "").strip()
        mode_cfg = str(getattr(self.model_config, "sam2_mask_mode", "manual_only") or "manual_only").strip().lower()
        request_mode = str(prompt.get("mode", "cleanup") or "cleanup").strip().lower()
        if mode_cfg == "manual_only":
            return {"ok": False, "status": "disabled", "error": "SAM2 mask mode is manual_only."}
        if request_mode == "cleanup" and mode_cfg not in {"cleanup_assist", "container_assist"}:
            return {"ok": False, "status": "disabled", "error": "SAM2 cleanup assist is not enabled."}
        if request_mode in {"container", "protect"} and mode_cfg != "container_assist":
            return {"ok": False, "status": "disabled", "error": "SAM2 container/protect assist is not enabled."}
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        if not (0 <= int(region_idx) < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")

        block = self._regions[int(region_idx)]
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        ok_img, img_buf = cv2.imencode(".png", self._raw_cv)
        if not ok_img:
            raise RuntimeError("Could not encode page image for SAM2.")
        payload = {
            "page_idx": page_idx,
            "region_id": self._region_id_for_cleanup_patch(int(region_idx)),
            "mode": request_mode,
            "bbox": [int(v) for v in (prompt.get("bbox") or block.bbox())],
            "positive_clicks": prompt.get("positive_clicks") or [],
            "negative_clicks": prompt.get("negative_clicks") or [],
            "current_manual_mask": prompt.get("current_manual_mask"),
            "image_b64": base64.b64encode(img_buf.tobytes()).decode("utf-8"),
            "sam2_model_path": str(getattr(self.model_config, "sam2_model_path", "") or ""),
            "sam2_checkpoint_path": str(getattr(self.model_config, "sam2_checkpoint_path", "") or ""),
            "sam2_device": str(getattr(self.model_config, "sam2_device", "auto") or "auto"),
        }
        if url:
            timeout = max(1.0, float(getattr(self.model_config, "sam2_timeout_sec", 30) or 30))
            errors: List[str] = []
            candidates = [url.rstrip("/")]
            if not candidates[0].endswith("/propose_cleanup_mask"):
                candidates.append(candidates[0] + "/propose_cleanup_mask")
            response_data: Dict[str, Any] = {}
            for endpoint in candidates:
                try:
                    resp = requests.post(endpoint, json=payload, timeout=timeout)
                    resp.raise_for_status()
                    response_data = resp.json()
                    break
                except Exception as exc:
                    errors.append(f"{endpoint}: {exc}")
            if not response_data:
                return {"ok": False, "status": "unavailable", "error": "; ".join(errors)}
        else:
            response_data = sam2_mask.propose_mask(
                self._raw_cv,
                tuple(int(v) for v in payload["bbox"]),
                positive_clicks=payload["positive_clicks"],
                negative_clicks=payload["negative_clicks"],
                mode=request_mode,
                config=self.model_config,
            )
            if not bool(response_data.get("ok", False)):
                if _config_bool(getattr(self.model_config, "sam2_required", False)):
                    return {
                        "ok": False,
                        "status": str(response_data.get("status") or "failed"),
                        "error": f"Required embedded SAM2 setup failed: {response_data.get('error') or 'unknown error'}",
                    }
                return response_data

        mask_b64 = str(
            response_data.get("mask_b64")
            or response_data.get("mask_crop_b64")
            or response_data.get("b64")
            or ""
        )
        mask = self._decode_mask_png_b64(mask_b64)
        bbox = response_data.get("bbox") or response_data.get("mask_bbox") or payload["bbox"]
        if mask is None or len(bbox) != 4:
            return {
                "ok": False,
                "status": "error",
                "error": str(response_data.get("error") or "SAM2 response did not include a valid mask."),
                "reason": str(response_data.get("reason") or ""),
            }
        x, y, w, h = [int(v) for v in bbox]
        ih, iw = self._raw_cv.shape[:2]
        if mask.shape[:2] == (ih, iw):
            full = mask
        else:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            if x2 <= x1 or y2 <= y1:
                return {"ok": False, "status": "error", "error": "SAM2 returned an invalid mask bbox."}
            full = np.zeros((ih, iw), dtype=np.uint8)
            full[y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
        mask_px = int(np.count_nonzero(full))
        bx, by, bw, bh = [int(v) for v in payload["bbox"]]
        prompt_area = max(1, bw * bh)
        mask_bbox = self._mask_bbox(full, (bx, by, bw, bh))
        full_rectish = mask_px / prompt_area > 0.85 or (
            abs(mask_bbox[0] - bx) <= 2 and abs(mask_bbox[1] - by) <= 2
            and abs(mask_bbox[2] - bw) <= 4 and abs(mask_bbox[3] - bh) <= 4
        )
        if full_rectish and not bool(prompt.get("force_allow", False)):
            return {
                "ok": False,
                "status": "rejected",
                "bbox": [int(v) for v in mask_bbox],
                "confidence": float(response_data.get("confidence", 0.0) or 0.0),
                "reason": "SAM2 returned a near-full-rect mask; refine clicks or force allow.",
                "error": "SAM2 mask was rejected by cleanup safety checks.",
            }
        encoded = self._encode_mask_crop(full, (bx, by, bw, bh))
        return {
            "ok": True,
            "status": str(response_data.get("status") or "ok"),
            "mask_b64": encoded["b64"],
            "bbox": encoded["bbox"],
            "confidence": float(response_data.get("confidence", 0.0) or 0.0),
            "reason": str(response_data.get("reason") or "sam2_mask_proposal"),
            "error": "",
        }

    def get_sam2_status(self, load: bool = False) -> Dict[str, Any]:
        if not _config_bool(getattr(self.model_config, "sam2_enabled", False)):
            return {"ok": True, "status": "disabled", "loaded": False, "error": "SAM2 mask assist is disabled in settings."}
        url = str(getattr(self.model_config, "sam2_backend_url", "") or "").strip()
        if url:
            return {"ok": True, "status": "service", "loaded": False, "backend_url": url, "error": ""}
        state = sam2_mask.load(self.model_config) if bool(load) else sam2_mask.status(self.model_config)
        return {"ok": True, **state}

    def get_region_cleanup_debug(self, region_idx: int, manual_mask: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        if not (0 <= int(region_idx) < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        block = self._regions[int(region_idx)]
        page = self.chapter_mgr.current_page
        patch = self._cleanup_patch_for_region(page, int(region_idx)) if page is not None else None
        plan = build_cleanup_plan(
            self._raw_cv,
            block,
            page_index=int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
            region_id=f"R-{int(region_idx) + 1:02d}",
            cleanup_debug_artifacts=False,
            cleanup_debug_dir="",
            auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
            model_config=self.model_config,
        )
        self._apply_manual_cleanup_mask(plan, manual_mask, self._raw_cv.shape, block)
        group_mask = None
        try:
            base = self._rebuild_cleaned_from_cleanup_patches(page, self._raw_cv, exclude_region_idx=int(region_idx)) if page is not None else self._raw_cv.copy()
            group = self._try_grouped_fallback(int(region_idx), plan, base, base.copy(), bool(manual_mask))
            group_mask = group.get("combined_mask") if bool(group.get("used", False)) else None
        except Exception:
            group_mask = None
        fallback = tuple(int(v) for v in (getattr(plan, "container_bbox", None) or block.bbox()))
        manual_full = self._manual_mask_full(manual_mask, self._raw_cv.shape)
        quality = (getattr(plan, "debug_metrics", {}) or {}).get("quality", {}) or {}
        mask_metrics = (getattr(plan, "debug_metrics", {}) or {}).get("mask", {}) or {}
        halo_info = (getattr(plan, "debug_metrics", {}) or {}).get("halo_mask", {}) or {}
        solid = (getattr(plan, "debug_metrics", {}) or {}).get("solid_bubble_override", None)
        retry_used = (getattr(plan, "debug_metrics", {}) or {}).get("retry_used", None)
        grouped_used = bool((patch or {}).get("grouped_inpaint", False))
        override = getattr(block, "override", None)
        override_summary = override.to_dict() if override is not None and hasattr(override, "to_dict") else {}
        return {
            "ok": True,
            "analysis": {
                "page_index": int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
                "region_id": self._region_id_for_cleanup_patch(int(region_idx)),
                "region_label": f"R-{int(region_idx) + 1:02d}",
                "region_type": str(getattr(plan, "region_class", "") or ""),
                "effective_cleanup_action": str(getattr(plan, "cleanup_strategy", "") or ""),
                "effective_cleanup_mode": str(getattr(plan, "inpaint_method", "") or ""),
                "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                "skip_reason": str(getattr(plan, "skip_reason", "") or ""),
                "background_model": str(getattr(plan, "background_model", "") or ""),
                "container_confidence": round(float(getattr(plan, "container_confidence", 0.0) or 0.0), 4),
                "text_mask_confidence": round(float(getattr(plan, "text_mask_confidence", 0.0) or 0.0), 4),
                "mask_container_ratio": round(float(quality.get("mask_container_ratio", 0.0) or 0.0), 4),
                "mask_region_ratio": round(float(quality.get("mask_region_ratio", 0.0) or 0.0), 4),
                "mask_area": int(quality.get("mask_area", mask_metrics.get("mask_area", 0)) or 0),
                "border_touch_ratio": round(float(quality.get("border_touch_ratio", 0.0) or 0.0), 4),
                "border_collision_bbox_source": str(quality.get("safety_bbox_source", "") or ""),
                "rectangularity": round(float(quality.get("rectangularity", 0.0) or 0.0), 4),
                "cleanup_mask_rejected": bool((getattr(plan, "debug_metrics", {}) or {}).get("cleanup_mask_rejected", False)),
                "cleanup_mask_rejection_reason": str((getattr(plan, "debug_metrics", {}) or {}).get("cleanup_mask_rejection_reason", "") or ""),
                "selected_text_mask_candidate_source": str((getattr(plan, "debug_metrics", {}) or {}).get("selected_text_mask_candidate_source", "") or ""),
                "solid_fill_eligible": bool(solid) if solid is not None else None,
                "halo_mask_used": bool(halo_info.get("enabled", False)) if halo_info else None,
                "residual_retry_used": bool(retry_used) if retry_used is not None else None,
                "grouped_fallback_used": grouped_used,
                "selected_cleanup_candidate": str(
                    (patch or {}).get("candidate_id")
                    or (getattr(plan, "debug_metrics", {}) or {}).get("selected_candidate", "")
                    or ""
                ),
                "last_patch_status": str((patch or {}).get("cleanup_status", "") or ""),
                "last_patch_reason": str((patch or {}).get("cleanup_reason", "") or (patch or {}).get("fallback_error", "") or ""),
                "bbox": [int(v) for v in block.bbox()],
                "detector_text_bbox": [int(v) for v in getattr(block, "detector_text_bbox", None)] if getattr(block, "detector_text_bbox", None) else None,
                "container_bbox": [int(v) for v in (getattr(block, "cleanup_container_bbox", None) or getattr(plan, "container_bbox", None) or getattr(block, "bubble_bbox", None))] if (getattr(block, "cleanup_container_bbox", None) or getattr(plan, "container_bbox", None) or getattr(block, "bubble_bbox", None)) else None,
                "cleanup_override": self._json_safe(override_summary),
            },
            "boxes": {
                "detector_text_bbox": [int(v) for v in getattr(block, "detector_text_bbox", None)] if getattr(block, "detector_text_bbox", None) else None,
                "editable_bbox": [int(v) for v in block.bbox()],
                "container_bbox": [int(v) for v in (getattr(block, "cleanup_container_bbox", None) or getattr(plan, "container_bbox", None) or getattr(block, "bubble_bbox", None))] if (getattr(block, "cleanup_container_bbox", None) or getattr(plan, "container_bbox", None) or getattr(block, "bubble_bbox", None)) else None,
                "patch_bbox": [int(v) for v in (patch or {}).get("bbox", [])] if patch and len((patch or {}).get("bbox", [])) == 4 else None,
            },
            "labels": {
                "detector": f"{getattr(block, 'detector_source', 'detector')} {float(getattr(block, 'confidence', 0.0) or 0.0):.2f}",
                "container": f"conf {float(getattr(plan, 'container_confidence', 0.0) or 0.0):.2f}",
                "patch": str((patch or {}).get("cleanup_status") or (patch or {}).get("strategy") or ""),
            },
            "masks": {
                "text_mask": self._encode_mask_crop(getattr(plan, "text_mask", None), fallback),
                "cleanup_mask": self._encode_mask_crop(getattr(plan, "cleanup_mask", None), fallback),
                "halo_mask": self._encode_mask_crop(getattr(plan, "halo_mask", None), fallback),
                "manual_mask": self._encode_mask_crop(manual_full, fallback),
                "grouped_mask": self._encode_mask_crop(group_mask, fallback),
            },
        }

    def _mask_qa_dir(self) -> str:
        base = self.chapter_mgr.chapter_dir or os.getcwd()
        out_dir = os.path.join(base, "mask_qa")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _normalize_mask_qa_label(label: str) -> str:
        key = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {
            "good",
            "bad_glyph_mask",
            "bad_container_mask",
            "unsafe_cleanup_validation",
            "bad_fill_inpaint",
            "bad_routing_strategy",
            "legacy_candidate_wrong",
        }
        return key if key in allowed else ""

    def record_mask_qa_label(self, region_idx: int, label: str, notes: str = "") -> Dict[str, Any]:
        label_key = self._normalize_mask_qa_label(label)
        if not label_key:
            raise ValueError(f"Unknown mask QA label: {label!r}")
        debug = self.get_region_cleanup_debug(int(region_idx))
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        region_label = f"R-{int(region_idx) + 1:02d}"
        meta_path = os.path.join(os.getcwd(), "debug_cleanup", f"page_{page_idx:03d}", f"{region_label}_meta.json")
        record = {
            "label": label_key,
            "notes": str(notes or ""),
            "page_index": page_idx,
            "region_index": int(region_idx),
            "region_id": self._region_id_for_cleanup_patch(int(region_idx)),
            "region_label": region_label,
            "meta_path": meta_path if os.path.exists(meta_path) else "",
            "analysis": self._json_safe((debug.get("analysis") or {}) if isinstance(debug, dict) else {}),
            "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        labels_path = os.path.join(self._mask_qa_dir(), "mask_qa_labels.jsonl")
        with open(labels_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "label": label_key, "labels_path": labels_path, "record": record}

    def export_mask_qa_dataset(self) -> Dict[str, Any]:
        out_dir = self._mask_qa_dir()
        labels_path = os.path.join(out_dir, "mask_qa_labels.jsonl")
        records = 0
        labels: Dict[str, int] = {}
        if os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    label = str(item.get("label", "") or "")
                    if label:
                        labels[label] = labels.get(label, 0) + 1
                        records += 1
        manifest = {
            "labels_path": labels_path,
            "records": records,
            "labels": labels,
            "feature_source": "cleanup debug analysis/meta",
            "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        manifest_path = os.path.join(out_dir, "mask_qa_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        return {"ok": True, "dataset_dir": out_dir, "labels_path": labels_path, "manifest": manifest_path, "records": records, "labels": labels}

    def train_mask_qa_model(self) -> Dict[str, Any]:
        dataset = self.export_mask_qa_dataset()
        out_dir = str(dataset["dataset_dir"])
        labels_path = str(dataset["labels_path"])
        model_path = os.path.join(out_dir, "mask_qa_model.json")
        script_path = os.path.join(os.getcwd(), "tools", "train_mask_qa_model.py")
        cmd = [sys.executable, script_path, "--labels", labels_path, "--out", model_path]
        proc = subprocess.run(cmd, cwd=os.getcwd(), text=True, capture_output=True)
        log_path = os.path.join(out_dir, "mask_qa_training.log")
        with open(log_path, "w", encoding="utf-8") as fh:
            if proc.stdout:
                fh.write(proc.stdout)
            if proc.stderr:
                fh.write(proc.stderr)
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr or proc.stdout or "mask QA training failed", "log": log_path}
        model = {}
        try:
            with open(model_path, encoding="utf-8") as fh:
                model = json.load(fh)
        except Exception:
            model = {}
        return {"ok": True, "model_path": model_path, "log": log_path, "records": int(model.get("records", 0) or 0), "labels": model.get("centroids", {})}

    def apply_region_cleanup(self, region_idx: int, rerun: bool = False, manual_mask: Optional[Dict[str, Any]] = None) -> dict:
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")
        self._push_undo_snapshot()
        run = self._run_selected_region_cleanup(int(region_idx), mutate_block=True, manual_mask=manual_mask)
        if bool(run.get("cross_page", False)):
            block = self._regions[int(region_idx)]
            pages_touched = [int(v) for v in (run.get("cross_page_pages") or [int(getattr(self.chapter_mgr, "current_idx", 0) or 0)])]
            self._remove_cleanup_patches_for_region(pages_touched, int(region_idx))
            created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            for page_result in run.get("page_results", []) or []:
                pidx = int(page_result["page_idx"])
                x, y, w, h = [int(v) for v in page_result["bbox"]]
                patch = {
                    "page_idx": pidx,
                    "region_id": self._region_id_for_cleanup_patch(int(region_idx)),
                    "region_idx": int(region_idx),
                    "bbox": [x, y, w, h],
                    "strategy": str(getattr(run["plan"], "cleanup_strategy", "") or ""),
                    "backend": str(getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
                    "inpaint_method": str(getattr(run["plan"], "inpaint_method", "") or ""),
                    "mask_hash": str(run.get("mask_hash", "") or ""),
                    "manual_mask_used": bool(run.get("manual_mask_used", False)),
                    "grouped_inpaint": False,
                    "cross_page": True,
                    "cross_page_group_id": str(getattr(block, "cross_page_group_id", "") or self._region_id_for_cleanup_patch(int(region_idx))),
                    "cross_page_pages": pages_touched,
                    "composite_bbox": [int(v) for v in getattr(block, "composite_bbox", [])] if getattr(block, "composite_bbox", None) else None,
                    "page_local_bboxes": {
                        str(int(k)): [int(vv) for vv in v]
                        for k, v in (getattr(block, "page_local_bboxes", {}) or {}).items()
                    },
                    "created_at": created_at,
                    "review_required": bool((getattr(block, "cleanup_meta", {}) or {}).get("review_required", False)),
                    "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                    "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                    "rerun": bool(rerun),
                    "patch_png_b64": self._encode_cv_png_b64(page_result["crop"]),
                }
                if not self._attach_cleanup_patch_mask_crop(patch, page_result.get("mask_crop")):
                    continue
                self._append_cleanup_patch_to_page(pidx, patch)
            for pidx in pages_touched:
                self._rebuild_page_cleaned(pidx)
            self._flush_working_state_to_page()
            self.chapter_mgr.save_state()
            self._notify("Cross-page cleanup applied.", 1, 1, updated_pages=pages_touched)
            return self.get_bootstrap()
        group_indices = [int(v) for v in (run.get("group_region_indices") or [region_idx])]
        group_plans = list(run.get("plans") or [run["plan"]])
        replace_ids = {self._region_id_for_cleanup_patch(i) for i in group_indices}
        page.cleanup_patches = [
            p for p in (getattr(page, "cleanup_patches", []) or [])
            if not (p.get("region_id") in replace_ids or int(p.get("region_idx", -1) or -1) in group_indices)
        ]
        created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        patches: List[Dict[str, Any]] = []
        for idx, patch_plan in zip(group_indices, group_plans):
            block = self._regions[idx]
            if idx != int(region_idx):
                _write_cleanup_metadata_to_block(
                    block,
                    patch_plan,
                    self._raw_cv,
                    cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
                )
            x, y, w, h = self._mask_bbox(getattr(patch_plan, "cleanup_mask", None), block.bbox())
            crop = run["result"][y:y + h, x:x + w].copy()
            mask_bytes = b""
            if getattr(patch_plan, "cleanup_mask", None) is not None:
                try:
                    mask_bytes = np.ascontiguousarray(patch_plan.cleanup_mask).tobytes()
                except Exception:
                    mask_bytes = b""
            patch = {
                "page_idx": int(getattr(self.chapter_mgr, "current_idx", 0) or 0),
                "region_id": self._region_id_for_cleanup_patch(idx),
                "region_idx": int(idx),
                "bbox": [int(x), int(y), int(w), int(h)],
                "strategy": str(getattr(patch_plan, "cleanup_strategy", "") or ""),
                "backend": str((run.get("group") or {}).get("backend") or getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
                "inpaint_method": str(getattr(patch_plan, "inpaint_method", "") or ""),
                "mask_hash": hashlib.sha1(mask_bytes).hexdigest()[:16] if mask_bytes else "",
                "manual_mask_used": bool(run.get("manual_mask_used", False) and idx == int(region_idx)),
                "grouped_inpaint": bool((run.get("group") or {}).get("used", False)),
                "group_id": str((run.get("group") or {}).get("group_id", "") or ""),
                "group_region_ids": list((run.get("group") or {}).get("region_ids", []) or []),
                "group_backend": str((run.get("group") or {}).get("backend", "") or ""),
                "group_reason": str((run.get("group") or {}).get("reason", "") or ""),
                "fallback_error": str((run.get("group") or {}).get("error", "") or ""),
                "created_at": created_at,
                "review_required": bool((getattr(block, "cleanup_meta", {}) or {}).get("review_required", False)),
                "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                "rerun": bool(rerun),
                "patch_png_b64": self._encode_cv_png_b64(crop),
            }
            if not self._attach_cleanup_patch_mask(patch, getattr(patch_plan, "cleanup_mask", None), (x, y, w, h), run["result"].shape):
                continue
            patches.append(patch)
        page.cleanup_patches.extend(patches)
        page.cleaned_cv = self._rebuild_cleaned_from_cleanup_patches(page, self._raw_cv)
        page.typeset_pil = None
        page.render_dirty = True
        page.bump_render_version()
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        self._notify("Region cleanup applied.", 1, 1, updated_pages=[int(getattr(self.chapter_mgr, "current_idx", 0) or 0)])
        return self.get_bootstrap()

    def rerun_region_cleanup(self, region_idx: int) -> dict:
        return self.apply_region_cleanup(region_idx, rerun=True)

    def delete_region_cleanup(self, region_idx: int) -> dict:
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")
        self._push_undo_snapshot()
        block = self._regions[region_idx] if 0 <= region_idx < len(self._regions) else None
        pages_touched = [int(getattr(self.chapter_mgr, "current_idx", 0) or 0)]
        if block is not None and bool(getattr(block, "cross_page", False)):
            pages_touched = [int(v) for v in (getattr(block, "cross_page_pages", []) or pages_touched)]
            before_counts = {
                pidx: len(getattr(self.chapter_mgr.pages[pidx], "cleanup_patches", []) or [])
                for pidx in pages_touched
                if 0 <= pidx < len(self.chapter_mgr.pages)
            }
            self._remove_cleanup_patches_for_region(pages_touched, region_idx)
            changed = [
                pidx for pidx, before in before_counts.items()
                if len(getattr(self.chapter_mgr.pages[pidx], "cleanup_patches", []) or []) != before
            ]
            if not changed:
                return self.get_bootstrap()
            for pidx in changed:
                self._rebuild_page_cleaned(pidx)
            self._flush_working_state_to_page()
            self.chapter_mgr.save_state()
            self._notify("Cross-page cleanup removed.", 1, 1, updated_pages=changed)
            return self.get_bootstrap()
        region_id = self._region_id_for_cleanup_patch(region_idx)
        before = len(getattr(page, "cleanup_patches", []) or [])
        page.cleanup_patches = [
            p for p in (getattr(page, "cleanup_patches", []) or [])
            if not (p.get("region_id") == region_id or int(p.get("region_idx", -1) or -1) == int(region_idx))
        ]
        if len(page.cleanup_patches) == before:
            return self.get_bootstrap()
        if self._raw_cv is None:
            self._load_page_into_working_state()
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        page.cleaned_cv = (
            self._rebuild_cleaned_from_cleanup_patches(page, self._raw_cv)
            if page.cleanup_patches else None
        )
        page.typeset_pil = None
        page.render_dirty = True
        page.bump_render_version()
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        self._notify("Region cleanup removed.", 1, 1, updated_pages=[int(getattr(self.chapter_mgr, "current_idx", 0) or 0)])
        return self.get_bootstrap()

    # ── Memory initialisation ────────────────────────────────────────────────

    def _init_memory(
        self,
        series_title: str,
        chapter_id: str,
        aliases: Optional[List[str]] = None,
        display_title: str = "",
    ) -> None:
        """
        Initialise all three memory scopes for the current series/chapter.

        New series always start with EMPTY stores.  No migration is performed
        here — call migrate_legacy_series() explicitly for legacy projects.
        """
        self._series_title = series_title
        self._display_series_title = display_title or series_title
        self._chapter_id   = chapter_id
        self._memory_aliases = [
            a for a in (aliases or [])
            if a and a != series_title and a != chapter_id
        ]
        if not _HAS_MEMORY:
            if _MEMORY_IMPORT_ERROR:
                self._notify(f"Memory unavailable: {_MEMORY_IMPORT_ERROR}")
            return
        try:
            self._global_glossary = GlossaryStore(self._memory_root, "_global")
            self._global_names    = NameMemory(self._memory_root, "_global")
            self._global_blocked  = BlockedMappingStore(self._memory_root, "_global")
            self._glossary        = GlossaryStore(self._memory_root, series_title)
            self._name_mem        = NameMemory(self._memory_root, series_title)
            self._blocked         = BlockedMappingStore(self._memory_root, series_title)
            self._chapter_tm      = ChapterTM(self._memory_root, series_title, chapter_id)
            self._alias_glossaries = []
            self._alias_names      = []
            self._alias_blocked    = []
            try:
                from memory.storage import series_slug
            except Exception:
                series_slug = lambda value: str(value or "")  # type: ignore
            for alias in self._memory_aliases:
                alias_dir = os.path.join(self._memory_root, series_slug(alias))
                if not os.path.isdir(alias_dir):
                    continue
                self._alias_glossaries.append(GlossaryStore(self._memory_root, alias))
                self._alias_names.append(NameMemory(self._memory_root, alias))
                self._alias_blocked.append(BlockedMappingStore(self._memory_root, alias))
            print(
                f"[memory] series={series_title!r} chapter={chapter_id!r} | "
                f"glossary={len(self._glossary.all_entries())} "
                f"names={len(self._name_mem.all_entries())} "
                f"blocked={len(self._blocked.all_entries())} "
                f"tm={len(self._chapter_tm.all_entries())} "
                f"aliases={self._memory_aliases!r}"
            )
        except Exception as exc:
            print(f"[memory] init failed: {exc}")
            traceback.print_exc()
            self._global_glossary = None
            self._global_names    = None
            self._global_blocked  = None
            self._glossary        = None
            self._name_mem        = None
            self._blocked         = None
            self._chapter_tm      = None
            self._alias_glossaries = []
            self._alias_names      = []
            self._alias_blocked    = []
            self._notify(f"Memory init failed: {exc}")

    def _merged_blocked(self) -> List[Any]:
        """Return merged global + series blocked entries."""
        out: List[Any] = []
        if self._global_blocked:
            out.extend(self._global_blocked.all_entries())
        if self._blocked:
            out.extend(self._blocked.all_entries())
        for store in self._alias_blocked:
            try:
                out.extend(store.all_entries())
            except Exception:
                pass
        return out

    def _read_source_metadata(self, folder: str) -> Dict[str, Any]:
        path = os.path.join(folder, "source_metadata.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            debug_print(f"[sync] source_metadata load failed: {exc}")
            return {}

    def _write_source_metadata(self, folder: str, series: Dict[str, Any], chapter: Dict[str, Any]) -> None:
        if not folder:
            return
        source = str(series.get("source") or "")
        source_id = str(series.get("source_id") or "")
        if not source or source == "local" or not source_id:
            return
        data = {
            "source": source,
            "source_id": source_id,
            "series_title_en": series.get("title_en") or series.get("title") or "",
            "series_title_ko": series.get("title_ko") or "",
            "memory_key": series.get("memory_key") or f"source:{source}:{source_id}",
            "memory_fs_key": series.get("memory_fs_key") or _memory_slug(series.get("memory_key") or f"source:{source}:{source_id}"),
            "memory_aliases": series.get("memory_aliases") or [],
            "chapter_memory_key": chapter.get("chapter_memory_key") or (
                f"source:{source}:{source_id}:{chapter.get('source_id') or chapter.get('episode_no') or os.path.basename(folder)}"
            ),
            "chapter_memory_fs_key": chapter.get("chapter_memory_fs_key") or _memory_slug(chapter.get("chapter_memory_key") or ""),
            "episode_no": chapter.get("episode_no") or "",
            "source_chapter_id": chapter.get("source_id") or "",
            "source_url": chapter.get("source_url") or series.get("source_url") or "",
        }
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "source_metadata.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            debug_print(f"[sync] source_metadata save failed: {exc}")

    def _memory_context_for_folder(self, folder: str, fallback_series: str, fallback_chapter: str) -> Dict[str, Any]:
        series = self.series_db.find_series_for_folder(folder)
        chapter = self.series_db.find_chapter_for_folder(folder)
        metadata = self._read_source_metadata(folder)

        memory_key = ""
        aliases: List[str] = []
        chapter_key = ""
        display_title = fallback_series

        if series:
            display_title = str(series.get("title_en") or series.get("title") or fallback_series)
            memory_key = str(series.get("memory_key") or "")
            aliases = [str(a) for a in (series.get("memory_aliases") or []) if str(a).strip()]
        if chapter:
            chapter_key = str(chapter.get("chapter_memory_key") or "")

        if metadata:
            display_title = str(metadata.get("series_title_en") or metadata.get("series_title_ko") or display_title)
            memory_key = str(metadata.get("memory_key") or memory_key)
            chapter_key = str(metadata.get("chapter_memory_key") or chapter_key)
            aliases.extend(str(a) for a in (metadata.get("memory_aliases") or []) if str(a).strip())

        if not memory_key:
            memory_key = fallback_series
        if not chapter_key:
            chapter_key = fallback_chapter

        deduped_aliases: List[str] = []
        for alias in aliases + [fallback_series, display_title]:
            alias = str(alias or "").strip()
            if alias and alias != memory_key and alias not in deduped_aliases:
                deduped_aliases.append(alias)

        return {
            "display_title": display_title,
            "memory_key": memory_key,
            "chapter_memory_key": chapter_key,
            "memory_aliases": deduped_aliases,
        }

    # ── Chapter management ──────────────────────────────────────────────────

    def import_chapter(self, folder: str) -> dict:
        if not os.path.isdir(folder):
            raise ValueError(f"Selected folder does not exist: {folder}")
        if _count_images(folder) == 0:
            raise ValueError(
                "Selected folder has no image files. "
                "If this is a source series folder, open a chapter folder under chapters/."
            )
        n = self.chapter_mgr.load_from_folder(folder)
        if n == 0:
            raise ValueError(
                "Selected folder has no image files. "
                "If this is a source series folder, open a chapter folder under chapters/."
            )
        self._undo_stack = []
        self.chapter_mgr.load_state()
        chapter_name  = os.path.basename(folder)
        series_title  = os.path.basename(os.path.dirname(folder)) or chapter_name
        self.series_db.register_chapter(series_title, folder, chapter_name, n)
        memory_ctx = self._memory_context_for_folder(folder, series_title, chapter_name)
        self._load_page_into_working_state()
        # Initialise memory. Source-linked chapters use stable source keys;
        # local-only chapters preserve the historical folder-title behavior.
        self._init_memory(
            memory_ctx["memory_key"],
            memory_ctx["chapter_memory_key"],
            aliases=memory_ctx["memory_aliases"],
            display_title=memory_ctx["display_title"],
        )
        self._consistency_warnings = []
        self._memory_hits = {}
        self._last_batch_ctx = None
        self._notify(f"Loaded {n} pages from {chapter_name}")
        return self.get_bootstrap()

    # ── Source sync ─────────────────────────────────────────────────────────

    def _get_series_base_folder(self, series_title: str) -> str:
        for s in self.series_db.series:
            if s.get("title") == series_title:
                source = str(s.get("source") or "")
                source_id = str(s.get("source_id") or "")
                if source and source != "local" and source_id:
                    key = str(s.get("memory_fs_key") or _memory_slug(f"source:{source}:{source_id}"))
                    return os.path.join(str(_PROJECT_ROOT), key)
                for ch in s.get("chapters", []) or []:
                    folder = ch.get("folder") or ""
                    parent = os.path.dirname(os.path.dirname(folder)) if folder else ""
                    if parent and os.path.isdir(parent):
                        return parent
        slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", series_title).strip("._")
        return os.path.join(str(_PROJECT_ROOT), slug or "source_series")

    def _safe_source_chapter_folder(self, detail: Dict[str, Any], chapter: Dict[str, Any]) -> str:
        source = str(detail.get("source") or "")
        source_id = str(detail.get("source_id") or "")
        series_key = str(detail.get("memory_fs_key") or _memory_slug(f"source:{source}:{source_id}"))
        ch_source_id = str(chapter.get("source_id") or chapter.get("episode_no") or "chapter")
        episode_no = chapter.get("episode_no") or ""
        ep_str = f"{int(episode_no):04d}" if str(episode_no).isdigit() else ch_source_id
        slug = f"{ep_str}-{ch_source_id}" if ch_source_id and ch_source_id != ep_str else ep_str
        slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", slug).strip("._") or "chapter"
        return os.path.join(str(_PROJECT_ROOT), series_key, "chapters", slug)

    def _ensure_safe_source_folder(self, series_title: str, detail: Dict[str, Any], chapter: Dict[str, Any]) -> str:
        current = str(chapter.get("folder") or "")
        safe = self._safe_source_chapter_folder(detail, chapter)
        if os.path.abspath(current or safe) == os.path.abspath(safe):
            return safe
        if _count_images(current) > 0 and _count_images(safe) == 0:
            os.makedirs(safe, exist_ok=True)
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
            for name in os.listdir(current):
                src = os.path.join(current, name)
                if os.path.isfile(src) and os.path.splitext(name.lower())[1] in image_exts:
                    shutil.copy2(src, os.path.join(safe, name))
            state_path = os.path.join(current, ".ml_state.json")
            if os.path.isfile(state_path):
                shutil.copy2(state_path, os.path.join(safe, ".ml_state.json"))
            meta_path = os.path.join(current, "source_metadata.json")
            if os.path.isfile(meta_path):
                shutil.copy2(meta_path, os.path.join(safe, "source_metadata.json"))
        safe_count = _count_images(safe)
        if safe_count > 0:
            self.series_db.mark_chapter_imported(series_title, str(chapter.get("source_id") or ""), safe, safe_count)
        else:
            chapter["folder"] = safe
            try:
                self.series_db.save()
            except Exception:
                pass
        chapter["folder"] = safe
        return safe

    def list_sources(self) -> dict:
        return {"ok": True, "sources": list_providers()}

    def browse_source_series(self, source: str, query: str = "") -> dict:
        debug_print(f"[sync] browse_source_series source={source!r} query={query!r}")
        provider = get_provider(source)
        if provider is None:
            return {"ok": False, "error": f"Unknown source: {source!r}", "cards": []}
        try:
            return {"ok": True, "error": "", "cards": provider.search_series(query or "")}
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc), "cards": []}

    def update_series_metadata(self, series_title: str, updates: dict) -> dict:
        debug_print(f"[sync] update_series_metadata series={series_title!r} keys={list((updates or {}).keys())}")
        try:
            self.series_db.update_series_metadata(series_title, updates or {})
            return {"ok": True, "error": ""}
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    def sync_series_metadata(self, series_title: str, source: str = "", source_id: str = "") -> dict:
        debug_print(f"[sync] sync_series_metadata series={series_title!r} source={source!r} source_id={source_id!r}")
        detail = self.series_db.get_series_detail(series_title)
        if detail:
            source = source or detail.get("source", "")
            source_id = source_id or str(detail.get("source_id") or "")
        if not source or source == "local" or not source_id:
            return {"ok": False, "error": "Series has no remote source configured."}
        provider = get_provider(source)
        if provider is None:
            return {"ok": False, "error": f"Unknown source: {source!r}"}
        try:
            self._notify(f"Fetching metadata for {series_title}...", 0, 2)
            meta = provider.get_series_metadata(source_id)
            if not meta.get("ok"):
                return {"ok": False, "error": meta.get("error", "Metadata fetch failed.")}
            self._notify("Fetching chapter list...", 1, 2)
            chapters = provider.get_chapter_list(source_id)
            entry = self.series_db.upsert_source_series(
                series_title, source, source_id, meta, chapters,
                self._get_series_base_folder(series_title),
            )
            self._notify(f"Chapter index synced - {len(chapters)} chapters.", 2, 2)
            return {"ok": True, "error": "", "chapter_count": len(chapters), "series": entry}
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    def select_browse_series(self, series_title: str, source: str, source_id: str, card: dict) -> dict:
        debug_print(f"[sync] select_browse_series series={series_title!r} source={source!r} source_id={source_id!r}")
        provider = get_provider(source)
        if provider is None:
            return {"ok": False, "error": f"Unknown source: {source!r}"}
        try:
            self._notify(f"Fetching chapter index for {series_title}...", 0, 2)
            meta = provider.get_series_metadata(source_id)
            if not meta.get("ok"):
                meta = {
                    "ok": True,
                    "error": "",
                    "source": source,
                    "source_id": source_id,
                    "title_ko": card.get("title_ko", ""),
                    "title_en": card.get("title_en", ""),
                    "synopsis_ko": "",
                    "synopsis_en": "",
                    "thumbnail_url": card.get("thumbnail_url", ""),
                    "source_url": card.get("source_url", ""),
                }
            self._notify("Fetching chapter list...", 1, 2)
            chapters = provider.get_chapter_list(source_id)
            self.series_db.upsert_source_series(
                series_title, source, source_id, meta, chapters,
                self._get_series_base_folder(series_title),
            )
            self._notify(f"Series saved - {len(chapters)} chapters indexed.", 2, 2)
            return {"ok": True, "error": "", "series_title": series_title, "chapter_count": len(chapters)}
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    def sync_source_chapter(self, series_title: str, chapter_source_id: str) -> dict:
        detail = self.series_db.get_series_detail(series_title)
        if not detail:
            return {"ok": False, "error": f"Series {series_title!r} not found."}
        chapters = self.series_db.get_chapter_list_for_series(series_title)
        ch = next((c for c in chapters if str(c.get("source_id") or "") == str(chapter_source_id)), None)
        if ch is None:
            return {"ok": False, "error": f"Chapter {chapter_source_id!r} not found in index."}
        folder = self._ensure_safe_source_folder(series_title, detail, ch)
        if not folder:
            return {"ok": False, "error": "Chapter has no local folder assigned."}
        if _has_ml_state(folder):
            return {"ok": True, "error": "", "pages_synced": _count_images(folder), "skipped": True}
        source = detail.get("source", "local")
        source_id = str(detail.get("source_id") or "")
        provider = get_provider(source)
        if provider is None:
            return {"ok": False, "error": f"Unknown source: {source!r}"}
        debug_print(f"[sync] sync_source_chapter series={series_title!r} ch={chapter_source_id!r}")
        result = provider.sync_chapter_images(source_id, chapter_source_id, folder, overwrite=False)
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error", "Sync failed.")}
        pages = int(result.get("pages_synced", -1) or -1)
        self.series_db.mark_chapter_imported(series_title, chapter_source_id, folder, pages)
        self._write_source_metadata(folder, detail, ch)
        return {"ok": True, "error": "", "pages_synced": pages, "folder": folder}

    def sync_series_chapters(self, series_title: str, mode: str = "missing") -> dict:
        debug_print(f"[sync] sync_series_chapters series={series_title!r} mode={mode!r}")
        detail = self.series_db.get_series_detail(series_title)
        if not detail:
            return {"ok": False, "error": f"Series {series_title!r} not found in DB."}
        source = detail.get("source", "local")
        source_id = str(detail.get("source_id") or "")
        if source == "local" or not source_id:
            return {"ok": False, "error": "Series has no remote source configured."}
        provider = get_provider(source)
        if provider is None:
            return {"ok": False, "error": f"Unknown source: {source!r}"}

        chapters = self.series_db.get_chapter_list_for_series(series_title)
        to_sync = [ch for ch in chapters if ch.get("missing_raw") or not ch.get("imported")] if mode == "missing" else list(chapters)
        total = len(to_sync)
        synced = 0
        errors: List[str] = []
        for i, ch in enumerate(to_sync):
            ch_source_id = str(ch.get("source_id") or "")
            folder = self._ensure_safe_source_folder(series_title, detail, ch)
            if not folder:
                errors.append(f"Chapter {ch_source_id}: no local folder defined.")
                continue
            if _has_ml_state(folder):
                synced += 1
                continue
            self._notify(f"Syncing chapter {i + 1}/{total}...", i, total)
            result = provider.sync_chapter_images(source_id, ch_source_id, folder, overwrite=False)
            if result.get("ok"):
                self.series_db.mark_chapter_imported(
                    series_title, ch_source_id, folder, int(result.get("pages_synced", -1) or -1)
                )
                self._write_source_metadata(folder, detail, ch)
                synced += 1
            else:
                errors.append(f"Chapter {ch_source_id}: {result.get('error', '?')}")
        self._notify(f"Sync complete - {synced}/{total} chapters.", total, total)
        return {
            "ok": not errors,
            "error": "; ".join(errors) if errors else "",
            "synced": synced,
            "total": total,
            "errors": errors,
        }

    def import_source_chapter(self, series_title: str, chapter_source_id: str) -> dict:
        debug_print(f"[sync] import_source_chapter series={series_title!r} ch={chapter_source_id!r}")
        detail = self.series_db.get_series_detail(series_title)
        if not detail:
            return {"ok": False, "error": f"Series {series_title!r} not found."}
        chapters = self.series_db.get_chapter_list_for_series(series_title)
        ch = next((c for c in chapters if str(c.get("source_id") or "") == str(chapter_source_id)), None)
        if ch is None:
            return {"ok": False, "error": f"Chapter {chapter_source_id!r} not found in index."}
        folder = self._ensure_safe_source_folder(series_title, detail, ch)
        if not folder:
            return {"ok": False, "error": "Chapter has no local folder assigned."}

        source = detail.get("source", "local")
        source_id = str(detail.get("source_id") or "")
        if not ch.get("imported") or _count_images(folder) == 0:
            provider = get_provider(source)
            if provider is None:
                return {"ok": False, "error": f"Unknown source: {source!r}"}
            self._notify(f"Importing chapter {chapter_source_id}...", 0, 1)
            result = provider.sync_chapter_images(source_id, chapter_source_id, folder, overwrite=False)
            if not result.get("ok"):
                return {"ok": False, "error": result.get("error", "Sync failed.")}
            self.series_db.mark_chapter_imported(
                series_title, chapter_source_id, folder, int(result.get("pages_synced", -1) or -1)
            )
        self._write_source_metadata(folder, detail, ch)
        try:
            bootstrap = self.import_chapter(folder)
            abs_folder = os.path.abspath(folder)
            return {
                "ok": True,
                "error": "",
                **bootstrap,
                "folder": abs_folder,
                "page_count": int(bootstrap.get("meta", {}).get("totalPages", 0) or _count_images(abs_folder)),
                "opened": True,
                "bootstrap": bootstrap,
            }
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    def delete_series(self, series_title: str, source: str = "", source_id: str = "", delete_files: bool = False) -> dict:
        debug_print(
            f"[sync] delete_series series={series_title!r} source={source!r} "
            f"source_id={source_id!r} delete_files={delete_files!r}"
        )
        if delete_files:
            return {
                "ok": False,
                "error": "Deleting local files is not implemented. Remove from library keeps local files and memory.",
            }
        removed = self.series_db.delete_series(series_title, source, source_id)
        if not removed:
            return {"ok": False, "error": f"Series {series_title!r} not found."}
        return {
            "ok": True,
            "error": "",
            "removed": removed,
            "message": "Removed from library. Local files and memory were preserved.",
        }

    def sync_missing_thumbnails(self, series_title: str) -> dict:
        return {"ok": False, "error": "sync_missing_thumbnails: not yet implemented"}

    def get_thumbnail_b64(self, url: str = "", path: str = "") -> dict:
        """Fetch a thumbnail by URL or local path and return as a base64 data-URI.
        Sends Naver Referer header to bypass hotlink protection."""
        import base64
        import pathlib

        _THUMB_HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://comic.naver.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }

        _EXT_TO_MIME = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
        }

        # ── 1. Try local path first ──────────────────────────────────────────
        if path:
            p = pathlib.Path(path)
            if not p.is_absolute():
                p = pathlib.Path(".") / p
            if p.exists() and p.is_file():
                try:
                    data = p.read_bytes()
                    ext = p.suffix.lower().lstrip(".")
                    mime = _EXT_TO_MIME.get(ext, "image/jpeg")
                    b64 = base64.b64encode(data).decode()
                    return {"ok": True, "b64": f"data:{mime};base64,{b64}", "source": "path"}
                except Exception:
                    pass

        # ── 2. Try remote URL ────────────────────────────────────────────────
        if url and url.startswith("http"):
            try:
                import requests as _req
                resp = _req.get(url, headers=_THUMB_HEADERS, timeout=12, stream=False)
                if resp.ok:
                    ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    if not ct.startswith("image/"):
                        ct = "image/jpeg"
                    b64 = base64.b64encode(resp.content).decode()
                    return {"ok": True, "b64": f"data:{ct};base64,{b64}", "source": "url"}
            except Exception:
                pass

        return {"ok": False, "error": "No thumbnail available", "b64": ""}

    def translate_series_metadata(self, series_title: str) -> dict:
        detail = self.series_db.get_series_detail(series_title)
        if not detail:
            return {"ok": False, "error": f"Series {series_title!r} not found."}
        ko_title = detail.get("title_ko") or detail.get("title", "")
        ko_synopsis = detail.get("synopsis_ko", "")
        if not ko_title and not ko_synopsis:
            return {"ok": False, "error": "No Korean title or synopsis to translate."}
        try:
            results: Dict[str, str] = {}
            if ko_title and not detail.get("title_en"):
                raw = self.client.chat_raw(
                    model=self.model_config.translate_model,
                    prompt="Translate this Korean webtoon/manhwa title to natural English. Output only the English title, nothing else.\n\n" + ko_title,
                    keep_alive=self.model_config.keep_alive,
                )
                results["title_en"] = (raw or "").strip()
            if ko_synopsis and not detail.get("synopsis_en"):
                raw = self.client.chat_raw(
                    model=self.model_config.translate_model,
                    prompt="Translate this Korean manhwa synopsis to natural English. Output only the English translation.\n\n" + ko_synopsis,
                    keep_alive=self.model_config.keep_alive,
                )
                results["synopsis_en"] = (raw or "").strip()
            if results:
                self.series_db.update_series_metadata(series_title, results)
            return {"ok": True, "error": "", "translated": results}
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    def retranslate_series(self, series_title: str) -> dict:
        return {"ok": False, "error": "retranslate_series: not yet implemented in this version."}

    def get_series_list(self) -> dict:
        try:
            return {"ok": True, "series": self.series_db.list_series_summary()}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "series": []}

    def get_series_detail(self, series_title: str) -> dict:
        try:
            detail = self.series_db.get_series_detail(series_title)
            if detail is None:
                return {"ok": False, "error": f"Series {series_title!r} not found."}
            detail["chapters"] = self.series_db.get_chapter_list_for_series(series_title)
            detail["stats"] = self.series_db.compute_series_stats(series_title)
            return {"ok": True, "detail": detail}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def go_to_page(self, idx: int) -> dict:
        self._flush_working_state_to_page()
        self.chapter_mgr.go_to(idx)
        self._load_page_into_working_state()
        self._consistency_warnings = []
        self._memory_hits = {}
        self._last_batch_ctx = None
        self._notify(f"Page {idx + 1} / {self.chapter_mgr.total_pages()}")
        return self.get_bootstrap()

    # ── Pipeline steps ──────────────────────────────────────────────────────

    def detect_current_page(self) -> dict:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded — import a chapter first.")
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._set_progress(stage="detect", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
        backend = str(getattr(self.model_config, "detector_backend", "ocr") or "ocr").lower()
        if backend == "ocr" and self._ocr_proc is None:
            raise RuntimeError("OCR detector selected but EasyOCR compatibility OCR is not ready.")

        self._begin_active_operation("detect")
        self._notify("Detecting text regions…", 0, 1)
        tmp = "_detect_tmp.png"
        cv2.imwrite(tmp, self._raw_cv)
        try:
            blocks = self._detect_regions(tmp)
            n = len(blocks)
            for i, block in enumerate(blocks):
                self._notify(f"Enriching region {i+1}/{n}…", i, n, region_idx=i, region_total=n)
                self._enrich_region(block)
            self._regions      = blocks
            self._translations = [""] * n
            page = self.chapter_mgr.current_page
            if page is not None:
                page.detected = True
            self._bump_region_mutation_version()
            self._log_cross_page_candidates(page_idx, self._regions, self._raw_cv)
            self._invalidate_page_outputs(preserve_cleanup=False)
            self._flush_working_state_to_page()
            self._notify(f"Detected {n} region(s).", n, n, updated_pages=[page_idx], region_idx=n, region_total=n)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
            self._end_active_operation("detect")
        return self.get_bootstrap()

    def ocr_current_page(self) -> dict:
        if not self._regions:
            if self._current_page_detected():
                page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
                self._set_progress(stage="ocr", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
                self._notify("OCR skipped - no regions.", 1, 1, updated_pages=[page_idx])
                return self.get_bootstrap()
            raise RuntimeError("Run Detect first.")
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._set_progress(stage="ocr", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
        total = len(self._regions)
        snap_len = total
        snap_ids = [id(r) for r in self._regions]
        snap_version = int(getattr(self, "_region_mutation_version", 0) or 0)
        self._notify(f"OCR: 0/{total}…", 0, total)
        self._begin_active_operation("ocr")
        try:
            # Pass 4: honor the SFX master toggle — skip OCR for SFX blocks.
            process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
            for idx in range(total):
                curr_page = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
                if curr_page != page_idx:
                    debug_print(f"[OCR_STALE] idx={idx} reason=page_changed snap={page_idx} current={curr_page}")
                    return self.get_bootstrap()
                block_now = self._regions[idx] if idx < len(self._regions) else None
                if block_now is not None and self._is_cross_page_secondary(block_now):
                    debug_print(
                        f"[CROSS_PAGE_SKIP_BYPASS] stage=ocr idx={idx} "
                        "reason=ocr_allowed_for_inspector_text"
                    )
                if not process_sfx and block_now is not None and _is_pipeline_sfx(block_now):
                    block_now.ocr_status = "skipped_sfx"
                    block_now.ocr_status_reason = "process_sfx_regions_disabled"
                    debug_print(f"[SFX_SKIP] stage=ocr idx={idx} role={getattr(block_now, 'bubble_role', '')!r}")
                    self._notify(f"OCR region {idx+1}/{total} (SFX skipped)", idx + 1, total, region_idx=idx, region_total=total)
                    continue
                self._notify(f"OCR region {idx+1}/{total}…", idx, total, region_idx=idx, region_total=total)
                text = self._ocr_one_region(idx)
                curr_page = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
                if curr_page != page_idx:
                    debug_print(f"[OCR_STALE] idx={idx} reason=page_changed_after_ocr snap={page_idx} current={curr_page}")
                    return self.get_bootstrap()
                if (
                    len(self._regions) != snap_len
                    or id(self._regions[idx]) != snap_ids[idx]
                    or int(getattr(self, "_region_mutation_version", 0) or 0) != snap_version
                ):
                    debug_print(f"[OCR_STALE] idx={idx} reason=regions_mutated snap_len={snap_len} curr_len={len(self._regions)}")
                    return self.get_bootstrap()
                status = str(getattr(self._regions[idx], "ocr_status", "") or "")
                reason = str(getattr(self._regions[idx], "ocr_status_reason", "") or "")
                if status == "ok":
                    self._regions[idx].text = text
                elif status in {"empty", "failed", "cached_empty"}:
                    self._regions[idx].text = ""
                debug_print(f"[OCR_RESULT] idx={idx} status={status or 'unknown'} has_text={bool((text or '').strip())} reason={reason!r}")
            self._notify("OCR: checking split-page regions...", total, total, region_idx=total, region_total=total)
            self._try_cross_page_ocr_groups(page_idx)
            self._invalidate_page_outputs(preserve_cleanup=False)
            self._flush_working_state_to_page()
            ok_count = sum(1 for b in self._regions if str(getattr(b, "ocr_status", "") or "") == "ok")
            empty_count = sum(1 for b in self._regions if str(getattr(b, "ocr_status", "") or "") in {"empty", "failed", "cached_empty"})
            skipped_count = sum(1 for b in self._regions if str(getattr(b, "ocr_status", "") or "").startswith("skipped"))
            msg = f"OCR complete — {ok_count}/{total} text region(s)."
            if empty_count:
                msg += f" {empty_count} empty/failed."
            if skipped_count:
                msg += f" {skipped_count} skipped."
            self._notify(msg, total, total, updated_pages=[page_idx], region_idx=total, region_total=total)
            return self.get_bootstrap()
        finally:
            self._end_active_operation("ocr")

    def _normalise_ocr_backend(self) -> str:
        backend = str(getattr(self.model_config, "ocr_backend", "cascade") or "cascade").strip().lower()
        if backend not in {"cascade", "qwen_vl", "paddleocr", "easyocr"}:
            return "cascade"
        return backend

    def _ocr_cache_enabled(self) -> bool:
        return _config_bool(getattr(self.model_config, "ocr_cache_enabled", True))

    def _ocr_cache_key(self, crop: np.ndarray, backend: str) -> str:
        ok, buf = cv2.imencode(".png", crop)
        payload = buf.tobytes() if ok else crop.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        if backend.startswith("qwen_vl"):
            model_id = (
                getattr(self.model_config, "qwen_ocr_model", "")
                or getattr(self.model_config, "ocr_model", "")
                or getattr(self.model_config, "vision_model", "")
            )
        elif backend.startswith("paddleocr") or backend.startswith("cascade"):
            model_id = (
                str(getattr(self.model_config, "paddleocr_service_url", "") or "")
                + "|"
                + str(getattr(self.model_config, "paddleocr_lang", "korean") or "korean")
            )
        else:
            model_id = backend
        return f"{backend}:{model_id}:{digest}"

    def _restore_ocr_cache_entry(self, block: OCRBlock, entry: Dict[str, Any]) -> str:
        text = normalize_ocr_korean(str(entry.get("text", "") or ""))
        if not text.strip():
            block.ocr_status = "cached_empty"
            block.ocr_status_reason = "empty_cache_entry"
            return ""
        block.ocr_confidence = float(entry.get("confidence", 0.0) or 0.0)
        setattr(block, "ocr_backend", str(entry.get("backend", "") or ""))
        if entry.get("qwen_text_blocks") is not None:
            setattr(block, "qwen_ocr", {"text_blocks": entry.get("qwen_text_blocks") or []})
            setattr(block, "qwen_text_blocks", entry.get("qwen_text_blocks") or [])
        if entry.get("boxes") is not None:
            block.boxes = entry.get("boxes") or []
        if getattr(block, "detector_source", "") == "yolo" and str(entry.get("backend", "")) != "easyocr":
            block.boxes = []
            block.text_mask = None
        block.ocr_status = "ok"
        block.ocr_status_reason = "cache"
        return text

    def _read_ocr_cache(self, crop: np.ndarray, backend: str, block: Optional[OCRBlock]) -> Optional[str]:
        if not self._ocr_cache_enabled():
            return None
        cache = getattr(self, "_ocr_cache", None)
        if not isinstance(cache, dict):
            return None
        key = self._ocr_cache_key(crop, backend)
        entry = cache.get(key)
        if not isinstance(entry, dict):
            return None
        if not normalize_ocr_korean(str(entry.get("text", "") or "")).strip():
            debug_print(f"ocr_cache_empty_miss backend={backend} key={key[:24]}")
            return None
        debug_print(f"ocr_cache_hit backend={backend} key={key[:24]}")
        if block is None:
            return normalize_ocr_korean(str(entry.get("text", "") or ""))
        return self._restore_ocr_cache_entry(block, entry)

    def _write_ocr_cache(self, crop: np.ndarray, backend: str, block: Optional[OCRBlock], text: str) -> None:
        if not self._ocr_cache_enabled():
            return
        if not normalize_ocr_korean(text or "").strip():
            return
        if not hasattr(self, "_ocr_cache") or not isinstance(self._ocr_cache, dict):
            self._ocr_cache = {}
        key = self._ocr_cache_key(crop, backend)
        self._ocr_cache[key] = {
            "text": normalize_ocr_korean(text or ""),
            "confidence": float(getattr(block, "ocr_confidence", 0.0) or getattr(block, "confidence", 0.0) or 0.0) if block is not None else 0.0,
            "backend": str(getattr(block, "ocr_backend", backend) or backend) if block is not None else backend,
            "qwen_text_blocks": copy.deepcopy(getattr(block, "qwen_text_blocks", None)) if block is not None else None,
            "boxes": copy.deepcopy(getattr(block, "boxes", [])) if block is not None else [],
        }

    def _parse_paddleocr_payload(self, payload: Any) -> Tuple[str, float, List[Any]]:
        texts: List[str] = []
        confidences: List[float] = []
        boxes: List[Any] = []

        def add_text(value: Any, conf: Any = None, box: Any = None) -> None:
            text = normalize_ocr_korean(str(value or "").strip())
            if not text:
                return
            texts.append(text)
            try:
                confidences.append(float(conf))
            except Exception:
                confidences.append(0.0)
            if box is not None:
                boxes.append(box)

        if isinstance(payload, dict):
            if isinstance(payload.get("res"), dict):
                nested_text, nested_confidence, nested_boxes = self._parse_paddleocr_payload(payload.get("res"))
                if nested_text:
                    texts.append(nested_text)
                    confidences.append(nested_confidence)
                    boxes.extend(nested_boxes)
            for key in ("text", "source_text"):
                if payload.get(key):
                    add_text(payload.get(key), payload.get("confidence"))
            rec_texts = payload.get("rec_texts")
            rec_scores = payload.get("rec_scores") or []
            rec_polys = payload.get("rec_polys") or payload.get("dt_polys") or []
            if isinstance(rec_texts, list):
                for i, rec_text in enumerate(rec_texts):
                    conf = rec_scores[i] if i < len(rec_scores) else None
                    box = rec_polys[i] if i < len(rec_polys) else None
                    add_text(rec_text, conf, box)
            raw_blocks = payload.get("text_blocks")
            if isinstance(raw_blocks, list):
                for item in raw_blocks:
                    if isinstance(item, dict):
                        add_text(item.get("source_text") or item.get("text"), item.get("confidence"), item.get("box") or item.get("bbox"))
            raw_results = payload.get("results") or payload.get("data")
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict):
                        add_text(item.get("text") or item.get("source_text"), item.get("confidence") or item.get("score"), item.get("box") or item.get("bbox"))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                            add_text(item[1][0], item[1][1], item[0])
                        else:
                            add_text(item[0], item[1])
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    nested_text, nested_confidence, nested_boxes = self._parse_paddleocr_payload(item)
                    if nested_text:
                        texts.append(nested_text)
                        confidences.append(nested_confidence)
                        boxes.extend(nested_boxes)
                elif hasattr(item, "json") and isinstance(getattr(item, "json"), dict):
                    nested_text, nested_confidence, nested_boxes = self._parse_paddleocr_payload(getattr(item, "json"))
                    if nested_text:
                        texts.append(nested_text)
                        confidences.append(nested_confidence)
                        boxes.extend(nested_boxes)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        add_text(item[1][0], item[1][1], item[0])
                    else:
                        add_text(item[0], item[1])

        text = normalize_ocr_korean(" ".join(texts))
        confidence = max(confidences) if confidences else 0.0
        return text, float(max(0.0, min(1.0, confidence))), boxes

    def _run_paddleocr_on_crop(self, crop: np.ndarray) -> Dict[str, Any]:
        url = str(getattr(self.model_config, "paddleocr_service_url", "") or "").strip()
        lang = str(getattr(self.model_config, "paddleocr_lang", "korean") or "korean").strip() or "korean"
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return {"ok": False, "error": "encode_failed"}
        if url:
            try:
                payload = {
                    "image_b64": base64.b64encode(buf.tobytes()).decode("utf-8"),
                    "lang": lang,
                }
                resp = requests.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                text, confidence, boxes = self._parse_paddleocr_payload(resp.json())
                return {"ok": bool(text), "text": text, "confidence": confidence, "boxes": boxes}
            except Exception as exc:
                return {"ok": False, "error": f"service:{exc}"}

        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:
            return {"ok": False, "error": f"import:{exc}"}

        try:
            if getattr(self, "_paddle_ocr", None) is None or getattr(self, "_paddle_ocr_lang", "") != lang:
                try:
                    kwargs: Dict[str, Any] = {
                        "lang": lang,
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                    }
                    if lang.lower() in {"korean", "ko"}:
                        kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
                        kwargs["text_recognition_model_name"] = "korean_PP-OCRv5_mobile_rec"
                    self._paddle_ocr = PaddleOCR(**kwargs)
                except (TypeError, ValueError):
                    # Older PaddleOCR builds use the legacy argument name.
                    self._paddle_ocr = PaddleOCR(lang=lang, use_angle_cls=False)
                self._paddle_ocr_lang = lang
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if hasattr(self._paddle_ocr, "predict"):
                raw = self._paddle_ocr.predict(rgb)
            else:
                raw = self._paddle_ocr.ocr(rgb, cls=False)
            flattened = raw[0] if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list) else raw
            text, confidence, boxes = self._parse_paddleocr_payload(flattened)
            return {"ok": bool(text), "text": text, "confidence": confidence, "boxes": boxes}
        except Exception as exc:
            return {"ok": False, "error": f"runtime:{exc}"}

    def _apply_paddleocr_result(self, block: OCRBlock, result: Dict[str, Any], x: int, y: int) -> str:
        text = normalize_ocr_korean(str(result.get("text", "") or ""))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        boxes = []
        for box in result.get("boxes", []) or []:
            try:
                pts = [[float(px) + x, float(py) + y] for px, py in box]
                if pts:
                    boxes.append(pts)
            except Exception:
                continue
        if boxes and getattr(block, "detector_source", "ocr") != "yolo":
            block.boxes = boxes
        elif getattr(block, "detector_source", "") == "yolo":
            block.boxes = []
            block.text_mask = None
        block.ocr_confidence = confidence
        setattr(block, "ocr_backend", "paddleocr")
        debug_print(f"paddle_ocr has_text={bool(text)} confidence={confidence:.3f} boxes={len(boxes)}")
        block.ocr_status = "ok" if text else "empty"
        block.ocr_status_reason = "paddleocr" if text else "paddleocr_no_text"
        return text

    def _paddle_needs_vlm_fallback(self, block: OCRBlock, text: str, confidence: float) -> bool:
        if not text.strip():
            return True
        try:
            threshold = float(getattr(self.model_config, "ocr_vlm_fallback_confidence", 0.70) or 0.70)
        except Exception:
            threshold = 0.70
        if confidence < threshold:
            return True
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return True
        if len(compact) >= 3 and not contains_hangul(compact):
            return True
        if "�" in compact or compact.count("?") >= max(2, len(compact) // 2):
            return True
        if str(getattr(block, "yolo_kind", "") or "").lower() in {"shout", "sfx"}:
            return True
        return False

    def _ocr_one_region(self, idx: int) -> str:
        block = self._regions[idx]
        x, y, w, h = block.bbox()
        crop = self._raw_cv[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            block.ocr_status = "empty"
            block.ocr_status_reason = "empty_crop"
            return ""
        backend = self._normalise_ocr_backend()

        if backend == "easyocr":
            cached = self._read_ocr_cache(crop, "easyocr", block)
            if cached is not None:
                return cached
            text = self._ocr_one_region_easyocr_fallback(idx, crop, x, y)
            self._write_ocr_cache(crop, "easyocr", block, text)
            return text

        if backend in {"cascade", "paddleocr"}:
            cached = self._read_ocr_cache(crop, backend, block)
            if cached is not None:
                return cached
            paddle = self._run_paddleocr_on_crop(crop)
            if paddle.get("ok"):
                text = self._apply_paddleocr_result(block, paddle, x, y)
                if backend == "paddleocr" or not self._paddle_needs_vlm_fallback(
                    block,
                    text,
                    float(paddle.get("confidence", 0.0) or 0.0),
                ):
                    self._write_ocr_cache(crop, backend, block, text)
                    return text
                debug_print(
                    "paddle_ocr cascade_fallback_to_qwen "
                    f"confidence={float(paddle.get('confidence', 0.0) or 0.0):.3f}"
                )
            else:
                debug_print(f"paddle_ocr failed: {paddle.get('error', '')}")
                if backend == "paddleocr":
                    block.ocr_status = "failed"
                    block.ocr_status_reason = str(paddle.get("error", "") or "paddleocr_failed")
                    return ""

        cached = self._read_ocr_cache(crop, "qwen_vl", block)
        if cached is not None:
            return cached
        _, buf = cv2.imencode(".png", crop)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        try:
            resp = self.client.chat_json(
                model=(
                    getattr(self.model_config, "qwen_ocr_model", "")
                    or getattr(self.model_config, "ocr_model", "")
                    or getattr(self.model_config, "vision_model", "")
                ),
                prompt=(
                    "You are OCR/localization for one manhwa region crop. "
                    "Return strict JSON only. Read source text, usually Korean, "
                    "and classify role/style. Do not output pixel coordinates, "
                    "bounding boxes, polygons, masks, or placement geometry. "
                    "Use spatial_hint only as weak text-position metadata. "
                    "Do not translate or explain."
                ),
                schema=QWEN_OCR_SCHEMA,
                image_b64=b64,
                keep_alive=self.model_config.keep_alive,
            )
            setattr(block, "ocr_backend", "qwen_vl")
            raw_blocks = resp.get("text_blocks") if isinstance(resp, dict) else []
            if not isinstance(raw_blocks, list):
                raw_blocks = []
            sanitized_blocks: List[Dict[str, Any]] = []
            for raw_block in raw_blocks:
                if not isinstance(raw_block, dict):
                    continue
                try:
                    order = int(raw_block.get("reading_order", len(sanitized_blocks) + 1))
                except Exception:
                    order = len(sanitized_blocks) + 1
                source_text = normalize_ocr_korean(str(raw_block.get("source_text") or "").strip())
                role = str(raw_block.get("role") or "unknown").strip().lower()
                if role not in {"dialogue", "sfx", "caption", "narration", "unknown"}:
                    role = "unknown"
                hint = str(raw_block.get("spatial_hint") or "unknown").strip().lower()
                if hint not in {
                    "top", "bottom", "left", "right", "center",
                    "top_left", "top_right", "bottom_left", "bottom_right", "unknown",
                }:
                    hint = "unknown"
                try:
                    conf = float(raw_block.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf = 0.0
                sanitized_blocks.append({
                    "source_text": source_text,
                    "role": role,
                    "confidence": max(0.0, min(1.0, conf)),
                    "reading_order": order,
                    "spatial_hint": hint,
                    "notes": str(raw_block.get("notes") or ""),
                })

            sanitized_blocks.sort(key=lambda item: (int(item["reading_order"]), item["spatial_hint"]))
            setattr(block, "qwen_ocr", {"text_blocks": sanitized_blocks})
            setattr(block, "qwen_text_blocks", sanitized_blocks)
            texts = [b["source_text"] for b in sanitized_blocks if b["source_text"]]
            roles = [b["role"] for b in sanitized_blocks if b["role"] != "unknown"]
            role = roles[0] if roles else "unknown"
            if role == "sfx":
                block.bubble_role = "sfx"
                block.region_kind = RegionKind.SFX_OVER_ART
            elif role == "caption":
                block.bubble_role = "dialog"
                if str(getattr(block, "yolo_kind", "") or "").lower() in {"sfx", "shout"}:
                    block.yolo_kind = "dialogue"
                    block.yolo_class_id = 0
                if getattr(block, "detector_source", "") != "yolo":
                    block.region_kind = RegionKind.CAPTION_BOX
            elif role in {"dialogue", "narration"}:
                block.bubble_role = "dialog"
                if str(getattr(block, "yolo_kind", "") or "").lower() in {"sfx", "shout"}:
                    block.yolo_kind = "dialogue"
                    block.yolo_class_id = 0

            confidences = [float(b["confidence"]) for b in sanitized_blocks]
            if getattr(block, "detector_source", "") == "yolo":
                block.boxes = []
                block.text_mask = None
            if texts:
                # Pass 6: OCR text confidence writes to ocr_confidence; the
                # detector confidence on `block.confidence` (YOLO score) is
                # preserved so the UI can report both independently.
                if confidences:
                    block.ocr_confidence = float(max(confidences))
                if len(texts) != 1:
                    block.flag("multiple_qwen_text_blocks", {"count": len(texts)})
            elif getattr(block, "detector_source", "") == "yolo":
                block.text_mask = None
                block.flag("ocr_missing_but_yolo_detected_text", {"backend": "qwen_vl"})

            text = normalize_ocr_korean(" ".join(texts))
            debug_print(
                "qwen_ocr "
                f"region={idx} has_text={bool(texts)} role={role!r} "
                f"text_blocks={len(sanitized_blocks)} "
                f"confidence={round(max(confidences), 3) if confidences else 0.0} "
                f"spatial_hints={[b['spatial_hint'] for b in sanitized_blocks]}"
            )
            final_text = text
            block.ocr_status = "ok" if final_text else "empty"
            block.ocr_status_reason = "qwen_vl" if final_text else "qwen_vl_no_text"
            self._write_ocr_cache(crop, "qwen_vl", block, final_text)
            if backend == "cascade":
                self._write_ocr_cache(crop, "cascade", block, final_text)
            return final_text
        except Exception as exc:
            debug_print(f"qwen_ocr region={idx} failed={exc}")
            fallback_enabled = str(
                getattr(self.model_config, "easyocr_fallback_enabled", False)
            ).strip().lower() in {"1", "true", "yes", "on"}
            if fallback_enabled and self._ocr_proc is not None:
                return self._ocr_one_region_easyocr_fallback(idx, crop, x, y)
            block.ocr_status = "failed"
            block.ocr_status_reason = str(exc)
            return ""

    def _ocr_one_region_easyocr_fallback(self, idx: int, crop: np.ndarray, x: int, y: int) -> str:
        block = self._regions[idx]
        try:
            tmp = f"_ocr_region_{idx}.png"
            cv2.imwrite(tmp, crop)
            try:
                local_blocks = self._ocr_proc.detect(tmp) if self._ocr_proc is not None else []
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            boxes = []
            texts = []
            confidences = []
            for local in local_blocks:
                for box in getattr(local, "boxes", []) or []:
                    pts = [[float(px) + x, float(py) + y] for px, py in box]
                    if pts:
                        boxes.append(pts)
                if getattr(local, "text", ""):
                    texts.append(str(local.text))
                confidences.append(float(getattr(local, "confidence", 0.0) or 0.0))
            if boxes:
                block.boxes = boxes
                # Pass 6: for "ocr" detector source (EasyOCR), the EasyOCR
                # detector score *is* the detector confidence. For YOLO-sourced
                # blocks that fall back to EasyOCR, preserve YOLO detector
                # score and record EasyOCR's score as ocr_confidence.
                if confidences:
                    if getattr(block, "detector_source", "ocr") == "yolo":
                        block.ocr_confidence = float(max(confidences))
                    else:
                        block.confidence = float(max(confidences))
            else:
                block.boxes = []
                block.text_mask = None
            debug_print(
                "easyocr_fallback "
                f"region={idx} boxes={len(boxes)} texts={len(texts)}"
            )
            text = normalize_ocr_korean(" ".join(texts))
            block.ocr_status = "ok" if text else "empty"
            block.ocr_status_reason = "easyocr" if text else "easyocr_no_text"
            return text
        except Exception as exc:
            debug_print(f"easyocr_fallback region={idx} failed={exc}")
            block.ocr_status = "failed"
            block.ocr_status_reason = str(exc)
            return ""

    def translate_current_page(self) -> dict:
        if not self._regions:
            if self._current_page_detected():
                page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
                self._set_progress(stage="translate", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
                self._translations = []
                self._flush_working_state_to_page()
                self._notify("Translate skipped - no regions.", 1, 1, updated_pages=[page_idx])
                return self.get_bootstrap()
            raise RuntimeError("Run Detect first.")
        self._consistency_warnings = []
        self._memory_hits = {}
        self._last_batch_ctx = None
        # Pass 4: when the SFX master toggle is OFF, blank out SFX inputs so
        # they never hit the translator; index alignment with self._translations
        # is preserved by using "" for skipped slots.
        process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
        texts: List[str] = []
        sfx_mask: List[bool] = []
        for b in self._regions:
            if self._is_cross_page_secondary(b):
                if str(getattr(b, "text", "") or "").strip():
                    debug_print("[CROSS_PAGE_SKIP_BYPASS] stage=translate reason=secondary_has_existing_text")
                    sfx_mask.append(False)
                    texts.append(b.text or "")
                else:
                    b.text = ""
                    sfx_mask.append(True)
                    texts.append("")
                continue
            is_sfx = (not process_sfx) and _is_pipeline_sfx(b)
            sfx_mask.append(is_sfx)
            texts.append("" if is_sfx else (b.text or ""))
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._set_progress(stage="translate", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
        active_count = sum(1 for sk in sfx_mask if not sk)
        self._notify(
            f"Translating {active_count} region(s)…"
            + (f" ({len(texts) - active_count} SFX skipped)" if (len(texts) - active_count) else ""),
            0, len(texts),
        )
        self._begin_active_operation("translate")
        try:
            results = self._translate_texts(texts, page_idx=page_idx)
            # Force "" at SFX positions regardless of what the translator returned.
            for i, is_sfx in enumerate(sfx_mask):
                if is_sfx and i < len(results):
                    results[i] = ""
            for i, t in enumerate(results):
                cleaned = (t or "").strip()
                # Pass 4: for SFX slots, explicitly overwrite any previously
                # stored translation so toggling the SFX master OFF removes
                # old visible SFX text. For non-SFX slots, keep the existing
                # "don't clobber a non-empty translation with empty" guard.
                if sfx_mask[i]:
                    if i < len(self._translations):
                        self._translations[i] = ""
                    else:
                        self._translations.append("")
                    continue
                if not cleaned and i < len(self._translations) and self._translations[i].strip():
                    continue
                if i < len(self._translations):
                    self._translations[i] = cleaned
                else:
                    self._translations.append(cleaned)
            attempted = [
                i for i, source in enumerate(texts)
                if (source or "").strip() and i < len(sfx_mask) and not sfx_mask[i]
            ]
            empty_results = [
                i for i in attempted
                if i < len(results) and not str(results[i] or "").strip()
            ]
            self._flag_translations(list(range(len(results))))
            self._post_translate(texts, results, self._last_batch_ctx, page_idx)
            self._invalidate_page_outputs(preserve_cleanup=True)
            self._flush_working_state_to_page()
            flagged = sum(1 for b in self._regions if getattr(b, "is_flagged", False))
            drift   = len(self._consistency_warnings)
            msg = f"Translation complete — {len(results)} line(s)."
            if empty_results:
                msg = f"Translation incomplete — {len(empty_results)}/{len(attempted)} non-empty source line(s) returned no result."
            if flagged:
                msg += f"  ⚑ {flagged} region(s) flagged."
            if drift:
                msg += f"  ⚠ {drift} memory warning(s)."
            self._notify(msg, len(results), len(results), updated_pages=[page_idx], region_idx=len(results), region_total=len(results))
            return self.get_bootstrap()
        finally:
            self._end_active_operation("translate")

    def _translate_texts(self, texts: List[str], page_idx: int = 0) -> List[str]:
        """
        Translate a list of Korean strings.

        Phase 4: retrieve_batch() is called first to build a bounded exact
        name/glossary constraint block. Chapter TM examples are not injected.

        Phase 6: blocked mappings are checked post-translation via
        _post_translate(), which also stores results to ChapterTM.

        Provider routing: if model_config.translation_provider == "deepseek",
        the DeepSeek HTTP API is used instead of Ollama.  Set the env variable
        named by model_config.deepseek_api_key_env before starting the server.
        If deepseek_fallback_to_ollama is True (default), any DeepSeek error
        falls back transparently to the Ollama path.
        """
        provider = str(getattr(self.model_config, "translation_provider", "ollama") or "ollama").lower()
        if provider == "deepseek":
            return self._translate_texts_deepseek(texts, page_idx=page_idx)

        has_full_pipeline = hasattr(self.client, "chat_json")

        # ── Phase 4: build bounded memory context for this batch ──────────────
        batch_ctx     = self._retrieve_batch_context(texts)
        self._last_batch_ctx = batch_ctx
        prompt_prefix = self._build_prompt_prefix(batch_ctx)

        results: List[str] = []

        if has_full_pipeline:
            try:
                # Batch translate via Ollama (mirrors _run_translate_all logic)
                numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
                prompt = (
                    f"{prompt_prefix}"
                    f"Translate each numbered Korean line to natural English manga dialogue.\n"
                    f"Output JSON with 'translated_lines' array in the same order.\n\n"
                    f"{numbered}"
                )
                resp = self.client.chat_json(
                    model=self.model_config.translate_model,
                    prompt=prompt,
                    schema=TRANSLATOR_SCHEMA,
                    keep_alive=self.model_config.keep_alive,
                )
                raw_lines = resp.get("translated_lines", [])
                for i, (kr, en) in enumerate(zip(texts, raw_lines)):
                    heuristic = heuristic_localize_line(kr)
                    cleaned   = sanitize_final_translation(kr, en, heuristic)
                    results.append(cleaned)
                    self._notify(f"Translated {i+1}/{len(texts)}…", i + 1, len(texts))
                # Pad if model returned fewer lines
                while len(results) < len(texts):
                    results.append("")
                return results
            except Exception as exc:
                print(f"[translate] Ollama batch failed: {exc} — trying per-region")

        # Fallback: translate one at a time
        for i, kr in enumerate(texts):
            self._notify(f"Translating {i+1}/{len(texts)}…", i, len(texts))
            heuristic = heuristic_localize_line(kr)
            if heuristic:
                results.append(heuristic)
                continue
            # Per-region prefix: only the constraints relevant to this one line.
            region_prefix = self._build_region_prefix(batch_ctx, i)
            try:
                raw = self.client.chat_raw(
                    model=self.model_config.translate_model,
                    prompt=(
                        f"{region_prefix}"
                        f"Translate this Korean manhwa dialogue to natural English. "
                        f"Output only the English translation, nothing else.\n\n{kr}"
                    ),
                    keep_alive=self.model_config.keep_alive,
                )
                results.append(sanitize_final_translation(kr, raw))
            except Exception as exc:
                print(f"[translate] region {i} failed: {exc}")
                results.append(heuristic_localize_line(kr) or "")

        return results

    def _translate_texts_deepseek(self, texts: List[str], page_idx: int = 0) -> List[str]:
        """Translate via the DeepSeek HTTP API.  Falls back to Ollama on error
        if model_config.deepseek_fallback_to_ollama is True (default).

        Empty-string slots (SFX-masked regions) are filtered out before the API
        call so they never consume tokens or distort line-count matching.
        """
        from backend.core.deepseek_translate import (
            translate_batch as _ds_translate,
            DeepSeekConfigError,
            DeepSeekAPIError,
        )
        from backend.core.text_utils import sanitize_final_translation, heuristic_localize_line

        # ── Filter empty (SFX-masked) slots before sending to DeepSeek ───────
        # translate_current_page() already puts "" in SFX positions so they
        # never get translated; we honour that here by skipping them entirely
        # rather than sending an empty numbered line that wastes tokens and
        # confuses the model's line-count alignment.
        active_indices = [i for i, t in enumerate(texts) if (t or "").strip()]
        active_texts   = [texts[i] for i in active_indices]

        if not active_texts:
            debug_print("[DEEPSEEK] all texts empty (all SFX masked) — skipping API call")
            return [""] * len(texts)

        debug_print(
            f"[DEEPSEEK] sending {len(active_texts)}/{len(texts)} texts "
            f"({len(texts) - len(active_texts)} empty slots suppressed)"
        )

        batch_ctx     = self._retrieve_batch_context(active_texts)
        self._last_batch_ctx = batch_ctx
        prompt_prefix = self._build_prompt_prefix(batch_ctx)
        fallback = bool(getattr(self.model_config, "deepseek_fallback_to_ollama", True))
        try:
            raw = _ds_translate(active_texts, config=self.model_config, prompt_prefix=prompt_prefix)
            # Map DeepSeek results back to original slot positions.
            results: List[str] = [""] * len(texts)
            for pos, orig_i in enumerate(active_indices):
                kr = texts[orig_i]
                en = raw[pos] if pos < len(raw) else ""
                heuristic = heuristic_localize_line(kr)
                results[orig_i] = sanitize_final_translation(kr, en, heuristic) if en else ""
            return results
        except (DeepSeekConfigError, DeepSeekAPIError) as exc:
            debug_print(f"[DEEPSEEK] error={exc}; fallback_to_ollama={fallback}")
            if fallback:
                self._notify(f"DeepSeek failed ({type(exc).__name__}); falling back to Ollama…", 0, len(texts))
                return self._translate_texts_ollama(texts, prompt_prefix=prompt_prefix)
            raise

    def _translate_texts_ollama(self, texts: List[str], prompt_prefix: str = "") -> List[str]:
        """Inner Ollama translation path, extracted so DeepSeek fallback can call it."""
        from backend.core.text_utils import sanitize_final_translation, heuristic_localize_line
        has_full_pipeline = hasattr(self.client, "chat_json")
        results: List[str] = []
        if has_full_pipeline:
            try:
                numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
                prompt = (
                    f"{prompt_prefix}"
                    f"Translate each numbered Korean line to natural English manga dialogue.\n"
                    f"Output JSON with 'translated_lines' array in the same order.\n\n"
                    f"{numbered}"
                )
                resp = self.client.chat_json(
                    model=self.model_config.translate_model,
                    prompt=prompt,
                    schema=TRANSLATOR_SCHEMA,
                    keep_alive=self.model_config.keep_alive,
                )
                raw_lines = resp.get("translated_lines", [])
                for i, (kr, en) in enumerate(zip(texts, raw_lines)):
                    heuristic = heuristic_localize_line(kr)
                    cleaned   = sanitize_final_translation(kr, en, heuristic)
                    results.append(cleaned)
                    self._notify(f"Translated {i+1}/{len(texts)}…", i + 1, len(texts))
                while len(results) < len(texts):
                    results.append("")
                return results
            except Exception as exc:
                print(f"[translate] Ollama batch failed: {exc} — trying per-region")
        for i, kr in enumerate(texts):
            self._notify(f"Translating {i+1}/{len(texts)}…", i, len(texts))
            try:
                region_prefix = self._build_region_prefix(i, batch_ctx=self._last_batch_ctx)
                prompt = (
                    f"{prompt_prefix}{region_prefix}"
                    f"Translate to natural English manga dialogue. "
                    f"Output JSON with 'translation', 'confidence', 'notes' fields.\n\n{kr}"
                )
                resp = self.client.chat_json(
                    model=self.model_config.translate_model,
                    prompt=prompt,
                    schema={"type": "object", "properties": {"translation": {"type": "string"}, "confidence": {"type": "number"}, "notes": {"type": "array", "items": {"type": "string"}}}},
                    keep_alive=self.model_config.keep_alive,
                )
                en = str(resp.get("translation", "") or "").strip()
                heuristic = heuristic_localize_line(kr)
                results.append(sanitize_final_translation(kr, en, heuristic) if en else "")
            except Exception:
                results.append(heuristic_localize_line(kr) or "")
        return results

    # ── Memory retrieval helpers ──────────────────────────────────────────────

    def _retrieve_batch_context(self, texts: List[str]) -> Any:
        """
        Call retrieve_batch() with all loaded store data.
        Returns None when memory is unavailable or retrieval fails.
        """
        if not _HAS_MEMORY:
            return None
        try:
            g_global   = self._global_glossary.all_entries() if self._global_glossary else []
            g_series   = self._glossary.all_entries()        if self._glossary        else []
            for store in self._alias_glossaries:
                g_series.extend(store.all_entries())
            n_global   = self._global_names.all_entries()    if self._global_names    else []
            n_series   = self._name_mem.all_entries()        if self._name_mem        else []
            for store in self._alias_names:
                n_series.extend(store.all_entries())
            blocked    = self._merged_blocked()
            chapter_tm = self._chapter_tm.retrievable_entries() if self._chapter_tm else []
            return retrieve_batch(
                texts,
                g_global, g_series,
                n_global, n_series,
                chapter_tm,
                blocked,
            )
        except Exception as exc:
            print(f"[memory] retrieve_batch failed: {exc}")
            return None

    def _build_prompt_prefix(self, batch_ctx: Any) -> str:
        """
        Assemble the full memory prefix for a batch prompt.
        Returns "" when there is nothing to inject.
        """
        if not _HAS_MEMORY or batch_ctx is None:
            return ""
        parts = []
        if batch_ctx.constraint_block:
            parts.append(batch_ctx.constraint_block)
        return "\n\n".join(parts) + "\n\n" if parts else ""

    def _build_region_prefix(self, batch_ctx: Any, region_idx: int) -> str:
        """
        Assemble a per-region constraint prefix (fallback single-region path).
        Uses only the glossary/name hits for this specific line.
        Returns "" when there is nothing to inject.
        """
        if not _HAS_MEMORY or batch_ctx is None:
            return ""
        try:
            from memory.retrieval import _build_constraint_block
            g = (batch_ctx.per_line_glossary[region_idx]
                 if region_idx < len(batch_ctx.per_line_glossary) else [])
            n = (batch_ctx.per_line_names[region_idx]
                 if region_idx < len(batch_ctx.per_line_names) else [])
            cb = _build_constraint_block(g, n)
            return cb + "\n\n" if cb else ""
        except Exception:
            return ""

    def _post_translate(
        self,
        texts:     List[str],
        results:   List[str],
        batch_ctx: Any,         # BatchRetrievalResult | None
        page_idx:  int,
    ) -> None:
        """
        Run post-translation consistency checks (name drift, glossary drift,
        blocked output) and store results to ChapterTM as machine-trust entries.

        Populates self._consistency_warnings and self._memory_hits; called once
        per translate pass.
        Skips when memory is unavailable; import/init diagnostics are exposed
        through status and memoryStats.
        """
        if not _HAS_MEMORY:
            return

        new_warnings: List[Dict[str, Any]] = []
        new_hits: Dict[int, List[Dict[str, Any]]] = {}

        for i, (kr, en) in enumerate(zip(texts, results)):
            g_hits = (batch_ctx.per_line_glossary[i]
                      if batch_ctx and i < len(batch_ctx.per_line_glossary) else [])
            n_hits = (batch_ctx.per_line_names[i]
                      if batch_ctx and i < len(batch_ctx.per_line_names) else [])
            region_hits: List[Dict[str, Any]] = []
            for e in n_hits:
                region_hits.append({
                    "type":       "name",
                    "kr":         e.kr_name,
                    "en":         e.en_name,
                    "trust":      e.trust,
                    "matched_by": "exact",
                    "scope":      e.scope,
                })
            for e in g_hits:
                region_hits.append({
                    "type":       "glossary",
                    "kr":         e.source_kr,
                    "en":         e.target_en,
                    "trust":      e.trust,
                    "matched_by": "exact",
                    "scope":      e.scope,
                })
            new_hits[i] = region_hits

            # Consistency checks
            w_name     = check_name_drift(kr, en, n_hits, page_idx, i)
            w_glossary = check_glossary_drift(kr, en, g_hits, page_idx, i)

            # Phase 6: blocked output check (series store takes precedence)
            fired: List[Any] = []
            if self._blocked:
                fired = self._blocked.matches(kr, en)
            if not fired:
                for store in self._alias_blocked:
                    try:
                        fired = store.matches(kr, en)
                    except Exception:
                        fired = []
                    if fired:
                        break
            if not fired and self._global_blocked:
                fired = self._global_blocked.matches(kr, en)
            w_blocked = check_blocked_output(kr, en, fired, page_idx, i)

            label = f"R-{i+1:02d}"
            for w in (w_name + w_glossary + w_blocked):
                new_warnings.append({
                    "id":     f"mem-p{page_idx}-r{i}-{w.warning_type}",
                    "sev":    "err" if w.warning_type == "blocked_output" else "warn",
                    "msg":    (
                        f"[{w.warning_type.replace('_', ' ')}] "
                        f"expected \"{w.expected}\""
                    ),
                    "region": label,
                    "page":   page_idx + 1,
                })

            # Store to ChapterTM — machine trust, pending status, write-only in Phase 3
            if self._chapter_tm and kr.strip() and en.strip():
                block_flagged = False
                if i < len(self._regions):
                    review = getattr(self._regions[i], "review", None)
                    block_flagged = (
                        bool(review and getattr(review, "flagged", False))
                        or bool(getattr(self._regions[i], "is_flagged", False))
                    )
                try:
                    self._chapter_tm.store(
                        kr, en, page_idx, i,
                        flagged=block_flagged or bool(w_blocked),
                    )
                except Exception as exc:
                    print(f"[memory] ChapterTM.store failed: {exc}")

        self._consistency_warnings = new_warnings
        self._memory_hits = new_hits

    def _flag_translations(self, indices: List[int]) -> None:
        """Heuristic post-translate flagging for suspicious translations."""
        PLACEHOLDER_PHRASES = {
            "i'm sorry, i can't assist with that.",
            "i cannot assist with that request.",
            "i'm not able to translate this.",
        }
        for idx in indices:
            if idx >= len(self._regions):
                continue
            block = self._regions[idx]
            tl = self._translations[idx] if idx < len(self._translations) else ""
            reasons: List[str] = []
            if not tl.strip():
                reasons.append("empty_translation")
            elif contains_hangul(tl):
                reasons.append("possibly_untranslated")
            elif tl.lower().strip() in PLACEHOLDER_PHRASES:
                reasons.append("placeholder_output")
            elif (len(block.text) > 0 and
                  (len(tl) > len(block.text) * 6 or len(tl) < len(block.text) * 0.1)):
                reasons.append("length_mismatch")
            elif is_likely_garbage_literal(block.text, tl):
                reasons.append("garbage_output")
            if reasons and hasattr(block, "flag"):
                for r in reasons:
                    block.flag(r)

    def cleanup_current_page(self) -> dict:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        if not self._regions:
            if self._current_page_detected():
                page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
                self._set_progress(stage="cleanup", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
                render_debug("cleanup", action="cleanup_noop", baseLayer="raw", **self._debug_page_state())
                self._notify("Cleanup skipped - no regions.", 1, 1, updated_pages=[page_idx])
                return self.get_bootstrap()
            raise RuntimeError("Run Detect first.")

        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._set_progress(stage="cleanup", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())
        self._notify("Running cleanup…", 0, 1)
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")

        self._begin_active_operation("cleanup")
        try:
            debug_print(f'cleanup_route="planned" page={page_idx} action="cleanup_current_page"')
            # Pass 4: hide SFX from cleanup when the master toggle is OFF.
            process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
            cleanup_indices = [
                idx for idx, block in enumerate(self._regions)
                if (
                    not self._is_cross_page_secondary(block)
                    and
                    (process_sfx or not _is_pipeline_sfx(block) or _cleanup_override_allows_pipeline_sfx(block))
                    and _can_destructively_clean_region(block, "", self.model_config, operation="whole_page")[0]
                )
            ]
            if len(cleanup_indices) != len(self._regions):
                debug_print(f"[SFX_SKIP] stage=cleanup kept={len(cleanup_indices)}/{len(self._regions)}")
            if not cleanup_indices:
                render_debug("cleanup", action="cleanup_noop", baseLayer="raw", **self._debug_page_state())
                self._notify("Cleanup skipped - no eligible regions.", 1, 1, updated_pages=[page_idx])
                return self.get_bootstrap()
            has_cross_page = any(bool(getattr(self._regions[idx], "cross_page", False)) for idx in cleanup_indices)
            if has_cross_page:
                debug_print(f'cleanup_route="split_patches" page={page_idx} action="cleanup_current_page"')
                page.cleanup_patches = []
                updated_pages = {page_idx}
                created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                for idx in cleanup_indices:
                    run = self._run_selected_region_cleanup(idx, mutate_block=True)
                    block = self._regions[idx]
                    if bool(run.get("cross_page", False)):
                        pages_touched = [int(v) for v in (run.get("cross_page_pages") or [page_idx])]
                        self._remove_cleanup_patches_for_region(pages_touched, idx)
                        for page_result in run.get("page_results", []) or []:
                            pidx = int(page_result["page_idx"])
                            x, y, w, h = [int(v) for v in page_result["bbox"]]
                            patch = {
                                "page_idx": pidx,
                                "region_id": self._region_id_for_cleanup_patch(idx),
                                "region_idx": int(idx),
                                "bbox": [x, y, w, h],
                                "strategy": str(getattr(run["plan"], "cleanup_strategy", "") or ""),
                                "backend": str(getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
                                "inpaint_method": str(getattr(run["plan"], "inpaint_method", "") or ""),
                                "mask_hash": str(run.get("mask_hash", "") or ""),
                                "manual_mask_used": False,
                                "grouped_inpaint": False,
                                "cross_page": True,
                                "cross_page_group_id": str(getattr(block, "cross_page_group_id", "") or self._region_id_for_cleanup_patch(idx)),
                                "cross_page_pages": pages_touched,
                                "composite_bbox": [int(v) for v in getattr(block, "composite_bbox", [])] if getattr(block, "composite_bbox", None) else None,
                                "page_local_bboxes": {
                                    str(int(k)): [int(vv) for vv in v]
                                    for k, v in (getattr(block, "page_local_bboxes", {}) or {}).items()
                                },
                                "created_at": created_at,
                                "review_required": bool((getattr(block, "cleanup_meta", {}) or {}).get("review_required", False)),
                                "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                                "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                                "rerun": False,
                                "patch_png_b64": self._encode_cv_png_b64(page_result["crop"]),
                            }
                            if not self._attach_cleanup_patch_mask_crop(patch, page_result.get("mask_crop")):
                                continue
                            self._append_cleanup_patch_to_page(pidx, patch)
                            updated_pages.add(pidx)
                    else:
                        x, y, w, h = self._mask_bbox(getattr(run["plan"], "cleanup_mask", None), block.bbox())
                        crop = run["result"][y:y + h, x:x + w].copy()
                        patch = {
                            "page_idx": page_idx,
                            "region_id": self._region_id_for_cleanup_patch(idx),
                            "region_idx": int(idx),
                            "bbox": [x, y, w, h],
                            "strategy": str(getattr(run["plan"], "cleanup_strategy", "") or ""),
                            "backend": str(getattr(self.model_config, "cleanup_backend", "opencv") or "opencv"),
                            "inpaint_method": str(getattr(run["plan"], "inpaint_method", "") or ""),
                            "mask_hash": str(run.get("mask_hash", "") or ""),
                            "manual_mask_used": False,
                            "grouped_inpaint": bool((run.get("group") or {}).get("used", False)),
                            "created_at": created_at,
                            "review_required": bool((getattr(block, "cleanup_meta", {}) or {}).get("review_required", False)),
                            "cleanup_status": str(getattr(block, "cleanup_status", "") or ""),
                            "cleanup_reason": str(getattr(block, "cleanup_reason", "") or ""),
                            "rerun": False,
                            "patch_png_b64": self._encode_cv_png_b64(crop),
                        }
                        if not self._attach_cleanup_patch_mask(patch, getattr(run["plan"], "cleanup_mask", None), (x, y, w, h), run["result"].shape):
                            continue
                        self._append_cleanup_patch_to_page(page_idx, patch)
                for pidx in sorted(updated_pages):
                    self._rebuild_page_cleaned(pidx)
                render_debug("cleanup", action="cleanup_split_patches", baseLayer="raw", updated_pages=sorted(updated_pages), **self._debug_page_state())
                self._notify("Cleanup complete ✓", 1, 1, updated_pages=sorted(updated_pages))
            else:
                cleanup_regions = [self._regions[idx] for idx in cleanup_indices]
                cleaned = erase_text_region_planned(
                    self._raw_cv,
                    cleanup_regions,
                    page_index=page_idx,
                    cleanup_backend=getattr(self.model_config, "cleanup_backend", "opencv"),
                    iopaint_url=getattr(self.model_config, "iopaint_url", ""),
                    cleanup_debug_artifacts=_config_bool(getattr(self.model_config, "cleanup_debug_artifacts", False)),
                    cleanup_debug_dir=getattr(self.model_config, "cleanup_debug_dir", ""),
                    auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
                    cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
                    model_config=self.model_config,
                )
                page.cleaned_cv = cleaned
                page.cleanup_patches = []
                page.typeset_pil = None
                page.render_dirty = True
                page.bump_render_version()
                render_debug("cleanup", action="cleanup", baseLayer="raw", **self._debug_page_state())
                self._notify("Cleanup complete ✓", 1, 1, updated_pages=[page_idx])
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Cleanup failed: {exc}") from exc
        finally:
            self._end_active_operation("cleanup")

        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        return self.get_bootstrap()

    def _apply_cross_page_typeset(
        self,
        current_page_idx: int,
        current_pil: Image.Image,
    ) -> List[int]:
        updated: List[int] = []
        current_cv = cv2.cvtColor(np.array(current_pil), cv2.COLOR_RGB2BGR)
        for idx, block in enumerate(self._regions):
            if not bool(getattr(block, "cross_page", False)):
                continue
            if idx >= len(self._translations):
                continue
            trans = sanitize_final_translation(getattr(block, "text", "") or "", self._translations[idx] or "")
            if not trans.strip() or not getattr(block, "visible", True):
                continue
            bbox = tuple(int(v) for v in block.bbox())
            composite, offsets, pages, comp_bbox, local = self._cross_page_context_for_block(current_page_idx, block)
            if composite is None or len(pages) <= 1 or current_page_idx not in offsets:
                continue
            base, render_offsets = self._build_stitched_render_base(pages, current_override=current_cv)
            if base is None:
                continue
            comp_block = copy.deepcopy(block)
            comp_block.bbox_override = comp_bbox
            comp_block.bubble_bbox = comp_bbox
            comp_block.cross_page = False
            comp_block.manually_adjusted = True
            old_regions, old_translations = self._regions, self._translations
            try:
                self._regions = [comp_block]
                self._translations = [trans]
                rendered = self._typeset_image(base)
            finally:
                self._regions = old_regions
                self._translations = old_translations
            rendered_cv = cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)
            for pidx in pages:
                if pidx not in render_offsets:
                    continue
                page = self.chapter_mgr.pages[pidx]
                raw_page = self._page_render_base_cv(pidx)
                if raw_page is None:
                    continue
                ph = int(raw_page.shape[0])
                oy = render_offsets[pidx]
                crop = rendered_cv[oy:oy + ph, :, :].copy()
                page.typeset_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                page.render_dirty = False
                page.bump_render_version()
                updated.append(int(pidx))
                if pidx == current_page_idx:
                    current_cv = crop
            block.typeset_meta["cross_page_typeset_split"] = True
            block.typeset_status = "typeset_ok"
            block.typeset_reason = ""
        return sorted(set(updated))

    def typeset_current_page(self) -> dict:
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._set_progress(stage="typeset", page_idx=page_idx, page_total=self.chapter_mgr.total_pages())

        if page.cleaned_cv is None and self._raw_cv is not None and self._regions:
            debug_print(f'cleanup_route="planned" page={page_idx} action="typeset_missing_cleaned"')
            # Pass 4: same SFX filter as cleanup_current_page.
            process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
            cleanup_regions = [
                b for b in self._regions
                if (
                    not self._is_cross_page_secondary(b)
                    and (process_sfx or not _is_pipeline_sfx(b) or _cleanup_override_allows_pipeline_sfx(b))
                    and _can_destructively_clean_region(b, "", self.model_config, operation="typeset_missing_cleaned")[0]
                )
            ]
            page.cleaned_cv = erase_text_region_planned(
                self._raw_cv,
                cleanup_regions,
                page_index=page_idx,
                cleanup_backend=getattr(self.model_config, "cleanup_backend", "opencv"),
                iopaint_url=getattr(self.model_config, "iopaint_url", ""),
                cleanup_debug_artifacts=_config_bool(getattr(self.model_config, "cleanup_debug_artifacts", False)),
                cleanup_debug_dir=getattr(self.model_config, "cleanup_debug_dir", ""),
                auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
                cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
                model_config=self.model_config,
            )
            page.cleanup_patches = []

        base_cv = page.cleaned_cv if page.cleaned_cv is not None else self._raw_cv
        if base_cv is None:
            raise RuntimeError("No image loaded.")
        if not any(t and t.strip() for t in self._translations):
            process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
            has_typeset_candidate = any(
                getattr(block, "visible", True)
                and not self._is_cross_page_secondary(block)
                and (process_sfx or not _is_pipeline_sfx(block))
                and not (
                    hasattr(block, "effective_skip_typeset")
                    and block.effective_skip_typeset()
                )
                for block in self._regions
            )
            if not has_typeset_candidate:
                if page.typeset_pil is not None:
                    self._notify("Typeset skipped - no eligible regions.", 1, 1, updated_pages=[page_idx])
                    return self.get_bootstrap()
                page.typeset_pil = Image.fromarray(cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB))
                page.render_dirty = False
                page.bump_render_version()
                self._flush_working_state_to_page()
                self.chapter_mgr.save_state()
                self._notify("Typeset skipped — no eligible regions.", 1, 1, updated_pages=[page_idx])
                return self.get_bootstrap()
            raise RuntimeError("Run Translate first.")

        self._begin_active_operation("typeset")
        self._notify("Rendering translations…", 0, 1)
        try:
            pil_out = self._typeset_image(base_cv)
            page.typeset_pil = pil_out
            page.render_dirty = False
            page.bump_render_version()
            updated_pages = [page_idx]
            cross_updated = self._apply_cross_page_typeset(page_idx, pil_out)
            if cross_updated:
                updated_pages = sorted(set(updated_pages + cross_updated))
            render_debug(
                "typeset",
                action="typeset",
                baseLayer="cleaned" if page.cleaned_cv is not None else "raw",
                regions=[self._debug_region_style(b) for b in self._regions],
                **self._debug_page_state(),
            )
            self._notify("Typeset complete ✓", 1, 1, updated_pages=updated_pages)
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Typeset failed: {exc}") from exc
        finally:
            self._end_active_operation("typeset")

        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        return self.get_bootstrap()

    def _text_bbox_px(self, draw: ImageDraw.Draw, text: str,
                      font: ImageFont.ImageFont, stroke_w: int = 0) -> tuple[int, int, tuple[int, int, int, int]]:
        if not text:
            return 0, 0, (0, 0, 0, 0)
        try:
            bb = draw.textbbox((0, 0), text, font=font, stroke_width=max(0, int(stroke_w)))
        except TypeError:
            bb = draw.textbbox((0, 0), text, font=font)
        return int(bb[2] - bb[0]), int(bb[3] - bb[1]), (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))

    def _break_token_to_width(self, token: str, font: ImageFont.ImageFont,
                              max_width: int) -> list[str]:
        if not token:
            return [""]
        parts: list[str] = []
        current = ""
        for ch in token:
            trial = current + ch
            if current and _text_width(font, trial) > max_width:
                parts.append(current)
                current = ch
            else:
                current = trial
        if current:
            parts.append(current)
        return parts or [token]

    def _wrap_text_for_width(self, text: str, font: ImageFont.ImageFont,
                             max_width: int) -> list[str]:
        max_width = max(1, int(max_width))
        out: list[str] = []
        paragraphs = text.splitlines() or [text]
        sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        avg_char = max(1, int(round(_text_width(font, sample) / max(1, len(sample)))))
        approx_chars = max(1, int((max_width / avg_char) * 1.10))

        for para in paragraphs:
            para = para.strip()
            if not para:
                out.append("")
                continue
            rough = textwrap.wrap(
                para,
                width=approx_chars,
                break_long_words=False,
                break_on_hyphens=False,
                drop_whitespace=True,
            ) or [para]
            for chunk in rough:
                greedy = _wrap_text(chunk, font, max_width)
                for line in greedy:
                    if _text_width(font, line) <= max_width:
                        out.append(line)
                    else:
                        words = line.split()
                        if len(words) <= 1:
                            out.extend(self._break_token_to_width(line, font, max_width))
                        else:
                            cur = ""
                            for word in words:
                                candidate = f"{cur} {word}".strip()
                                if cur and _text_width(font, candidate) > max_width:
                                    out.append(cur)
                                    if _text_width(font, word) > max_width:
                                        out.extend(self._break_token_to_width(word, font, max_width))
                                        cur = ""
                                    else:
                                        cur = word
                                else:
                                    cur = candidate
                            if cur:
                                out.append(cur)
        return out

    def _measure_lines(self, draw: ImageDraw.Draw, lines: list[str],
                       font: ImageFont.ImageFont, line_h: int,
                       stroke_w: int = 0) -> tuple[int, int, tuple[int, int, int, int] | None]:
        if not lines:
            return 0, 0, None
        union = None
        cursor_y = 0
        for line in lines:
            _w, _h, bb = self._text_bbox_px(draw, line, font, stroke_w)
            rendered = (bb[0], cursor_y + bb[1], bb[2], cursor_y + bb[3])
            if union is None:
                union = rendered
            else:
                union = (
                    min(union[0], rendered[0]), min(union[1], rendered[1]),
                    max(union[2], rendered[2]), max(union[3], rendered[3]),
                )
            cursor_y += line_h
        if union is None:
            return 0, 0, None
        return int(union[2] - union[0]), int(union[3] - union[1]), union

    def _get_typeset_box(self, block: OCRBlock) -> Tuple[int, int, int, int]:
        # 1. User/editor override wins. A detector-provided YOLO bbox is only a
        # proposal, so it must not become the default text placement box.
        detector_bbox = getattr(block, "detector_text_bbox", None)
        if detector_bbox is not None:
            base_bbox = tuple(int(v) for v in detector_bbox)
        else:
            base_bbox = tuple(int(v) for v in block.bbox())
        rx, ry, rw, rh = base_bbox
        ignored_manual_box = False
        rejected_auto_typeset_rect = False

        def _box_implausible(rect: Tuple[int, int, int, int], source: str) -> Tuple[bool, str]:
            _x, _y, w, h = [int(v) for v in rect]
            ref_area = max(1, int(rw) * int(rh))
            area_ratio = (max(1, w) * max(1, h)) / float(ref_area)
            height_ratio = max(1, h) / float(max(1, rh))
            page_fraction = 0.0
            if self._raw_cv is not None:
                ph, pw = self._raw_cv.shape[:2]
                page_fraction = (max(1, w) * max(1, h)) / float(max(1, pw * ph))
            if source == "manual" and bool(getattr(block, "locked", False)):
                return False, ""
            if page_fraction > 0.32:
                return True, f"page_fraction_too_large({page_fraction:.3f})"
            if area_ratio > 4.0 and height_ratio > 2.5:
                return True, f"area_height_ratio_too_large(area={area_ratio:.2f},height={height_ratio:.2f})"
            if height_ratio > 5.0:
                return True, f"height_ratio_too_large({height_ratio:.2f})"
            return False, ""

        if getattr(block, 'bbox_override', None) is not None and getattr(block, 'manually_adjusted', False):
            manual_box = tuple(int(v) for v in block.bbox_override)
            bad_manual, bad_reason = _box_implausible(manual_box, "manual")
            if not bad_manual:
                debug_print(f"[TYPESET_BOX] source=manual box={block.bbox_override}")
                return block.bbox_override
            ignored_manual_box = True
            debug_print(
                f"[TYPESET_BOX_REJECT] source=manual box={manual_box} "
                f"base={base_bbox} reason={bad_reason}"
            )
            meta = getattr(block, "typeset_meta", None)
            if isinstance(meta, dict):
                meta["box_reject_reason"] = bad_reason
                meta["box_rejected_source"] = "manual"
        text_box = None
        try:
            pts = np.array([pt for box in (getattr(block, "boxes", []) or []) for pt in box], dtype=np.int32)
            if pts.size:
                text_box = tuple(int(v) for v in cv2.boundingRect(pts))
        except Exception:
            text_box = None

        def _plausible_yolo_typeset_rect(rect: Tuple[int, int, int, int], label: str) -> bool:
            if getattr(block, "detector_source", "") != "yolo" or (
                getattr(block, "manually_adjusted", False) and not ignored_manual_box
            ):
                return True
            _x, _y, w, h = [int(v) for v in rect]
            rect_area = max(1, int(w) * int(h))
            region_area = max(1, int(rw) * int(rh))
            ratio = rect_area / float(region_area)
            if ratio <= 3.0:
                return True
            debug_print(
                f"[TYPESET_BOX] {label}={rect} rejected; "
                f"ratio={ratio:.2f} region={(rx, ry, rw, rh)}"
            )
            return False

        # 2. Phase 2: prefer safe_rect (computed by compute_placement).
        #    This is the closest thing we have to a bubble-interior placement
        #    box; it is intentionally preferred over detector rectangles.
        safe_rect = getattr(block, 'safe_rect', None)
        if safe_rect is not None:
            sx, sy, sw, sh = safe_rect
            weak_yolo_container = False
            try:
                bm = getattr(block, "bubble_mask", None)
                weak_yolo_container = (
                    getattr(block, "detector_source", "") == "yolo"
                    and bm is not None
                    and (np.count_nonzero(bm) / max(1, int(bm.size))) > 0.92
                )
            except Exception:
                weak_yolo_container = getattr(block, "detector_source", "") == "yolo"
            safe_area = int(sw) * int(sh)
            region_area = max(1, int(rw) * int(rh))
            clearly_larger_than_region = safe_area >= int(region_area * 1.35)
            safe_region_ratio = safe_area / float(region_area)
            plausible_yolo_safe = _plausible_yolo_typeset_rect(safe_rect, "safe_rect")
            if sw >= 24 and sh >= 16 and plausible_yolo_safe and (not weak_yolo_container or clearly_larger_than_region):
                debug_print(
                    f"[TYPESET_BOX] source=LIR safe_rect={safe_rect} "
                    f"region={(rx, ry, rw, rh)}"
                )
                meta = getattr(block, "typeset_meta", None)
                if isinstance(meta, dict):
                    meta["box_source"] = "safe_rect"
                return safe_rect
            debug_print(
                f"[TYPESET_BOX] safe_rect={safe_rect} rejected; falling back "
                f"weak_yolo_container={weak_yolo_container} ratio={safe_region_ratio:.2f}"
            )
            rejected_auto_typeset_rect = True

        cleanup_safe = getattr(block, "cleanup_safe_rect", None)
        cleanup_safe_conf = float(getattr(block, "cleanup_safe_rect_confidence", 0.0) or 0.0)
        if cleanup_safe is not None and cleanup_safe_conf >= 0.45:
            x, y, w, h = [int(v) for v in cleanup_safe]
            if w >= 24 and h >= 16 and _plausible_yolo_typeset_rect((x, y, w, h), "cleanup_safe_rect"):
                debug_print(
                    f"[TYPESET_BOX] source=cleanup_safe_rect "
                    f"safe={(x, y, w, h)} conf={cleanup_safe_conf:.2f}"
                )
                meta = getattr(block, "typeset_meta", None)
                if isinstance(meta, dict):
                    meta["box_source"] = "cleanup_safe_rect"
                    meta["box_confidence"] = round(cleanup_safe_conf, 3)
                return (x, y, w, h)
            rejected_auto_typeset_rect = True

        cleanup_cont = getattr(block, "cleanup_container_bbox", None)
        cleanup_conf = float(getattr(block, "cleanup_container_confidence", 0.0) or 0.0)
        if cleanup_cont is not None and cleanup_conf >= 0.40:
            cx, cy, cw, ch = [int(v) for v in cleanup_cont]
            inset_x = max(10, int(cw * 0.08))
            inset_y = max(10, int(ch * 0.08))
            derived = (
                cx + inset_x,
                cy + inset_y,
                max(1, cw - inset_x * 2),
                max(1, ch - inset_y * 2),
            )
            if derived[2] >= 24 and derived[3] >= 16 and _plausible_yolo_typeset_rect(derived, "cleanup_container"):
                debug_print(
                    f"[TYPESET_BOX] source=cleanup_container "
                    f"container={cleanup_cont} derived={derived}"
                )
                meta = getattr(block, "typeset_meta", None)
                if isinstance(meta, dict):
                    meta["box_source"] = "cleanup_container_derived"
                    meta["box_confidence"] = round(cleanup_conf, 3)
                return derived

        yolo_box = getattr(block, "detector_text_bbox", None) or getattr(block, "bbox_override", None)
        if getattr(block, "detector_source", "") == "yolo" and yolo_box is not None:
            x, y, w, h = [int(v) for v in yolo_box]
            if w >= 24 and h >= 16:
                if rejected_auto_typeset_rect:
                    return (x, y, w, h)
                pad_x = max(8, int(round(w * 0.08)))
                pad_y = max(6, int(round(h * 0.18)))
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = w + pad_x * 2
                h = h + pad_y * 2
                debug_print(
                    f"[TYPESET_BOX] source=yolo_region_fallback "
                    f"box={(x, y, w, h)} bubble={getattr(block, 'bubble_bbox', None)}"
                )
                meta = getattr(block, "typeset_meta", None)
                if isinstance(meta, dict):
                    meta["box_source"] = "yolo_region_fallback"
                return (x, y, w, h)

        # 3. No reliable bubble: use CV/manual text-cluster geometry when available.
        if not getattr(block, 'bubble_bbox', None):
            if text_box is not None:
                tx, ty, tw, th = text_box
                pad_x = max(8, int(round(tw * 0.35)))
                pad_y = max(8, int(round(th * 0.45)))
                safe_box = (
                    max(0, tx - pad_x),
                    max(0, ty - pad_y),
                    max(1, tw + pad_x * 2),
                    max(1, th + pad_y * 2),
                )
                debug_print(f"[TYPESET_BOX] source=text_bbox_fallback text={text_box} safe={safe_box}")
                meta = getattr(block, "typeset_meta", None)
                if isinstance(meta, dict):
                    meta["box_source"] = "text_bbox_fallback"
                return safe_box
            FLAT_PAD = 6
            inset_x = max(FLAT_PAD, int(round(rw * 0.15)))
            inset_y = max(FLAT_PAD, int(round(rh * 0.15)))
            safe_box = (
                rx + inset_x, ry + inset_y,
                max(1, rw - inset_x * 2),
                max(1, rh - inset_y * 2),
            )
            debug_print(f"[TYPESET_BOX] source=weak_container_center_fallback region={(rx, ry, rw, rh)} safe={safe_box}")
            meta = getattr(block, "typeset_meta", None)
            if isinstance(meta, dict):
                meta["box_source"] = "weak_container_center_fallback"
            return safe_box

        # 4. Bubble bbox is weak placement geometry unless supported by CV/manual
        #    text-cluster geometry. Never use the full YOLO rectangle as a dialogue box.
        bx, by, bw, bh = block.bubble_bbox
        if text_box is not None:
            tx, ty, tw, th = text_box
            max_expand_x = max(10, int(round(tw * 0.55)))
            max_expand_y = max(10, int(round(th * 0.70)))
            new_x1 = max(bx, tx - max_expand_x)
            new_y1 = max(by, ty - max_expand_y)
            new_x2 = min(bx + bw, tx + tw + max_expand_x)
            new_y2 = min(by + bh, ty + th + max_expand_y)
            safe_box = (new_x1, new_y1, max(1, new_x2 - new_x1), max(1, new_y2 - new_y1))
            debug_print(
                f"[TYPESET_BOX] source=text_bbox_fallback text={text_box} "
                f"bubble={(bx, by, bw, bh)} safe={safe_box}"
            )
            meta = getattr(block, "typeset_meta", None)
            if isinstance(meta, dict):
                meta["box_source"] = "text_bbox_fallback"
            return safe_box

        inset_x = max(10, int(round(bw * 0.16)))
        inset_y = max(10, int(round(bh * 0.18)))
        safe_box = (bx + inset_x, by + inset_y, max(1, bw - inset_x * 2), max(1, bh - inset_y * 2))
        debug_print(
            f"[TYPESET_BOX] source=weak_container_center_fallback "
            f"bubble={(bx, by, bw, bh)} safe={safe_box}"
        )
        meta = getattr(block, "typeset_meta", None)
        if isinstance(meta, dict):
            meta["box_source"] = "weak_container_center_fallback"
        return safe_box

    def _minimum_typeset_font_size(self, role: str) -> int:
        cfg = getattr(self, "model_config", None)
        if role == "sfx":
            return int(getattr(cfg, "minimum_sfx_font_size", 16) or 16)
        return int(getattr(cfg, "minimum_dialog_font_size", 14) or 14)

    def _should_skip_auto_typeset_fit(
        self,
        block: OCRBlock,
        fit_result: _FitResult,
        box: Tuple[int, int, int, int],
        role: str,
        manually_overridden: bool,
    ) -> Tuple[bool, str]:
        if manually_overridden:
            return False, ""
        _x, _y, w, h = [int(v) for v in box]
        if w < 24 or h < 16:
            return True, "typeset_box_too_small"
        minimum_size = self._minimum_typeset_font_size(role)
        if int(getattr(fit_result, "font_size", 0) or 0) < minimum_size:
            return True, "font_below_minimum"
        if bool(getattr(fit_result, "overflow", False)):
            return True, "text_overflow"
        return False, ""

    def _record_typeset_outcome(
        self,
        block: OCRBlock,
        status: str,
        reason: str,
        meta: Optional[Dict[str, Any]] = None,
        flag_review: bool = False,
    ) -> None:
        block.typeset_status = status
        block.typeset_reason = reason
        if not isinstance(getattr(block, "typeset_meta", None), dict):
            block.typeset_meta = {}
        if meta:
            for key, val in meta.items():
                if isinstance(val, (str, int, float, bool)) or val is None:
                    block.typeset_meta[str(key)] = val
                elif (
                    isinstance(val, (list, tuple))
                    and all(isinstance(item, (str, int, float, bool)) or item is None for item in val)
                ):
                    block.typeset_meta[str(key)] = list(val)
        if flag_review and hasattr(block, "flag") and not bool(getattr(block, "is_flagged", False)):
            block.flag(
                "typeset_fit_review",
                {
                    "reason": reason,
                    "font_size": int((meta or {}).get("font_size", 0) or 0),
                    "box": list((meta or {}).get("box", [])),
                },
            )

    def _typeset_image(self, base_cv: np.ndarray) -> Image.Image:
        """Draw all translations onto base_cv and return PIL image.

        Phase 3 changes:
          - Uses block.effective_style() for all styling decisions
          - Draws drop-shadow, outline, gradient or solid fill via _draw_line_with_style()
          - Draws optional background plate behind the text block
          - Applies contrast safety check for auto-styled regions
          - Respects effective_skip_typeset() and effective_erase_only()
        """
        pil_img = Image.fromarray(cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)

        # Pass 4: skip SFX regions in the final typeset when the SFX master
        # toggle is OFF. They also have empty translations at this point
        # (translate_current_page clears them), but this guard also prevents
        # any leftover cached typeset_pil from being regenerated with SFX text.
        process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
        for idx, (block, trans) in enumerate(zip(self._regions, self._translations)):
            if bool(getattr(block, "cross_page", False)):
                continue
            if self._is_cross_page_secondary(block):
                continue
            if not getattr(block, 'visible', True):
                continue
            if not process_sfx and _is_pipeline_sfx(block):
                continue
            if not _has_explicit_typeset_override(block) and _is_visual_sfx_like(block):
                self._record_typeset_outcome(
                    block,
                    "typeset_skipped",
                    _typeset_cleanup_skip_reason(block),
                    {
                        "role": getattr(block, "bubble_role", None) or "dialog",
                        "cleanup_tier": int(getattr(block, "cleanup_tier", 0) or 0),
                    },
                )
                continue
            trans = sanitize_final_translation(getattr(block, "text", "") or "", trans or "")
            if idx < len(self._translations):
                self._translations[idx] = trans
            # Phase 3: honour explicit skip_typeset override
            if (hasattr(block, 'effective_skip_typeset')
                    and block.effective_skip_typeset()):
                continue
            # Phase 3: honour effective_erase_only (override wins over field)
            erase_only = (block.effective_erase_only()
                          if hasattr(block, 'effective_erase_only')
                          else getattr(block, 'erase_only', False))
            if not trans or not trans.strip() or erase_only:
                continue

            bx, by, bw, bh = self._get_typeset_box(block)
            role          = getattr(block, 'bubble_role', None) or "dialog"
            override_name = getattr(block, 'font_name', '')
            override_size = getattr(block, 'font_size', 0)
            manually_overridden = _has_explicit_typeset_override(block)

            # Phase 3: resolve full style (override → block style → auto-derived)
            style = (block.effective_style()
                     if hasattr(block, 'effective_style')
                     else None)
            if style is None:
                ow = getattr(block, 'outline_width', 0) or 0
                oc = getattr(block, 'outline_color', (255,255,255)) or (255,255,255)
                # Reconstruct minimal TextStyle from legacy fields
                fg = block.fg_color or (0, 0, 0)
                style_outline_w = max(1, int(ow)) if ow > 0 else 1
                style_outline_c = oc if ow > 0 else (
                    (255,255,255) if sum(fg) < 382 else (0,0,0)
                )
                # Build inline since TextStyle may not be importable here
                class _S:
                    pass
                s = _S()
                s.fg_color = fg; s.outline_color = style_outline_c
                s.outline_width = style_outline_w
                s.gradient_on = False; s.shadow_on = False; s.plate_on = False
                s.gradient_start = fg; s.gradient_end = fg; s.gradient_angle = 90
                s.shadow_color = (0,0,0); s.shadow_offset = (1,2); s.shadow_opacity = 0.55
                s.plate_color = (255,255,255); s.plate_opacity = 0.78; s.plate_pad = 4
                s.source = "auto"
                style = s  # type: ignore

            if role == "sfx" and sum(tuple(style.fg_color)) > 700 and sum(tuple(style.outline_color)) > 700:
                style.outline_color = (0, 0, 0)
                style.outline_width = max(2, int(getattr(style, "outline_width", 1) or 1))
                style.shadow_on = True
                style.shadow_color = (0, 0, 0)

            outline_w = int(style.outline_width) if style.outline_width else 1

            # Compute content area BEFORE fitting
            pad_x     = max(8, int(round(bw * 0.06)))
            pad_y     = max(8, int(round(bh * 0.06)))
            content_x = bx + pad_x
            content_y = by + pad_y
            content_w = max(1, bw - pad_x * 2)
            content_h = max(1, bh - pad_y * 2)

            if override_size and override_size > 0:
                font  = self._load_fit_font(role, override_size, override_name)
                lines = self._wrap_text_for_width(trans, font, content_w)
                line_h = max(1, int(round(override_size * 1.15)))
                used_w, used_h, _ = self._measure_lines(draw, lines, font, line_h, outline_w)
                fit = _FitResult(
                    lines=lines, font_size=override_size,
                    used_width=used_w, used_height=used_h,
                    overflow=used_w > content_w or used_h > content_h,
                )
                debug_print(
                    f"[TYPESET_OVERRIDE] idx={idx} box={(bx,by,bw,bh)} "
                    f"size={override_size} used={(used_w,used_h)} "
                    f"overflow={fit.overflow} lines={lines}"
                )
            else:
                fit  = self._fit_font_size(trans, content_w, content_h, role,
                                           override_name, outline_w)
                font = self._load_fit_font(role, fit.font_size, override_name)

            if not fit.lines:
                continue
            block.typeset_overflow = bool(fit.overflow)
            block.typeset_fit = {
                "font_size": fit.font_size,
                "used_width": fit.used_width,
                "used_height": fit.used_height,
                "box_width": content_w,
                "box_height": content_h,
            }
            if not isinstance(getattr(block, "typeset_meta", None), dict):
                block.typeset_meta = {}
            box_source = str(block.typeset_meta.get("box_source", "") or "")
            typeset_meta = {
                "box": [int(bx), int(by), int(bw), int(bh)],
                "box_source": box_source,
                "font_size": int(fit.font_size),
                "overflow": bool(fit.overflow),
                "role": role,
                "cleanup_tier": int(getattr(block, "cleanup_tier", 0) or 0),
            }
            skip_fit, skip_reason = self._should_skip_auto_typeset_fit(
                block,
                fit,
                (bx, by, bw, bh),
                role,
                manually_overridden,
            )
            if skip_fit:
                self._record_typeset_outcome(
                    block,
                    "typeset_review_fit_failed",
                    skip_reason,
                    typeset_meta,
                    flag_review=True,
                )
                continue
            cleanup_tier = int(getattr(block, "cleanup_tier", 0) or 0)
            if cleanup_tier == 2:
                self._record_typeset_outcome(
                    block,
                    "typeset_review",
                    "cleanup_tier_2_review",
                    typeset_meta,
                    flag_review=False,
                )
            else:
                self._record_typeset_outcome(block, "typeset_ok", "", typeset_meta)

            line_h  = max(1, int(round(fit.font_size * 1.15)))
            used_w, used_h, union0 = self._measure_lines(draw, fit.lines, font,
                                                          line_h, outline_w)
            union0  = union0 or (0, 0, 0, 0)
            cursor_y = content_y + max(0, int(round((content_h - used_h) / 2)))

            # ── Phase 3: optional background plate ───────────────────────────
            is_caption_box = getattr(getattr(block, "region_kind", None), "name", "") == "CAPTION_BOX"
            if getattr(style, 'plate_on', False) and is_caption_box:
                pad = getattr(style, 'plate_pad', 4)
                # Compute where the whole text block will land
                approx_x1 = content_x - pad
                approx_y1 = cursor_y  - pad
                approx_x2 = content_x + used_w + pad
                approx_y2 = cursor_y  + used_h + pad
                _render_plate(
                    pil_img,
                    approx_x1, approx_y1, approx_x2, approx_y2,
                    getattr(style, 'plate_color', (255,255,255)),
                    getattr(style, 'plate_opacity', 0.78),
                )
            elif getattr(style, 'plate_on', False):
                render_debug(
                    "typeset",
                    action="ignored_plate",
                    region=idx,
                    reason="normal_bubble",
                    kind=getattr(getattr(block, "region_kind", None), "name", ""),
                )

            # ── Phase 3: contrast safety (auto-sourced styles only) ──────────
            fg_color = style.fg_color
            if getattr(style, 'source', 'auto') == 'auto' and not getattr(style, 'gradient_on', False):
                fg_color = _check_text_contrast(
                    fg_color, pil_img,
                    content_x, cursor_y, used_w, used_h,
                )

            # Build a local effective style with the (possibly corrected) fg_color
            # so _draw_line_with_style uses it
            class _EffStyle:
                pass
            eff = _EffStyle()
            eff.fg_color       = fg_color
            eff.outline_color  = style.outline_color
            eff.outline_width  = outline_w
            eff.gradient_on    = getattr(style, 'gradient_on',    False)
            eff.gradient_start = getattr(style, 'gradient_start', fg_color)
            eff.gradient_end   = getattr(style, 'gradient_end',   fg_color)
            eff.gradient_angle = getattr(style, 'gradient_angle', 90)
            eff.shadow_on      = getattr(style, 'shadow_on',      False)
            eff.shadow_color   = getattr(style, 'shadow_color',   (0,0,0))
            eff.shadow_offset  = getattr(style, 'shadow_offset',  (1,2))
            eff.shadow_opacity = getattr(style, 'shadow_opacity',  0.55)

            align       = getattr(block, 'align', 'center') or 'center'
            actual_union = None

            for line_text in fit.lines:
                line_w, _line_h_px, line_bb = self._text_bbox_px(draw, line_text,
                                                                   font, outline_w)
                if align == "left":
                    target_left = content_x
                elif align == "right":
                    target_left = content_x + content_w - line_w
                else:
                    target_left = content_x + int(round((content_w - line_w) / 2))
                x_pos = target_left - line_bb[0]
                y_pos = cursor_y   - line_bb[1]
                rendered_bb = (
                    int(x_pos + line_bb[0]), int(y_pos + line_bb[1]),
                    int(x_pos + line_bb[2]), int(y_pos + line_bb[3]),
                )
                if actual_union is None:
                    actual_union = rendered_bb
                else:
                    actual_union = (
                        min(actual_union[0], rendered_bb[0]),
                        min(actual_union[1], rendered_bb[1]),
                        max(actual_union[2], rendered_bb[2]),
                        max(actual_union[3], rendered_bb[3]),
                    )

                # Phase 3: full line render (shadow + outline + fill/gradient)
                _draw_line_with_style(
                    pil_img, draw, line_text, x_pos, y_pos, font,
                    eff, outline_w,  # type: ignore[arg-type]
                )
                cursor_y += line_h

            debug_print(
                f"[TYPESET_DRAW] idx={idx} role={role} text={trans!r} "
                f"box={(bx,by,bw,bh)} content={(content_x,content_y,content_w,content_h)} "
                f"size={fit.font_size} used={(fit.used_width,fit.used_height)} "
                f"actual={actual_union} lines={fit.lines} "
                f"style_src={getattr(style,'source','auto')!r}"
            )

        return pil_img

    def get_region_preview_sprite(self, region_idx: int, draft: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Render one region's English text as a transparent PNG without mutating state."""
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No page loaded.")

        block = copy.deepcopy(self._regions[region_idx])
        trans = self._translations[region_idx] if region_idx < len(self._translations) else ""
        draft = draft or {}
        if "tl" in draft:
            trans = str(draft.get("tl") or "")
        if "src" in draft:
            block.text = str(draft.get("src") or "")
        if any(k in draft for k in ("x", "y", "w", "h")):
            bx, by, bw, bh = block.bbox()
            bbox = self._clamp_bbox(
                draft.get("x", bx), draft.get("y", by),
                draft.get("w", bw), draft.get("h", bh),
            )
            block.boxes = [self._bbox_to_box(*bbox)]
            block.bbox_override = bbox
            block.bubble_bbox = bbox
            block.bubble_mask = build_ellipse_mask(bbox[2], bbox[3], inset=4)
            block.safe_rect = None
        if "font" in draft:
            block.font_name = str(draft.get("font") or "")
        if "size" in draft:
            block.font_size = max(0, min(96, int(float(draft.get("size") or 0))))
        if "align" in draft:
            align = str(draft.get("align") or "center")
            block.align = align if align in {"left", "center", "right"} else "center"
        if "fg" in draft:
            block.fg_color = self._parse_hex_color(draft.get("fg"))
            if block.style is not None:
                block.style.fg_color = block.fg_color
        if "bg" in draft and getattr(getattr(block, "region_kind", None), "name", "") == "CAPTION_BOX":
            block.bg_color = self._parse_hex_color(draft.get("bg"))
        if "outline" in draft:
            block.outline_color = self._parse_hex_color(draft.get("outline"))
            if block.style is not None:
                block.style.outline_color = block.outline_color
        if "outline_width" in draft:
            block.outline_width = max(0, min(8, int(float(draft.get("outline_width") or 0))))
            if block.style is not None:
                block.style.outline_width = block.outline_width
        if "shadow" in draft:
            if block.style is None:
                block.style = block.effective_style()
            block.style.shadow_color = self._parse_hex_color(draft.get("shadow"))
        if "shadow_on" in draft:
            if block.style is None:
                block.style = block.effective_style()
            block.style.shadow_on = bool(draft.get("shadow_on"))
        if "visible" in draft:
            block.visible = bool(draft.get("visible"))

        rendered = self._render_region_preview_sprite(block, trans, region_idx)
        render_debug(
            "preview.sprite",
            action="get_region_preview_sprite",
            region=int(region_idx),
            bbox=rendered.get("bbox"),
            font=rendered.get("font"),
            resolved_font_size=rendered.get("resolved_font_size"),
            fg=rendered.get("fg"),
            outline=rendered.get("outline"),
            shadow=rendered.get("shadow"),
            sprite={k: rendered.get(k) for k in ("x", "y", "w", "h")},
            has_b64=bool(rendered.get("b64")),
            **self._debug_page_state(),
        )
        return rendered

    def _render_region_preview_sprite(self, block: OCRBlock, trans: str, idx: int) -> Dict[str, Any]:
        if not getattr(block, 'visible', True):
            return _empty_preview_response()
        # Pass 4: also suppress preview sprite for SFX when master toggle OFF
        if (
            not _config_bool(getattr(self.model_config, "process_sfx_regions", False))
            and _is_pipeline_sfx(block)
        ):
            debug_print(f"[TYPESET_SKIP] preview region={idx} reason=process_sfx_regions_disabled")
            return _empty_preview_response("typeset_skipped_sfx_master_disabled")
        trans = sanitize_final_translation(getattr(block, "text", "") or "", trans or "")
        erase_only = (
            block.effective_erase_only()
            if hasattr(block, 'effective_erase_only')
            else getattr(block, 'erase_only', False)
        )
        skip_reason = ""
        if not trans.strip():
            skip_reason = "empty_translation"
        elif erase_only:
            skip_reason = "erase_only"
        elif hasattr(block, 'effective_skip_typeset') and block.effective_skip_typeset():
            skip_reason = "explicit_skip_typeset"
        elif not _has_explicit_typeset_override(block) and _is_visual_sfx_like(block):
            skip_reason = _typeset_cleanup_skip_reason(block)
        if skip_reason:
            debug_print(f"[TYPESET_SKIP] preview region={idx} reason={skip_reason}")
            return _empty_preview_response(skip_reason)

        page = self.chapter_mgr.current_page
        sprite_y_offset = 0
        if bool(getattr(block, "cross_page", False)):
            page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
            bbox = tuple(int(v) for v in block.bbox())
            _composite, _offsets, pages, comp_bbox, _local = self._cross_page_context_for_bbox(page_idx, bbox)
            base_cv, render_offsets = self._build_stitched_render_base(pages or [page_idx])
            if base_cv is not None and page_idx in render_offsets:
                sprite_y_offset = int(render_offsets[page_idx])
                block.bbox_override = tuple(int(v) for v in comp_bbox)
                block.bubble_bbox = tuple(int(v) for v in comp_bbox)
                block.cross_page = False
                base_pil = Image.fromarray(cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB))
            elif page is not None and page.cleaned_cv is not None:
                base_pil = Image.fromarray(cv2.cvtColor(page.cleaned_cv, cv2.COLOR_BGR2RGB))
            elif self._raw_cv is not None:
                base_pil = Image.fromarray(cv2.cvtColor(self._raw_cv, cv2.COLOR_BGR2RGB))
            else:
                base_pil = Image.new("RGB", (1, 1), "white")
        elif page is not None and page.cleaned_cv is not None:
            base_pil = Image.fromarray(cv2.cvtColor(page.cleaned_cv, cv2.COLOR_BGR2RGB))
        elif self._raw_cv is not None:
            base_pil = Image.fromarray(cv2.cvtColor(self._raw_cv, cv2.COLOR_BGR2RGB))
        else:
            try:
                base_pil = Image.open(page.image_path).convert("RGB") if page else Image.new("RGB", (1, 1), "white")
            except Exception:
                base_pil = Image.new("RGB", (1, 1), "white")

        layer = Image.new("RGBA", base_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        bx, by, bw, bh = self._get_typeset_box(block)
        role = getattr(block, 'bubble_role', None) or "dialog"
        override_name = getattr(block, 'font_name', '')
        override_size = getattr(block, 'font_size', 0)
        manually_overridden = _has_explicit_typeset_override(block)
        style = block.effective_style() if hasattr(block, 'effective_style') else None
        if style is None:
            style = TextStyle(
                fg_color=getattr(block, 'fg_color', None) or (0, 0, 0),
                outline_color=getattr(block, 'outline_color', None) or (255, 255, 255),
                outline_width=max(1, int(getattr(block, 'outline_width', 1) or 1)),
                source="auto",
            )
        if role == "sfx" and sum(tuple(style.fg_color)) > 700 and sum(tuple(style.outline_color)) > 700:
            style.outline_color = (0, 0, 0)
            style.outline_width = max(2, int(getattr(style, "outline_width", 1) or 1))
            style.shadow_on = True
            style.shadow_color = (0, 0, 0)

        outline_w = int(style.outline_width) if style.outline_width else 1

        pad_x = max(8, int(round(bw * 0.06)))
        pad_y = max(8, int(round(bh * 0.06)))
        content_x = bx + pad_x
        content_y = by + pad_y
        content_w = max(1, bw - pad_x * 2)
        content_h = max(1, bh - pad_y * 2)

        if override_size and override_size > 0:
            font = self._load_fit_font(role, override_size, override_name)
            lines = self._wrap_text_for_width(trans, font, content_w)
            line_h = max(1, int(round(override_size * 1.15)))
            used_w, used_h, _ = self._measure_lines(draw, lines, font, line_h, outline_w)
            fit = _FitResult(
                lines=lines,
                font_size=override_size,
                used_width=used_w,
                used_height=used_h,
                overflow=used_w > content_w or used_h > content_h,
            )
        else:
            fit = self._fit_font_size(trans, content_w, content_h, role, override_name, outline_w)
            font = self._load_fit_font(role, fit.font_size, override_name)
        if not fit.lines:
            return _empty_preview_response("typeset_no_fit_lines")
        skip_fit, skip_reason = self._should_skip_auto_typeset_fit(
            block,
            fit,
            (bx, by, bw, bh),
            role,
            manually_overridden,
        )
        if skip_fit:
            return _empty_preview_response(skip_reason)

        line_h = max(1, int(round(fit.font_size * 1.15)))
        used_w, used_h, _union0 = self._measure_lines(draw, fit.lines, font, line_h, outline_w)
        cursor_y = content_y + max(0, int(round((content_h - used_h) / 2)))
        actual_union = None

        is_caption_box = getattr(getattr(block, "region_kind", None), "name", "") == "CAPTION_BOX"
        if getattr(style, 'plate_on', False) and is_caption_box:
            pad = getattr(style, 'plate_pad', 4)
            plate_bb = (
                int(content_x - pad), int(cursor_y - pad),
                int(content_x + used_w + pad), int(cursor_y + used_h + pad),
            )
            _render_plate(
                layer, *plate_bb,
                getattr(style, 'plate_color', (255, 255, 255)),
                getattr(style, 'plate_opacity', 0.78),
            )
            actual_union = plate_bb
        elif getattr(style, 'plate_on', False):
            render_debug(
                "preview.sprite",
                action="ignored_plate",
                region=idx,
                reason="normal_bubble",
                kind=getattr(getattr(block, "region_kind", None), "name", ""),
            )

        fg_color = style.fg_color
        if getattr(style, 'source', 'auto') == 'auto' and not getattr(style, 'gradient_on', False):
            fg_color = _check_text_contrast(fg_color, base_pil, content_x, cursor_y, used_w, used_h)

        class _EffStyle:
            pass
        eff = _EffStyle()
        eff.fg_color = fg_color
        eff.outline_color = style.outline_color
        eff.outline_width = outline_w
        eff.gradient_on = getattr(style, 'gradient_on', False)
        eff.gradient_start = getattr(style, 'gradient_start', fg_color)
        eff.gradient_end = getattr(style, 'gradient_end', fg_color)
        eff.gradient_angle = getattr(style, 'gradient_angle', 90)
        eff.shadow_on = getattr(style, 'shadow_on', False)
        eff.shadow_color = getattr(style, 'shadow_color', (0, 0, 0))
        eff.shadow_offset = getattr(style, 'shadow_offset', (1, 2))
        eff.shadow_opacity = getattr(style, 'shadow_opacity', 0.55)

        align = getattr(block, 'align', 'center') or 'center'
        for line_text in fit.lines:
            line_w, _line_h_px, line_bb = self._text_bbox_px(draw, line_text, font, outline_w)
            if align == "left":
                target_left = content_x
            elif align == "right":
                target_left = content_x + content_w - line_w
            else:
                target_left = content_x + int(round((content_w - line_w) / 2))
            x_pos = target_left - line_bb[0]
            y_pos = cursor_y - line_bb[1]
            rendered_bb = (
                int(x_pos + line_bb[0]), int(y_pos + line_bb[1]),
                int(x_pos + line_bb[2]), int(y_pos + line_bb[3]),
            )
            actual_union = rendered_bb if actual_union is None else (
                min(actual_union[0], rendered_bb[0]),
                min(actual_union[1], rendered_bb[1]),
                max(actual_union[2], rendered_bb[2]),
                max(actual_union[3], rendered_bb[3]),
            )
            _draw_line_with_style(layer, draw, line_text, x_pos, y_pos, font, eff, outline_w)  # type: ignore[arg-type]
            cursor_y += line_h

        if actual_union is None:
            return {"b64": None, "x": 0, "y": 0, "w": 0, "h": 0}
        shadow = getattr(eff, 'shadow_offset', (0, 0)) if getattr(eff, 'shadow_on', False) else (0, 0)
        extra = max(4, outline_w * 3, abs(int(shadow[0])) + 4, abs(int(shadow[1])) + 4)
        x1 = max(0, int(actual_union[0]) - extra)
        y1 = max(0, int(actual_union[1]) - extra)
        x2 = min(layer.width, int(actual_union[2]) + extra)
        y2 = min(layer.height, int(actual_union[3]) + extra)
        if x1 >= x2 or y1 >= y2:
            return {"b64": None, "x": 0, "y": 0, "w": 0, "h": 0}

        sprite = layer.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        sprite.save(buf, format="PNG", optimize=False)
        font_name = getattr(font, "path", "") or (override_name or role)
        return {
            "b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "x": int(x1), "y": int(y1 - sprite_y_offset), "w": int(x2 - x1), "h": int(y2 - y1),
            "bbox": [int(bx), int(by - sprite_y_offset), int(bw), int(bh)],
            "resolved_font_size": int(fit.font_size),
            "line_count": int(len(fit.lines)),
            "overflow": bool(fit.overflow),
            "font": os.path.basename(str(font_name)) if font_name else str(role),
            "role": str(role),
            "fg": _hex_color(fg_color, "#111111"),
            "outline": _hex_color(getattr(eff, 'outline_color', None), "#ffffff"),
            "outline_width": int(outline_w),
            "shadow": _hex_color(getattr(eff, 'shadow_color', None), "#000000"),
            "shadow_on": bool(getattr(eff, 'shadow_on', False)),
            "align": str(align),
        }

    def list_font_options(self) -> Dict[str, List[str]]:
        roles = ["auto", "dialog", "bold", "thought", "sfx"]
        fonts = self.font_lib.list_fonts() if self.font_lib else []
        return {"roles": roles, "fonts": [str(name) for name in fonts]}

    @staticmethod
    def _font_visible_sanity(font: ImageFont.ImageFont, sample: str = "SFX") -> Tuple[bool, str]:
        try:
            img = Image.new("L", (256, 96), 0)
            draw = ImageDraw.Draw(img)
            draw.text((8, 8), sample, font=font, fill=255)
            arr = np.array(img)
            ink = int(np.count_nonzero(arr))
            if ink < 12:
                return False, "empty_glyph_bbox"
            ys, xs = np.where(arr > 0)
            if xs.size == 0 or ys.size == 0:
                return False, "empty_glyph_bbox"
            w = int(xs.max() - xs.min() + 1)
            h = int(ys.max() - ys.min() + 1)
            if w <= 2 or h <= 2:
                return False, "glyph_too_small"
            if ink / max(1, w * h) > 0.92:
                return False, "solid_or_tofu_like_glyph"
            return True, "ok"
        except Exception as exc:
            return False, f"render_error:{exc}"

    def _load_fit_font(self, role: str, size: int, override_name: str = '') -> ImageFont.ImageFont:
        lib = self.font_lib or ComicFontLibrary("")
        override = str(override_name or "").strip()
        if override and override.lower() != "auto":
            role_override = override.lower()
            if role_override in {"dialog", "bold", "thought", "sfx"}:
                font = lib.get_native(role_override, size)
            else:
                font = lib.get_by_name(override, size)
        else:
            font = lib.get(role, size)
        if role == "sfx":
            ok, reason = self._font_visible_sanity(font)
            if ok:
                debug_print(f"SFX font selected: size={size} sanity={reason}")
            else:
                debug_print(f"SFX font fallback: reason={reason} size={size}")
                font = lib.get("bold", size)
        return font

    def _fit_font_size(self, text: str, box_w: int, box_h: int,
                       role: str, override_name: str = '',
                       stroke_w: int = 1) -> _FitResult:
        text = (text or '').strip()
        # Caller now passes content_w / content_h (padding already stripped), so use
        # box_w / box_h directly.  The old -16 conflicted with the percentage-based
        # padding applied at the draw site, causing the fitter to work against a wider
        # area than the region text is actually drawn in.
        usable_w = max(1, int(box_w))
        usable_h = max(1, int(box_h))
        if not text:
            return _FitResult(lines=[], font_size=8, used_width=0, used_height=0, overflow=False)

        probe_img = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(probe_img)
        MIN_SIZE = 8
        MAX_SIZE = 72 if str(role or "").lower() == "sfx" else 56
        size = max(MIN_SIZE, min(MAX_SIZE, int(min(48, max(12, usable_h * 0.28, usable_w * 0.12)))))
        best_fit: _FitResult | None = None
        best_overflow: _FitResult | None = None

        for step in range(8):
            font = self._load_fit_font(role, size, override_name)
            lines = self._wrap_text_for_width(text, font, usable_w)
            line_h = max(1, int(round(size * 1.15)))
            used_w, used_h, _union = self._measure_lines(draw, lines, font, line_h, stroke_w)
            overflow = used_w > usable_w or used_h > usable_h
            result = _FitResult(
                lines=lines,
                font_size=size,
                used_width=used_w,
                used_height=used_h,
                overflow=overflow,
            )
            scale_w = usable_w / max(1, used_w)
            scale_h = usable_h / max(1, used_h)
            scale_ratio = min(scale_w, scale_h)
            debug_print(
                f"[FIT_STEP] role={role} text={text!r} step={step} size={size} "
                f"box={(box_w, box_h)} usable={(usable_w, usable_h)} used={(used_w, used_h)} "
                f"scale=({scale_w:.3f},{scale_h:.3f})->{scale_ratio:.3f} overflow={overflow} lines={lines}"
            )
            if not overflow:
                if best_fit is None or result.font_size > best_fit.font_size:
                    best_fit = result
            else:
                if best_overflow is None or result.font_size < best_overflow.font_size or best_overflow.used_height * best_overflow.used_width > used_w * used_h:
                    best_overflow = result

            new_size = max(MIN_SIZE, min(MAX_SIZE, int(round(size * scale_ratio * 0.97))))
            if abs(new_size - size) <= 1:
                break
            size = new_size

        center = best_fit.font_size if best_fit is not None else (best_overflow.font_size if best_overflow is not None else size)
        for cand in range(max(MIN_SIZE, center - 6), min(MAX_SIZE, center + 6) + 1):
            font = self._load_fit_font(role, cand, override_name)
            lines = self._wrap_text_for_width(text, font, usable_w)
            line_h = max(1, int(round(cand * 1.15)))
            used_w, used_h, _union = self._measure_lines(draw, lines, font, line_h, stroke_w)
            overflow = used_w > usable_w or used_h > usable_h
            result = _FitResult(lines=lines, font_size=cand, used_width=used_w, used_height=used_h, overflow=overflow)
            debug_print(
                f"[FIT_REFINE] role={role} text={text!r} size={cand} used={(used_w, used_h)} "
                f"overflow={overflow} lines={lines}"
            )
            if not overflow:
                if best_fit is None or cand > best_fit.font_size:
                    best_fit = result
            else:
                if best_overflow is None:
                    best_overflow = result

        final = best_fit or best_overflow or _FitResult(lines=[text], font_size=MIN_SIZE, used_width=usable_w, used_height=usable_h, overflow=True)
        if final.font_size >= MAX_SIZE and str(role or "").lower() != "sfx":
            debug_print(f"[FIT_CAP] role={role} max_size={MAX_SIZE} text={text!r}")
        debug_print(
            f"[FIT_FINAL] role={role} text={text!r} box={(box_w, box_h)} usable={(usable_w, usable_h)} "
            f"chosen_size={final.font_size} used={(final.used_width, final.used_height)} overflow={final.overflow} lines={final.lines}"
        )
        return final

    def run_current_page_steps(self) -> dict:
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._set_progress(
            running=True,
            job="run_page",
            stage="detect",
            page_idx=page_idx,
            page_total=self.chapter_mgr.total_pages(),
            updated_pages=[],
        )
        self._begin_active_operation("run_page")
        try:
            self.detect_current_page()
            step_region_version = int(getattr(self, "_region_mutation_version", 0) or 0)
            self.ocr_current_page()
            if int(getattr(self, "_region_mutation_version", 0) or 0) != step_region_version:
                debug_print("[RUN_PAGE_ABORT] reason=regions_mutated_during_ocr")
                return self.get_bootstrap()
            self.translate_current_page()
            if int(getattr(self, "_region_mutation_version", 0) or 0) != step_region_version:
                debug_print("[RUN_PAGE_ABORT] reason=regions_mutated_during_translate")
                return self.get_bootstrap()
            self.cleanup_current_page()
            if int(getattr(self, "_region_mutation_version", 0) or 0) != step_region_version:
                debug_print("[RUN_PAGE_ABORT] reason=regions_mutated_during_cleanup")
                return self.get_bootstrap()
            self.typeset_current_page()
            self.chapter_mgr.save_state()
            self._notify(f"Run Page complete — Page {page_idx + 1}", 1, 1, running=False, updated_pages=[page_idx])
            bootstrap = self.get_bootstrap()
            bootstrap["processedPageIdx"] = page_idx
            return bootstrap
        finally:
            self._end_active_operation("run_page")

    def detect_all_pages(self) -> dict:
        if not self.chapter_mgr.pages:
            raise RuntimeError("No chapter loaded.")
        start_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        total = self.chapter_mgr.total_pages()
        updated_pages: List[int] = []
        self._set_progress(running=True, job="detect_all", stage="detect", page_idx=start_idx, page_total=total, updated_pages=[])
        self._begin_active_operation("detect_all")
        try:
            for idx in range(total):
                self._set_progress(running=True, job="detect_all", stage="detect", page_idx=idx, page_total=total, updated_pages=[])
                self._notify(f"Detect All: Page {idx + 1}/{total}...", idx, total)
                self.go_to_page(idx)
                self.detect_current_page()
                self.chapter_mgr.save_state()
                updated_pages.append(idx)
            self.go_to_page(min(start_idx, max(0, total - 1)))
            self._notify(f"Detect All complete - {total}/{total} pages.", total, total, running=False, updated_pages=updated_pages)
            bootstrap = self.get_bootstrap()
            bootstrap["processedPageIdx"] = start_idx
            bootstrap["updatedPages"] = updated_pages
            return bootstrap
        finally:
            self._end_active_operation("detect_all")

    def run_all_steps(self) -> dict:
        return self._run_all_steps_from(None, 0)

    def continue_run_all_steps(self) -> dict:
        checkpoint = getattr(self.chapter_mgr, "run_all_checkpoint", {}) or {}
        if not checkpoint:
            raise RuntimeError("No Run All checkpoint to continue.")
        phase = str(checkpoint.get("phase", "detect") or "detect")
        try:
            page_idx = int(checkpoint.get("page_idx", 0) or 0)
        except Exception:
            page_idx = 0
        return self._run_all_steps_from(phase, page_idx)

    def _run_all_steps_from(self, start_phase: Optional[str], start_page_idx: int) -> dict:
        if not self.chapter_mgr.pages:
            raise RuntimeError("No chapter loaded.")
        start_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        total = self.chapter_mgr.total_pages()
        updated_pages: List[int] = []
        phases = ["detect", "ocr", "translate", "cleanup", "typeset"]
        if start_phase not in phases:
            start_phase = "detect"
            start_page_idx = 0
            if hasattr(self.chapter_mgr, "clear_run_all_checkpoint"):
                self.chapter_mgr.clear_run_all_checkpoint()
        start_phase_pos = phases.index(start_phase)
        start_page_idx = max(0, min(total - 1, int(start_page_idx or 0)))
        self._set_progress(running=True, job="run_all", stage=start_phase, page_idx=start_page_idx, page_total=total, updated_pages=[])
        self._begin_active_operation("run_all")
        try:
            def mark_checkpoint(phase: str, idx: int, next_phase: Optional[str] = None) -> None:
                if next_phase is None:
                    next_phase = phase
                    next_idx = min(total - 1, idx + 1)
                    if idx + 1 >= total:
                        phase_pos = phases.index(phase)
                        next_phase = phases[min(len(phases) - 1, phase_pos + 1)]
                        next_idx = 0
                else:
                    next_idx = 0
                if phase == "typeset" and idx + 1 >= total:
                    return
                if hasattr(self.chapter_mgr, "save_run_all_checkpoint"):
                    self.chapter_mgr.save_run_all_checkpoint({
                        "active": True,
                        "phase": next_phase,
                        "page_idx": int(next_idx),
                        "page_total": int(total),
                        "message": f"Continue Run All from {next_phase} page {next_idx + 1}/{total}",
                    })

            def phase_start(phase: str) -> int:
                return start_page_idx if phases.index(phase) == start_phase_pos else 0

            if start_phase_pos <= phases.index("detect"):
                for idx in range(phase_start("detect"), total):
                    self._set_progress(running=True, job="run_all", stage="detect", page_idx=idx, page_total=total, updated_pages=[])
                    self._notify(f"Run All Detect: Page {idx + 1}/{total}...", idx, total)
                    self.go_to_page(idx)
                    self.detect_current_page()
                    self._notify(f"Run All Detect: checkpointing page {idx + 1}/{total}...", idx + 1, total)
                    self.chapter_mgr.save_state()
                    mark_checkpoint("detect", idx)
            if start_phase_pos <= phases.index("ocr"):
                for idx in range(phase_start("ocr"), total):
                    self._set_progress(running=True, job="run_all", stage="ocr", page_idx=idx, page_total=total, updated_pages=[])
                    self._notify(f"Run All OCR: Page {idx + 1}/{total}...", idx, total)
                    self.go_to_page(idx)
                    self.ocr_current_page()
                    self._notify(f"Run All OCR: checkpointing page {idx + 1}/{total}...", idx + 1, total)
                    self.chapter_mgr.save_state()
                    mark_checkpoint("ocr", idx)
            if start_phase_pos <= phases.index("translate"):
                for idx in range(phase_start("translate"), total):
                    self._set_progress(running=True, job="run_all", stage="translate", page_idx=idx, page_total=total, updated_pages=[])
                    self._notify(f"Run All Translate: Page {idx + 1}/{total}...", idx, total)
                    self.go_to_page(idx)
                    self.translate_current_page()
                    self._notify(f"Run All Translate: checkpointing page {idx + 1}/{total}...", idx + 1, total)
                    self.chapter_mgr.save_state()
                    mark_checkpoint("translate", idx)
            if start_phase_pos <= phases.index("cleanup"):
                for idx in range(phase_start("cleanup"), total):
                    self._set_progress(running=True, job="run_all", stage="cleanup", page_idx=idx, page_total=total, updated_pages=[])
                    self._notify(f"Run All Cleanup: Page {idx + 1}/{total}...", idx, total)
                    self.go_to_page(idx)
                    self.cleanup_current_page()
                    mark_checkpoint("cleanup", idx)
            if start_phase_pos <= phases.index("typeset"):
                for idx in range(phase_start("typeset"), total):
                    self._set_progress(running=True, job="run_all", stage="typeset", page_idx=idx, page_total=total, updated_pages=[])
                    self._notify(f"Run All Typeset: Page {idx + 1}/{total}...", idx, total)
                    self.go_to_page(idx)
                    self.typeset_current_page()
                    updated_pages.append(idx)
                    mark_checkpoint("typeset", idx)
                    self._notify(f"Run All: Page {idx + 1}/{total} complete", idx + 1, total, updated_pages=[idx])
            if hasattr(self.chapter_mgr, "clear_run_all_checkpoint"):
                self.chapter_mgr.clear_run_all_checkpoint()
            self.go_to_page(min(start_idx, max(0, total - 1)))
            self._notify(f"Run complete - {total}/{total} pages.", total, total, running=False, updated_pages=updated_pages)
            bootstrap = self.get_bootstrap()
            bootstrap["processedPageIdx"] = start_idx
            bootstrap["updatedPages"] = updated_pages
            return bootstrap
        finally:
            self._end_active_operation("run_all")

    def export_chapter(self, export_dir: Optional[str] = None) -> str:
        if not self.chapter_mgr.pages:
            raise RuntimeError("No chapter loaded.")
        if export_dir is None:
            export_dir = self.chapter_mgr.export_dir or os.path.join(os.getcwd(), "translated")
        os.makedirs(export_dir, exist_ok=True)

        import shutil
        total = self.chapter_mgr.total_pages()
        saved = 0
        for i, page in enumerate(self.chapter_mgr.pages):
            self._notify(f"Exporting {i+1}/{total}…", i, total)
            base = os.path.splitext(os.path.basename(page.image_path))[0]
            if page.typeset_pil:
                page.typeset_pil.save(os.path.join(export_dir, f"{base}_translated.png"))
            elif page.cleaned_cv is not None:
                cv2.imwrite(os.path.join(export_dir, f"{base}_cleaned.png"), page.cleaned_cv)
            else:
                shutil.copy2(page.image_path,
                             os.path.join(export_dir, os.path.basename(page.image_path)))
            saved += 1
        self.chapter_mgr.save_state()
        self._notify(f"Exported {saved}/{total} pages to {os.path.basename(export_dir)}",
                     saved, total)
        return export_dir

    # ── Region editing ──────────────────────────────────────────────────────

    def _clamp_bbox(self, x: Any, y: Any, w: Any, h: Any) -> Tuple[int, int, int, int]:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded.")
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        pages = getattr(self.chapter_mgr, "pages", []) or []
        img_h, img_w = self._raw_cv.shape[:2]
        min_w = min(8, max(1, int(img_w)))
        min_h = min(8, max(1, int(img_h)))
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        iw = max(min_w, int(round(float(w))))
        ih = max(min_h, int(round(float(h))))
        iw = min(iw, max(1, int(img_w)))
        max_cross_h = int(img_h)
        if page_idx > 0:
            prev_img = cv2.imread(pages[page_idx - 1].image_path) if page_idx - 1 < len(pages) else None
            if prev_img is not None:
                max_cross_h += int(prev_img.shape[0])
        if page_idx + 1 < len(pages):
            next_img = cv2.imread(pages[page_idx + 1].image_path)
            if next_img is not None:
                max_cross_h += int(next_img.shape[0])
        ih = min(ih, max(1, max_cross_h))
        ix = max(0, min(ix, max(0, int(img_w) - iw)))
        min_y = 0
        max_y = max(0, int(img_h) - min_h)
        if page_idx > 0:
            prev_img = cv2.imread(pages[page_idx - 1].image_path) if page_idx - 1 < len(pages) else None
            if prev_img is not None:
                min_y = -int(prev_img.shape[0])
        if page_idx + 1 < len(pages):
            max_y = int(img_h) - min_h
        iy = max(min_y, min(iy, max_y))
        if iy + ih < min_h:
            ih = min(max_cross_h, min_h - iy)
        if iy > int(img_h) - min_h:
            iy = int(img_h) - min_h
        return ix, iy, iw, ih

    @staticmethod
    def _bbox_to_box(x: int, y: int, w: int, h: int) -> List[List[float]]:
        return [
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ]

    @staticmethod
    def _parse_hex_color(value: Any) -> Tuple[int, int, int]:
        text = str(value or "").strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6:
            raise ValueError(f"Invalid color: {value!r}")
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))

    def _invalidate_page_outputs(self, preserve_cleanup: bool = False) -> None:
        page = self.chapter_mgr.current_page
        if page is not None:
            before = {
                "dirty": bool(getattr(page, "render_dirty", False)),
                "has_cleaned": page.cleaned_cv is not None,
                "has_typeset": page.typeset_pil is not None,
            }
            page.typeset_pil = None
            if preserve_cleanup:
                page.render_dirty = True
            else:
                page.cleaned_cv = None
                page.cleanup_patches = []
                page.render_dirty = False
                idx = getattr(self.chapter_mgr, "current_idx", -1)
                if idx >= 0:
                    self.chapter_mgr.delete_artifact(idx, "cleaned")
                    self.chapter_mgr.delete_artifact(idx, "typeset")
            page.bump_render_version()
            render_debug(
                "invalidate",
                action="invalidate",
                preserve_cleanup=preserve_cleanup,
                before=before,
                **self._debug_page_state(),
            )
        for block in self._regions:
            if hasattr(block, "typeset_overflow"):
                block.typeset_overflow = False

    def _touch_region_geometry(self, block: OCRBlock, bbox: Tuple[int, int, int, int]) -> None:
        x, y, w, h = bbox
        block.bbox_override = bbox
        block.bubble_bbox = bbox
        block.manually_adjusted = True
        block.bubble_mask = build_ellipse_mask(w, h, inset=4)
        block.text_mask = None
        block.safe_text_mask = None
        block.safe_center = None
        block.safe_rect = None
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._update_cross_page_metadata(block, page_idx, bbox)
        if self._raw_cv is not None and not bool(getattr(block, "cross_page", False)):
            try:
                compute_placement(self._raw_cv, block)
            except Exception as exc:
                debug_print(f"region geometry refresh failed: {exc}")

    def update_region_field(self, region_idx: int, field: str, value: Any) -> dict:
        self._begin_active_operation("mutate")
        try:
            return self._update_region_field_active(region_idx, field, value)
        finally:
            self._end_active_operation("mutate")

    def _update_region_field_active(self, region_idx: int, field: str, value: Any) -> dict:
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        block = self._regions[region_idx]
        if field == "cleanup_override":
            existing = (
                block.override.to_dict()
                if getattr(block, "override", None) is not None
                else {}
            )
            next_value = value if isinstance(value, dict) else {}
        elif field == "reset_cleanup_override":
            existing = (
                block.override.to_dict()
                if getattr(block, "override", None) is not None
                else {}
            )
            next_value = {}
        elif field == "translation":
            existing = self._translations[region_idx] if region_idx < len(self._translations) else ""
            next_value = sanitize_final_translation(block.text, str(value or "")) if str(value or "").strip() else ""
        elif field == "font_size":
            existing = int(getattr(block, "font_size", 0) or 0)
            next_value = max(0, min(96, int(float(value or 0))))
        elif field in {"font_name", "text", "align"}:
            existing = str(getattr(block, field if field != "text" else "text", "") or "")
            next_value = str(value or "")
            if field == "align":
                next_value = next_value if next_value in {"left", "center", "right"} else "center"
        elif field in {"visible", "locked", "shadow_on"}:
            existing = bool(getattr(block, field, False))
            next_value = bool(value)
        else:
            existing = object()
            next_value = None
        if existing == next_value:
            render_debug("mutation", action="update_region_field_skipped", region=region_idx, field=field, reason="unchanged", **self._debug_page_state())
            return self.get_bootstrap()

        self._push_undo_snapshot()
        before_style = self._debug_region_style(block)
        if field == "cleanup_override":
            allowed = {
                "cleanup_override_mode",
                "cleanup_region_class",
                "cleanup_halo_max_px",
                "cleanup_residual_retry_enabled",
                "cleanup_residual_retry_dilate_px",
                "cleanup_min_container_confidence",
                "cleanup_max_mask_container_ratio",
                "cleanup_max_mask_region_ratio",
                "cleanup_max_border_touch_ratio",
                "cleanup_max_rectangularity",
                "cleanup_allow_low_confidence",
                "cleanup_allow_texture_inpaint",
                "cleanup_allow_translucent_caption",
            }
            if block.override is None:
                block.override = RegionOverride()
            for key, raw in (value if isinstance(value, dict) else {}).items():
                if key not in allowed or not hasattr(block.override, key):
                    continue
                if raw == "" or raw is None:
                    setattr(block.override, key, None)
                elif key in {
                    "cleanup_halo_max_px",
                    "cleanup_residual_retry_dilate_px",
                }:
                    setattr(block.override, key, int(float(raw)))
                elif key in {
                    "cleanup_min_container_confidence",
                    "cleanup_max_mask_container_ratio",
                    "cleanup_max_mask_region_ratio",
                    "cleanup_max_border_touch_ratio",
                    "cleanup_max_rectangularity",
                }:
                    setattr(block.override, key, float(raw))
                elif key in {
                    "cleanup_residual_retry_enabled",
                    "cleanup_allow_low_confidence",
                    "cleanup_allow_texture_inpaint",
                    "cleanup_allow_translucent_caption",
                }:
                    setattr(block.override, key, bool(raw))
                else:
                    setattr(block.override, key, str(raw))
            if block.override is not None and block.override.is_empty():
                block.override = None
            self._invalidate_page_outputs(preserve_cleanup=False)
        elif field == "reset_cleanup_override":
            if block.override is not None:
                for key in (
                    "cleanup_override_mode",
                    "cleanup_region_class",
                    "cleanup_halo_max_px",
                    "cleanup_residual_retry_enabled",
                    "cleanup_residual_retry_dilate_px",
                    "cleanup_min_container_confidence",
                    "cleanup_max_mask_container_ratio",
                    "cleanup_max_mask_region_ratio",
                    "cleanup_max_border_touch_ratio",
                    "cleanup_max_rectangularity",
                    "cleanup_allow_low_confidence",
                    "cleanup_allow_texture_inpaint",
                    "cleanup_allow_translucent_caption",
                ):
                    setattr(block.override, key, None)
                if block.override.is_empty():
                    block.override = None
            self._invalidate_page_outputs(preserve_cleanup=False)
        elif field == "translation":
            while len(self._translations) <= region_idx:
                self._translations.append("")
            raw_value = str(value or "")
            self._translations[region_idx] = (
                sanitize_final_translation(block.text, raw_value)
                if raw_value.strip()
                else ""
            )
            block.typeset_override = bool(raw_value.strip())
            self._invalidate_page_outputs(preserve_cleanup=True)
        elif field in {"text", "font_name", "font_size", "align", "visible", "locked", "fg_color", "bg_color", "outline_color", "outline_width", "shadow_color", "shadow_on"}:
            if field == "bg_color" and getattr(getattr(block, "region_kind", None), "name", "") != "CAPTION_BOX":
                render_debug(
                    "style_migration",
                    action="ignored_bg_for_cleanup",
                    region=region_idx,
                    bg=value,
                    reason="normal_bubble",
                    **self._debug_page_state(),
                )
                return self.get_bootstrap()
            if field == "font_size":
                block.font_size = max(0, min(96, int(float(value or 0))))
            elif field in {"visible", "locked"}:
                setattr(block, field, bool(value))
            elif field in {"fg_color", "bg_color", "outline_color", "shadow_color"}:
                color = self._parse_hex_color(value)
                if field == "shadow_color":
                    if block.style is None:
                        block.style = block.effective_style()
                    block.style.shadow_color = color
                    block.style.source = "manual"
                else:
                    setattr(block, field, color)
                    if field in {"fg_color", "outline_color"}:
                        if block.style is None:
                            block.style = block.effective_style()
                        setattr(block.style, field, color)
                        block.style.source = "manual"
            elif field == "outline_width":
                block.outline_width = max(0, min(8, int(float(value or 0))))
                if block.style is None:
                    block.style = block.effective_style()
                block.style.outline_width = block.outline_width
                block.style.source = "manual"
            elif field == "shadow_on":
                if block.style is None:
                    block.style = block.effective_style()
                block.style.shadow_on = bool(value)
                block.style.source = "manual"
            elif field == "align":
                block.align = str(value or "center") if str(value or "center") in {"left", "center", "right"} else "center"
            else:
                setattr(block, field, str(value or ""))
            if field == "text":
                self._memory_hits.pop(region_idx, None)
                block.ocr_status = "ok" if str(value or "").strip() else ""
                block.ocr_status_reason = "manual_edit" if str(value or "").strip() else ""
            if field == "translation" or (field == "text" and str(value or "").strip()):
                block.typeset_override = True
            self._invalidate_page_outputs(preserve_cleanup=(field != "visible"))
        render_debug(
            "mutation",
            action="update_region_field",
            region=region_idx,
            field=field,
            beforeStyle=before_style,
            afterStyle=self._debug_region_style(block),
            **self._debug_page_state(),
        )
        self._flush_working_state_to_page()
        self.chapter_mgr.request_save_state()
        return self.get_bootstrap()

    def update_region_bbox(self, region_idx: int, x: Any, y: Any, w: Any, h: Any) -> dict:
        self._begin_active_operation("mutate")
        try:
            return self._update_region_bbox_active(region_idx, x, y, w, h)
        finally:
            self._end_active_operation("mutate")

    def _update_region_bbox_active(self, region_idx: int, x: Any, y: Any, w: Any, h: Any) -> dict:
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        bbox = self._clamp_bbox(x, y, w, h)
        block = self._regions[region_idx]
        current_bbox = tuple(int(v) for v in block.bbox())
        if current_bbox == bbox:
            render_debug(
                "mutation",
                action="update_region_bbox_skipped",
                region=region_idx,
                reason="unchanged",
                bbox=bbox,
                **self._debug_page_state(),
            )
            return self.get_bootstrap()
        self._push_undo_snapshot()
        before_style = self._debug_region_style(block)
        self._touch_region_geometry(block, bbox)
        self._bump_region_mutation_version()
        self._invalidate_page_outputs(preserve_cleanup=True)
        render_debug(
            "mutation",
            action="update_region_bbox",
            region=region_idx,
            beforeStyle=before_style,
            afterStyle=self._debug_region_style(block, bbox),
            **self._debug_page_state(),
        )
        self._flush_working_state_to_page()
        self.chapter_mgr.request_save_state()
        return self.get_bootstrap()

    def add_region(self, x: Any, y: Any, w: Any, h: Any, text: str = "") -> dict:
        self._begin_active_operation("mutate")
        try:
            return self._add_region_active(x, y, w, h, text)
        finally:
            self._end_active_operation("mutate")

    def _add_region_active(self, x: Any, y: Any, w: Any, h: Any, text: str = "") -> dict:
        self._push_undo_snapshot()
        bbox = self._clamp_bbox(x, y, w, h)
        bx, by, bw, bh = bbox
        block = OCRBlock(
            text=str(text or ""),
            boxes=[],
            confidence=1.0 if text else 0.0,
            detector_source="manual",
            bubble_bbox=bbox,
            bubble_mask=build_ellipse_mask(bw, bh, inset=4),
        )
        self._touch_region_geometry(block, bbox)
        self._regions.append(block)
        self._translations.append("")
        self._bump_region_mutation_version()
        self._invalidate_page_outputs(preserve_cleanup=False)
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        return self.get_bootstrap()

    def _yolo_finetune_dir(self) -> str:
        base = self.chapter_mgr.chapter_dir or os.getcwd()
        out_dir = os.path.join(base, "yolo_finetune")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _normalize_yolo_reject_reason(reason: str) -> str:
        key = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {
            "not_sfx",
            "not_caption",
            "not_bubble",
            "wrongly_detected_art",
            "exclamation_questionmark",
        }
        return key if key in allowed else ""

    @staticmethod
    def _yolo_class_for_block(block: Any) -> int:
        try:
            train_class_id = int(getattr(block, "yolo_train_class_id", -1))
            if train_class_id in {0, 1, 2, 3}:
                return train_class_id
        except Exception:
            pass
        try:
            class_id = int(getattr(block, "yolo_class_id", -1))
            if class_id in {0, 1, 2, 3}:
                return class_id
        except Exception:
            pass
        kind = str(getattr(block, "yolo_kind", "") or "").strip().lower()
        role = str(getattr(block, "bubble_role", "") or "").strip().lower()
        if kind == "narration" or role in {"caption", "narration", "thought"}:
            return 1
        if kind == "sfx" or role in {"sfx", "sound", "sound_effect", "impact"}:
            return 2
        if kind == "shout" or role in {"shout", "bold"}:
            return 3
        return 0

    def set_yolo_train_class(self, region_idx: int, class_id: int) -> dict:
        if not (0 <= int(region_idx) < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        class_id = int(class_id)
        if class_id not in {0, 1, 2, 3}:
            raise ValueError(f"Unknown YOLO training class: {class_id}")
        self._push_undo_snapshot()
        block = self._regions[int(region_idx)]
        setattr(block, "yolo_train_class_id", class_id)
        class_kind = {0: "dialogue", 1: "narration", 2: "sfx", 3: "shout"}[class_id]
        setattr(block, "yolo_kind", class_kind)
        setattr(block, "yolo_class_id", class_id)
        if class_id == 2:
            block.bubble_role = "sfx"
            block.region_kind = RegionKind.SFX_OVER_ART
            block.background_kind = BackgroundKind.ART
        elif class_id == 3:
            block.bubble_role = "bold"
        elif class_id == 1:
            block.bubble_role = "thought"
            block.region_kind = RegionKind.CAPTION_BOX
        else:
            block.bubble_role = "dialog"
        self._invalidate_page_outputs(preserve_cleanup=False)
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        return self.get_bootstrap()

    def _record_yolo_negative(self, region_idx: int, block: Any, reason: str) -> None:
        reason_key = self._normalize_yolo_reject_reason(reason)
        if not reason_key:
            return
        page = self.chapter_mgr.current_page
        if page is None:
            return
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        x, y, w, h = [int(v) for v in block.bbox()]
        img_w = img_h = 0
        try:
            if self._raw_cv is not None:
                img_h, img_w = self._raw_cv.shape[:2]
        except Exception:
            img_w = img_h = 0
        record = {
            "page_index": page_idx,
            "image_path": str(getattr(page, "image_path", "") or ""),
            "region_index": int(region_idx),
            "bbox": [x, y, w, h],
            "image_size": [int(img_w), int(img_h)] if img_w and img_h else None,
            "detected_class_id": self._yolo_class_for_block(block),
            "yolo_train_class_id": getattr(block, "yolo_train_class_id", None),
            "detected_kind": str(getattr(block, "yolo_kind", "") or ""),
            "role": str(getattr(block, "bubble_role", "") or ""),
            "reason": reason_key,
            "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path = os.path.join(self._yolo_finetune_dir(), "negative_corrections.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_yolo_finetune_dataset(self) -> Dict[str, Any]:
        if not self.chapter_mgr.pages:
            raise RuntimeError("No chapter loaded.")
        out_dir = self._yolo_finetune_dir()
        images_dir = os.path.join(out_dir, "images")
        labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        manifest: Dict[str, Any] = {
            "classes": ["dialogue", "narration", "sfx", "shout"],
            "negative_corrections": "negative_corrections.jsonl",
            "pages": [],
        }
        for page_idx, page in enumerate(self.chapter_mgr.pages):
            image_path = str(getattr(page, "image_path", "") or "")
            if not image_path or not os.path.exists(image_path):
                continue
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            stem = f"page_{page_idx:04d}"
            ext = os.path.splitext(image_path)[1].lower() or ".png"
            image_name = f"{stem}{ext}"
            label_name = f"{stem}.txt"
            shutil.copy2(image_path, os.path.join(images_dir, image_name))
            lines: List[str] = []
            box_records: List[Dict[str, Any]] = []
            resized_boxes = 0
            for block in getattr(page, "regions", []) or []:
                try:
                    x, y, w, h = [float(v) for v in block.bbox()]
                except Exception:
                    continue
                if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                    continue
                x1 = max(0.0, min(float(img_w), x))
                y1 = max(0.0, min(float(img_h), y))
                x2 = max(0.0, min(float(img_w), x + w))
                y2 = max(0.0, min(float(img_h), y + h))
                cw = x2 - x1
                ch = y2 - y1
                if cw <= 1.0 or ch <= 1.0:
                    continue
                cx = (x1 + cw / 2.0) / float(img_w)
                cy = (y1 + ch / 2.0) / float(img_h)
                nw = cw / float(img_w)
                nh = ch / float(img_h)
                vals = [max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh)]
                class_id = self._yolo_class_for_block(block)
                lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in vals))
                manually_adjusted = bool(getattr(block, "manually_adjusted", False))
                has_override = getattr(block, "bbox_override", None) is not None
                if manually_adjusted:
                    resized_boxes += 1
                box_records.append({
                    "class_id": class_id,
                    "class_name": manifest["classes"][class_id] if 0 <= class_id < len(manifest["classes"]) else str(class_id),
                    "bbox": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
                    "clipped_bbox": [int(round(x1)), int(round(y1)), int(round(cw)), int(round(ch))],
                    "normalized": [round(float(v), 6) for v in vals],
                    "manually_adjusted": manually_adjusted,
                    "bbox_source": "manual_resize" if manually_adjusted else ("bbox_override" if has_override else "detector"),
                    "detector_text_bbox": (
                        [int(v) for v in getattr(block, "detector_text_bbox", None)]
                        if getattr(block, "detector_text_bbox", None) else None
                    ),
                    "yolo_train_class_id": getattr(block, "yolo_train_class_id", None),
                })
            with open(os.path.join(labels_dir, label_name), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            manifest["pages"].append({
                "page_index": page_idx,
                "image": f"images/{image_name}",
                "label": f"labels/{label_name}",
                "positive_boxes": len(lines),
                "resized_boxes": resized_boxes,
                "boxes": box_records,
            })
        manifest_path = os.path.join(out_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        return {"ok": True, "dataset_dir": out_dir, "manifest": manifest_path, "pages": len(manifest["pages"])}

    def _yolo_training_status_path(self) -> str:
        return os.path.join(self._yolo_finetune_dir(), "training_status.json")

    def get_yolo_training_status(self) -> Dict[str, Any]:
        status_path = self._yolo_training_status_path()
        status: Dict[str, Any] = {"status": "idle", "running": False, "status_path": status_path}
        if os.path.exists(status_path):
            try:
                with open(status_path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    status.update(loaded)
            except Exception as exc:
                status.update({"status": "unknown", "error": str(exc)})
        proc = self._yolo_training_proc
        if proc is not None:
            code = proc.poll()
            status["running"] = code is None
            status["returncode"] = code
            if code is not None and status.get("status") in {"training", "preparing", "exporting"}:
                status["status"] = "complete" if code == 0 else "error"
        return status

    @staticmethod
    def _coerce_train_int(value: Any, default: int, min_value: int, max_value: int) -> int:
        try:
            out = int(float(value))
        except Exception:
            out = default
        return max(min_value, min(max_value, out))

    def train_yolo_detector(self) -> Dict[str, Any]:
        current = self.get_yolo_training_status()
        if bool(current.get("running", False)):
            return {"ok": True, "message": "YOLO training is already running.", **current}
        dataset = self.export_yolo_finetune_dataset()
        dataset_dir = str(dataset["dataset_dir"])
        status_path = self._yolo_training_status_path()
        log_path = os.path.join(dataset_dir, "training.log")
        script_path = os.path.join(os.getcwd(), "tools", "train_yolo_detector.py")
        base_model = str(getattr(self.model_config, "yolo_training_base_model", "yolov8n.pt") or "yolov8n.pt")
        epochs = self._coerce_train_int(getattr(self.model_config, "yolo_training_epochs", 30), 30, 1, 1000)
        imgsz = self._coerce_train_int(getattr(self.model_config, "yolo_training_imgsz", 640), 640, 320, 2048)
        batch = self._coerce_train_int(getattr(self.model_config, "yolo_training_batch", 8), 8, 1, 128)
        device = str(getattr(self.model_config, "yolo_training_device", "") or "")
        cmd = [
            sys.executable,
            script_path,
            "--dataset-dir", dataset_dir,
            "--base-model", base_model,
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--status", status_path,
        ]
        if device.strip():
            cmd.extend(["--device", device.strip()])
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump({
                "status": "starting",
                "running": True,
                "dataset_dir": dataset_dir,
                "log": log_path,
                "command": cmd,
                "updated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, fh, ensure_ascii=False, indent=2)
        log_fh = open(log_path, "a", encoding="utf-8")
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        self._yolo_training_proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
        )
        return {
            "ok": True,
            "status": "starting",
            "running": True,
            "dataset_dir": dataset_dir,
            "status_path": status_path,
            "log": log_path,
            "pid": self._yolo_training_proc.pid,
            "pages": dataset.get("pages", 0),
        }

    def delete_region(self, region_idx: int, yolo_reject_reason: str = "") -> dict:
        self._begin_active_operation("mutate")
        try:
            return self._delete_region_active(region_idx, yolo_reject_reason)
        finally:
            self._end_active_operation("mutate")

    def _delete_region_active(self, region_idx: int, yolo_reject_reason: str = "") -> dict:
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        self._push_undo_snapshot()
        block = self._regions[region_idx]
        page = self.chapter_mgr.current_page
        has_cleanup_patch = self._cleanup_patch_for_region(page, region_idx) is not None if page is not None else False
        self._record_yolo_negative(region_idx, block, yolo_reject_reason)
        del self._regions[region_idx]
        if region_idx < len(self._translations):
            del self._translations[region_idx]
        self._bump_region_mutation_version()
        self._memory_hits = {
            (idx if idx < region_idx else idx - 1): hits
            for idx, hits in self._memory_hits.items()
            if idx != region_idx
        }
        self._invalidate_page_outputs(preserve_cleanup=not has_cleanup_patch)
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        return self.get_bootstrap()

    def ocr_region(self, region_idx: int) -> dict:
        self._begin_active_operation("ocr_region")
        try:
            return self._ocr_region_active(region_idx)
        finally:
            self._end_active_operation("ocr_region")

    def _ocr_region_active(self, region_idx: int) -> dict:
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        # Pass 4: honour the SFX master toggle for single-region OCR too.
        # A manual call on an SFX block while the toggle is OFF is a no-op —
        # same as ocr_current_page which skips SFX slots.  This prevents the
        # user from accidentally populating OCR text that the translate /
        # typeset stages will then refuse to process.
        process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
        block_now = self._regions[region_idx]
        if not process_sfx and _is_pipeline_sfx(block_now):
            debug_print(
                f"[SFX_SKIP] stage=ocr_region idx={region_idx} "
                f"role={getattr(block_now, 'bubble_role', '')!r}"
            )
            return self.get_bootstrap()
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        snap_region_id = id(self._regions[region_idx])
        self._push_undo_snapshot()
        text = self._ocr_one_region(region_idx)
        curr_page = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        if curr_page != page_idx:
            debug_print(f"[OCR_STALE] region={region_idx} reason=page_changed snap={page_idx} current={curr_page}")
            return self.get_bootstrap()
        if region_idx >= len(self._regions) or id(self._regions[region_idx]) != snap_region_id:
            debug_print(f"[OCR_STALE] region={region_idx} reason=region_gone")
            return self.get_bootstrap()
        status = str(getattr(self._regions[region_idx], "ocr_status", "") or "")
        reason = str(getattr(self._regions[region_idx], "ocr_status_reason", "") or "")
        if status == "ok":
            self._regions[region_idx].text = text
        elif status in {"empty", "failed", "cached_empty"}:
            self._regions[region_idx].text = ""
        debug_print(f"[OCR_RESULT] idx={region_idx} status={status or 'unknown'} has_text={bool((text or '').strip())} reason={reason!r}")
        self._try_cross_page_ocr_groups(page_idx)
        self._invalidate_page_outputs(preserve_cleanup=False)
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        if status == "ok":
            self._notify(f"OCR region {region_idx + 1} complete.", 1, 1)
        elif status in {"empty", "cached_empty"}:
            self._notify(f"OCR region {region_idx + 1}: no text found.", 1, 1)
        elif status == "failed":
            self._notify(f"OCR region {region_idx + 1} failed.", 1, 1)
        return self.get_bootstrap()

    def translate_region(self, region_idx: int) -> dict:
        self._begin_active_operation("translate_region")
        try:
            return self._translate_region_active(region_idx)
        finally:
            self._end_active_operation("translate_region")

    def _translate_region_active(self, region_idx: int) -> dict:
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        # Pass 4: honour the SFX master toggle for single-region translate too.
        process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
        block_now = self._regions[region_idx]
        if not process_sfx and _is_pipeline_sfx(block_now):
            debug_print(
                f"[SFX_SKIP] stage=translate_region idx={region_idx} "
                f"role={getattr(block_now, 'bubble_role', '')!r}"
            )
            return self.get_bootstrap()
        self._push_undo_snapshot()
        while len(self._translations) <= region_idx:
            self._translations.append("")
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        text = self._regions[region_idx].text
        if not str(text or "").strip():
            self._notify(f"Translate skipped — region {region_idx + 1} has no source text.", 1, 1)
            return self.get_bootstrap()
        self._notify(f"Translating region {region_idx + 1}…", 0, 1)
        result = self._translate_texts([text], page_idx=page_idx)[0]
        if (result or "").strip():
            self._translations[region_idx] = result.strip()
        self._flag_translations([region_idx])
        self._invalidate_page_outputs(preserve_cleanup=True)
        self._flush_working_state_to_page()
        self.chapter_mgr.save_state()
        self._notify(f"Translated region {region_idx + 1}.", 1, 1)
        return self.get_bootstrap()

    # ── Image access ─────────────────────────────────────────────────────────

    def _ensure_page_artifact(self, page: Any, idx: int, kind: str) -> None:
        if self._is_active_operation():
            debug_print(
                f"[STARTUP_RESTORE] skipped page={idx} kind={kind!r} "
                f"active_ops={self._active_op_names!r}"
            )
            return
        if idx in self._restoring_pages:
            debug_print(f"[STARTUP_RESTORE] reentrant skip page={idx} kind={kind!r}")
            return
        page_regions = getattr(page, "regions", None) or []
        has_ocr_text = any(str(getattr(r, "text", "") or "").strip() for r in page_regions)
        rebuilt = False
        if kind == "cleaned" and page.cleaned_cv is None and page_regions and has_ocr_text:
            raw_cv = cv2.imread(page.image_path)
            if raw_cv is not None:
                if getattr(page, "cleanup_patches", None):
                    page.cleaned_cv = self._rebuild_cleaned_from_cleanup_patches(page, raw_cv)
                    rebuilt = True
                    page.bump_render_version()
                    render_debug(
                        "startup_restore",
                        action="rebuilt_cleaned_from_patches",
                        page=idx,
                        dirty=bool(getattr(page, "render_dirty", False)),
                        has_cleaned=True,
                        has_typeset=page.typeset_pil is not None,
                    )
                    return
                if any(bool(getattr(r, "cross_page", False)) for r in page_regions):
                    debug_print(f"[STARTUP_RESTORE] skipped page={idx} kind='cleaned' reason=cross_page_requires_split_cleanup")
                    return
                debug_print(f'cleanup_route="planned" page={idx} action="startup_restore_cleaned"')
                # Pass 4: honor SFX master toggle on startup restore too.
                process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))
                cleanup_regions = [
                    b for b in page.regions
                    if (
                        (process_sfx or not _is_pipeline_sfx(b) or _cleanup_override_allows_pipeline_sfx(b))
                        and _can_destructively_clean_region(b, "", self.model_config, operation="startup_restore")[0]
                    )
                ]
                page.cleaned_cv = erase_text_region_planned(
                    raw_cv,
                    cleanup_regions,
                    page_index=idx,
                    cleanup_backend=getattr(self.model_config, "cleanup_backend", "opencv"),
                    iopaint_url=getattr(self.model_config, "iopaint_url", ""),
                    cleanup_debug_artifacts=_config_bool(getattr(self.model_config, "cleanup_debug_artifacts", False)),
                    cleanup_debug_dir=getattr(self.model_config, "cleanup_debug_dir", ""),
                    auto_clean_sfx=_config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
                    cleanup_mode=getattr(self.model_config, "cleanup_mode", "balanced"),
                    model_config=self.model_config,
                )
                page.cleanup_patches = []
                rebuilt = True
                page.bump_render_version()
                render_debug(
                    "startup_restore",
                    action="rebuilt_cleaned",
                    page=idx,
                    dirty=bool(getattr(page, "render_dirty", False)),
                    has_cleaned=True,
                    has_typeset=page.typeset_pil is not None,
                )
        if (
            kind == "typeset"
            and page.typeset_pil is None
            and not getattr(page, "render_dirty", False)
            and any((t or "").strip() for t in getattr(page, "translations", []) or [])
            and has_ocr_text
        ):
            self._restoring_pages.add(idx)
            try:
                self._ensure_page_artifact(page, idx, "cleaned")
                raw_cv = cv2.imread(page.image_path)
                base_cv = page.cleaned_cv if page.cleaned_cv is not None else raw_cv
                if base_cv is None:
                    return
                old_raw, old_regions, old_translations = self._raw_cv, self._regions, self._translations
                try:
                    self._raw_cv = raw_cv
                    self._regions = page.regions
                    self._translations = page.translations
                    page.typeset_pil = self._typeset_image(base_cv)
                    rebuilt = True
                    page.bump_render_version()
                    render_debug(
                        "startup_restore",
                        action="rebuilt_typeset",
                        page=idx,
                        dirty=False,
                        has_cleaned=page.cleaned_cv is not None,
                        has_typeset=True,
                    )
                finally:
                    self._raw_cv, self._regions, self._translations = old_raw, old_regions, old_translations
            finally:
                self._restoring_pages.discard(idx)
        if rebuilt:
            self.chapter_mgr.save_state()

    def get_page_image_b64(self, idx: int, mode: str = "best") -> Optional[str]:
        """
        Return the best available image for page `idx` as a base64-encoded PNG.
        Priority: typeset > cleaned > raw unless mode is "raw".
        Returns None if no image is available.
        """
        pages = self.chapter_mgr.pages
        if not (0 <= idx < len(pages)):
            return None
        page = pages[idx]

        pil: Optional[Image.Image] = None

        mode = (mode or "best").strip().lower()
        if mode == "raw":
            base_layer = "raw"
            reason = "explicit"
            try:
                pil = Image.open(page.image_path)
            except Exception:
                return None
        elif mode == "cleaned":
            self._ensure_page_artifact(page, idx, "cleaned")
            if page.cleaned_cv is not None:
                base_layer = "cleaned"
                reason = "explicit"
                pil = Image.fromarray(cv2.cvtColor(page.cleaned_cv, cv2.COLOR_BGR2RGB))
            else:
                base_layer = "raw"
                reason = "cleaned-missing"
                try:
                    pil = Image.open(page.image_path)
                except Exception:
                    return None
        elif mode == "typeset":
            self._ensure_page_artifact(page, idx, "typeset")
            if page.typeset_pil is not None and not getattr(page, "render_dirty", False):
                base_layer = "typeset"
                reason = "explicit"
                pil = page.typeset_pil
            elif page.cleaned_cv is not None:
                base_layer = "cleaned"
                reason = "typeset-missing"
                pil = Image.fromarray(cv2.cvtColor(page.cleaned_cv, cv2.COLOR_BGR2RGB))
            else:
                base_layer = "raw"
                reason = "typeset-and-cleaned-missing"
                try:
                    pil = Image.open(page.image_path)
                except Exception:
                    return None
        else:
            self._ensure_page_artifact(page, idx, "cleaned")
            self._ensure_page_artifact(page, idx, "typeset")
            if page.typeset_pil is not None and not getattr(page, "render_dirty", False):
                base_layer = "typeset"
                reason = "restored-typeset"
                pil = page.typeset_pil
            elif page.cleaned_cv is not None:
                base_layer = "cleaned"
                reason = "dirty-or-no-typeset"
                pil = Image.fromarray(cv2.cvtColor(page.cleaned_cv, cv2.COLOR_BGR2RGB))
            else:
                base_layer = "raw"
                reason = "fallback-raw"
                try:
                    pil = Image.open(page.image_path)
                except Exception:
                    return None
        render_debug(
            "image",
            action="image_mode_resolve",
            requestedMode=mode,
            baseLayer=base_layer,
            reason=reason,
            page=idx,
            dirty=bool(getattr(page, "render_dirty", False)),
            has_cleaned=page.cleaned_cv is not None,
            has_typeset=page.typeset_pil is not None,
            activeOperation=self._is_active_operation(),
            activeOps=list(getattr(self, "_active_op_names", []) or []),
            restoring=idx in getattr(self, "_restoring_pages", set()),
        )

        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Phase 5: TM approval API ─────────────────────────────────────────────

    def approve_tm_entry(self, entry_id: str) -> Dict[str, Any]:
        """Approve a ChapterTM entry for explicit human review/promote flows."""
        if not _HAS_MEMORY or self._chapter_tm is None:
            return {"ok": False, "msg": "Memory not available."}
        ok, msg = approve_entry(self._chapter_tm, entry_id)
        return {"ok": ok, "msg": msg}

    def reject_tm_entry(self, entry_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Reject a ChapterTM entry.

        Automatically creates a blocked mapping so the same (kr, en) pair
        cannot be retrieved in future sessions for this series (Phase 6).
        """
        if not _HAS_MEMORY or self._chapter_tm is None:
            return {"ok": False, "msg": "Memory not available."}
        scope = f"series:{self._series_title}" if self._series_title else "series:unknown"
        ok, msg = reject_entry(
            self._chapter_tm, entry_id,
            blocked_store=self._blocked,
            series_scope=scope,
            reason=reason,
        )
        return {"ok": ok, "msg": msg}

    def mark_tm_reviewed(self, entry_id: str) -> Dict[str, Any]:
        """Mark a TM entry as reviewed (seen but not yet approved)."""
        if not _HAS_MEMORY or self._chapter_tm is None:
            return {"ok": False, "msg": "Memory not available."}
        ok, msg = mark_reviewed(self._chapter_tm, entry_id)
        return {"ok": ok, "msg": msg}

    def promote_tm_entry(
        self,
        entry_id:    str,
        scope:       str           = "series",   # "series" | "global"
        as_name:     bool          = False,
        kr_override: Optional[str] = None,
        en_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Promote an approved ChapterTM entry to series or global memory.

        Entry must be approved first.
        as_name=True → NameEntry; False → GlossaryEntry.
        """
        if not _HAS_MEMORY or self._chapter_tm is None:
            return {"ok": False, "msg": "Memory not available."}

        if scope == "global":
            g_store = self._global_glossary
            n_store = self._global_names
            if g_store is None or n_store is None:
                return {"ok": False, "msg": "Global memory stores not initialised."}
            ok, msg = promote_entry_to_global(
                self._chapter_tm, entry_id, g_store, n_store,
                as_name=as_name,
                kr_canonical=kr_override,
                en_canonical=en_override,
            )
        else:
            g_store = self._glossary
            n_store = self._name_mem
            if g_store is None or n_store is None:
                return {"ok": False, "msg": "Series memory stores not initialised."}
            ok, msg = promote_entry_to_series(
                self._chapter_tm, entry_id, g_store, n_store,
                self._series_title,
                as_name=as_name,
                kr_canonical=kr_override,
                en_canonical=en_override,
            )
        return {"ok": ok, "msg": msg}

    def edit_tm_entry(self, entry_id: str, new_en: str) -> Dict[str, Any]:
        """
        Replace the English text of a TM entry and reset status to 'reviewed'.
        Human must re-approve before the entry can be promoted.
        """
        if not _HAS_MEMORY or self._chapter_tm is None:
            return {"ok": False, "msg": "Memory not available."}
        ok = self._chapter_tm.update_translation(entry_id, new_en)
        return {"ok": ok, "msg": "Updated." if ok else f"Entry {entry_id} not found."}

    # ── Phase 6: blocked mapping API ─────────────────────────────────────────

    def add_blocked_mapping(
        self,
        source_kr:    str,
        blocked_en:   str,
        reason:       str  = "",
        global_scope: bool = False,
    ) -> Dict[str, Any]:
        """Add a blocked mapping to the series (or global) store."""
        if not _HAS_MEMORY:
            return {"ok": False, "msg": "Memory not available."}
        store = self._global_blocked if global_scope else self._blocked
        if store is None:
            return {"ok": False, "msg": "Blocked mapping store not initialised."}
        scope = "global" if global_scope else f"series:{self._series_title}"
        entry = store.add(source_kr, blocked_en, scope, reason)
        return {"ok": True, "id": entry.id, "msg": f"Blocked mapping added: '{blocked_en}'."}

    def remove_blocked_mapping(
        self,
        entry_id:     str,
        global_scope: bool = False,
    ) -> Dict[str, Any]:
        """Remove a blocked mapping by id."""
        if not _HAS_MEMORY:
            return {"ok": False, "msg": "Memory not available."}
        store = self._global_blocked if global_scope else self._blocked
        if store is None:
            return {"ok": False, "msg": "Store not initialised."}
        ok = store.remove(entry_id)
        return {"ok": ok, "msg": "Removed." if ok else f"Entry {entry_id} not found."}

    # ── Series-scoped memory CRUD ────────────────────────────────────────────

    def _ensure_series_memory(self) -> bool:
        """Create/load the current series stores when memory is available."""
        if not _HAS_MEMORY:
            return False
        if self._glossary is not None and self._name_mem is not None:
            return True
        series_title = self._current_series_title()
        chapter_id = self._chapter_id or (
            os.path.basename(getattr(self.chapter_mgr, "chapter_dir", "") or "")
            or "chapter"
        )
        folder = getattr(self.chapter_mgr, "chapter_dir", "") or ""
        ctx = self._memory_context_for_folder(folder, series_title, chapter_id) if folder else {
            "memory_key": series_title,
            "chapter_memory_key": chapter_id,
            "memory_aliases": [],
            "display_title": series_title,
        }
        self._init_memory(
            ctx["memory_key"],
            ctx["chapter_memory_key"],
            aliases=ctx["memory_aliases"],
            display_title=ctx["display_title"],
        )
        return self._glossary is not None and self._name_mem is not None

    @staticmethod
    def _split_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).split(",") if v.strip()]

    @staticmethod
    def _entry_dict(entry: Any) -> Dict[str, Any]:
        try:
            return asdict(entry)
        except Exception:
            return dict(getattr(entry, "__dict__", {}))

    def list_series_memory(self) -> Dict[str, Any]:
        """Return current series name/glossary entries only. Never includes global memory."""
        if not _HAS_MEMORY:
            return {
                "available": False,
                "error": _MEMORY_IMPORT_ERROR,
                "series_title": self._current_series_title(),
                "memory_key": self._series_title,
                "memory_fs_key": "",
                "memory_aliases": self._memory_aliases,
                "names": [],
                "glossary": [],
            }
        self._ensure_series_memory()
        alias_names: List[Any] = []
        alias_glossary: List[Any] = []
        for store in self._alias_names:
            alias_names.extend(store.all_entries())
        for store in self._alias_glossaries:
            alias_glossary.extend(store.all_entries())
        return {
            "available": True,
            "series_title": self._display_series_title or self._current_series_title(),
            "memory_key": self._series_title or self._current_series_title(),
            "memory_fs_key": _memory_slug(self._series_title or self._current_series_title()),
            "memory_aliases": self._memory_aliases,
            "names": [
                self._entry_dict(e)
                for e in ((self._name_mem.all_entries() if self._name_mem else []) + alias_names)
            ],
            "glossary": [
                self._entry_dict(e)
                for e in ((self._glossary.all_entries() if self._glossary else []) + alias_glossary)
            ],
        }

    def add_series_name(
        self,
        kr_name: str,
        en_name: str,
        aliases_kr: Optional[Any] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        if not self._ensure_series_memory() or self._name_mem is None:
            return self.get_bootstrap()
        kr = (kr_name or "").strip()
        en = (en_name or "").strip()
        if not kr or not en:
            raise ValueError("Korean and English names are required.")
        now = now_iso()
        self._name_mem.add(NameEntry(
            id=make_id(),
            kr_name=kr,
            en_name=en,
            aliases_kr=self._split_list(aliases_kr),
            trust="manual",
            scope=f"series:{self._series_title or self._current_series_title()}",
            note=(note or "").strip(),
            created_at=now,
            updated_at=now,
        ))
        return self.get_bootstrap()

    def _detect_regions(self, image_path: str) -> List[OCRBlock]:
        backend = (getattr(self.model_config, "detector_backend", "ocr") or "ocr").strip().lower()
        detector: Any
        if backend == "yolo":
            raw_model_path = getattr(self.model_config, "yolo_model_path", "") or ""
            model_path = self._resolve_yolo_model_path(raw_model_path)
            exists = os.path.isfile(model_path)
            debug_print(
                "yolo_model_path_raw="
                f"{raw_model_path!r} yolo_model_path_resolved={model_path!r} "
                f"exists={exists}"
            )
            allow_fallback = str(
                getattr(self.model_config, "detector_allow_fallback", "false") or ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not exists:
                msg = (
                    "YOLO detector configured but model file is missing: "
                    f"{model_path}"
                )
                self._notify(msg)
                if allow_fallback:
                    debug_print(f"{msg}; explicit fallback enabled, using OCR.")
                else:
                    raise RuntimeError(msg)
            try:
                # Pass 7: user-configurable YOLO thresholds. Rebuild detector
                # when model path, confidence, or nms_iou changes.
                try:
                    yolo_conf = float(getattr(self.model_config, "yolo_confidence", 0.25) or 0.25)
                except Exception:
                    yolo_conf = 0.25
                yolo_conf = max(0.01, min(0.95, yolo_conf))
                try:
                    yolo_nms = float(getattr(self.model_config, "yolo_nms_iou", 0.45) or 0.45)
                except Exception:
                    yolo_nms = 0.45
                yolo_nms = max(0.01, min(0.95, yolo_nms))
                needs_rebuild = (
                    self._yolo_detector is None
                    or self._yolo_model_path != model_path
                    or float(getattr(self._yolo_detector, "confidence", -1.0)) != yolo_conf
                    or float(getattr(self._yolo_detector, "nms_iou", -1.0)) != yolo_nms
                )
                if needs_rebuild:
                    self._yolo_detector = YoloV6RegionDetector(
                        model_path, confidence=yolo_conf, nms_iou=yolo_nms
                    )
                    self._yolo_model_path = model_path
                detector = self._yolo_detector
                debug_print(f"Detector backend: yolo model={model_path!r}")
                blocks = detector.detect(image_path)
                for block in blocks:
                    block.detector_source = "yolo"
                if blocks:
                    return blocks
                debug_print("YOLO detector returned no regions; not falling back to OCR.")
                return []
            except Exception as exc:
                msg = f"YOLO detector unavailable: {exc}"
                self._notify(msg)
                if allow_fallback:
                    debug_print(f"{msg}; explicit fallback enabled, using OCR.")
                else:
                    raise RuntimeError(msg) from exc
        debug_print(f"Detector backend: ocr requested={backend!r}")
        if self._ocr_proc is None:
            raise RuntimeError("OCR detector/fallback requested but EasyOCR compatibility OCR is disabled.")
        detector = OCRRegionDetector(self._ocr_proc)
        blocks = group_ocr_blocks(detector.detect(image_path))
        for block in blocks:
            block.detector_source = "ocr"
        return blocks

    def _resolve_yolo_model_path(self, raw_path: str) -> str:
        text = os.path.expandvars(os.path.expanduser(str(raw_path or "").strip()))
        if not text:
            return text
        p = pathlib.Path(text)
        if p.is_absolute():
            return str(p)
        candidates = [
            _PROJECT_ROOT / p,
            pathlib.Path.cwd() / p,
            pathlib.Path(__file__).resolve().parent / p,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return str(candidates[0])

    def _enrich_region(self, block: OCRBlock) -> None:
        if self._raw_cv is None:
            return
        bb = block.bbox()
        bg_rgb = estimate_initial_bg_color(self._raw_cv, bb)
        if getattr(block, "detector_source", "") == "yolo":
            yolo_kind = getattr(block, "yolo_kind", "")
            if yolo_kind != "sfx":
                detected_bbox, _detected_mask = detect_bubble_region(self._raw_cv, bb, bg_rgb)
                old_area = max(1, bb[2] * bb[3])
                new_area = max(1, detected_bbox[2] * detected_bbox[3])
                if 1.25 <= (new_area / old_area) <= 10.0:
                    debug_print(
                        f"YOLO bubble detected kind={yolo_kind or 'dialogue'} "
                        f"region={old_area}px bubble={new_area}px bubble_bbox={detected_bbox}"
                    )
                    block.bubble_bbox = detected_bbox
                else:
                    block.bubble_bbox = bb
            else:
                block.bubble_bbox = bb
            # CRITICAL: bubble_mask shape must match bubble_bbox (w, h), NOT the
            # original region bbox (bb).  When detect_bubble_region expands the box,
            # using bb dimensions produces a shape mismatch that crashes
            # extract_block_colors with "could not broadcast input array from shape
            # (bb_h, bb_w) into shape (bubble_h, bubble_w)".
            _, _, _bw_mask, _bh_mask = block.bubble_bbox
            block.bubble_mask = np.full(
                (max(1, _bh_mask), max(1, _bw_mask)), 255, dtype=np.uint8
            )
        else:
            bub_bbox, bub_mask = detect_bubble_region(self._raw_cv, bb, bg_rgb)
            block.bubble_bbox = bub_bbox
            block.bubble_mask = bub_mask
        block.bg_color, block.fg_color = extract_block_colors(self._raw_cv, block)
        if getattr(block, "detector_source", "") == "yolo":
            role_by_kind = {
                "dialogue": "dialog",
                "narration": "thought",
                "sfx": "sfx",
                "shout": "bold",
            }
            block.bubble_role = role_by_kind.get(getattr(block, "yolo_kind", ""), block.bubble_role)
        elif self.font_lib:
            block.bubble_role = self.font_lib.pick_role(self._raw_cv, bb, block.text)
        classify_region(self._raw_cv, block)
        if getattr(block, "detector_source", "") == "yolo" and getattr(block, "yolo_kind", "") == "sfx":
            block.region_kind = RegionKind.SFX_OVER_ART
            block.background_kind = BackgroundKind.ART
            block.region_confidence = 0.85

        # CleanupPlan derives and validates YOLO glyph masks at cleanup time.
        # Keep detector containers out of block.text_mask so restored/preview
        # paths cannot accidentally treat a whole YOLO bbox as glyph ink.
        if getattr(block, "detector_source", "") == "yolo":
            block.text_mask = None
            debug_print(
                f"_enrich_region: YOLO text_mask deferred_to_cleanup_plan "
                f"bbox={block.bbox()} "
                f"kind={getattr(getattr(block,'region_kind',None),'name','?')!r} "
                f"role={getattr(block,'bubble_role','?')!r}"
            )
        decide_cleanup_strategy(block)
        compute_placement(self._raw_cv, block)

        # ── Pass 7: bbox separation guard ────────────────────────────────────
        # bubble_bbox may have been expanded by detect_bubble_region above;
        # bbox_override must stay equal to detector_text_bbox so the overlay
        # shows the tight detector box, not the cleanup container.
        # Exception: if the user has manually dragged the box, honour that edit.
        if getattr(block, "detector_source", "") == "yolo":
            dtb = getattr(block, "detector_text_bbox", None)
            is_manual = getattr(block, "manually_adjusted", False)
            if dtb is not None and not is_manual:
                block.bbox_override = dtb
                debug_print(
                    f"[BBOX_SOURCE] source=detector_text_bbox "
                    f"overlay={dtb} bubble_bbox={getattr(block, 'bubble_bbox', None)}"
                )
            else:
                debug_print(
                    f"[BBOX_SOURCE] source={'manual' if is_manual else 'bbox_override'} "
                    f"overlay={block.bbox_override} "
                    f"bubble_bbox={getattr(block, 'bubble_bbox', None)}"
                )

    def update_series_name(self, id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        if not self._ensure_series_memory() or self._name_mem is None:
            return self.get_bootstrap()
        allowed = {"kr_name", "en_name", "aliases_kr", "note"}
        found = False
        for entry in self._name_mem._entries:
            if entry.id != id:
                continue
            for key, value in (fields or {}).items():
                if key not in allowed:
                    continue
                setattr(entry, key, self._split_list(value) if key == "aliases_kr" else str(value).strip())
            entry.trust = "manual"
            entry.scope = f"series:{self._series_title or self._current_series_title()}"
            entry.updated_at = now_iso()
            found = True
            break
        if not found:
            raise ValueError(f"Name entry {id!r} not found.")
        self._name_mem.save()
        return self.get_bootstrap()

    def delete_series_name(self, id: str) -> Dict[str, Any]:
        if not self._ensure_series_memory() or self._name_mem is None:
            return self.get_bootstrap()
        before = len(self._name_mem._entries)
        self._name_mem._entries = [e for e in self._name_mem._entries if e.id != id]
        if len(self._name_mem._entries) == before:
            raise ValueError(f"Name entry {id!r} not found.")
        self._name_mem.save()
        return self.get_bootstrap()

    def add_series_glossary(
        self,
        source_kr: str,
        target_en: str,
        alternatives_en: Optional[Any] = None,
        aliases_kr: Optional[Any] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        if not self._ensure_series_memory() or self._glossary is None:
            return self.get_bootstrap()
        kr = (source_kr or "").strip()
        en = (target_en or "").strip()
        if not kr or not en:
            raise ValueError("Korean source and English target are required.")
        now = now_iso()
        self._glossary.add(GlossaryEntry(
            id=make_id(),
            source_kr=kr,
            target_en=en,
            alternatives_en=self._split_list(alternatives_en),
            aliases_kr=self._split_list(aliases_kr),
            trust="manual",
            scope=f"series:{self._series_title or self._current_series_title()}",
            note=(note or "").strip(),
            created_at=now,
            updated_at=now,
        ))
        return self.get_bootstrap()

    def update_series_glossary(self, id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        if not self._ensure_series_memory() or self._glossary is None:
            return self.get_bootstrap()
        allowed = {"source_kr", "target_en", "alternatives_en", "aliases_kr", "note"}
        found = False
        for entry in self._glossary._entries:
            if entry.id != id:
                continue
            for key, value in (fields or {}).items():
                if key not in allowed:
                    continue
                setattr(
                    entry,
                    key,
                    self._split_list(value) if key in {"alternatives_en", "aliases_kr"} else str(value).strip(),
                )
            entry.trust = "manual"
            entry.scope = f"series:{self._series_title or self._current_series_title()}"
            entry.updated_at = now_iso()
            found = True
            break
        if not found:
            raise ValueError(f"Glossary entry {id!r} not found.")
        self._glossary.save()
        return self.get_bootstrap()

    def delete_series_glossary(self, id: str) -> Dict[str, Any]:
        if not self._ensure_series_memory() or self._glossary is None:
            return self.get_bootstrap()
        before = len(self._glossary._entries)
        self._glossary._entries = [e for e in self._glossary._entries if e.id != id]
        if len(self._glossary._entries) == before:
            raise ValueError(f"Glossary entry {id!r} not found.")
        self._glossary.save()
        return self.get_bootstrap()

    # ── Legacy migration (opt-in, call once for legacy project only) ──────────

    def migrate_legacy_series(self, series_title: str) -> Dict[str, Any]:
        """
        One-time seed of built-in NAME_MAP and GLOSSARY_ANCHORS
        into the per-series memory stores for *series_title*.

        NEVER called automatically.  Call once for the known legacy project.
        Subsequent calls are idempotent (returns 0, 0).
        """
        if not _HAS_MEMORY:
            return {"migrated_glossary": 0, "migrated_names": 0,
                    "error": "memory package unavailable"}
        try:
            g = GlossaryStore(self._memory_root, series_title)
            n = NameMemory(self._memory_root, series_title)
            n_g = g.migrate_from_anchors(GLOSSARY_ANCHORS)
            n_n = n.migrate_from_name_map(NAME_MAP)
            print(f"[memory] migrate_legacy_series({series_title!r}): "
                  f"+{n_g} glossary, +{n_n} names")
            return {"migrated_glossary": n_g, "migrated_names": n_n}
        except Exception as exc:
            return {"migrated_glossary": 0, "migrated_names": 0, "error": str(exc)}

    # ── Memory introspection ──────────────────────────────────────────────────

    def get_memory_stats(self) -> Dict[str, Any]:
        if not _HAS_MEMORY:
            return {"available": False, "error": _MEMORY_IMPORT_ERROR}
        def _c(s: Any) -> int:
            return len(s.all_entries()) if s else 0
        return {
            "available":              True,
            "global_glossary":        _c(self._global_glossary),
            "global_names":           _c(self._global_names),
            "global_blocked":         _c(self._global_blocked),
            "series_glossary":        _c(self._glossary),
            "series_names":           _c(self._name_mem),
            "series_blocked":         _c(self._blocked),
            "chapter_tm_total":       _c(self._chapter_tm),
            "chapter_tm_pending":     (
                len(self._chapter_tm.pending_review()) if self._chapter_tm else 0),
            "chapter_tm_retrievable": (
                len(self._chapter_tm.retrievable_entries()) if self._chapter_tm else 0),
        }

    def get_pending_review(self) -> List[Dict[str, Any]]:
        """Return all ChapterTM entries awaiting human review (pending + reviewed)."""
        if not _HAS_MEMORY or self._chapter_tm is None:
            return []
        return [
            {
                "id":         e.id,
                "kr":         e.kr_text,
                "en":         e.en_text,
                "status":     e.status,
                "flagged":    e.flagged,
                "page_idx":   e.page_idx,
                "region_idx": e.region_idx,
            }
            for e in self._chapter_tm.pending_review()
        ]

    # ── Bootstrap serialiser (ported from bridge.py _build_bootstrap_on_ui) ─

    def get_bootstrap(self) -> dict:
        chapter_mgr = self.chapter_mgr
        pages       = chapter_mgr.pages or []

        # ── Series / chapter hierarchy ──────────────────────────────────────
        series_entries: List[Dict[str, Any]] = []
        chapters_by_series: Dict[str, List[Dict[str, Any]]] = {}
        active_series_id:  Optional[str] = None
        active_chapter_id: Optional[str] = None

        current_series_title = self._current_series_title()

        for i, series in enumerate(self.series_db.series or []):
            sid   = f"s{i+1}"
            title = series.get("title") or f"Series {i+1}"
            if title == current_series_title:
                active_series_id = sid
            ch_entries: List[Dict[str, Any]] = []
            for chapter in series.get("chapters", []) or []:
                folder = chapter.get("folder", "")
                cid = f"{sid}:{os.path.basename(folder) or chapter.get('name', 'chapter')}"
                if folder and folder == getattr(chapter_mgr, "chapter_dir", ""):
                    active_chapter_id = cid
                    active_series_id = sid
                progress, status = self._chapter_progress(
                    folder, pages, chapter_mgr)
                ch_entries.append({
                    "id":       cid,
                    "title":    chapter.get("name") or os.path.basename(folder) or "Chapter",
                    "pages":    int(chapter.get("page_count") or 0),
                    "progress": progress,
                    "status":   status,
                })
            chapters_by_series[sid] = ch_entries
            series_entries.append({
                "id":       sid,
                "title":    title,
                "subtitle": series.get("source") or "local",
                "source":   series.get("source") or "local",
                "source_id": str(series.get("source_id") or ""),
                "thumbnail_url": series.get("thumbnail_url", ""),
                "thumbnail_path": series.get("thumbnail_path", ""),
                "lang":     "ko→en",
                "chapters": len(series.get("chapters", []) or []),
                "color":    ["#8b7cf8", "#e8a454", "#4ec9b4", "#6090e8"][i % 4],
            })

        if not series_entries:
            active_series_id = "s1"
            series_entries = [{
                "id": "s1", "title": current_series_title,
                "subtitle": "local", "lang": "ko→en",
                "source": "local", "source_id": "",
                "thumbnail_url": "", "thumbnail_path": "",
                "chapters": 1 if getattr(chapter_mgr, "chapter_dir", "") else 0,
                "color": "#8b7cf8",
            }]
            chapters_by_series = {"s1": []}

        if getattr(chapter_mgr, "chapter_dir", ""):
            if active_series_id is None:
                active_series_id = series_entries[0]["id"]
            if active_chapter_id is None:
                active_chapter_id = (
                    f"{active_series_id}:"
                    f"{os.path.basename(chapter_mgr.chapter_dir) or 'chapter'}"
                )
            chs = chapters_by_series.setdefault(active_series_id, [])
            if not any(c["id"] == active_chapter_id for c in chs):
                progress, status = self._chapter_progress(
                    chapter_mgr.chapter_dir, pages, chapter_mgr)
                chs.append({
                    "id":       active_chapter_id,
                    "title":    os.path.basename(chapter_mgr.chapter_dir) or "Chapter",
                    "pages":    len(pages),
                    "progress": progress,
                    "status":   status,
                })

        # ── Page summaries ──────────────────────────────────────────────────
        page_summaries: List[Dict[str, Any]] = []
        for idx, page in enumerate(pages):
            has_reg   = len(getattr(page, "regions", []) or []) > 0
            has_text  = has_reg and all(
                (getattr(r, "text", "") or "").strip()
                for r in getattr(page, "regions", []) or [])
            pg_tl     = getattr(page, "translations", []) or []
            has_trans = has_reg and len(pg_tl) >= len(getattr(page, "regions", []) or []) \
                        and all((t or "").strip() for t in pg_tl[:len(getattr(page, "regions", []) or [])])
            page_summaries.append({
                "id":      f"p{idx}",
                "idx":     idx,
                "regions": len(getattr(page, "regions", []) or []),
                "dirty":   bool(getattr(page, "render_dirty", False)),
                "render_version": int(getattr(page, "render_version", 0) or 0),
                "status":  [
                    "done" if has_reg   else "pend",
                    "done" if has_text  else "pend",
                    "done" if has_trans else "pend",
                    "done" if getattr(page, "cleaned_cv",  None) is not None else "pend",
                    "done" if getattr(page, "typeset_pil", None) is not None else "pend",
                ],
            })

        # ── Region list for current page ────────────────────────────────────
        active_page_idx = int(getattr(chapter_mgr, "current_idx", 0) or 0)
        current_page    = chapter_mgr.current_page
        regions_src = self._regions
        trans_src   = self._translations

        # Pass 4: compute master SFX toggle once for this bootstrap.
        process_sfx = _config_bool(getattr(self.model_config, "process_sfx_regions", False))

        region_entries: List[Dict[str, Any]] = []
        issues:         List[Dict[str, Any]] = []
        cleanup_patch_by_region: Dict[str, Dict[str, Any]] = {}
        if current_page is not None:
            for patch in getattr(current_page, "cleanup_patches", []) or []:
                rid = str(patch.get("region_id", "") or "")
                if rid:
                    cleanup_patch_by_region[rid] = self._cleanup_patch_summary(patch)

        for idx, block in enumerate(regions_src):
            x, y, w, h = block.bbox()
            tl    = trans_src[idx] if idx < len(trans_src) else ""
            rid   = f"r{idx+1}"
            label = f"R-{idx+1:02d}"
            font_name = getattr(block, "font_name", "") or getattr(block, "bubble_role", "dialog") or "auto"
            style = block.effective_style() if hasattr(block, "effective_style") else None
            region_entries.append({
                "id":      rid,
                "label":   label,
                "idx":     idx,
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "src":     getattr(block, "text", "") or "",
                "tl":      tl,
                "conf":    int(round(float(getattr(block, "confidence", 0.0) or 0.0) * 100)),
                "font":    font_name,
                "size":    int(getattr(block, "font_size", 0) or 0),
                "leading": 1.15,
                "fg":      _hex_color(getattr(style, "fg_color", None) if style else getattr(block, "fg_color", None), "#111111"),
                "bg":      _hex_color(getattr(block, "bg_color", None), "#ffffff"),
                "outline": _hex_color(getattr(style, "outline_color", None) if style else getattr(block, "outline_color", None), "#ffffff"),
                "outline_width": int((getattr(style, "outline_width", None) if style else None) or getattr(block, "outline_width", 1) or 1),
                "shadow":  _hex_color(getattr(style, "shadow_color", None), "#000000"),
                "shadow_on": bool(getattr(style, "shadow_on", False)),
                "align":   getattr(block, "align", "center") or "center",
                "visible": bool(getattr(block, "visible", True)),
                "locked":  bool(getattr(block, "locked", False)),
                "detector_source": getattr(block, "detector_source", "ocr") or "ocr",
                "manually_adjusted": bool(getattr(block, "manually_adjusted", False)),
                # ── Pass 4: SFX master toggle surface ──────────────────
                "role":              (getattr(block, "bubble_role", None) or "dialog"),
                "pipeline_disabled": bool(
                    (not process_sfx)
                    and _is_pipeline_sfx(block)
                    and not _cleanup_override_allows_pipeline_sfx(block)
                ),
                # ── Pass 6: per-stage confidence / status fields ──────
                "detector_confidence": float(getattr(block, "confidence", 0.0) or 0.0),
                "ocr_confidence":      float(getattr(block, "ocr_confidence", 0.0) or 0.0),
                "yolo_kind":           getattr(block, "yolo_kind", None),
                "yolo_class_id":       (int(getattr(block, "yolo_class_id", -1))
                                         if getattr(block, "yolo_class_id", None) is not None else None),
                "yolo_train_class_id": (int(getattr(block, "yolo_train_class_id", -1))
                                        if getattr(block, "yolo_train_class_id", None) is not None else None),
                "cleanup_tier":        int(getattr(block, "cleanup_tier", 0) or 0),
                "cleanup_status":      str(getattr(block, "cleanup_status", "") or ""),
                "cleanup_reason":      str(getattr(block, "cleanup_reason", "") or ""),
                "cleanup_patch":       cleanup_patch_by_region.get(rid),
                "cleanup_container_confidence": float(getattr(block, "cleanup_container_confidence", 0.0) or 0.0),
                "cleanup_safe_rect_confidence": float(getattr(block, "cleanup_safe_rect_confidence", 0.0) or 0.0),
                "detector_text_bbox":  [int(v) for v in getattr(block, "detector_text_bbox", None)] if getattr(block, "detector_text_bbox", None) else None,
                "bbox_override":       [int(v) for v in getattr(block, "bbox_override", None)] if getattr(block, "bbox_override", None) else None,
                "cleanup_container_bbox": [int(v) for v in getattr(block, "cleanup_container_bbox", None)] if getattr(block, "cleanup_container_bbox", None) else None,
                "container_bbox":      [int(v) for v in getattr(block, "bubble_bbox", None)] if getattr(block, "bubble_bbox", None) else None,
                "cross_page":          bool(getattr(block, "cross_page", False)),
                "cross_page_group_id": getattr(block, "cross_page_group_id", None),
                "cross_page_pages":    [int(v) for v in (getattr(block, "cross_page_pages", []) or [])],
                "composite_bbox":      [int(v) for v in getattr(block, "composite_bbox", None)] if getattr(block, "composite_bbox", None) else None,
                "page_local_bboxes":   {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in (getattr(block, "page_local_bboxes", {}) or {}).items()
                    if isinstance(v, (list, tuple)) and len(v) == 4
                },
                "typeset_status":      str(getattr(block, "typeset_status", "") or ""),
                "typeset_reason":      str(getattr(block, "typeset_reason", "") or ""),
                "translation_status":  (
                    "skipped_sfx" if (
                        (not process_sfx)
                        and _is_pipeline_sfx(block)
                        and not _cleanup_override_allows_pipeline_sfx(block)
                    )
                    else ("flagged" if getattr(block, "is_flagged", False)
                          else ("ok" if (tl or "").strip()
                                else "pending"))
                ),
                # ── Phase 1 classification fields (safe to ignore if unused by UI) ──
                "cleanup_strategy": getattr(block, "cleanup_strategy", "auto") or "auto",
                "cleanup_override": (
                    block.override.to_dict()
                    if getattr(block, "override", None) is not None
                    else {}
                ),
                "region_kind":      (getattr(block.region_kind, "name", None)
                                     if getattr(block, "region_kind", None) is not None
                                     else None),
                "memory_hits":      self._memory_hits.get(idx, []),
                "layers": [
                    {"id": f"{rid}-region",  "type": "region",  "name": f"Region – {idx+1}",  "vis": True,  "locked": False},
                    {"id": f"{rid}-text",    "type": "text",    "name": f"Text – {idx+1}",    "vis": bool(getattr(block, "visible", True)), "locked": bool(getattr(block, "locked", False))},
                    {"id": f"{rid}-cleanup", "type": "cleanup", "name": f"Cleanup – {idx+1}", "vis": True,  "locked": False},
                ],
            })

            if bool(getattr(block, "typeset_overflow", False)):
                issues.append({
                    "id": f"typeset-overflow-{idx}",
                    "sev": "warn",
                    "msg": "Typeset text did not fully fit inside the edited region.",
                    "region": label,
                    "page": active_page_idx + 1,
                })

            review = getattr(block, "review", None)
            if review and getattr(review, "flagged", False):
                sev    = "warn"
                reason = getattr(review, "flag_reason", "review") or "review"
                if reason in {"low_confidence", "empty_translation", "placeholder_output"}:
                    sev = "err"
                elif reason in {"unknown_region"}:
                    sev = "warn"
                issues.append({
                    "id": f"issue-{idx+1}", "sev": sev,
                    "msg":    reason.replace("_", " ").capitalize(),
                    "region": label, "page": active_page_idx + 1,
                })
            elif float(getattr(block, "confidence", 0.0) or 0.0) < 0.4:
                issues.append({
                    "id":     f"issue-conf-{idx+1}",
                    "sev":    "warn",
                    "msg":    f"OCR confidence below threshold ({int(round(float(getattr(block, 'confidence', 0.0) or 0.0) * 100))}%)",
                    "region": label,
                    "page":   active_page_idx + 1,
                })

        if not region_entries:
            issues.append({
                "id": "no-regions", "sev": "info",
                "msg": ("No chapter imported yet." if not pages else "No regions on this page."),
                "region": None, "page": active_page_idx + 1,
            })

        # Append memory consistency warnings (Phases 4–6) to the issues list.
        # These use the same dict shape the UI already renders; the "mem-" id
        # prefix lets the frontend distinguish them if needed.
        issues.extend(self._consistency_warnings)

        # ── Chapter progress ─────────────────────────────────────────────────
        chapter_progress, _ = self._chapter_progress(
            getattr(chapter_mgr, "chapter_dir", ""), pages, chapter_mgr)

        return {
            "series":   series_entries,
            "chapters": chapters_by_series,
            "pages":    page_summaries,
            "regions":  region_entries,
            "issues":   issues,
            "memory":   self.list_series_memory(),
            "meta": {
                "activeSeriesId":  active_series_id or (series_entries[0]["id"] if series_entries else "s1"),
                "activeChapterId": active_chapter_id,
                "activePageIdx":   active_page_idx,
                "busy":            self.busy,
                "status":          self.status,
                "chapterDir":      getattr(chapter_mgr, "chapter_dir", "") or "",
                "totalPages":      len(page_summaries),
                "chapterProgress": chapter_progress,
                "runAllCheckpoint": dict(getattr(chapter_mgr, "run_all_checkpoint", {}) or {}),
                # Phase 4–6 additions — safe for the UI to ignore if not yet consumed
                "memoryStats":          self.get_memory_stats(),
                "pendingReviewCount":   (
                    len(self._chapter_tm.pending_review())
                    if (_HAS_MEMORY and self._chapter_tm) else 0
                ),
                # Pass 4 / Pass 8: settings surface so the frontend can show
                # current values without a separate fetch. Writes go through
                # the existing update_model_config API (if wired).
                "settings": {
                    "process_sfx_regions":    bool(process_sfx),
                    "ocr_backend":             str(getattr(self.model_config, "ocr_backend", "cascade") or "cascade"),
                    "qwen_ocr_model":          str(getattr(self.model_config, "qwen_ocr_model", "") or ""),
                    "paddleocr_service_url":   str(getattr(self.model_config, "paddleocr_service_url", "") or ""),
                    "paddleocr_lang":          str(getattr(self.model_config, "paddleocr_lang", "korean") or "korean"),
                    "ocr_vlm_fallback_confidence": getattr(self.model_config, "ocr_vlm_fallback_confidence", 0.70),
                    "ocr_cache_enabled":       _config_bool(getattr(self.model_config, "ocr_cache_enabled", True)),
                    "translation_provider":   str(getattr(self.model_config, "translation_provider", "ollama") or "ollama"),
                    "deepseek_model":         str(getattr(self.model_config, "deepseek_model", "") or ""),
                    "deepseek_configured":    bool(
                        os.environ.get(
                            str(getattr(self.model_config, "deepseek_api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY"),
                            "",
                        ).strip()
                    ),
                    "cleanup_mode":           str(getattr(self.model_config, "cleanup_mode", "balanced") or "balanced"),
                    "auto_clean_sfx":         _config_bool(getattr(self.model_config, "auto_clean_sfx", False)),
                    "auto_typeset_sfx":       _config_bool(getattr(self.model_config, "auto_typeset_sfx", False)),
                    "auto_clean_text_over_art": _config_bool(getattr(self.model_config, "auto_clean_text_over_art", False)),
                    "cleanup_allow_sfx_cleanup": _config_bool(getattr(self.model_config, "cleanup_allow_sfx_cleanup", False)),
                    "cleanup_allow_text_over_art": _config_bool(getattr(self.model_config, "cleanup_allow_text_over_art", False)),
                    "cleanup_solid_bubble_fill_enabled": _config_bool(getattr(self.model_config, "cleanup_solid_bubble_fill_enabled", True)),
                    "cleanup_solid_bubble_min_container_confidence": getattr(self.model_config, "cleanup_solid_bubble_min_container_confidence", 0.60),
                    "cleanup_solid_bubble_max_mask_container_ratio": getattr(self.model_config, "cleanup_solid_bubble_max_mask_container_ratio", 0.15),
                    "cleanup_solid_bubble_max_rectangularity": getattr(self.model_config, "cleanup_solid_bubble_max_rectangularity", 0.45),
                    "cleanup_halo_mask_enabled": _config_bool(getattr(self.model_config, "cleanup_halo_mask_enabled", True)),
                    "cleanup_halo_max_px": getattr(self.model_config, "cleanup_halo_max_px", 2),
                    "cleanup_residual_retry_enabled": _config_bool(getattr(self.model_config, "cleanup_residual_retry_enabled", True)),
                    "cleanup_residual_retry_dilate_px": getattr(self.model_config, "cleanup_residual_retry_dilate_px", 1),
                    "cleanup_allow_grouped_inpaint": _config_bool(getattr(self.model_config, "cleanup_allow_grouped_inpaint", False)),
                    "cleanup_manual_review_only": _config_bool(getattr(self.model_config, "cleanup_manual_review_only", False)),
                    "cleanup_min_container_confidence": getattr(self.model_config, "cleanup_min_container_confidence", 0.0),
                    "cleanup_max_mask_container_ratio": getattr(self.model_config, "cleanup_max_mask_container_ratio", 0.50),
                    "cleanup_max_mask_region_ratio": getattr(self.model_config, "cleanup_max_mask_region_ratio", 0.28),
                    "cleanup_max_border_touch_ratio": getattr(self.model_config, "cleanup_max_border_touch_ratio", 0.35),
                    "cleanup_max_rectangularity": getattr(self.model_config, "cleanup_max_rectangularity", 0.88),
                    "cleanup_allow_translucent_caption": _config_bool(getattr(self.model_config, "cleanup_allow_translucent_caption", False)),
                    "cleanup_allow_texture_inpaint": _config_bool(getattr(self.model_config, "cleanup_allow_texture_inpaint", False)),
                    "cleanup_risky_action": str(getattr(self.model_config, "cleanup_risky_action", "skip") or "skip"),
                    "cleanup_fallback_backend": str(getattr(self.model_config, "cleanup_fallback_backend", "telea") or "telea"),
                    "cleanup_verbose_logs": _config_bool(getattr(self.model_config, "cleanup_verbose_logs", False)),
                    "cleanup_show_diagnostics": _config_bool(getattr(self.model_config, "cleanup_show_diagnostics", False)),
                    "sam2_enabled": _config_bool(getattr(self.model_config, "sam2_enabled", False)),
                    "sam2_load_mode": str(getattr(self.model_config, "sam2_load_mode", "lazy") or "lazy"),
                    "sam2_required": _config_bool(getattr(self.model_config, "sam2_required", False)),
                    "sam2_backend_url": str(getattr(self.model_config, "sam2_backend_url", "") or ""),
                    "sam2_timeout_sec": getattr(self.model_config, "sam2_timeout_sec", 30),
                    "sam2_model_path": str(getattr(self.model_config, "sam2_model_path", "") or ""),
                    "sam2_checkpoint_path": str(getattr(self.model_config, "sam2_checkpoint_path", "") or ""),
                    "sam2_device": str(getattr(self.model_config, "sam2_device", "auto") or "auto"),
                    "sam2_mask_mode": str(getattr(self.model_config, "sam2_mask_mode", "manual_only") or "manual_only"),
                    "sam2_status": self.get_sam2_status(False),
                },
            },
        }

    def _current_series_title(self) -> str:
        chapter_dir = getattr(self.chapter_mgr, "chapter_dir", "") or ""
        if chapter_dir:
            try:
                series = self.series_db.find_series_for_folder(chapter_dir)
                if series:
                    return (
                        series.get("title_en")
                        or series.get("title")
                        or os.path.basename(os.path.dirname(chapter_dir))
                        or os.path.basename(chapter_dir)
                        or "Local Series"
                    )
            except Exception:
                pass
            return (os.path.basename(os.path.dirname(chapter_dir))
                    or os.path.basename(chapter_dir) or "Local Series")
        try:
            title = self.series_db.current_series_title()
            if title:
                return title
        except Exception:
            pass
        return "Local Series"

    @staticmethod
    def _chapter_progress(folder: str, pages: List[Any],
                           chapter_mgr: ChapterManager) -> Tuple[int, str]:
        if folder != getattr(chapter_mgr, "chapter_dir", "") or not pages:
            return 0, "idle"
        total = len(pages)
        done    = sum(1 for p in pages if getattr(p, "typeset_pil",  None) is not None)
        cleaned = sum(1 for p in pages if getattr(p, "cleaned_cv",   None) is not None)
        trans   = sum(1 for p in pages if any((t or "").strip() for t in getattr(p, "translations", []) or []))
        dets    = sum(1 for p in pages if len(getattr(p, "regions",  []) or []) > 0)
        if done == total and total > 0:
            return 100, "typeset"
        if done > 0:
            return int(done / total * 100), "typeset"
        if cleaned > 0:
            return int(cleaned / total * 100), "cleanup"
        if trans > 0:
            return int(trans / total * 100), "translate"
        if dets > 0:
            return int(dets / total * 100), "detect"
        return 0, "idle"

    # ── Settings ─────────────────────────────────────────────────────────────

    def update_model_config(self, updates: Dict[str, str]) -> dict:
        for k, v in updates.items():
            if hasattr(self.model_config, k):
                setattr(self.model_config, k, v if isinstance(v, list) else str(v))
                if k == "yolo_model_path":
                    self._yolo_detector = None
                    self._yolo_model_path = ""
                if k in {
                    "ocr_backend",
                    "qwen_ocr_model",
                    "ocr_model",
                    "vision_model",
                    "paddleocr_service_url",
                    "paddleocr_lang",
                    "ocr_vlm_fallback_confidence",
                }:
                    self._ocr_cache = {}
                    self._paddle_ocr = None
                    self._paddle_ocr_lang = ""
        self.model_config.save()
        return {"ok": True, "config": self.model_config.to_dict()}

    def get_model_config(self) -> dict:
        return self.model_config.to_dict()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        try:
            self._flush_working_state_to_page()
            self.chapter_mgr.save_state()
            self.model_config.save()
        except Exception:
            pass
        if self._ocr_proc and hasattr(self._ocr_proc, "shutdown"):
            try:
                self._ocr_proc.shutdown()
            except Exception:
                pass
