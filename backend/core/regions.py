from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto as _auto
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.core.constants import debug_print

class RegionKind(Enum):
    PLAIN_BUBBLE      = _auto()   # clean solid-fill bubble — safe to flat-fill
    TEXTURED_BUBBLE   = _auto()   # halftone / dotted bubble — Phase 2 will add texture_clone
    GRADIENT_BUBBLE   = _auto()   # smooth colour-ramp fill — Phase 2 will add gradient_fill
    CAPTION_BOX       = _auto()   # rectangular caption panel
    SFX_OVER_ART      = _auto()   # sound effect on art, no enclosing bubble
    DIALOGUE_OVER_ART = _auto()   # dialogue text directly on art (bubble detection failed)
    UNKNOWN           = _auto()   # ambiguous — flagged for human review

class BackgroundKind(Enum):
    CLEAN    = _auto()   # uniform fill — flat-fill is safe
    TEXTURED = _auto()   # halftone / repeating pattern
    GRADIENT = _auto()   # smooth luminance ramp
    ART      = _auto()   # complex art / line art present
    UNKNOWN  = _auto()

@dataclass(slots=True)
class _FitResult:
    lines: list[str]
    font_size: int
    used_width: int
    used_height: int
    overflow: bool

@dataclass
class CharacterMemory:
    history: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def update(self, characters: List[Dict[str, Any]]) -> None:
        debug_print(f"CharacterMemory.update: received {len(characters)} character(s)")
        for char in characters:
            name = char.get("name", "").strip()
            if not name:
                debug_print(f"CharacterMemory.update: skipping unnamed entry {char!r}")
                continue
            if name not in self.history:
                self.history[name] = {
                    "description": char.get("description", ""),
                    "emotion":     char.get("emotion", ""),
                }
                debug_print(f"CharacterMemory.update: added {name!r}")
            else:
                if char.get("description"):
                    self.history[name]["description"] = char["description"]
                if char.get("emotion"):
                    self.history[name]["emotion"] = char["emotion"]
                debug_print(f"CharacterMemory.update: refreshed {name!r}")

    def context(self) -> str:
        return json.dumps(self.history, ensure_ascii=False, indent=2)

    def is_empty(self) -> bool:
        return len(self.history) == 0

@dataclass
class TextStyle:
    """
    All text rendering parameters for one OCR region.  Fields default to
    values that reproduce the pre-Phase-3 behaviour so existing regions are
    unaffected until a style is explicitly set.

    source: tracks provenance
        "auto"          — derived from OCRBlock.fg_color / outline_color at render time
        "role:<role>"   — role-based preset applied automatically
        "preset:<name>" — named style preset applied by the user
        "manual"        — edited directly in the inspector
    """
    # ── Solid text color ──────────────────────────────────────────────────────
    fg_color:       Tuple[int, int, int] = (0, 0, 0)

    # ── Outline / stroke ─────────────────────────────────────────────────────
    outline_color:  Tuple[int, int, int] = (255, 255, 255)
    outline_width:  int                  = 1          # 0 = no outline

    # ── Gradient fill (only glyph interior; outline drawn first) ──────────────
    gradient_on:    bool                 = False
    gradient_start: Tuple[int, int, int] = (0, 0, 0)
    gradient_end:   Tuple[int, int, int] = (60, 60, 60)
    gradient_angle: int                  = 90         # 0=L→R, 90=T→B, 180=R→L, 270=B→T

    # ── Drop shadow ───────────────────────────────────────────────────────────
    shadow_on:      bool                 = False
    shadow_color:   Tuple[int, int, int] = (0, 0, 0)
    shadow_offset:  Tuple[int, int]      = (1, 2)
    shadow_opacity: float                = 0.55       # 0.0–1.0

    # ── Background plate (behind text, boosts legibility on complex art) ──────
    plate_on:       bool                 = False
    plate_color:    Tuple[int, int, int] = (255, 255, 255)
    plate_opacity:  float                = 0.78
    plate_pad:      int                  = 4          # px padding around text bbox

    # ── Provenance ────────────────────────────────────────────────────────────
    source:         str                  = "auto"

    def to_dict(self) -> dict:
        return {
            "fg_color":       list(self.fg_color),
            "outline_color":  list(self.outline_color),
            "outline_width":  self.outline_width,
            "gradient_on":    self.gradient_on,
            "gradient_start": list(self.gradient_start),
            "gradient_end":   list(self.gradient_end),
            "gradient_angle": self.gradient_angle,
            "shadow_on":      self.shadow_on,
            "shadow_color":   list(self.shadow_color),
            "shadow_offset":  list(self.shadow_offset),
            "shadow_opacity": self.shadow_opacity,
            "plate_on":       self.plate_on,
            "plate_color":    list(self.plate_color),
            "plate_opacity":  self.plate_opacity,
            "plate_pad":      self.plate_pad,
            "source":         self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextStyle":
        def _t3(key: str, default: list) -> Tuple[int, int, int]:
            v = d.get(key, default)
            return (int(v[0]), int(v[1]), int(v[2])) if len(v) >= 3 else tuple(default)
        def _t2(key: str, default: list) -> Tuple[int, int]:
            v = d.get(key, default)
            return (int(v[0]), int(v[1])) if len(v) >= 2 else tuple(default)
        return cls(
            fg_color       = _t3("fg_color",       [0,   0,   0  ]),
            outline_color  = _t3("outline_color",  [255, 255, 255]),
            outline_width  = int(d.get("outline_width",  1)),
            gradient_on    = bool(d.get("gradient_on",   False)),
            gradient_start = _t3("gradient_start", [0,   0,   0  ]),
            gradient_end   = _t3("gradient_end",   [60,  60,  60 ]),
            gradient_angle = int(d.get("gradient_angle",  90)),
            shadow_on      = bool(d.get("shadow_on",      False)),
            shadow_color   = _t3("shadow_color",   [0,   0,   0  ]),
            shadow_offset  = _t2("shadow_offset",  [1,   2       ]),
            shadow_opacity = float(d.get("shadow_opacity", 0.55)),
            plate_on       = bool(d.get("plate_on",       False)),
            plate_color    = _t3("plate_color",    [255, 255, 255]),
            plate_opacity  = float(d.get("plate_opacity",  0.78)),
            plate_pad      = int(d.get("plate_pad",        4)),
            source         = str(d.get("source",          "auto")),
        )

    def is_default(self) -> bool:
        """True when this style is functionally equivalent to auto-derived defaults."""
        return (
            not self.gradient_on and not self.shadow_on and not self.plate_on
            and self.source == "auto"
        )

STYLE_PRESETS: Dict[str, "TextStyle"] = {
    "dialog_light":  TextStyle(
        fg_color=(15, 15, 15), outline_color=(255, 255, 255), outline_width=1,
        source="preset:dialog_light"),
    "dialog_dark":   TextStyle(
        fg_color=(235, 235, 235), outline_color=(10, 10, 10), outline_width=1,
        source="preset:dialog_dark"),
    "thought_soft":  TextStyle(
        fg_color=(45, 45, 90), outline_color=(200, 200, 225), outline_width=1,
        source="preset:thought_soft"),
    "narration":     TextStyle(
        fg_color=(20, 10, 5), outline_color=(215, 205, 190), outline_width=1,
        source="preset:narration"),
    "shout":         TextStyle(
        fg_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=2,
        gradient_on=True, gradient_start=(30, 30, 30), gradient_end=(0, 0, 0),
        gradient_angle=90, source="preset:shout"),
    "sfx_impact":    TextStyle(
        fg_color=(255, 255, 255), outline_color=(0, 0, 0), outline_width=3,
        source="preset:sfx_impact"),
    "sfx_color":     TextStyle(
        fg_color=(255, 220, 0), outline_color=(0, 0, 0), outline_width=2,
        gradient_on=True, gradient_start=(255, 240, 50), gradient_end=(210, 120, 0),
        gradient_angle=90, source="preset:sfx_color"),
    "sfx_dark":      TextStyle(
        fg_color=(20, 20, 20), outline_color=(255, 80, 0), outline_width=2,
        source="preset:sfx_dark"),
}

ROLE_DEFAULT_PRESET: Dict[str, str] = {
    "dialog":  "dialog_light",
    "bold":    "shout",
    "thought": "thought_soft",
    "sfx":     "sfx_impact",
}

@dataclass
class RegionReview:
    """Human-review metadata for one OCR region.

    Stored as ``OCRBlock.review``; defaults to ``None`` on old blocks so
    existing code that never touches review state is unaffected.

    Fields
    ------
    flagged        : bool           — region needs human attention
    flag_reason    : str            — machine-readable tag, e.g. ``"low_confidence"``,
                                      ``"translation_suspect"``, ``"manual"``
    flag_details   : dict           — optional structured data per flag
                                      (e.g. ``{"conf": 0.31, "threshold": 0.40}``)
    reviewed       : bool           — a human has looked at this region
    approved       : bool | None    — True=approved, False=needs rework, None=undecided
    reviewer_notes : str            — freeform human annotation
    """
    flagged:        bool            = False
    flag_reason:    str             = ""
    flag_details:   dict            = field(default_factory=dict)
    reviewed:       bool            = False
    approved:       Optional[bool]  = None   # None = not yet decided
    reviewer_notes: str             = ""
    fix_note:       str             = ""     # Phase 3: what specifically needs manual fixing

    # ── Workflow helpers (Phase 3) ────────────────────────────────────────────
    def mark_ok(self, notes: str = "") -> None:
        """Mark as reviewed and acceptable — clears flagged state."""
        self.flagged        = False
        self.reviewed       = True
        self.approved       = True
        if notes:
            self.reviewer_notes = notes

    def mark_needs_fix(self, note: str = "") -> None:
        """Mark as reviewed but requiring manual correction."""
        self.flagged        = True
        self.reviewed       = True
        self.approved       = False
        if note:
            self.fix_note   = note

    def reset(self) -> None:
        """Clear all review state — reverts the region to unreviewed."""
        self.flagged        = False
        self.flag_reason    = ""
        self.flag_details   = {}
        self.reviewed       = False
        self.approved       = None
        self.reviewer_notes = ""
        self.fix_note       = ""

@dataclass
class RegionOverride:
    """
    Explicit per-region override values  (Phase 3).

    Any field that is not None takes priority over automatic classification /
    strategy selection.  Setting every field back to None (via
    OCRBlock.reset_override()) restores fully automatic behaviour.

    Serialised as the "override" key in .ml_state.json block dicts.
    All fields default to None / False so missing fields in old state files
    are safely ignored.
    """
    region_kind:        Optional[str]        = None  # RegionKind.name
    background_kind:    Optional[str]        = None  # BackgroundKind.name
    cleanup_strategy:   Optional[str]        = None  # e.g. "flat_fill"
    placement_strategy: Optional[str]        = None  # e.g. "bubble_center"
    erase_only:         Optional[bool]       = None
    skip_typeset:       bool                 = False  # exclude from auto typeset
    style:              Optional[TextStyle]  = field(default=None)
    cleanup_override_mode: Optional[str]     = None
    cleanup_region_class: Optional[str]      = None
    cleanup_halo_max_px: Optional[int]       = None
    cleanup_residual_retry_enabled: Optional[bool] = None
    cleanup_residual_retry_dilate_px: Optional[int] = None
    cleanup_min_container_confidence: Optional[float] = None
    cleanup_max_mask_container_ratio: Optional[float] = None
    cleanup_max_mask_region_ratio: Optional[float] = None
    cleanup_max_border_touch_ratio: Optional[float] = None
    cleanup_max_rectangularity: Optional[float] = None
    cleanup_allow_low_confidence: Optional[bool] = None
    cleanup_allow_texture_inpaint: Optional[bool] = None
    cleanup_allow_translucent_caption: Optional[bool] = None

    def is_empty(self) -> bool:
        """True when no field carries an actual override value."""
        return (
            self.region_kind is None
            and self.background_kind is None
            and self.cleanup_strategy is None
            and self.placement_strategy is None
            and self.erase_only is None
            and not self.skip_typeset
            and self.style is None
            and self.cleanup_override_mode is None
            and self.cleanup_region_class is None
            and self.cleanup_halo_max_px is None
            and self.cleanup_residual_retry_enabled is None
            and self.cleanup_residual_retry_dilate_px is None
            and self.cleanup_min_container_confidence is None
            and self.cleanup_max_mask_container_ratio is None
            and self.cleanup_max_mask_region_ratio is None
            and self.cleanup_max_border_touch_ratio is None
            and self.cleanup_max_rectangularity is None
            and self.cleanup_allow_low_confidence is None
            and self.cleanup_allow_texture_inpaint is None
            and self.cleanup_allow_translucent_caption is None
        )

    def to_dict(self) -> dict:
        d: dict = {}
        if self.region_kind        is not None: d["region_kind"]        = self.region_kind
        if self.background_kind    is not None: d["background_kind"]    = self.background_kind
        if self.cleanup_strategy   is not None: d["cleanup_strategy"]   = self.cleanup_strategy
        if self.placement_strategy is not None: d["placement_strategy"] = self.placement_strategy
        if self.erase_only         is not None: d["erase_only"]         = self.erase_only
        if self.skip_typeset:                   d["skip_typeset"]       = True
        if self.style              is not None: d["style"]              = self.style.to_dict()
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
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RegionOverride":
        sty = None
        if "style" in d:
            try:
                sty = TextStyle.from_dict(d["style"])
            except Exception:
                pass
        return cls(
            region_kind        = d.get("region_kind"),
            background_kind    = d.get("background_kind"),
            cleanup_strategy   = d.get("cleanup_strategy"),
            placement_strategy = d.get("placement_strategy"),
            erase_only         = d.get("erase_only"),
            skip_typeset       = bool(d.get("skip_typeset", False)),
            style              = sty,
            cleanup_override_mode = d.get("cleanup_override_mode"),
            cleanup_region_class = d.get("cleanup_region_class"),
            cleanup_halo_max_px = d.get("cleanup_halo_max_px"),
            cleanup_residual_retry_enabled = d.get("cleanup_residual_retry_enabled"),
            cleanup_residual_retry_dilate_px = d.get("cleanup_residual_retry_dilate_px"),
            cleanup_min_container_confidence = d.get("cleanup_min_container_confidence"),
            cleanup_max_mask_container_ratio = d.get("cleanup_max_mask_container_ratio"),
            cleanup_max_mask_region_ratio = d.get("cleanup_max_mask_region_ratio"),
            cleanup_max_border_touch_ratio = d.get("cleanup_max_border_touch_ratio"),
            cleanup_max_rectangularity = d.get("cleanup_max_rectangularity"),
            cleanup_allow_low_confidence = d.get("cleanup_allow_low_confidence"),
            cleanup_allow_texture_inpaint = d.get("cleanup_allow_texture_inpaint"),
            cleanup_allow_translucent_caption = d.get("cleanup_allow_translucent_caption"),
        )

@dataclass
class OCRBlock:
    text:       str
    boxes:      List[List[List[float]]]
    # `confidence` is the DETECTOR confidence (YOLO score, or EasyOCR detector
    # score for legacy ocr-detected regions). Pass 6: OCR text confidence is
    # now tracked separately on `ocr_confidence` so the two values never
    # clobber each other and the UI can report them independently.
    confidence: float
    ocr_confidence: float = 0.0
    ocr_status: str = ""
    ocr_status_reason: str = ""
    bg_color:   Tuple[int, int, int] = (255, 255, 255)
    fg_color:   Tuple[int, int, int] = (0, 0, 0)
    bubble_bbox: Optional[Tuple[int, int, int, int]] = None
    bubble_mask: Optional[np.ndarray] = None
    # ── Pass 7: tight detector output — never modified by bubble expansion ────
    # Set once in YoloV8RegionDetector.detect(); preserved through _enrich_region.
    # Cleanup / container hints live in bubble_bbox / cleanup_container_bbox.
    # bbox_override (user-draggable overlay) starts equal to this and diverges
    # only when the user drags the box.
    detector_text_bbox: Optional[Tuple[int, int, int, int]] = None
    bubble_role: str = "dialog"
    erase_only: bool = False   # True = erase Korean but skip translation (low OCR confidence)
    detector_source: str = "ocr"  # "ocr" | "yolo" | "manual"
    manually_adjusted: bool = False

    # ── Layer-inspector properties (editable at runtime) ──────────────────────
    font_name:     str                    = ""              # "" = auto via ComicFontLibrary
    font_size:     int                    = 0               # 0  = auto-size
    align:         str                    = "center"        # "left" | "center" | "right"
    outline_color: Tuple[int, int, int]   = (255, 255, 255)
    outline_width: int                    = 0               # 0 = no outline
    visible:       bool                   = True
    locked:        bool                   = False
    # bbox override: when set, replaces boxes for rendering (inspector X/Y/W/H edits)
    bbox_override: Optional[Tuple[int, int, int, int]] = None

    # ── Review metadata (populated lazily — None on old/unreviewed blocks) ────
    review: Optional[RegionReview] = field(default=None)

    # ── Classification results (populated by classify_region / decide_cleanup_strategy) ──
    region_kind:        Optional[RegionKind]             = field(default=None)
    background_kind:    Optional[BackgroundKind]         = field(default=None)
    region_confidence:  float                            = 0.0
    cleanup_strategy:   str                              = "auto"
    # "auto"             → legacy bg_std branch (safe default for unclassified)
    # "flat_fill"        → plain solid-fill erase (plain bubbles, caption boxes)
    # "mask_only_inpaint"→ tight stroke mask + TELEA (SFX / over-art)
    # "review"           → skip erase entirely, leave for human (unknown regions)
    # "texture_clone"    → local TELEA on bubble ROI (halftone / screentone)
    # "gradient_fill"    → gradient-aware local TELEA (smooth gradient bubbles)
    placement_strategy: str                              = "bubble_center"

    # ── Cleanup outcome metadata (Pass 2) ────────────────────────────────────
    cleanup_tier:   int = 0    # 0=not yet run, 1=auto_safe, 2=cautious, 3=skipped
    cleanup_status: str = ""
    cleanup_reason: str = ""
    cleanup_meta:   dict = field(default_factory=dict)
    cleanup_container_bbox: Optional[Tuple[int, int, int, int]] = None
    cleanup_container_confidence: float = 0.0
    computed_text_bbox: Optional[Tuple[int, int, int, int]] = None
    cleanup_safe_rect: Optional[Tuple[int, int, int, int]] = None
    cleanup_safe_rect_confidence: float = 0.0
    typeset_status: str = ""
    typeset_reason: str = ""
    typeset_meta: dict = field(default_factory=dict)
    typeset_override: bool = False
    cross_page: bool = False
    cross_page_group_id: Optional[str] = None
    cross_page_pages: List[int] = field(default_factory=list)
    composite_bbox: Optional[Tuple[int, int, int, int]] = None
    page_local_bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)

    # ── Mask cache (populated by classify_region; None until then) ────────────
    text_mask:       Optional[np.ndarray] = field(default=None)
    safe_text_mask:  Optional[np.ndarray] = field(default=None)  # reserved Phase 2

    # ── Placement geometry (reserved for Phase 2 typesetting redesign) ────────
    safe_center:     Optional[Tuple[int, int]]           = field(default=None)
    safe_rect:       Optional[Tuple[int, int, int, int]] = field(default=None)
    text_angle:      float                               = 0.0

    # ── Phase 3: manual overrides + text style ────────────────────────────────
    # override=None → fully automatic; any non-None field in the override wins.
    override:   Optional[RegionOverride] = field(default=None)
    # style=None → auto-derived from fg_color / outline_color at render time.
    style:      Optional[TextStyle]      = field(default=None)

    # ── Core geometry helpers ─────────────────────────────────────────────────
    def all_points(self) -> np.ndarray:
        return np.array([pt for box in self.boxes for pt in box], dtype=np.int32)

    def bbox(self) -> Tuple[int, int, int, int]:
        if self.bbox_override is not None:
            return self.bbox_override
        pts = self.all_points()
        if pts.size == 0:
            return (0, 0, 1, 1)
        return cv2.boundingRect(pts)

    def merge(self, other: "OCRBlock") -> None:
        self.text       = (self.text + " " + other.text).strip() if self.text else other.text
        self.boxes.extend(other.boxes)
        self.confidence = min(self.confidence, other.confidence)

    # ── Review helpers (lazy-init, safe to call on any block) ─────────────────
    def ensure_review(self) -> RegionReview:
        """Return the review object, creating it if this block has none yet."""
        if self.review is None:
            self.review = RegionReview()
        return self.review

    def flag(self, reason: str, details: Optional[dict] = None) -> None:
        """Mark region as needing review.  Idempotent; safe to call in pipeline."""
        r = self.ensure_review()
        r.flagged     = True
        r.flag_reason = reason
        if details:
            r.flag_details.update(details)

    def approve(self, notes: str = "") -> None:
        """Record that a reviewer approved this region."""
        r = self.ensure_review()
        r.reviewed = True
        r.approved = True
        if notes:
            r.reviewer_notes = notes

    def reject(self, notes: str = "") -> None:
        """Record that a reviewer marked this region as needing rework."""
        r = self.ensure_review()
        r.reviewed = True
        r.approved = False
        if notes:
            r.reviewer_notes = notes

    # ── Convenience read properties ───────────────────────────────────────────
    @property
    def is_flagged(self) -> bool:
        return self.review is not None and self.review.flagged

    @property
    def is_approved(self) -> bool:
        return self.review is not None and self.review.approved is True

    @property
    def needs_review(self) -> bool:
        """Flagged but not yet reviewed by a human."""
        return self.review is not None and self.review.flagged and not self.review.reviewed

    # ── Phase 3: override helpers ─────────────────────────────────────────────

    def set_override(self, **kwargs) -> None:
        """Set one or more override fields.  Creates the override object if needed."""
        if self.override is None:
            self.override = RegionOverride()
        for k, v in kwargs.items():
            if hasattr(self.override, k):
                setattr(self.override, k, v)

    def reset_override(self) -> None:
        """Remove all overrides — restores fully automatic classification."""
        self.override = None

    def has_override(self, field_name: str) -> bool:
        """True when a specific field carries an explicit override value."""
        if self.override is None:
            return False
        val = getattr(self.override, field_name, None)
        return val is not None

    # ── Phase 3: effective value accessors ───────────────────────────────────
    # Always use these in pipeline code so overrides are automatically honoured.

    def effective_cleanup_strategy(self) -> str:
        if self.override is not None and self.override.cleanup_strategy is not None:
            return self.override.cleanup_strategy
        return self.cleanup_strategy

    def effective_placement_strategy(self) -> str:
        if self.override is not None and self.override.placement_strategy is not None:
            return self.override.placement_strategy
        return self.placement_strategy

    def effective_erase_only(self) -> bool:
        if self.override is not None and self.override.erase_only is not None:
            return self.override.erase_only
        return self.erase_only

    def effective_skip_typeset(self) -> bool:
        if self.override is not None and self.override.skip_typeset:
            return True
        return False

    def effective_style(self) -> TextStyle:
        """
        Return the TextStyle to use for rendering.

        Priority (highest first):
          1. override.style  — set via inspector "Style" tab with manual edits
          2. self.style      — block-level style (preset or manual without override)
          3. auto-derived    — built from legacy fg_color / outline_color fields

        The auto-derived path ensures zero behaviour change for blocks that have
        never been styled through Phase 3 controls.
        """
        if self.override is not None and self.override.style is not None:
            return self.override.style
        if self.style is not None:
            return self.style
        # Auto-derive from legacy fields — preserves Phase 1/2 rendering exactly
        ow = max(0, self.outline_width or 0)
        oc = self.outline_color if (self.outline_color and ow > 0) else (255, 255, 255)
        return TextStyle(
            fg_color      = self.fg_color or (0, 0, 0),
            outline_color = oc,
            outline_width = max(1, ow) if ow > 0 else 1,
            source        = "auto",
        )

    # ── Phase 3: style helpers ────────────────────────────────────────────────

    def set_style(self, **kwargs) -> None:
        """Apply style fields and mark source='manual'."""
        if self.style is None:
            self.style = TextStyle()
        for k, v in kwargs.items():
            if hasattr(self.style, k):
                setattr(self.style, k, v)
        self.style.source = "manual"

    def apply_preset(self, preset_name: str) -> bool:
        """Apply a named preset from STYLE_PRESETS.  Returns False if unknown."""
        preset = STYLE_PRESETS.get(preset_name)
        if preset is None:
            return False
        import copy
        self.style = copy.copy(preset)
        return True

    def reset_style(self) -> None:
        """Remove style override — reverts to auto-derived rendering."""
        self.style = None
        if self.override is not None:
            self.override.style = None

    # ── Phase 3: review workflow shortcuts ───────────────────────────────────

    def mark_review_ok(self, notes: str = "") -> None:
        """Mark reviewed + acceptable.  Clears flagged state."""
        self.ensure_review().mark_ok(notes)

    def mark_review_needs_fix(self, note: str = "") -> None:
        """Mark reviewed + needs manual correction."""
        self.ensure_review().mark_needs_fix(note)

    def reset_review(self) -> None:
        """Clear all review state — reverts to unreviewed."""
        if self.review is not None:
            self.review.reset()

def _block_to_dict(block: "OCRBlock") -> dict:
    """Serialise an OCRBlock to a JSON-safe dict.
    Numpy arrays (bubble_mask, text_mask) are intentionally excluded — they are
    recomputed by the detect pipeline and are too large to store."""
    d: dict = {
        "text":             block.text,
        "confidence":       block.confidence,
        "ocr_confidence":   float(getattr(block, "ocr_confidence", 0.0) or 0.0),
        "ocr_status":       str(getattr(block, "ocr_status", "") or ""),
        "ocr_status_reason": str(getattr(block, "ocr_status_reason", "") or ""),
        "boxes":            block.boxes,
        "bubble_role":      block.bubble_role,
        "erase_only":       block.erase_only,
        "detector_source":  getattr(block, "detector_source", "ocr") or "ocr",
        "manually_adjusted": bool(getattr(block, "manually_adjusted", False)),
        "font_name":        block.font_name,
        "font_size":        block.font_size,
        "align":            block.align,
        "visible":          block.visible,
        "locked":           block.locked,
        "outline_width":    block.outline_width,
        "cleanup_strategy": block.cleanup_strategy,
        "placement_strategy": block.placement_strategy,
    }
    if block.bg_color:          d["bg_color"]       = list(block.bg_color)
    if block.fg_color:          d["fg_color"]       = list(block.fg_color)
    if block.outline_color:     d["outline_color"]  = list(block.outline_color)
    if block.bbox_override:     d["bbox_override"]  = list(block.bbox_override)
    if block.bubble_bbox:       d["bubble_bbox"]    = list(block.bubble_bbox)
    if getattr(block, "detector_text_bbox", None):
        d["detector_text_bbox"] = list(block.detector_text_bbox)
    if getattr(block, "yolo_train_class_id", None) is not None:
        try:
            d["yolo_train_class_id"] = int(getattr(block, "yolo_train_class_id"))
        except Exception:
            pass
    if block.region_kind:       d["region_kind"]    = block.region_kind.name
    if block.background_kind:   d["background_kind"] = block.background_kind.name
    if getattr(block, "cleanup_safe_rect", None):
        d["cleanup_safe_rect"] = [int(v) for v in block.cleanup_safe_rect]
    safe_rect_conf = float(getattr(block, "cleanup_safe_rect_confidence", 0.0) or 0.0)
    if safe_rect_conf > 0:
        d["cleanup_safe_rect_confidence"] = safe_rect_conf
    cleanup_tier = int(getattr(block, "cleanup_tier", 0) or 0)
    cleanup_status = str(getattr(block, "cleanup_status", "") or "")
    meta = {
        "tier": cleanup_tier,
        "status": cleanup_status,
        "reason": str(getattr(block, "cleanup_reason", "") or ""),
    }
    existing_meta = getattr(block, "cleanup_meta", {}) or {}
    persisted_meta_flags = False
    if isinstance(existing_meta, dict):
        for key in ("review_required", "typeset_box_source", "cross_page_cleanup_limited", "cross_page_cleanup_split", "cross_page_secondary"):
            val = existing_meta.get(key)
            if isinstance(val, (str, int, float, bool)) or val is None:
                meta[key] = val
                persisted_meta_flags = persisted_meta_flags or bool(val)
    if cleanup_tier != 0 or cleanup_status or persisted_meta_flags:
        d["cleanup_meta"] = meta
    typeset_status = str(getattr(block, "typeset_status", "") or "")
    typeset_reason = str(getattr(block, "typeset_reason", "") or "")
    if typeset_status:
        d["typeset_status"] = typeset_status
    if typeset_reason:
        d["typeset_reason"] = typeset_reason
    if bool(getattr(block, "typeset_override", False)):
        d["typeset_override"] = True
    typeset_meta = getattr(block, "typeset_meta", {}) or {}
    if isinstance(typeset_meta, dict):
        safe_typeset_meta = {}
        for key, val in typeset_meta.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                safe_typeset_meta[str(key)] = val
            elif (
                isinstance(val, (list, tuple))
                and all(isinstance(item, (str, int, float, bool)) or item is None for item in val)
            ):
                safe_typeset_meta[str(key)] = list(val)
        if safe_typeset_meta:
            d["typeset_meta"] = safe_typeset_meta
    if bool(getattr(block, "cross_page", False)):
        d["cross_page"] = True
        group_id = getattr(block, "cross_page_group_id", None)
        if group_id:
            d["cross_page_group_id"] = str(group_id)
        pages = getattr(block, "cross_page_pages", []) or []
        if pages:
            d["cross_page_pages"] = [int(v) for v in pages]
        if getattr(block, "composite_bbox", None):
            d["composite_bbox"] = [int(v) for v in block.composite_bbox]
        local = getattr(block, "page_local_bboxes", {}) or {}
        if isinstance(local, dict) and local:
            d["page_local_bboxes"] = {
                str(int(k)): [int(vv) for vv in v]
                for k, v in local.items()
                if isinstance(v, (list, tuple)) and len(v) == 4
            }
    if block.review:
        d["review"] = {
            "flagged":        block.review.flagged,
            "flag_reason":    block.review.flag_reason,
            "flag_details":   block.review.flag_details,
            "reviewed":       block.review.reviewed,
            "approved":       block.review.approved,
            "reviewer_notes": block.review.reviewer_notes,
            "fix_note":       getattr(block.review, "fix_note", ""),
        }
    if block.override and not block.override.is_empty():
        d["override"] = block.override.to_dict()
    if block.style and not block.style.is_default():
        d["style"] = block.style.to_dict()
    return d

def _block_from_dict(d: dict) -> "OCRBlock":
    """Create an OCRBlock from persisted JSON-safe state."""
    boxes = d.get("boxes") or []
    detector_source = d.get("detector_source", "ocr") or "ocr"
    if not boxes and detector_source != "yolo" and d.get("bbox_override"):
        x, y, w, h = [int(v) for v in d["bbox_override"][:4]]
        boxes = [[
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ]]
    block = OCRBlock(
        text=str(d.get("text", "") or ""),
        boxes=boxes,
        confidence=float(d.get("confidence", 0.0) or 0.0),
    )
    _apply_block_dict(block, d)
    if (
        getattr(block, "detector_source", "") == "yolo"
        and block.bbox_override is not None
        and len(block.boxes or []) == 1
    ):
        bx, by, bw, bh = block.bbox_override
        try:
            pts = block.boxes[0]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            box_bbox = (
                int(round(min(xs))),
                int(round(min(ys))),
                int(round(max(xs) - min(xs))),
                int(round(max(ys) - min(ys))),
            )
            if all(abs(a - b) <= 2 for a, b in zip(box_bbox, block.bbox_override)):
                block.boxes = []
                debug_print(
                    f"_block_from_dict: cleared yolo bbox-shaped OCR boxes "
                    f"bbox={(bx, by, bw, bh)}"
                )
        except Exception:
            pass
    return block

def _apply_block_dict(block: "OCRBlock", d: dict) -> None:
    """Restore persisted fields from a dict onto an existing OCRBlock.
    Missing keys default safely — callers must never assume a key is present."""
    def _t3(key: str, fallback: tuple) -> Tuple[int, int, int]:
        v = d.get(key)
        return (int(v[0]), int(v[1]), int(v[2])) if v and len(v) >= 3 else fallback

    block.text        = d.get("text",      block.text)
    block.confidence  = float(d.get("confidence", block.confidence))
    try:
        block.ocr_confidence = float(d.get("ocr_confidence", getattr(block, "ocr_confidence", 0.0)) or 0.0)
    except Exception:
        block.ocr_confidence = 0.0
    block.ocr_status = str(d.get("ocr_status", getattr(block, "ocr_status", "")) or "")
    block.ocr_status_reason = str(d.get("ocr_status_reason", getattr(block, "ocr_status_reason", "")) or "")
    block.bubble_role = d.get("bubble_role", block.bubble_role)
    block.erase_only  = bool(d.get("erase_only", block.erase_only))
    block.detector_source = d.get("detector_source", getattr(block, "detector_source", "ocr") or "ocr")
    block.manually_adjusted = bool(d.get("manually_adjusted", getattr(block, "manually_adjusted", False)))
    block.font_name   = d.get("font_name",  block.font_name)
    block.font_size   = int(d.get("font_size", block.font_size))
    block.align       = d.get("align",     block.align)
    block.visible     = bool(d.get("visible", block.visible))
    block.locked      = bool(d.get("locked",  block.locked))
    block.outline_width = int(d.get("outline_width", block.outline_width))
    block.cleanup_strategy   = d.get("cleanup_strategy",   block.cleanup_strategy)
    block.placement_strategy = d.get("placement_strategy", block.placement_strategy)

    if "bg_color"      in d: block.bg_color      = _t3("bg_color",      block.bg_color)
    if "fg_color"      in d: block.fg_color       = _t3("fg_color",      block.fg_color)
    if "outline_color" in d: block.outline_color  = _t3("outline_color", block.outline_color)

    if "bbox_override" in d and d["bbox_override"]:
        v = d["bbox_override"]
        block.bbox_override = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    if "detector_text_bbox" in d and d["detector_text_bbox"]:
        v = d["detector_text_bbox"]
        block.detector_text_bbox = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    if "yolo_train_class_id" in d and d["yolo_train_class_id"] is not None:
        try:
            block.yolo_train_class_id = int(d["yolo_train_class_id"])  # type: ignore[attr-defined]
        except Exception:
            pass
    if "bubble_bbox" in d and d["bubble_bbox"]:
        v = d["bubble_bbox"]
        block.bubble_bbox = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    if "cleanup_safe_rect" in d and d["cleanup_safe_rect"]:
        try:
            v = d["cleanup_safe_rect"]
            block.cleanup_safe_rect = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        except Exception:
            block.cleanup_safe_rect = None
    try:
        block.cleanup_safe_rect_confidence = float(d.get("cleanup_safe_rect_confidence", 0.0) or 0.0)
    except Exception:
        block.cleanup_safe_rect_confidence = 0.0
    block.typeset_status = str(d.get("typeset_status", getattr(block, "typeset_status", "")) or "")
    block.typeset_reason = str(d.get("typeset_reason", getattr(block, "typeset_reason", "")) or "")
    block.typeset_override = bool(d.get("typeset_override", getattr(block, "typeset_override", False)))
    raw_typeset_meta = d.get("typeset_meta", {})
    if isinstance(raw_typeset_meta, dict):
        block.typeset_meta = {}
        for key, val in raw_typeset_meta.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                block.typeset_meta[str(key)] = val
            elif (
                isinstance(val, list)
                and all(isinstance(item, (str, int, float, bool)) or item is None for item in val)
            ):
                block.typeset_meta[str(key)] = list(val)

    block.cross_page = bool(d.get("cross_page", getattr(block, "cross_page", False)))
    block.cross_page_group_id = (
        str(d.get("cross_page_group_id"))
        if d.get("cross_page_group_id") not in (None, "")
        else None
    )
    block.cross_page_pages = []
    raw_pages = d.get("cross_page_pages", [])
    if isinstance(raw_pages, list):
        for page_idx in raw_pages:
            try:
                block.cross_page_pages.append(int(page_idx))
            except Exception:
                pass
    block.composite_bbox = None
    if d.get("composite_bbox"):
        try:
            v = d["composite_bbox"]
            block.composite_bbox = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        except Exception:
            block.composite_bbox = None
    block.page_local_bboxes = {}
    raw_local_bboxes = d.get("page_local_bboxes", {})
    if isinstance(raw_local_bboxes, dict):
        for key, value in raw_local_bboxes.items():
            try:
                if isinstance(value, (list, tuple)) and len(value) == 4:
                    block.page_local_bboxes[int(key)] = (
                        int(value[0]), int(value[1]), int(value[2]), int(value[3])
                    )
            except Exception:
                pass

    meta = d.get("cleanup_meta", {})
    if isinstance(meta, dict) and meta:
        try:
            block.cleanup_tier = int(meta.get("tier", 0) or 0)
            block.cleanup_status = str(meta.get("status", "") or "")
            block.cleanup_reason = str(meta.get("reason", "") or "")
            block.cleanup_meta = {
                "tier": block.cleanup_tier,
                "status": block.cleanup_status,
                "reason": block.cleanup_reason,
            }
            for key in ("review_required", "typeset_box_source", "cross_page_cleanup_limited", "cross_page_cleanup_split", "cross_page_secondary"):
                val = meta.get(key)
                if isinstance(val, (str, int, float, bool)) or val is None:
                    block.cleanup_meta[key] = val
        except Exception:
            block.cleanup_tier = 0
            block.cleanup_status = ""
            block.cleanup_reason = ""
            block.cleanup_meta = {}

    if "region_kind" in d:
        try:
            block.region_kind = RegionKind[d["region_kind"]]
        except KeyError:
            pass
    if "background_kind" in d:
        try:
            block.background_kind = BackgroundKind[d["background_kind"]]
        except KeyError:
            pass

    if "review" in d:
        r = d["review"]
        rev = block.ensure_review()
        rev.flagged        = bool(r.get("flagged",        False))
        rev.flag_reason    = r.get("flag_reason",    "")
        rev.flag_details   = r.get("flag_details",   {})
        rev.reviewed       = bool(r.get("reviewed",       False))
        rev.approved       = r.get("approved")
        rev.reviewer_notes = r.get("reviewer_notes", "")
        rev.fix_note       = r.get("fix_note",       "")

    if "override" in d and d["override"]:
        try:
            block.override = RegionOverride.from_dict(d["override"])
        except Exception as exc:
            debug_print(f"_apply_block_dict: override restore failed: {exc}")

    if "style" in d and d["style"]:
        try:
            block.style = TextStyle.from_dict(d["style"])
        except Exception as exc:
            debug_print(f"_apply_block_dict: style restore failed: {exc}")

