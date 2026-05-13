"""
backend/engine.py
─────────────────
LocalizerEngine: all pipeline logic with zero Tkinter dependency.

This module imports the pure-logic classes from translator_v14.py
(ModelConfig, ChapterManager, SeriesDB, OCRProcessor, ComicFontLibrary, etc.)
and wraps the pipeline steps in a clean, thread-safe API that the pywebview
layer can call directly.

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
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── Import pure-logic classes from the legacy file ──────────────────────────
# translator_v14.py lives at the project root.  We import only the headless
# classes — everything that does NOT touch tkinter.  The ManhwaLocalizerApp
# class is intentionally NOT imported; we never instantiate it.
import sys, pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from translator_v14 import (
    ModelConfig,
    ChapterPage,
    ChapterManager,
    SeriesDB,
    OCRProcessor,
    ComicFontLibrary,
    OCRBlock,
    RegionKind,
    BackgroundKind,
    group_ocr_blocks,
    estimate_initial_bg_color,
    detect_bubble_region,
    extract_block_colors,
    classify_region,
    decide_cleanup_strategy,
    compute_placement,
    erase_text_region,
    normalize_ocr_korean,
    heuristic_localize_line,
    sanitize_final_translation,
    clean_translation_text,
    contains_hangul,
    is_likely_garbage_literal,
    COMIC_FONTS_DIR,
    NLLB_MODEL_DIR,
    SFX_MAP,
    KR_SFX_MAP,
)

try:
    import ctranslate2
    import transformers
    _HAS_CTRANSLATE = True
except ImportError:
    _HAS_CTRANSLATE = False

try:
    from translator_v14 import NLLBTranslator
except ImportError:
    NLLBTranslator = None  # type: ignore

try:
    from translator_v14 import OllamaClient
except ImportError:
    import requests as _req
    class OllamaClient:  # type: ignore
        OLLAMA_URL = "http://localhost:11434/api/chat"
        TIMEOUT    = 180

        def chat_raw(self, model: str, prompt: str, image_b64: Optional[str] = None,
                     keep_alive: str = "5m") -> str:
            msgs = [{"role": "user", "content": prompt}]
            if image_b64:
                msgs = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ]}]
            body = {"model": model, "messages": msgs, "stream": False,
                    "keep_alive": keep_alive}
            resp = _req.post(self.OLLAMA_URL, json=body, timeout=self.TIMEOUT)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        def chat_json(self, model: str, prompt: str, schema: dict,
                      keep_alive: str = "5m") -> dict:
            body = {"model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False, "keep_alive": keep_alive,
                    "format": schema}
            resp = _req.post(self.OLLAMA_URL, json=body, timeout=self.TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            return json.loads(raw)


def _hex_color(rgb: Any, fallback: str = "#111111") -> str:
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        return fallback
    try:
        r, g, b = [max(0, min(255, int(v))) for v in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return fallback


# ── Memory system ─────────────────────────────────────────────────────────────
# The memory/ package lives at the project root alongside translator_v14.py.
# All imports are guarded: the engine degrades gracefully when the package is
# absent — translation still works, just without memory-assisted constraints.
_MEMORY_ROOT = str(_PROJECT_ROOT / "series_memory")

try:
    from memory import (
        GlossaryStore,
        NameMemory,
        ChapterTM,
        check_name_drift,
        check_glossary_drift,
    )
    _HAS_MEMORY = True
    _MEMORY_IMPORT_ERROR = ""
except ImportError as exc:
    _MEMORY_IMPORT_ERROR = str(exc)
    print(f"[memory] import failed: {exc}")
    traceback.print_exc()
    _HAS_MEMORY = False


# ────────────────────────────────────────────────────────────────────────────
class LocalizerEngine:
    """
    The single source of truth for all pipeline state.

    Public interface (called by api.py):
        import_chapter(folder)               → bootstrap dict
        go_to_page(idx)                      → bootstrap dict
        detect_current_page()                → bootstrap dict
        ocr_current_page()                   → bootstrap dict
        translate_current_page()             → bootstrap dict
        cleanup_current_page()               → bootstrap dict
        typeset_current_page()               → bootstrap dict
        run_all_steps()                      → bootstrap dict
        export_chapter(export_dir)           → export_dir str
        get_page_image_b64(idx)              → base64 PNG string
        get_bootstrap()                      → bootstrap dict
        update_region_field(id, field, value)→ bootstrap dict
        migrate_legacy_to_series(title)      → {"glossary": int, "names": int}
    """

    def __init__(self,
                 on_progress: Optional[Callable[[str, int, int], None]] = None) -> None:
        self._on_progress = on_progress or (lambda m, c, t: None)

        self.model_config  = ModelConfig.load()
        self.chapter_mgr   = ChapterManager()
        self.series_db     = SeriesDB()
        self.client        = OllamaClient()

        self._ocr_proc: Optional[OCRProcessor] = None
        self._nllb: Any = None
        self.font_lib: Optional[ComicFontLibrary] = None

        # Per-page working state  (always mirrors chapter_mgr.current_page)
        self._raw_cv: Optional[np.ndarray] = None
        self._regions: List[Any] = []          # List[OCRBlock]
        self._translations: List[str] = []

        self.busy   = False
        self.status = "Ready"

        self._lock = threading.Lock()

        # ── Memory system state ───────────────────────────────────────────────
        # Scope separation:
        #   _global_*   — cross-series rules; empty by default; very rarely used
        #   _series_*   — per-series rules; the primary store for names + terms
        #   _tm_store   — chapter-local provisional TM; NOT retrieved into prompts
        #
        # All four stores are None until import_chapter() is called.
        # Every memory hook checks _HAS_MEMORY and store is-not-None before use.
        self._global_glossary:    Optional[Any] = None   # GlossaryStore
        self._global_name_memory: Optional[Any] = None   # NameMemory
        self._series_glossary:    Optional[Any] = None   # GlossaryStore
        self._series_name_memory: Optional[Any] = None   # NameMemory
        self._tm_store:           Optional[Any] = None   # ChapterTM

        # Per-page post-translation data, keyed by 0-based page index.
        # Reset each time translate_current_page() runs on a given page.
        #
        # _consistency_warnings[page_idx] — list of issue-dicts for the UI
        # _memory_hits[page_idx][region_idx] — list of hit-dicts for the UI
        self._consistency_warnings: Dict[int, List[Dict[str, Any]]] = {}
        self._memory_hits:          Dict[int, Dict[int, List[Dict[str, Any]]]] = {}

        threading.Thread(target=self._init_models, daemon=True).start()

    # ── Model init ──────────────────────────────────────────────────────────

    def _init_models(self) -> None:
        self._notify("Loading EasyOCR…", 0, 2)
        try:
            self._ocr_proc = OCRProcessor()
            self._notify("EasyOCR ready", 1, 2)
        except Exception as exc:
            self._notify(f"EasyOCR failed: {exc}", 1, 2)

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

    def _notify(self, message: str, current: int = 0, total: int = 0) -> None:
        self.status = message
        self._on_progress(message, current, total)

    def _load_page_into_working_state(self) -> None:
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
        page = self.chapter_mgr.current_page
        if page is None:
            return
        page.regions      = self._regions
        page.translations = self._translations

    # ── Memory system ─────────────────────────────────────────────────────────

    def _init_memory(self, series_title: str) -> None:
        """
        Initialise memory stores for *series_title*.

        Called by import_chapter().  A new series always starts with empty
        stores — no automatic migration from NAME_MAP or GLOSSARY_ANCHORS.

        To seed a legacy project's series with the hardcoded dicts from
        translator_v14.py, call migrate_legacy_to_series(series_title)
        explicitly once after the chapter is loaded.

        Scope layout
        ────────────
        _global_*   → series_memory/_global/   — empty by default
        _series_*   → series_memory/<slug>/    — empty for a new series
        """
        if not _HAS_MEMORY:
            if _MEMORY_IMPORT_ERROR:
                self._notify(f"Memory unavailable: {_MEMORY_IMPORT_ERROR}")
            return
        try:
            self._global_glossary    = GlossaryStore(_MEMORY_ROOT, "_global")
            self._global_name_memory = NameMemory(_MEMORY_ROOT, "_global")
            self._series_glossary    = GlossaryStore(_MEMORY_ROOT, series_title)
            self._series_name_memory = NameMemory(_MEMORY_ROOT, series_title)

            n_sg = len(self._series_glossary.all_entries())
            n_sn = len(self._series_name_memory.all_entries())
            n_gg = len(self._global_glossary.all_entries())
            n_gn = len(self._global_name_memory.all_entries())
            self._notify(
                f"Memory: {n_sg} series glossary + {n_sn} series names"
                + (f" + {n_gg} global glossary + {n_gn} global names"
                   if n_gg or n_gn else "")
            )
        except Exception as exc:
            print(f"[memory] _init_memory failed: {exc}")
            self._global_glossary    = None
            self._global_name_memory = None
            self._series_glossary    = None
            self._series_name_memory = None

    def _init_chapter_tm(self, series_title: str, chapter_id: str) -> None:
        """
        Initialise the chapter-local provisional TM store.

        Called by import_chapter() after _init_memory().  Loads any entries
        already saved from a prior translation run on this chapter.
        """
        if not _HAS_MEMORY:
            return
        try:
            self._tm_store = ChapterTM(_MEMORY_ROOT, series_title, chapter_id)
        except Exception as exc:
            print(f"[memory] _init_chapter_tm failed: {exc}")
            traceback.print_exc()
            self._tm_store = None

    def _memory_stores_ready(self) -> bool:
        """True when all four memory stores are initialised."""
        return (
            _HAS_MEMORY and
            self._global_glossary    is not None and
            self._global_name_memory is not None and
            self._series_glossary    is not None and
            self._series_name_memory is not None
        )

    def _lookup_name_hits(self, kr_text: str) -> List[Any]:
        """
        Exact-match name lookup across series store (priority) then global.

        Series entries take precedence: if the same kr_name exists in both
        stores, only the series entry is returned.
        """
        hits: List[Any] = []
        seen: set = set()
        for entry in self._series_name_memory.exact_match(kr_text):
            if entry.kr_name not in seen:
                hits.append(entry)
                seen.add(entry.kr_name)
        for entry in self._global_name_memory.exact_match(kr_text):
            if entry.kr_name not in seen:
                hits.append(entry)
                seen.add(entry.kr_name)
        return hits

    def _lookup_gloss_hits(self, kr_text: str) -> List[Any]:
        """
        Exact-match glossary lookup across series store (priority) then global.
        """
        hits: List[Any] = []
        seen: set = set()
        for entry in self._series_glossary.exact_match(kr_text):
            if entry.source_kr not in seen:
                hits.append(entry)
                seen.add(entry.source_kr)
        for entry in self._global_glossary.exact_match(kr_text):
            if entry.source_kr not in seen:
                hits.append(entry)
                seen.add(entry.source_kr)
        return hits

    def _build_memory_constraint_block(self, texts: List[str]) -> str:
        """
        For a list of Korean source texts, retrieve exact glossary and name
        hits and format them as a compact TRANSLATION CONSTRAINTS block.

        Ordering
        ────────
        1. Name constraints first (higher semantic specificity).
        2. Glossary constraints second.
        Within each group: series entries before global; longer source_kr first.

        Deduplication: the same kr_name / source_kr is included at most once
        across the entire batch even if it matches in multiple input lines.

        Token budget
        ────────────
        Each constraint line is ~10–14 tokens.  With only exact matches and
        bounded hit sets, the block stays well within the ~250-token safety
        budget for TranslateGemma 12B on 12 GB VRAM.

        Returns "" when memory is unavailable or no hits found so the caller
        can use a simple ``if constraint_block:`` guard.
        """
        if not self._memory_stores_ready():
            return ""
        try:
            seen_name:  set = set()
            seen_gloss: set = set()
            name_hits:  List[Any] = []
            gloss_hits: List[Any] = []

            for text in texts:
                if not (text or "").strip():
                    continue
                for entry in self._lookup_name_hits(text):
                    if entry.kr_name not in seen_name:
                        name_hits.append(entry)
                        seen_name.add(entry.kr_name)
                for entry in self._lookup_gloss_hits(text):
                    if entry.source_kr not in seen_gloss:
                        gloss_hits.append(entry)
                        seen_gloss.add(entry.source_kr)

            lines: List[str] = []
            if name_hits:
                lines.extend(
                    NameMemory.to_prompt_block(name_hits).splitlines()
                )
            if gloss_hits:
                lines.extend(
                    GlossaryStore.to_prompt_block(gloss_hits).splitlines()
                )
            return "\n".join(lines)
        except Exception as exc:
            print(f"[memory] _build_memory_constraint_block failed: {exc}")
            return ""

    def _run_post_translation_memory(
        self,
        texts:       List[str],
        results:     List[str],
        page_idx:    int,
        chapter_dir: str,
    ) -> None:
        """
        Run all post-translation memory operations for one page.

        1. Per-region exact-match lookup  → populate self._memory_hits
        2. Consistency checks             → populate self._consistency_warnings
        3. TM storage                     → write to self._tm_store

        Kept as a single method so it can be called once from
        translate_current_page() without scattering try/except blocks.
        All errors are caught locally and printed; they never propagate
        to the translation pipeline.
        """
        if not self._memory_stores_ready():
            return

        page_hits:     Dict[int, List[Dict[str, Any]]] = {}
        page_warnings: List[Dict[str, Any]] = []

        # ── Per-region lookup + consistency check ─────────────────────────────
        try:
            for i, (kr, en) in enumerate(zip(texts, results)):
                region_hit_list: List[Dict[str, Any]] = []
                label = f"R-{i+1:02d}"

                if not (kr or "").strip():
                    page_hits[i] = region_hit_list
                    continue

                n_hits = self._lookup_name_hits(kr)
                g_hits = self._lookup_gloss_hits(kr)

                # Build memory_hits for UI surfacing
                for e in n_hits:
                    region_hit_list.append({
                        "type":       "name",
                        "kr":         e.kr_name,
                        "en":         e.en_name,
                        "trust":      e.trust,
                        "matched_by": "exact",
                        "scope":      e.scope,
                    })
                for e in g_hits:
                    region_hit_list.append({
                        "type":       "glossary",
                        "kr":         e.source_kr,
                        "en":         e.target_en,
                        "trust":      e.trust,
                        "matched_by": "exact",
                        "scope":      e.scope,
                    })
                page_hits[i] = region_hit_list

                # Consistency checks (only when a translation was produced)
                if not (en or "").strip():
                    continue

                for w in check_name_drift(kr, en, n_hits, page_idx, i):
                    page_warnings.append({
                        "id":     f"mem-{page_idx+1}-{i+1}-name_drift",
                        "sev":    "warn",
                        "msg":    (
                            f'Name drift — "{w.expected}" expected '
                            f"but not found in translation"
                        ),
                        "region": label,
                        "page":   page_idx + 1,
                    })

                for w in check_glossary_drift(kr, en, g_hits, page_idx, i):
                    page_warnings.append({
                        "id":     f"mem-{page_idx+1}-{i+1}-glossary_drift",
                        "sev":    "warn",
                        "msg":    (
                            f'Glossary drift — "{w.expected}" expected '
                            f"but not found in translation"
                        ),
                        "region": label,
                        "page":   page_idx + 1,
                    })

        except Exception as exc:
            print(f"[memory] per-region lookup/check failed: {exc}")

        self._memory_hits[page_idx]          = page_hits
        self._consistency_warnings[page_idx] = page_warnings

        # ── TM storage ────────────────────────────────────────────────────────
        if self._tm_store is None:
            return
        try:
            # Identify which regions were flagged by _flag_translations()
            flagged_idx: set = {
                i for i, block in enumerate(self._regions)
                if getattr(block, "is_flagged", False)
            }
            batch = [
                (kr, en, page_idx, i, i in flagged_idx)
                for i, (kr, en) in enumerate(zip(texts, results))
                if (kr or "").strip() and (en or "").strip()
            ]
            if batch:
                n_stored = self._tm_store.store_batch(batch, chapter_dir)
                if n_stored:
                    prov = self._tm_store.provisional_count()
                    print(
                        f"[memory] stored {n_stored} TM entries "
                        f"({prov} provisional total in chapter)"
                    )
        except Exception as exc:
            print(f"[memory] TM storage failed: {exc}")

    # ── Chapter management ──────────────────────────────────────────────────

    def import_chapter(self, folder: str) -> dict:
        n = self.chapter_mgr.load_from_folder(folder)
        if n == 0:
            raise ValueError("No image files found in that folder.")
        self.chapter_mgr.load_state()
        chapter_name = os.path.basename(folder)
        series_title = os.path.basename(os.path.dirname(folder)) or chapter_name
        self.series_db.register_chapter(series_title, folder, chapter_name, n)

        # Memory: load stores for this series.
        # New series → empty glossary + empty names.  No auto-migration.
        self._init_memory(series_title)
        self._init_chapter_tm(series_title, chapter_name)

        # Clear per-page memory caches when a new chapter is loaded.
        self._consistency_warnings.clear()
        self._memory_hits.clear()

        self._load_page_into_working_state()
        self._notify(f"Loaded {n} pages from {chapter_name}")
        return self.get_bootstrap()

    def go_to_page(self, idx: int) -> dict:
        self._flush_working_state_to_page()
        self.chapter_mgr.go_to(idx)
        self._load_page_into_working_state()
        self._notify(f"Page {idx + 1} / {self.chapter_mgr.total_pages()}")
        return self.get_bootstrap()

    # ── Pipeline steps ──────────────────────────────────────────────────────

    def detect_current_page(self) -> dict:
        if self._raw_cv is None:
            raise RuntimeError("No image loaded — import a chapter first.")
        if self._ocr_proc is None:
            raise RuntimeError("EasyOCR not ready yet — wait a moment.")

        self._notify("Detecting text regions…", 0, 1)
        tmp = "_detect_tmp.png"
        cv2.imwrite(tmp, self._raw_cv)
        try:
            blocks = self._ocr_proc.detect(tmp)
            blocks = group_ocr_blocks(blocks)
            n = len(blocks)
            for i, block in enumerate(blocks):
                self._notify(f"Enriching region {i+1}/{n}…", i, n)
                bb = block.bbox()
                bg_rgb = estimate_initial_bg_color(self._raw_cv, bb)
                bub_bbox, bub_mask = detect_bubble_region(self._raw_cv, bb, bg_rgb)
                block.bubble_bbox = bub_bbox
                block.bubble_mask = bub_mask
                block.bg_color, block.fg_color = extract_block_colors(self._raw_cv, block)
                if self.font_lib:
                    block.bubble_role = self.font_lib.pick_role(self._raw_cv, bb, block.text)
                classify_region(self._raw_cv, block)
                decide_cleanup_strategy(block)
                compute_placement(self._raw_cv, block)
            self._regions      = blocks
            self._translations = [""] * n
            self._flush_working_state_to_page()
            self._notify(f"Detected {n} region(s).", n, n)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return self.get_bootstrap()

    def ocr_current_page(self) -> dict:
        if not self._regions:
            raise RuntimeError("Run Detect first.")
        total = len(self._regions)
        self._notify(f"OCR: 0/{total}…", 0, total)
        for idx in range(total):
            self._notify(f"OCR region {idx+1}/{total}…", idx, total)
            text = self._ocr_one_region(idx)
            self._regions[idx].text = text
        self._flush_working_state_to_page()
        self._notify(f"OCR complete — {total} region(s).", total, total)
        return self.get_bootstrap()

    def _ocr_one_region(self, idx: int) -> str:
        block = self._regions[idx]
        x, y, w, h = block.bbox()
        crop = self._raw_cv[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            return block.text
        _, buf = cv2.imencode(".png", crop)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        try:
            raw = self.client.chat_raw(
                model=self.model_config.ocr_model,
                prompt=(
                    "Read ONLY the Korean text visible in this speech bubble image. "
                    "Output ONLY the exact Korean characters — no explanation, no translation, "
                    "no punctuation changes, no extra words."
                ),
                image_b64=b64,
                keep_alive=self.model_config.keep_alive,
            )
            text = normalize_ocr_korean(raw.strip())
            return text if text else block.text
        except Exception as exc:
            print(f"[OCR] region {idx} Ollama failed: {exc} — using EasyOCR text")
            return block.text

    def translate_current_page(self) -> dict:
        if not self._regions:
            raise RuntimeError("Run Detect first.")
        texts = [b.text for b in self._regions]
        self._notify(f"Translating {len(texts)} region(s)…", 0, len(texts))
        results = self._translate_texts(texts)
        for i, t in enumerate(results):
            if i < len(self._translations):
                self._translations[i] = t
            else:
                self._translations.append(t)
        self._flag_translations(list(range(len(results))))

        # ── Memory: post-translation checks + TM storage ─────────────────────
        # This runs entirely outside the model — pure Python string matching
        # and JSON writes.  Errors are caught inside; never affects translation.
        page_idx    = int(getattr(self.chapter_mgr, "current_idx", 0))
        chapter_dir = getattr(self.chapter_mgr, "chapter_dir", "") or ""
        self._run_post_translation_memory(texts, results, page_idx, chapter_dir)
        # ─────────────────────────────────────────────────────────────────────

        self._flush_working_state_to_page()

        flagged   = sum(1 for b in self._regions if getattr(b, "is_flagged", False))
        mem_warned = len(self._consistency_warnings.get(page_idx, []))
        msg = f"Translation complete — {len(results)} line(s)."
        if flagged:
            msg += f"  ⚑ {flagged} region(s) flagged."
        if mem_warned:
            msg += f"  ⚠ {mem_warned} memory warning(s)."
        self._notify(msg, len(results), len(results))
        return self.get_bootstrap()

    def _translate_texts(self, texts: List[str]) -> List[str]:
        """
        Translate a list of Korean strings.

        Memory integration
        ──────────────────
        Before building the prompt, exact glossary and name hits for the batch
        are retrieved and formatted as a TRANSLATION CONSTRAINTS block that is
        prepended.  This is bounded: only exact substring matches, no fuzzy
        retrieval, so prompt size stays stable regardless of chapter length.

        Series-specific entries take precedence over global entries.  A new
        series with empty stores produces no constraint block at all.
        """
        try:
            from translator_v14 import (
                TRANSLATOR_SCHEMA, VISION_SCHEMA, POLISHER_SCHEMA,
                CharacterMemory,
            )
            has_full_pipeline = True
        except ImportError:
            has_full_pipeline = False

        results: List[str] = []

        if has_full_pipeline and hasattr(self.client, "chat_json"):
            try:
                numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))

                # ── Memory: constraint block for the batch ────────────────────
                constraint_block = self._build_memory_constraint_block(texts)
                prompt = (
                    "Translate each numbered Korean line to natural English manga dialogue.\n"
                    "Output JSON with 'translated_lines' array in the same order.\n"
                )
                if constraint_block:
                    prompt += (
                        "\nTRANSLATION CONSTRAINTS (mandatory — apply to all lines):\n"
                        f"{constraint_block}\n"
                    )
                prompt += f"\n{numbered}"
                # ─────────────────────────────────────────────────────────────

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

        # Fallback: translate one at a time
        for i, kr in enumerate(texts):
            self._notify(f"Translating {i+1}/{len(texts)}…", i, len(texts))
            heuristic = heuristic_localize_line(kr)
            if heuristic:
                results.append(heuristic)
                continue
            try:
                cblock  = self._build_memory_constraint_block([kr])
                prompt  = (
                    "Translate this Korean manhwa dialogue to natural English. "
                    "Output only the English translation, nothing else."
                )
                if cblock:
                    prompt += f"\n\nTRANSLATION CONSTRAINTS (mandatory):\n{cblock}"
                prompt += f"\n\n{kr}"
                raw = self.client.chat_raw(
                    model=self.model_config.translate_model,
                    prompt=prompt,
                    keep_alive=self.model_config.keep_alive,
                )
                results.append(sanitize_final_translation(kr, raw))
            except Exception as exc:
                print(f"[translate] region {i} failed: {exc}")
                results.append(heuristic_localize_line(kr) or "")
        return results

    def _flag_translations(self, indices: List[int]) -> None:
        """Heuristic post-translate flagging — mirrors ManhwaLocalizerApp._flag_translations."""
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
            raise RuntimeError("Run Detect first.")

        self._notify("Running cleanup…", 0, 1)
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")

        try:
            cleaned = erase_text_region(self._raw_cv, self._regions)
            page.cleaned_cv = cleaned
            self._notify("Cleanup complete ✓", 1, 1)
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Cleanup failed: {exc}") from exc

        self._flush_working_state_to_page()
        return self.get_bootstrap()

    def typeset_current_page(self) -> dict:
        page = self.chapter_mgr.current_page
        if page is None:
            raise RuntimeError("No current page.")

        base_cv = page.cleaned_cv if page.cleaned_cv is not None else self._raw_cv
        if base_cv is None:
            raise RuntimeError("No image loaded.")
        if not any(t and t.strip() for t in self._translations):
            raise RuntimeError("Run Translate first.")

        self._notify("Rendering translations…", 0, 1)
        try:
            pil_out = self._typeset_image(base_cv)
            page.typeset_pil = pil_out
            self._notify("Typeset complete ✓", 1, 1)
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Typeset failed: {exc}") from exc

        self._flush_working_state_to_page()
        return self.get_bootstrap()

    def _typeset_image(self, base_cv: np.ndarray) -> Image.Image:
        try:
            from translator_v14 import ManhwaLocalizerApp
            class _Shim:
                pass
            shim = _Shim()
            shim._regions      = self._regions
            shim._translations = self._translations
            shim.font_lib      = self.font_lib
            shim.model_config  = self.model_config
            bound = ManhwaLocalizerApp._typeset_image.__get__(shim, type(shim))
            return bound(base_cv)
        except Exception:
            from PIL import ImageDraw
            pil = Image.fromarray(cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for block, trans in zip(self._regions, self._translations):
                if not trans:
                    continue
                x, y, w, h = block.bbox()
                draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
                draw.text((x + 4, y + 4), trans, fill=(0, 0, 0))
            return pil

    def run_all_steps(self) -> dict:
        self.detect_current_page()
        self.ocr_current_page()
        self.translate_current_page()
        self.cleanup_current_page()
        self.typeset_current_page()
        self.chapter_mgr.save_state()
        return self.get_bootstrap()

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

    def update_region_field(self, region_idx: int, field: str, value: Any) -> dict:
        if not (0 <= region_idx < len(self._regions)):
            raise IndexError(f"No region at index {region_idx}")
        block = self._regions[region_idx]
        if field == "translation":
            while len(self._translations) <= region_idx:
                self._translations.append("")
            self._translations[region_idx] = str(value)
        elif field in {"text", "font_name", "font_size", "align", "visible", "locked"}:
            setattr(block, field, value)
            page = self.chapter_mgr.current_page
            if page:
                page.typeset_pil = None
        self._flush_working_state_to_page()
        return self.get_bootstrap()

    # ── Image access ─────────────────────────────────────────────────────────

    def get_page_image_b64(self, idx: int) -> Optional[str]:
        pages = self.chapter_mgr.pages
        if not (0 <= idx < len(pages)):
            return None
        page = pages[idx]

        pil: Optional[Image.Image] = None
        if page.typeset_pil is not None:
            pil = page.typeset_pil
        elif page.cleaned_cv is not None:
            pil = Image.fromarray(cv2.cvtColor(page.cleaned_cv, cv2.COLOR_BGR2RGB))
        else:
            try:
                pil = Image.open(page.image_path)
            except Exception:
                return None

        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Bootstrap serialiser ─────────────────────────────────────────────────

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
                progress, status = self._chapter_progress(folder, pages, chapter_mgr)
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
                "lang":     "ko→en",
                "chapters": len(series.get("chapters", []) or []),
                "color":    ["#8b7cf8", "#e8a454", "#4ec9b4", "#6090e8"][i % 4],
            })

        if not series_entries:
            active_series_id = "s1"
            series_entries = [{
                "id": "s1", "title": current_series_title,
                "subtitle": "local", "lang": "ko→en",
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
            has_trans = (
                has_reg
                and len(pg_tl) >= len(getattr(page, "regions", []) or [])
                and all(
                    (t or "").strip()
                    for t in pg_tl[:len(getattr(page, "regions", []) or [])]
                )
            )
            page_summaries.append({
                "id":      f"p{idx}",
                "idx":     idx,
                "regions": len(getattr(page, "regions", []) or []),
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
        regions_src = self._regions
        trans_src   = self._translations

        # Memory data for the active page
        page_hits = self._memory_hits.get(active_page_idx, {})

        region_entries: List[Dict[str, Any]] = []
        issues:         List[Dict[str, Any]] = []

        for idx, block in enumerate(regions_src):
            x, y, w, h = block.bbox()
            tl    = trans_src[idx] if idx < len(trans_src) else ""
            rid   = f"r{idx+1}"
            label = f"R-{idx+1:02d}"
            font_name = (
                getattr(block, "font_name", "")
                or getattr(block, "bubble_role", "dialog")
                or "auto"
            )
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
                "fg":      _hex_color(getattr(block, "fg_color", None), "#111111"),
                "bg":      _hex_color(getattr(block, "bg_color", None), "#ffffff"),
                "align":   getattr(block, "align", "center") or "center",
                "visible": bool(getattr(block, "visible", True)),
                "locked":  bool(getattr(block, "locked", False)),
                "cleanup_strategy": getattr(block, "cleanup_strategy", "auto") or "auto",
                "region_kind": (
                    getattr(block.region_kind, "name", None)
                    if getattr(block, "region_kind", None) is not None
                    else None
                ),
                "layers": [
                    {"id": f"{rid}-region",  "type": "region",  "name": f"Region – {idx+1}",  "vis": True,  "locked": False},
                    {"id": f"{rid}-text",    "type": "text",    "name": f"Text – {idx+1}",    "vis": bool(getattr(block, "visible", True)), "locked": bool(getattr(block, "locked", False))},
                    {"id": f"{rid}-cleanup", "type": "cleanup", "name": f"Cleanup – {idx+1}", "vis": True,  "locked": False},
                ],
                # ── Memory data ─────────────────────────────────────────────
                # List of {"type", "kr", "en", "trust", "matched_by", "scope"}
                # Empty list when memory is unavailable or no hits for this region.
                # The UI can ignore this field if it doesn't yet render it.
                "memory_hits": page_hits.get(idx, []),
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
                    "id":     f"issue-{idx+1}",
                    "sev":    sev,
                    "msg":    reason.replace("_", " ").capitalize(),
                    "region": label,
                    "page":   active_page_idx + 1,
                })
            elif float(getattr(block, "confidence", 0.0) or 0.0) < 0.4:
                issues.append({
                    "id":     f"issue-conf-{idx+1}",
                    "sev":    "warn",
                    "msg":    (
                        f"OCR confidence below threshold "
                        f"({int(round(float(getattr(block, 'confidence', 0.0) or 0.0) * 100))}%)"
                    ),
                    "region": label,
                    "page":   active_page_idx + 1,
                })

        if not region_entries:
            issues.append({
                "id":     "no-regions",
                "sev":    "info",
                "msg":    ("No chapter imported yet." if not pages else "No regions on this page."),
                "region": None,
                "page":   active_page_idx + 1,
            })

        # ── Consistency warnings for the active page ─────────────────────────
        # Only the active page's warnings are shown; stale warnings from other
        # pages are not included here (they exist in self._consistency_warnings
        # and reappear if you navigate back to that page).
        for w in self._consistency_warnings.get(active_page_idx, []):
            issues.append(w)

        # ── Chapter progress ─────────────────────────────────────────────────
        chapter_progress, _ = self._chapter_progress(
            getattr(chapter_mgr, "chapter_dir", ""), pages, chapter_mgr)

        return {
            "series":   series_entries,
            "chapters": chapters_by_series,
            "pages":    page_summaries,
            "regions":  region_entries,
            "issues":   issues,
            "meta": {
                "activeSeriesId":  active_series_id or (series_entries[0]["id"] if series_entries else "s1"),
                "activeChapterId": active_chapter_id,
                "activePageIdx":   active_page_idx,
                "busy":            self.busy,
                "status":          self.status,
                "chapterDir":      getattr(chapter_mgr, "chapter_dir", "") or "",
                "totalPages":      len(page_summaries),
                "chapterProgress": chapter_progress,
                # TM stats for the current chapter (None when memory is disabled)
                "tmProvisional": (
                    self._tm_store.provisional_count()
                    if self._tm_store is not None else None
                ),
                "tmTotal": (
                    self._tm_store.total_count()
                    if self._tm_store is not None else None
                ),
            },
        }

    def _current_series_title(self) -> str:
        chapter_dir = getattr(self.chapter_mgr, "chapter_dir", "") or ""
        if chapter_dir:
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
        total   = len(pages)
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
            if hasattr(self.model_config, k) and v:
                setattr(self.model_config, k, str(v))
        self.model_config.save()
        return {"ok": True, "config": self.model_config.to_dict()}

    def get_model_config(self) -> dict:
        return self.model_config.to_dict()

    # ── Memory management (public API) ────────────────────────────────────────

    def migrate_legacy_to_series(self, series_title: str) -> Dict[str, int]:
        """
        One-time opt-in migration of the hardcoded NAME_MAP and
        GLOSSARY_ANCHORS from translator_v14.py into the given series'
        memory store.

        This is intentionally NOT called automatically.  Call it once for
        legacy projects that were built around those specific dicts:

            engine.migrate_legacy_to_series("A Knight Living Only for Today")

        Safe to call multiple times — both helpers are idempotent (existing
        entries are never overwritten).

        Returns
        -------
        dict
            {"glossary": <count of new entries>, "names": <count of new entries>}
            Both counts are 0 on subsequent calls.

        Raises
        ------
        RuntimeError
            If NAME_MAP or GLOSSARY_ANCHORS cannot be imported from
            translator_v14.py (e.g. wrong project).
        RuntimeError
            If the memory package is not available.
        """
        if not _HAS_MEMORY:
            raise RuntimeError(
                "Memory package not available — cannot migrate legacy data."
            )
        try:
            from translator_v14 import NAME_MAP, GLOSSARY_ANCHORS
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import NAME_MAP / GLOSSARY_ANCHORS from translator_v14.py: {exc}"
            ) from exc

        gs = GlossaryStore(_MEMORY_ROOT, series_title)
        nm = NameMemory(_MEMORY_ROOT, series_title)
        n_g = gs.migrate_from_anchors(GLOSSARY_ANCHORS)
        n_n = nm.migrate_from_name_map(NAME_MAP)

        # If this is the currently-loaded series, reload the live stores.
        current_title = self._current_series_title()
        if series_title == current_title:
            self._series_glossary    = gs
            self._series_name_memory = nm

        self._notify(
            f"Legacy migration: {n_g} glossary + {n_n} name entries "
            f"added to \"{series_title}\""
        )
        return {"glossary": n_g, "names": n_n}

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
