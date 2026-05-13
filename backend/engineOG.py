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
        Only approved + non-flagged entries are retrieved (Phase 4).

Phase 4  — retrieve_batch() builds a bounded prompt block per translate call.
Phase 5  — approve / reject / promote API; all require explicit calls.
Phase 6  — blocked mappings checked post-translation; rejections auto-block.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import sys
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
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BACKEND_DIR  = pathlib.Path(__file__).resolve().parent
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    # Fallback minimal client if the name differs in the file
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


# ── Memory package (optional — engine degrades gracefully if absent) ──────────
try:
    from memory import (
        GlossaryStore,
        NameMemory,
        ChapterTM,
        BlockedMappingStore,
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
    _HAS_MEMORY = True
except ImportError as _mem_err:
    print(f"[engine] memory package unavailable: {_mem_err} — running without memory")
    _HAS_MEMORY = False

_MEMORY_ROOT = str(_PROJECT_ROOT / "series_memory")


def _hex_color(rgb: Any, fallback: str = "#111111") -> str:
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        return fallback
    try:
        r, g, b = [max(0, min(255, int(v))) for v in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return fallback


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
                 on_progress: Optional[Callable[[str, int, int], None]] = None) -> None:
        """
        on_progress(message, current, total) is called whenever a step makes
        progress.  The api layer turns this into a JS event push.
        """
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

        # ── Memory stores (None until _init_memory is called on import_chapter)
        self._memory_root:     str = _MEMORY_ROOT
        self._series_title:    str = ""
        self._chapter_id:      str = ""

        self._global_glossary: Any = None   # GlossaryStore | None
        self._global_names:    Any = None   # NameMemory    | None
        self._global_blocked:  Any = None   # BlockedMappingStore | None
        self._glossary:        Any = None   # GlossaryStore | None
        self._name_mem:        Any = None   # NameMemory    | None
        self._blocked:         Any = None   # BlockedMappingStore | None
        self._chapter_tm:      Any = None   # ChapterTM     | None

        # Consistency warnings from the most-recent translate_current_page() call.
        # Cleared at the start of each call and appended to issues in get_bootstrap().
        self._consistency_warnings: List[Dict[str, Any]] = []

        # Kick model initialisation off the main thread
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

    # ── Memory initialisation ────────────────────────────────────────────────

    def _init_memory(self, series_title: str, chapter_id: str) -> None:
        """
        Initialise all three memory scopes for the current series/chapter.

        New series always start with EMPTY stores.  No migration is performed
        here — call migrate_legacy_series() explicitly for legacy projects.
        """
        self._series_title = series_title
        self._chapter_id   = chapter_id
        if not _HAS_MEMORY:
            return
        try:
            self._global_glossary = GlossaryStore(self._memory_root, "_global")
            self._global_names    = NameMemory(self._memory_root, "_global")
            self._global_blocked  = BlockedMappingStore(self._memory_root, "_global")
            self._glossary        = GlossaryStore(self._memory_root, series_title)
            self._name_mem        = NameMemory(self._memory_root, series_title)
            self._blocked         = BlockedMappingStore(self._memory_root, series_title)
            self._chapter_tm      = ChapterTM(self._memory_root, series_title, chapter_id)
            print(
                f"[memory] series={series_title!r} chapter={chapter_id!r} | "
                f"glossary={len(self._glossary.all_entries())} "
                f"names={len(self._name_mem.all_entries())} "
                f"blocked={len(self._blocked.all_entries())} "
                f"tm={len(self._chapter_tm.all_entries())}"
            )
        except Exception as exc:
            print(f"[memory] init failed: {exc}")

    def _merged_blocked(self) -> List[Any]:
        """Return merged global + series blocked entries."""
        out: List[Any] = []
        if self._global_blocked:
            out.extend(self._global_blocked.all_entries())
        if self._blocked:
            out.extend(self._blocked.all_entries())
        return out

    # ── Chapter management ──────────────────────────────────────────────────

    def import_chapter(self, folder: str) -> dict:
        n = self.chapter_mgr.load_from_folder(folder)
        if n == 0:
            raise ValueError("No image files found in that folder.")
        self.chapter_mgr.load_state()
        chapter_name  = os.path.basename(folder)
        series_title  = os.path.basename(os.path.dirname(folder)) or chapter_name
        self.series_db.register_chapter(series_title, folder, chapter_name, n)
        self._load_page_into_working_state()
        # Initialise memory — new series always starts empty.
        self._init_memory(series_title, chapter_name)
        self._consistency_warnings = []
        self._notify(f"Loaded {n} pages from {chapter_name}")
        return self.get_bootstrap()

    def go_to_page(self, idx: int) -> dict:
        self._flush_working_state_to_page()
        self.chapter_mgr.go_to(idx)
        self._load_page_into_working_state()
        self._consistency_warnings = []
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
                # ── Phase 1: classify region and assign cleanup strategy ──────────
                classify_region(self._raw_cv, block)
                decide_cleanup_strategy(block)
                # ── Phase 2: compute safe placement rect from bubble interior ────
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
        self._consistency_warnings = []
        texts = [b.text for b in self._regions]
        page_idx = int(getattr(self.chapter_mgr, "current_idx", 0) or 0)
        self._notify(f"Translating {len(texts)} region(s)…", 0, len(texts))
        results = self._translate_texts(texts, page_idx=page_idx)
        for i, t in enumerate(results):
            if i < len(self._translations):
                self._translations[i] = t
            else:
                self._translations.append(t)
        self._flag_translations(list(range(len(results))))
        self._flush_working_state_to_page()
        flagged = sum(1 for b in self._regions if getattr(b, "is_flagged", False))
        drift   = len(self._consistency_warnings)
        msg = f"Translation complete — {len(results)} line(s)."
        if flagged:
            msg += f"  ⚑ {flagged} region(s) flagged."
        if drift:
            msg += f"  ⚠ {drift} memory warning(s)."
        self._notify(msg, len(results), len(results))
        return self.get_bootstrap()

    def _translate_texts(self, texts: List[str], page_idx: int = 0) -> List[str]:
        """
        Translate a list of Korean strings.

        Phase 4: retrieve_batch() is called first to build a bounded memory
        block (constraint_block + tm_examples) injected into the prompt.
        Machine-only TM entries are never retrieved.

        Phase 6: blocked mappings are checked post-translation via
        _post_translate(), which also stores results to ChapterTM.
        """
        # Attempt to import the full translation pipeline from the legacy file
        try:
            from translator_v14 import (
                TRANSLATOR_SCHEMA, VISION_SCHEMA, POLISHER_SCHEMA,
                CharacterMemory,
            )
            has_full_pipeline = True
        except ImportError:
            has_full_pipeline = False

        # ── Phase 4: build bounded memory context for this batch ──────────────
        batch_ctx     = self._retrieve_batch_context(texts)
        prompt_prefix = self._build_prompt_prefix(batch_ctx)

        results: List[str] = []

        if has_full_pipeline and hasattr(self.client, "chat_json"):
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
                self._post_translate(texts, results, batch_ctx, page_idx)
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

        self._post_translate(texts, results, batch_ctx, page_idx)
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
            n_global   = self._global_names.all_entries()    if self._global_names    else []
            n_series   = self._name_mem.all_entries()        if self._name_mem        else []
            tm_entries = self._chapter_tm.retrievable_entries() if self._chapter_tm   else []
            blocked    = self._merged_blocked()
            return retrieve_batch(
                texts,
                g_global, g_series,
                n_global, n_series,
                tm_entries,
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
        if batch_ctx.tm_examples:
            parts.append(batch_ctx.tm_examples)
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

        Populates self._consistency_warnings; called once per translate pass.
        Silently skips when memory is unavailable.
        """
        if not _HAS_MEMORY:
            return

        new_warnings: List[Dict[str, Any]] = []
        blocked_entries = self._merged_blocked()

        for i, (kr, en) in enumerate(zip(texts, results)):
            g_hits = (batch_ctx.per_line_glossary[i]
                      if batch_ctx and i < len(batch_ctx.per_line_glossary) else [])
            n_hits = (batch_ctx.per_line_names[i]
                      if batch_ctx and i < len(batch_ctx.per_line_names) else [])

            # Consistency checks
            w_name     = check_name_drift(kr, en, n_hits, page_idx, i)
            w_glossary = check_glossary_drift(kr, en, g_hits, page_idx, i)

            # Phase 6: blocked output check (series store takes precedence)
            fired: List[Any] = []
            if self._blocked:
                fired = self._blocked.matches(kr, en)
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
            if self._chapter_tm and en.strip():
                block_flagged = False
                if i < len(self._regions):
                    review = getattr(self._regions[i], "review", None)
                    block_flagged = bool(review and getattr(review, "flagged", False))
                try:
                    self._chapter_tm.store(
                        kr, en, page_idx, i,
                        flagged=block_flagged or bool(w_blocked),
                    )
                except Exception as exc:
                    print(f"[memory] ChapterTM.store failed: {exc}")

        self._consistency_warnings = new_warnings

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
        """Render all translations onto base_cv.  Mirrors ManhwaLocalizerApp._typeset_image."""
        # Delegate to the legacy implementation via a thin shim to avoid duplicating
        # the complex font / wrap / draw logic.
        try:
            from translator_v14 import ManhwaLocalizerApp
            # Build a headless shim that only has the state _typeset_image needs
            class _Shim:
                pass
            shim = _Shim()
            shim._regions      = self._regions
            shim._translations = self._translations
            shim.font_lib      = self.font_lib
            shim.model_config  = self.model_config
            # Bind the unbound method onto the shim
            bound = ManhwaLocalizerApp._typeset_image.__get__(shim, type(shim))
            return bound(base_cv)
        except Exception as exc:
            # Last-resort fallback: PIL simple text overlay
            from PIL import ImageDraw, ImageFont
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
            # Invalidate typeset cache
            page = self.chapter_mgr.current_page
            if page:
                page.typeset_pil = None
        self._flush_working_state_to_page()
        return self.get_bootstrap()

    # ── Image access ─────────────────────────────────────────────────────────

    def get_page_image_b64(self, idx: int) -> Optional[str]:
        """
        Return the best available image for page `idx` as a base64-encoded PNG.
        Priority: typeset > cleaned > raw.
        Returns None if no image is available.
        """
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

    # ── Phase 5: TM approval API ─────────────────────────────────────────────

    def approve_tm_entry(self, entry_id: str) -> Dict[str, Any]:
        """Approve a ChapterTM entry so it becomes eligible for retrieval."""
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
        Human must re-approve before the entry becomes retrievable.
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

    # ── Legacy migration (opt-in, call once for legacy project only) ──────────

    def migrate_legacy_series(self, series_title: str) -> Dict[str, Any]:
        """
        One-time seed of NAME_MAP and GLOSSARY_ANCHORS from translator_v14.py
        into the per-series memory stores for *series_title*.

        NEVER called automatically.  Call once for the known legacy project.
        Subsequent calls are idempotent (returns 0, 0).
        """
        if not _HAS_MEMORY:
            return {"migrated_glossary": 0, "migrated_names": 0,
                    "error": "memory package unavailable"}
        try:
            from translator_v14 import NAME_MAP, GLOSSARY_ANCHORS
        except ImportError as exc:
            return {"migrated_glossary": 0, "migrated_names": 0, "error": str(exc)}
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
            return {"available": False}
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
            has_trans = has_reg and len(pg_tl) >= len(getattr(page, "regions", []) or []) \
                        and all((t or "").strip() for t in pg_tl[:len(getattr(page, "regions", []) or [])])
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
        current_page    = chapter_mgr.current_page
        regions_src = self._regions
        trans_src   = self._translations

        region_entries: List[Dict[str, Any]] = []
        issues:         List[Dict[str, Any]] = []

        for idx, block in enumerate(regions_src):
            x, y, w, h = block.bbox()
            tl    = trans_src[idx] if idx < len(trans_src) else ""
            rid   = f"r{idx+1}"
            label = f"R-{idx+1:02d}"
            font_name = getattr(block, "font_name", "") or getattr(block, "bubble_role", "dialog") or "auto"
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
                # ── Phase 1 classification fields (safe to ignore if unused by UI) ──
                "cleanup_strategy": getattr(block, "cleanup_strategy", "auto") or "auto",
                "region_kind":      (getattr(block.region_kind, "name", None)
                                     if getattr(block, "region_kind", None) is not None
                                     else None),
                "layers": [
                    {"id": f"{rid}-region",  "type": "region",  "name": f"Region – {idx+1}",  "vis": True,  "locked": False},
                    {"id": f"{rid}-text",    "type": "text",    "name": f"Text – {idx+1}",    "vis": bool(getattr(block, "visible", True)), "locked": bool(getattr(block, "locked", False))},
                    {"id": f"{rid}-cleanup", "type": "cleanup", "name": f"Cleanup – {idx+1}", "vis": True,  "locked": False},
                ],
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
            "meta": {
                "activeSeriesId":  active_series_id or (series_entries[0]["id"] if series_entries else "s1"),
                "activeChapterId": active_chapter_id,
                "activePageIdx":   active_page_idx,
                "busy":            self.busy,
                "status":          self.status,
                "chapterDir":      getattr(chapter_mgr, "chapter_dir", "") or "",
                "totalPages":      len(page_summaries),
                "chapterProgress": chapter_progress,
                # Phase 4–6 additions — safe for the UI to ignore if not yet consumed
                "memoryStats":          self.get_memory_stats(),
                "pendingReviewCount":   (
                    len(self._chapter_tm.pending_review())
                    if (_HAS_MEMORY and self._chapter_tm) else 0
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
            if hasattr(self.model_config, k) and v:
                setattr(self.model_config, k, str(v))
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