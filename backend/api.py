"""
backend/api.py
──────────────
PywebviewAPI: the object passed as js_api to webview.create_window().

Every PUBLIC method on this class becomes callable from JavaScript as:
    await window.pywebview.api.methodName(arg1, arg2)

Rules:
  • Methods must accept only JSON-serialisable arguments (str, int, float, bool, list, dict, None).
  • Return values must be JSON-serialisable.
  • Methods run in a background thread managed by pywebview — they can block.
  • To push events TO JavaScript (progress, notifications), call
    self._push_event(name, payload) which calls window.evaluate_js().

No Tkinter.  No FastAPI.  No CORS headers.  This is the only file that
imports webview.
"""

from __future__ import annotations

import json
import threading
import traceback
from typing import Any, Optional

import webview  # pywebview

from backend.engine import LocalizerEngine


class PywebviewAPI:
    """
    Thin adapter between pywebview's JS bridge and LocalizerEngine.

    Separation of concerns
    ──────────────────────
    - This class handles:  JS ↔ Python boundary, error wrapping, event pushing
    - LocalizerEngine handles:  all actual pipeline logic and state

    Every method returns a dict that always has:
        {"ok": True,  ...result fields...}    on success
        {"ok": False, "error": "...message"}  on failure
    This way the React side never has to catch exceptions from the API.
    """

    def __init__(self, engine: LocalizerEngine) -> None:
        self._engine = engine
        self._window: Optional[webview.Window] = None
        self._push_lock = threading.Lock()

        # Wire engine progress notifications to JS event pushes
        engine._on_progress = self._on_engine_progress

    def set_window(self, window: webview.Window) -> None:
        """Called by launcher.py after the window is created."""
        self._window = window

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _on_engine_progress(self, message: str, current: int, total: int, payload: Optional[dict] = None) -> None:
        self._push_event("ml:progress", {
            **(payload or {}),
            "message": message,
            "current": current,
            "total":   total,
        })

    def _push_event(self, name: str, payload: Any) -> None:
        """Dispatch a CustomEvent into the webview's JS context."""
        if self._window is None:
            return
        safe_payload = json.dumps(payload, ensure_ascii=False)
        # evaluate_js is thread-safe in pywebview >= 3.6
        with self._push_lock:
            try:
                self._window.evaluate_js(
                    f"window.dispatchEvent("
                    f"new CustomEvent({json.dumps(name)}, "
                    f"{{detail: {safe_payload}}})"
                    f")"
                )
            except Exception:
                pass  # window may have been closed

    @staticmethod
    def _ok(**kwargs: Any) -> dict:
        return {"ok": True, **kwargs}

    @staticmethod
    def _err(exc: Exception) -> dict:
        traceback.print_exc()
        return {"ok": False, "error": str(exc)}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def get_bootstrap(self) -> dict:
        """Called on startup and after every action to sync React state."""
        try:
            return self._ok(**self._engine.get_bootstrap())
        except Exception as exc:
            return self._err(exc)

    # ── Chapter / navigation ──────────────────────────────────────────────────

    def open_chapter_folder(self) -> dict:
        """
        Show a native folder picker and import the selected chapter.
        Returns the new bootstrap so React can re-render immediately.
        """
        try:
            result = self._window.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory="",
                allow_multiple=False,
            )
            if not result:
                return self._ok(cancelled=True)
            folder = result[0]
            bootstrap = self._engine.import_chapter(folder)
            return self._ok(cancelled=False, **bootstrap)
        except Exception as exc:
            return self._err(exc)

    def import_chapter(self, folder: str) -> dict:
        """Import a chapter from a known folder path (programmatic, no dialog)."""
        try:
            return self._ok(**self._engine.import_chapter(folder))
        except Exception as exc:
            return self._err(exc)

    def go_to_page(self, idx: int) -> dict:
        try:
            return self._ok(**self._engine.go_to_page(int(idx)))
        except Exception as exc:
            return self._err(exc)

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def run_step(self, step: str) -> dict:
        """
        Run a single pipeline step on the current page.
        step ∈ {"detect", "ocr", "translate", "cleanup", "typeset"}
        """
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))

        STEPS = {
            "detect":    self._engine.detect_current_page,
            "ocr":       self._engine.ocr_current_page,
            "translate": self._engine.translate_current_page,
            "cleanup":   self._engine.cleanup_current_page,
            "typeset":   self._engine.typeset_current_page,
        }
        fn = STEPS.get(step.lower().strip())
        if fn is None:
            return self._err(ValueError(f"Unknown step: {step!r}"))

        self._engine.busy = True
        self._push_event("ml:busy", {"busy": True})
        try:
            self._engine._set_progress(running=True, job=step.lower().strip(), stage=step.lower().strip(), updated_pages=[])
            bootstrap = fn()
            page_idx = int(getattr(self._engine.chapter_mgr, "current_idx", 0) or 0)
            self._push_event("ml:progress", {
                "running": False,
                "job": step.lower().strip(),
                "stage": step.lower().strip(),
                "page_idx": page_idx,
                "page_total": self._engine.chapter_mgr.total_pages(),
                "updated_pages": [page_idx],
                "message": f"{step.title()} complete",
            })
            return self._ok(**bootstrap)
        except Exception as exc:
            self._push_event("ml:progress", {"running": False, "error": str(exc), "message": f"Error: {exc}", "job": step})
            return self._err(exc)
        finally:
            self._engine.busy = False
            self._push_event("ml:busy", {"busy": False})

    def run_all(self) -> dict:
        """Run the full pipeline (detect→ocr→translate→cleanup→typeset) on every loaded page."""
        if self._engine.busy:
            return self._err(RuntimeError("Already running."))
        self._engine.busy = True
        self._push_event("ml:busy", {"busy": True})
        try:
            bootstrap = self._engine.run_all_steps()
            return self._ok(**bootstrap)
        except Exception as exc:
            self._push_event("ml:progress", {"running": False, "error": str(exc), "message": f"Error: {exc}", "job": "run_all"})
            return self._err(exc)
        finally:
            self._engine.busy = False
            self._push_event("ml:busy", {"busy": False})

    def run_current_page(self) -> dict:
        """Run the full pipeline on the current page only."""
        if self._engine.busy:
            return self._err(RuntimeError("Already running."))
        self._engine.busy = True
        self._push_event("ml:busy", {"busy": True})
        try:
            bootstrap = self._engine.run_current_page_steps()
            return self._ok(**bootstrap)
        except Exception as exc:
            self._push_event("ml:progress", {"running": False, "error": str(exc), "message": f"Error: {exc}", "job": "run_page"})
            return self._err(exc)
        finally:
            self._engine.busy = False
            self._push_event("ml:busy", {"busy": False})

    def continue_run_all(self) -> dict:
        """Continue a previously interrupted Run All checkpoint."""
        if self._engine.busy:
            return self._err(RuntimeError("Already running."))
        self._engine.busy = True
        self._push_event("ml:busy", {"busy": True})
        try:
            bootstrap = self._engine.continue_run_all_steps()
            return self._ok(**bootstrap)
        except Exception as exc:
            self._push_event("ml:progress", {"running": False, "error": str(exc), "message": f"Error: {exc}", "job": "run_all"})
            return self._err(exc)
        finally:
            self._engine.busy = False
            self._push_event("ml:busy", {"busy": False})

    def detect_all(self) -> dict:
        """Run YOLO/detector only across every loaded page."""
        if self._engine.busy:
            return self._err(RuntimeError("Already running."))
        self._engine.busy = True
        self._push_event("ml:busy", {"busy": True})
        try:
            bootstrap = self._engine.detect_all_pages()
            return self._ok(**bootstrap)
        except Exception as exc:
            self._push_event("ml:progress", {"running": False, "error": str(exc), "message": f"Error: {exc}", "job": "detect_all"})
            return self._err(exc)
        finally:
            self._engine.busy = False
            self._push_event("ml:busy", {"busy": False})

    def export_project(self, export_dir: Optional[str] = None) -> dict:
        if self._engine.busy:
            return self._err(RuntimeError("Already running."))
        self._engine.busy = True
        self._push_event("ml:busy", {"busy": True})
        try:
            out_dir = self._engine.export_chapter(export_dir)
            return self._ok(export_dir=out_dir)
        except Exception as exc:
            return self._err(exc)
        finally:
            self._engine.busy = False
            self._push_event("ml:busy", {"busy": False})

    def reveal_export_folder(self) -> dict:
        """Open the export folder in the system file manager."""
        import os, subprocess
        d = self._engine.chapter_mgr.export_dir or os.getcwd()
        try:
            if os.name == "nt":
                os.startfile(d)
            elif os.uname().sysname == "Darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
            return self._ok(path=d)
        except Exception as exc:
            return self._err(exc)

    # ── Image access ──────────────────────────────────────────────────────────

    def get_page_image(self, idx: int, mode: str = "best") -> dict:
        """
        Return the requested page image mode as a base64 PNG.
        mode ∈ {"best", "raw", "cleaned", "typeset"}.
        The React canvas sets:  <img src={`data:image/png;base64,${result.b64}`} />

        Pass 2: the response also carries ``render_version`` — a server-authored
        monotonically-increasing counter bumped whenever the page's cleaned/
        typeset state changes. The frontend uses it in cache keys so freshly
        produced images show up immediately without a client-side race.
        """
        try:
            idx_i = int(idx)
            b64 = self._engine.get_page_image_b64(idx_i, mode)
            render_version = 0
            try:
                pages = getattr(self._engine.chapter_mgr, "pages", None) or []
                if 0 <= idx_i < len(pages):
                    render_version = int(getattr(pages[idx_i], "render_version", 0) or 0)
            except Exception:
                render_version = 0
            if b64 is None:
                return self._ok(b64=None, render_version=render_version)
            return self._ok(b64=b64, render_version=render_version)
        except Exception as exc:
            return self._err(exc)

    # ── Region editing ────────────────────────────────────────────────────────

    def update_region(self, region_idx: int, field: str, value: Any) -> dict:
        """
        Update a single field on a region.
        field ∈ {"translation", "text", "font_name", "font_size", "align", "visible", "locked"}
        """
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))
        try:
            bootstrap = self._engine.update_region_field(int(region_idx), field, value)
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def update_region_bbox(self, region_idx: int, x: int, y: int, w: int, h: int, page_idx: Optional[int] = None) -> dict:
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))
        try:
            bootstrap = self._engine.update_region_bbox(int(region_idx), x, y, w, h, page_idx)
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def get_region_preview_sprite(self, region_idx: int, draft: Optional[dict] = None, page_idx: Optional[int] = None) -> dict:
        try:
            return self._ok(**self._engine.get_region_preview_sprite(int(region_idx), draft or {}, page_idx))
        except Exception as exc:
            return self._err(exc)

    def list_fonts(self) -> dict:
        try:
            return self._ok(**self._engine.list_font_options())
        except Exception as exc:
            return self._err(exc)

    def add_region(self, x: int, y: int, w: int, h: int, text: str = "") -> dict:
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))
        try:
            bootstrap = self._engine.add_region(x, y, w, h, text)
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def delete_region(self, region_idx: int, yolo_reject_reason: str = "") -> dict:
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))
        try:
            bootstrap = self._engine.delete_region(int(region_idx), str(yolo_reject_reason or ""))
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def export_yolo_finetune_dataset(self) -> dict:
        try:
            return self._engine.export_yolo_finetune_dataset()
        except Exception as exc:
            return self._err(exc)

    def set_yolo_train_class(self, region_idx: int, class_id: int) -> dict:
        try:
            bootstrap = self._engine.set_yolo_train_class(int(region_idx), int(class_id))
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def train_yolo_detector(self) -> dict:
        try:
            return self._engine.train_yolo_detector()
        except Exception as exc:
            return self._err(exc)

    def get_yolo_training_status(self) -> dict:
        try:
            return self._ok(**self._engine.get_yolo_training_status())
        except Exception as exc:
            return self._err(exc)

    def ocr_region(self, region_idx: int) -> dict:
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))
        try:
            bootstrap = self._engine.ocr_region(int(region_idx))
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def translate_region(self, region_idx: int) -> dict:
        if self._engine.busy:
            return self._err(RuntimeError("Already running — wait for the current step to finish."))
        try:
            bootstrap = self._engine.translate_region(int(region_idx))
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def preview_region_cleanup(self, region_idx: int, manual_mask: Optional[dict] = None) -> dict:
        try:
            return self._engine.preview_region_cleanup(int(region_idx), manual_mask if isinstance(manual_mask, dict) else None)
        except Exception as exc:
            return self._err(exc)

    def propose_cleanup_mask_sam2(self, region_idx: int, prompt: Optional[dict] = None) -> dict:
        try:
            return self._engine.propose_cleanup_mask_sam2(int(region_idx), prompt if isinstance(prompt, dict) else None)
        except Exception as exc:
            return self._err(exc)

    def get_sam2_status(self, load: bool = False) -> dict:
        try:
            return self._engine.get_sam2_status(bool(load))
        except Exception as exc:
            return self._err(exc)

    def get_region_cleanup_debug(self, region_idx: int, manual_mask: Optional[dict] = None) -> dict:
        try:
            return self._engine.get_region_cleanup_debug(int(region_idx), manual_mask if isinstance(manual_mask, dict) else None)
        except Exception as exc:
            return self._err(exc)

    def record_mask_qa_label(self, region_idx: int, label: str, notes: str = "") -> dict:
        try:
            return self._engine.record_mask_qa_label(int(region_idx), str(label or ""), str(notes or ""))
        except Exception as exc:
            return self._err(exc)

    def export_mask_qa_dataset(self) -> dict:
        try:
            return self._engine.export_mask_qa_dataset()
        except Exception as exc:
            return self._err(exc)

    def train_mask_qa_model(self) -> dict:
        try:
            return self._engine.train_mask_qa_model()
        except Exception as exc:
            return self._err(exc)

    def compare_region_cleanup_candidates(self, region_idx: int, manual_mask: Optional[dict] = None) -> dict:
        try:
            return self._engine.compare_region_cleanup_candidates(int(region_idx), manual_mask if isinstance(manual_mask, dict) else None)
        except Exception as exc:
            return self._err(exc)

    def apply_region_cleanup_candidate(self, region_idx: int, candidate_id: str, manual_mask: Optional[dict] = None) -> dict:
        try:
            bootstrap = self._engine.apply_region_cleanup_candidate(
                int(region_idx),
                str(candidate_id or "default"),
                manual_mask if isinstance(manual_mask, dict) else None,
            )
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def apply_region_cleanup(self, region_idx: int, manual_mask: Optional[dict] = None) -> dict:
        try:
            bootstrap = self._engine.apply_region_cleanup(int(region_idx), manual_mask=manual_mask if isinstance(manual_mask, dict) else None)
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def rerun_region_cleanup(self, region_idx: int, manual_mask: Optional[dict] = None) -> dict:
        try:
            bootstrap = self._engine.apply_region_cleanup(
                int(region_idx),
                rerun=True,
                manual_mask=manual_mask if isinstance(manual_mask, dict) else None,
            )
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def delete_region_cleanup(self, region_idx: int) -> dict:
        try:
            bootstrap = self._engine.delete_region_cleanup(int(region_idx))
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    def undo(self) -> dict:
        try:
            bootstrap = self._engine.undo_last_edit()
            return self._ok(**bootstrap)
        except Exception as exc:
            return self._err(exc)

    # ── Source / series sync ────────────────────────────────────────────────

    def list_sources(self) -> dict:
        try:
            return self._engine.list_sources()
        except Exception as exc:
            return self._err(exc)

    def browse_source_series(self, source: str, query: str = "") -> dict:
        try:
            return self._engine.browse_source_series(source, query)
        except Exception as exc:
            return self._err(exc)

    def select_browse_series(self, series_title: str, source: str, source_id: str, card: dict) -> dict:
        try:
            return self._engine.select_browse_series(series_title, source, source_id, card)
        except Exception as exc:
            return self._err(exc)

    def get_series_list(self) -> dict:
        try:
            return self._engine.get_series_list()
        except Exception as exc:
            return self._err(exc)

    def get_series_detail(self, series_title: str) -> dict:
        try:
            return self._engine.get_series_detail(series_title)
        except Exception as exc:
            return self._err(exc)

    def update_series_metadata(self, series_title: str, updates: dict) -> dict:
        try:
            return self._engine.update_series_metadata(series_title, updates)
        except Exception as exc:
            return self._err(exc)

    def sync_series_metadata(self, series_title: str, source: str = "", source_id: str = "") -> dict:
        try:
            return self._engine.sync_series_metadata(series_title, source, source_id)
        except Exception as exc:
            return self._err(exc)

    def sync_series_chapters(self, series_title: str, mode: str = "missing") -> dict:
        try:
            return self._engine.sync_series_chapters(series_title, mode)
        except Exception as exc:
            return self._err(exc)

    def sync_source_chapter(self, series_title: str, chapter_source_id: str) -> dict:
        try:
            return self._engine.sync_source_chapter(series_title, chapter_source_id)
        except Exception as exc:
            return self._err(exc)

    def import_source_chapter(self, series_title: str, chapter_source_id: str) -> dict:
        try:
            return self._engine.import_source_chapter(series_title, chapter_source_id)
        except Exception as exc:
            return self._err(exc)

    def delete_series(self, series_title: str, source: str = "", source_id: str = "", delete_files: bool = False) -> dict:
        try:
            return self._engine.delete_series(series_title, source, source_id, delete_files)
        except Exception as exc:
            return self._err(exc)

    def sync_missing_thumbnails(self, series_title: str) -> dict:
        try:
            return self._engine.sync_missing_thumbnails(series_title)
        except Exception as exc:
            return self._err(exc)

    def get_thumbnail_b64(self, url: str = "", path: str = "") -> dict:
        """Fetch a thumbnail (by URL or local path) and return it as a base64 data-URI.
        Handles Naver hotlink protection by sending the correct Referer header."""
        try:
            return self._engine.get_thumbnail_b64(url or "", path or "")
        except Exception as exc:
            return self._err(exc)

    def translate_series_metadata(self, series_title: str) -> dict:
        try:
            return self._engine.translate_series_metadata(series_title)
        except Exception as exc:
            return self._err(exc)

    def retranslate_series(self, series_title: str) -> dict:
        try:
            return self._engine.retranslate_series(series_title)
        except Exception as exc:
            return self._err(exc)

    # ── Series memory ────────────────────────────────────────────────────────

    def list_series_memory(self) -> dict:
        try:
            return self._ok(memory=self._engine.list_series_memory())
        except Exception as exc:
            return self._err(exc)

    def add_series_name(
        self,
        kr_name: str,
        en_name: str,
        aliases_kr: Optional[Any] = None,
        note: str = "",
    ) -> dict:
        try:
            return self._ok(**self._engine.add_series_name(kr_name, en_name, aliases_kr, note))
        except Exception as exc:
            return self._err(exc)

    def update_series_name(self, id: str, fields: dict) -> dict:
        try:
            return self._ok(**self._engine.update_series_name(id, fields))
        except Exception as exc:
            return self._err(exc)

    def delete_series_name(self, id: str) -> dict:
        try:
            return self._ok(**self._engine.delete_series_name(id))
        except Exception as exc:
            return self._err(exc)

    def add_series_glossary(
        self,
        source_kr: str,
        target_en: str,
        alternatives_en: Optional[Any] = None,
        aliases_kr: Optional[Any] = None,
        note: str = "",
    ) -> dict:
        try:
            return self._ok(**self._engine.add_series_glossary(
                source_kr, target_en, alternatives_en, aliases_kr, note
            ))
        except Exception as exc:
            return self._err(exc)

    def update_series_glossary(self, id: str, fields: dict) -> dict:
        try:
            return self._ok(**self._engine.update_series_glossary(id, fields))
        except Exception as exc:
            return self._err(exc)

    def delete_series_glossary(self, id: str) -> dict:
        try:
            return self._ok(**self._engine.delete_series_glossary(id))
        except Exception as exc:
            return self._err(exc)

    # ── Settings ──────────────────────────────────────────────────────────────

    def get_model_config(self) -> dict:
        try:
            return self._ok(config=self._engine.get_model_config())
        except Exception as exc:
            return self._err(exc)

    def update_model_config(self, updates: dict) -> dict:
        try:
            return self._engine.update_model_config(updates)
        except Exception as exc:
            return self._err(exc)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def on_window_closed(self) -> None:
        """Registered as pywebview closing callback."""
        self._engine.shutdown()
