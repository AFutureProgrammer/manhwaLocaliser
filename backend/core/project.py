from __future__ import annotations

import json
import os
import datetime
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from backend.core.constants import debug_print
from backend.core.regions import _apply_block_dict, _block_from_dict, _block_to_dict

def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

@dataclass
class ChapterPage:
    """State for a single image within a chapter."""
    image_path:   str
    regions:      List[Any]  = field(default_factory=list)   # List[OCRBlock]
    translations: List[str]  = field(default_factory=list)
    detected:     bool       = False
    cleaned_cv:   Optional[np.ndarray] = field(default=None, repr=False)
    typeset_pil:  Optional[Image.Image] = field(default=None, repr=False)
    cleanup_patches: List[Dict[str, Any]] = field(default_factory=list)
    render_dirty: bool = False
    # Pass 2: server-authored render version. Monotonically increasing;
    # bumped whenever cleaned_cv / typeset_pil / render_dirty is mutated.
    # The frontend uses this value in cache keys so cleaned/typeset output
    # shows immediately after Run Page without a client-side race.
    render_version: int = 0
    artifacts_dirty: bool = False
    artifacts_saved_version: int = -1

    def bump_render_version(self) -> int:
        """Increment and return render_version. Safe to call any number of times."""
        self.render_version = int(self.render_version or 0) + 1
        self.artifacts_dirty = True
        return self.render_version

class ChapterManager:
    """Manages a folder of images as a chapter with per-page state."""
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self) -> None:
        self.pages:        List[ChapterPage] = []
        self.current_idx:  int  = 0
        self.chapter_dir:  str  = ""
        self.export_dir:   str  = ""
        self.run_all_checkpoint: Dict[str, Any] = {}
        self._save_lock = threading.RLock()
        self._save_timer: Optional[threading.Timer] = None

    def request_save_state(self, delay_seconds: float = 0.35) -> None:
        with self._save_lock:
            if not self.chapter_dir:
                return
            if delay_seconds <= 0:
                self.save_state()
                return
            if self._save_timer is not None:
                try:
                    self._save_timer.cancel()
                except Exception:
                    pass
            timer = threading.Timer(float(delay_seconds), self._run_debounced_save)
            timer.daemon = True
            self._save_timer = timer
            timer.start()

    def flush_pending_save(self) -> None:
        with self._save_lock:
            timer = self._save_timer
            self._save_timer = None
            if timer is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
        self.save_state()

    def _run_debounced_save(self) -> None:
        with self._save_lock:
            self._save_timer = None
        self.save_state()

    def load_from_folder(self, folder: str) -> int:
        paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in self._IMAGE_EXTS
        )
        self.pages       = [ChapterPage(image_path=p) for p in paths]
        self.current_idx = 0
        self.chapter_dir = folder
        self.export_dir  = os.path.join(folder, "translated")
        debug_print(
            f"ChapterManager: loaded {len(self.pages)} pages "
            f"from chapter_dir={os.path.abspath(folder)!r}"
        )
        return len(self.pages)

    @property
    def current_page(self) -> Optional[ChapterPage]:
        return self.pages[self.current_idx] if 0 <= self.current_idx < len(self.pages) else None

    def total_pages(self) -> int:
        return len(self.pages)

    def go_to(self, idx: int) -> bool:
        if 0 <= idx < len(self.pages):
            self.current_idx = idx
            return True
        return False

    def next_page(self) -> bool:
        return self.go_to(self.current_idx + 1)

    def prev_page(self) -> bool:
        return self.go_to(self.current_idx - 1)

    def page_stats(self, idx: int) -> dict:
        """Return dict of region/translation counts for a page."""
        if not (0 <= idx < len(self.pages)):
            return {"regions": 0, "translated": 0, "cleaned": False, "typeset": False}
        p = self.pages[idx]
        return {
            "regions":    len(p.regions),
            "translated": sum(1 for t in p.translations if t and t.strip()),
            "cleaned":    p.cleaned_cv is not None,
            "typeset":    p.typeset_pil is not None,
        }

    def _artifact_dir(self) -> str:
        return os.path.join(self.chapter_dir, ".ml_artifacts")

    def _run_all_checkpoint_path(self) -> str:
        return os.path.join(self.chapter_dir, ".ml_run_all_checkpoint.json")

    def save_run_all_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.chapter_dir:
            return
        self.run_all_checkpoint = dict(checkpoint or {})
        try:
            with open(self._run_all_checkpoint_path(), "w", encoding="utf-8") as f:
                json.dump(self.run_all_checkpoint, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as exc:
            debug_print(f"ChapterManager.save_run_all_checkpoint failed: {exc}")

    def clear_run_all_checkpoint(self) -> None:
        self.run_all_checkpoint = {}
        if not self.chapter_dir:
            return
        try:
            path = self._run_all_checkpoint_path()
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            debug_print(f"ChapterManager.clear_run_all_checkpoint failed: {exc}")

    def load_run_all_checkpoint(self) -> None:
        self.run_all_checkpoint = {}
        if not self.chapter_dir:
            return
        try:
            with open(self._run_all_checkpoint_path(), encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("active"):
                self.run_all_checkpoint = data
        except FileNotFoundError:
            return
        except Exception as exc:
            debug_print(f"ChapterManager.load_run_all_checkpoint failed: {exc}")

    def _artifact_relpath(self, idx: int, kind: str) -> str:
        return os.path.join(".ml_artifacts", f"page_{idx:04d}_{kind}.png")

    def _artifact_abspath(self, relpath: str) -> str:
        return os.path.join(self.chapter_dir, relpath)
    
    def delete_artifact(self, idx: int, kind: str) -> None:
        """Delete a saved page artifact if it exists.

        Used when cleaned/typeset outputs are invalidated so stale PNGs
        cannot be restored from .ml_artifacts on the next startup.
        """
        rel = self._artifact_relpath(idx, kind)
        path = self._artifact_abspath(rel)
        try:
            if os.path.exists(path):
                os.remove(path)
                debug_print(f"ChapterManager.delete_artifact: removed {path!r}")
        except OSError as exc:
            debug_print(
                f"ChapterManager.delete_artifact: failed idx={idx} kind={kind!r}: {exc}"
            )

    def _save_artifacts(self, idx: int, page: ChapterPage, entry: dict) -> None:
        os.makedirs(self._artifact_dir(), exist_ok=True)
        render_version = int(getattr(page, "render_version", 0) or 0)
        dirty = bool(getattr(page, "artifacts_dirty", False))
        saved_version = int(getattr(page, "artifacts_saved_version", -1) or -1)
        can_reuse = (not dirty) and saved_version == render_version
        if page.cleaned_cv is not None:
            rel = self._artifact_relpath(idx, "cleaned")
            path = self._artifact_abspath(rel)
            if not (can_reuse and os.path.exists(path)):
                rgb = page.cleaned_cv[:, :, ::-1] if page.cleaned_cv.ndim == 3 else page.cleaned_cv
                Image.fromarray(rgb).save(path)
            entry["cleaned_artifact"] = rel
        if page.typeset_pil is not None:
            rel = self._artifact_relpath(idx, "typeset")
            path = self._artifact_abspath(rel)
            if not (can_reuse and os.path.exists(path)):
                page.typeset_pil.save(path)
            entry["typeset_artifact"] = rel
        page.artifacts_dirty = False
        page.artifacts_saved_version = render_version

    def _restore_artifacts(self, page: ChapterPage, entry: dict) -> None:
        cleaned = entry.get("cleaned_artifact")
        if cleaned:
            try:
                rgb = np.array(Image.open(self._artifact_abspath(cleaned)).convert("RGB"))
                page.cleaned_cv = rgb[:, :, ::-1].copy()
                page.artifacts_dirty = False
                page.artifacts_saved_version = int(getattr(page, "render_version", 0) or 0)
            except Exception as exc:
                debug_print(f"restore_fallback page={os.path.basename(page.image_path)} artifact=cleaned reason={exc}")
        typeset = entry.get("typeset_artifact")
        if typeset:
            try:
                page.typeset_pil = Image.open(self._artifact_abspath(typeset)).convert("RGB")
                page.artifacts_dirty = False
                page.artifacts_saved_version = int(getattr(page, "render_version", 0) or 0)
            except Exception as exc:
                debug_print(f"restore_fallback page={os.path.basename(page.image_path)} artifact=typeset reason={exc}")

    def _migrate_unsafe_bg(self, page: ChapterPage, page_idx: int) -> bool:
        changed = False
        for region_idx, block in enumerate(page.regions):
            kind = getattr(getattr(block, "region_kind", None), "name", "") or str(getattr(block, "region_kind", "") or "")
            if kind == "CAPTION_BOX":
                continue
            bg = tuple(int(v) for v in (getattr(block, "bg_color", None) or (255, 255, 255))[:3])
            if max(bg) - min(bg) > 120:
                debug_print(
                    "style_migration ignored_bg_for_cleanup "
                    f"page={page_idx} region={region_idx} bg={bg} reason=normal_bubble"
                )
                block.bg_color = (255, 255, 255)
                changed = True
        if changed:
            page.cleaned_cv = None
            page.typeset_pil = None
        return changed

    def _migrate_cleanup_v5(self, page: ChapterPage, page_idx: int, version: int) -> bool:
        if version >= 5:
            return False
        changed = False
        if page.cleaned_cv is not None or page.typeset_pil is not None:
            page.cleaned_cv = None
            page.typeset_pil = None
            page.render_dirty = bool(page.translations)
            changed = True
        for region_idx, block in enumerate(page.regions):
            if getattr(block, "detector_source", "") != "manual":
                if getattr(block, "ocr_backend", "") != "qwen_vl":
                    block.boxes = []
                block.text_mask = None
                block.safe_text_mask = None
                block.safe_rect = None
                block.safe_center = None
                changed = True
                debug_print(
                    "state_migration_v5 invalidate_easyocr_geometry "
                    f"page={page_idx} region={region_idx} state_version={version}"
                )
        if changed:
            debug_print(
                "startup_restore reset_bad_cleanup_state "
                f"page={page_idx} state_version={version} reason=v5_qwen_cv_cleanup"
            )
        return changed

    def save_state(self) -> None:
        """Persist page state as JSON (Phase 3: full block data including overrides/styles).

        Schema version 2 adds a "regions" list with full block dicts.
        Version 1 files (translations-only) are still produced when regions is empty.
        """
        with self._save_lock:
            timer = self._save_timer
            self._save_timer = None
            if timer is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
            if not self.chapter_dir:
                return
            state_path = os.path.join(self.chapter_dir, ".ml_state.json")
            pages_data = []
            for idx, p in enumerate(self.pages):
                page_entry: dict = {
                    "image_path":   p.image_path,
                    "translations": p.translations,
                    "detected":     bool(getattr(p, "detected", False)),
                    "render_dirty": bool(getattr(p, "render_dirty", False)),
                    "render_version": int(getattr(p, "render_version", 0) or 0),
                }
                if p.regions:
                    page_entry["regions"] = [_block_to_dict(b) for b in p.regions]
                if getattr(p, "cleanup_patches", None):
                    page_entry["cleanup_patches"] = list(getattr(p, "cleanup_patches", []) or [])
                self._save_artifacts(idx, p, page_entry)
                pages_data.append(page_entry)
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump({"version": 5, "pages": pages_data}, f,
                              ensure_ascii=False, indent=2, default=_json_default)
                debug_print(f"ChapterManager: state v5 saved to {state_path!r}")
            except Exception as e:
                debug_print(f"ChapterManager.save_state failed: {e}")

    def load_state(self) -> bool:
        """Restore state from .ml_state.json.

        Backward-compatible:
          v1 files (translations only) — restore translations, skip block data
          v2 files — also restore overrides, review state, style, bbox_override
        """
        if not self.chapter_dir:
            return False
        state_path = os.path.join(self.chapter_dir, ".ml_state.json")
        debug_print(f"ChapterManager: loading state from {state_path!r}")
        try:
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        except Exception as e:
            debug_print(f"ChapterManager.load_state failed: {e}")
            return False

        version      = int(data.get("version", 1))
        path_to_page = {p.image_path: p for p in self.pages}

        migrated = False
        for entry in data.get("pages", []):
            p = path_to_page.get(entry.get("image_path"))
            if not p:
                continue
            page_idx = self.pages.index(p)
            if entry.get("translations"):
                p.translations = entry["translations"]
            p.detected = bool(entry.get("detected", False) or entry.get("regions"))
            p.render_dirty = bool(entry.get("render_dirty", False))
            try:
                p.render_version = int(entry.get("render_version", 0) or 0)
            except Exception:
                p.render_version = 0
            if version >= 4:
                self._restore_artifacts(p, entry)
            elif entry.get("cleaned_artifact") or entry.get("typeset_artifact"):
                p.cleaned_cv = None
                p.typeset_pil = None
                p.render_dirty = bool(p.translations)
                migrated = True
                debug_print(
                    "startup_restore reset_bad_cleanup_state "
                    f"page={page_idx} state_version={version} "
                    "reason=pre_cleanup_plan_artifact"
                )
            debug_print(
                "startup_restore "
                f"page={page_idx} dirty={p.render_dirty} "
                f"has_cleaned={p.cleaned_cv is not None} has_typeset={p.typeset_pil is not None}"
            )
            # v2: restore full block data. Older runs may have saved regions
            # before this process recreated detector blocks, so rebuild when
            # the page has no in-memory regions yet.
            if version >= 2 and entry.get("regions"):
                if not p.regions:
                    restored = []
                    for bd in entry["regions"]:
                        try:
                            restored.append(_block_from_dict(bd))
                        except Exception as exc:
                            debug_print(f"load_state: block rebuild failed: {exc}")
                    p.regions = restored
                else:
                    for block, bd in zip(p.regions, entry["regions"]):
                        try:
                            _apply_block_dict(block, bd)
                        except Exception as exc:
                            debug_print(f"load_state: block restore failed: {exc}")
            patches = entry.get("cleanup_patches", [])
            p.cleanup_patches = patches if isinstance(patches, list) else []
            migrated = self._migrate_cleanup_v5(p, page_idx, version) or migrated
            migrated = self._migrate_unsafe_bg(p, page_idx) or migrated

        debug_print(f"ChapterManager: state v{version} restored")
        self.load_run_all_checkpoint()
        if migrated:
            self.save_state()
        return True

SERIES_DB_FILE = "series_db.json"

def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _stable_series_memory_key(source: str, source_id: str) -> str:
    return f"source:{source}:{source_id}"

def _stable_chapter_memory_key(source: str, series_source_id: str, chapter_source_id: str, episode_no: Any = "") -> str:
    chapter_key = str(chapter_source_id or "").strip()
    if not chapter_key:
        chapter_key = f"episode-{episode_no}" if episode_no not in (None, "") else "chapter"
    return f"source:{source}:{series_source_id}:{chapter_key}"

def _memory_fs_key(memory_key: str) -> str:
    slug = str(memory_key or "").lower().replace(":", "_")
    slug = re.sub(r"[^a-z0-9_\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug or "unnamed_series"

def _count_images(folder: str) -> int:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
    try:
        return sum(
            1 for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in image_exts
        )
    except OSError:
        return 0

def _has_ml_state(folder: str) -> bool:
    return bool(folder) and os.path.isfile(os.path.join(folder, ".ml_state.json"))

class SeriesDB:
    """
    Persists a list of series, each with a list of chapter folders.
    Stored in series_db.json next to the script.

    Schema:
    {
      "series": [
        {
          "title": "A Knight Living Only for Today",
          "source": "naver-comic",
          "source_id": "824543",
          "chapters": [
            {"folder": "/path/to/chapter_101", "name": "101화", "page_count": 92}
          ]
        }
      ]
    }
    """

    def __init__(self, path: str = SERIES_DB_FILE) -> None:
        self._path = path
        self._data: Dict[str, Any] = {"series": []}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)
            debug_print(f"SeriesDB loaded: {len(self._data.get('series', []))} series")
        except FileNotFoundError:
            debug_print("SeriesDB: no DB file found, starting fresh")
        except Exception as e:
            debug_print(f"SeriesDB._load failed: {e}")

    def save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            debug_print(f"SeriesDB.save failed: {e}")

    def _save(self) -> None:
        self.save()

    @property
    def series(self) -> List[Dict[str, Any]]:
        return self._data.get("series", [])

    def find_or_create_series(self, title: str) -> Dict[str, Any]:
        for s in self.series:
            if s.get("title") == title:
                return s
        new_series: Dict[str, Any] = {
            "title": title, "source": "local",
            "source_id": "", "memory_key": title,
            "memory_aliases": [], "chapters": []
        }
        self._data.setdefault("series", []).append(new_series)
        return new_series

    def find_series_by_source(self, source: str, source_id: str) -> Optional[Dict[str, Any]]:
        for s in self.series:
            if s.get("source") == source and str(s.get("source_id") or "") == str(source_id):
                return s
        return None

    def find_series_for_folder(self, folder: str) -> Optional[Dict[str, Any]]:
        target = os.path.abspath(folder)
        for s in self.series:
            for ch in s.get("chapters", []) or []:
                if os.path.abspath(ch.get("folder") or "") == target:
                    return s
        return None

    def find_chapter_for_folder(self, folder: str) -> Optional[Dict[str, Any]]:
        target = os.path.abspath(folder)
        for s in self.series:
            for ch in s.get("chapters", []) or []:
                if os.path.abspath(ch.get("folder") or "") == target:
                    return ch
        return None

    def _ensure_source_memory_fields(self, entry: Dict[str, Any], source: str, source_id: str) -> None:
        if not source or source == "local" or not source_id:
            entry.setdefault("memory_key", entry.get("title", ""))
            entry.setdefault("memory_aliases", [])
            return
        stable_key = _stable_series_memory_key(source, str(source_id))
        aliases = entry.setdefault("memory_aliases", [])
        if not isinstance(aliases, list):
            aliases = [str(aliases)]
            entry["memory_aliases"] = aliases
        old_keys = [
            entry.get("memory_key"),
            entry.get("title"),
            entry.get("title_en"),
            entry.get("title_ko"),
        ]
        for old in old_keys:
            old_s = str(old or "").strip()
            if old_s and old_s != stable_key and old_s not in aliases:
                aliases.append(old_s)
        entry["memory_key"] = stable_key
        entry["memory_fs_key"] = _memory_fs_key(stable_key)

    def register_chapter(self, series_title: str, folder: str,
                          name: str, page_count: int) -> None:
        existing_series = self.find_series_for_folder(folder)
        s = existing_series or self.find_or_create_series(series_title)
        for ch in s.get("chapters", []):
            if ch.get("folder") == folder:
                ch["page_count"] = page_count
                ch.setdefault("name", name)
                self.save()
                return
        s.setdefault("chapters", []).append({
            "folder": folder, "name": name, "page_count": page_count
        })
        self.save()
        debug_print(f"SeriesDB: registered chapter {name!r} in {series_title!r}")

    def get_chapters(self, series_title: str) -> List[Dict[str, Any]]:
        for s in self.series:
            if s.get("title") == series_title:
                return s.get("chapters", [])
        return []

    def update_series_metadata(self, series_title: str, updates: Dict[str, Any]) -> None:
        entry = self.find_or_create_series(series_title)
        for key, value in (updates or {}).items():
            if key in {"title", "chapters"}:
                continue
            entry[key] = value
        self._ensure_source_memory_fields(
            entry,
            str(entry.get("source") or ""),
            str(entry.get("source_id") or ""),
        )
        self.save()

    def upsert_source_series(
        self,
        series_title: str,
        source: str,
        source_id: str,
        metadata: Dict[str, Any],
        chapter_list: List[Dict[str, Any]],
        base_folder: str,
    ) -> Dict[str, Any]:
        entry = self.find_series_by_source(source, source_id) or self.find_or_create_series(series_title)
        entry.setdefault("title", series_title)
        entry.setdefault("chapters", [])
        entry["source"] = source
        entry["source_id"] = str(source_id)
        self._ensure_source_memory_fields(entry, source, str(source_id))

        for key in (
            "title_en", "title_ko", "synopsis_en", "synopsis_ko",
            "source_url", "thumbnail_url", "thumbnail_path", "source_metadata",
            "metadata_lang", "memory_key", "memory_fs_key",
        ):
            if key in metadata and metadata[key] not in (None, ""):
                entry[key] = metadata[key]
        entry["last_synced_at"] = _now_iso()
        entry["sync_status"] = str(metadata.get("sync_status") or entry.get("sync_status") or "ok")

        existing_by_source_id = {
            str(ch.get("source_id") or ""): ch
            for ch in entry.get("chapters", []) or []
            if str(ch.get("source_id") or "")
        }

        for ch_data in chapter_list:
            ch_source_id = str(ch_data.get("source_id") or ch_data.get("episode_no") or "").strip()
            episode_no = ch_data.get("episode_no") or ""
            ep_str = f"{int(episode_no):04d}" if str(episode_no).isdigit() else (ch_source_id or "chapter")
            slug = f"{ep_str}-{ch_source_id}" if ch_source_id and ch_source_id != ep_str else ep_str
            safe_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", slug).strip("._") or "chapter"
            suggested_folder = os.path.join(base_folder, "chapters", safe_slug)
            chapter_memory_key = _stable_chapter_memory_key(source, str(source_id), ch_source_id, episode_no)

            ch = existing_by_source_id.get(ch_source_id)
            if ch is None:
                folder = ch_data.get("folder") or suggested_folder
                ch = {
                    "source": source,
                    "source_id": ch_source_id,
                    "episode_no": episode_no,
                    "chapter_memory_key": chapter_memory_key,
                    "chapter_memory_fs_key": _memory_fs_key(chapter_memory_key),
                    "title_en": str(ch_data.get("title_en") or ""),
                    "title_ko": str(ch_data.get("title_ko") or ""),
                    "name": str(ch_data.get("title_en") or ch_data.get("title_ko") or ch_source_id or episode_no or "Chapter"),
                    "source_url": str(ch_data.get("source_url") or ""),
                    "thumbnail_url": str(ch_data.get("thumbnail_url") or ""),
                    "thumbnail_path": str(ch_data.get("thumbnail_path") or ""),
                    "page_count": int(ch_data.get("page_count") or -1),
                    "folder": folder,
                    "indexed": True,
                    "imported": _count_images(folder) > 0,
                    "translated": _has_ml_state(folder),
                    "last_synced_at": _now_iso(),
                }
                ch["missing_raw"] = not ch["imported"]
                ch["needs_sync"] = ch["missing_raw"]
                entry["chapters"].append(ch)
                existing_by_source_id[ch_source_id] = ch
                continue

            for key in ("title_en", "title_ko", "source_url", "thumbnail_url", "thumbnail_path", "episode_no"):
                if key in ch_data and ch_data[key] not in (None, ""):
                    ch[key] = ch_data[key]
            ch["source"] = source
            ch["source_id"] = ch_source_id
            ch["chapter_memory_key"] = chapter_memory_key
            ch["chapter_memory_fs_key"] = _memory_fs_key(chapter_memory_key)
            ch.setdefault("folder", ch_data.get("folder") or suggested_folder)
            ch.setdefault("name", str(ch.get("title_en") or ch.get("title_ko") or ch_source_id or "Chapter"))
            if ch_data.get("page_count", -1) != -1:
                ch["page_count"] = int(ch_data.get("page_count") or -1)
            ch["indexed"] = True
            ch["imported"] = _count_images(ch.get("folder") or "") > 0
            ch["translated"] = _has_ml_state(ch.get("folder") or "")
            ch["missing_raw"] = not ch["imported"]
            ch["needs_sync"] = ch["missing_raw"]
            ch["last_synced_at"] = _now_iso()

        self.save()
        return entry

    def mark_chapter_imported(self, series_title: str, chapter_source_id: str,
                              folder: str, page_count: int = -1) -> None:
        entry = self.find_or_create_series(series_title)
        for ch in entry.get("chapters", []) or []:
            if str(ch.get("source_id") or "") == str(chapter_source_id):
                ch["imported"] = True
                ch["missing_raw"] = False
                ch["needs_sync"] = False
                ch["folder"] = folder
                if page_count >= 0:
                    ch["page_count"] = page_count
                else:
                    inferred = _count_images(folder)
                    if inferred:
                        ch["page_count"] = inferred
                break
        self.save()

    def compute_series_stats(self, series_title: str) -> Dict[str, Any]:
        entry = self.find_or_create_series(series_title)
        chapters = entry.get("chapters", []) or []
        imported = translated = missing_raw = total_bytes = 0
        for ch in chapters:
            folder = ch.get("folder") or ""
            is_imported = _count_images(folder) > 0 if folder else bool(ch.get("imported"))
            is_translated = _has_ml_state(folder) if folder else bool(ch.get("translated"))
            if is_imported:
                imported += 1
                try:
                    for fn in os.listdir(folder):
                        path = os.path.join(folder, fn)
                        if os.path.isfile(path):
                            total_bytes += os.path.getsize(path)
                except OSError:
                    pass
            else:
                missing_raw += 1
            if is_translated:
                translated += 1
        return {
            "indexed": len(chapters),
            "imported": imported,
            "translated": translated,
            "missing_raw": missing_raw,
            "estimated_bytes": total_bytes,
        }

    def list_series_summary(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self.series:
            chapters = s.get("chapters", []) or []
            out.append({
                "title": s.get("title", ""),
                "title_en": s.get("title_en", ""),
                "title_ko": s.get("title_ko", ""),
                "source": s.get("source", "local"),
                "source_id": str(s.get("source_id") or ""),
                "memory_key": s.get("memory_key", s.get("title", "")),
                "memory_fs_key": s.get("memory_fs_key", _memory_fs_key(str(s.get("memory_key") or s.get("title") or ""))),
                "memory_aliases": s.get("memory_aliases", []),
                "source_url": s.get("source_url", ""),
                "thumbnail_url": s.get("thumbnail_url", ""),
                "thumbnail_path": s.get("thumbnail_path", ""),
                "chapter_count": len(chapters),
                "last_synced_at": s.get("last_synced_at", ""),
                "sync_status": s.get("sync_status", ""),
            })
        return out

    def get_series_detail(self, series_title: str) -> Optional[Dict[str, Any]]:
        for s in self.series:
            if s.get("title") == series_title:
                if s.get("memory_key") and not s.get("memory_fs_key"):
                    s["memory_fs_key"] = _memory_fs_key(str(s.get("memory_key") or ""))
                    self.save()
                return dict(s)
        return None

    def delete_series(self, series_title: str, source: str = "", source_id: str = "") -> Optional[Dict[str, Any]]:
        """Remove a SeriesDB entry only. Local files and memory are intentionally preserved."""
        source = source or ""
        source_id = str(source_id or "")
        match_idx: Optional[int] = None
        for idx, s in enumerate(self.series):
            if source and source_id:
                if s.get("source") == source and str(s.get("source_id") or "") == source_id:
                    match_idx = idx
                    break
            elif s.get("title") == series_title:
                match_idx = idx
                break
        if match_idx is None:
            return None
        removed = dict(self.series.pop(match_idx))
        self.save()
        return removed

    def get_chapter_list_for_series(self, series_title: str) -> List[Dict[str, Any]]:
        entry = self.find_or_create_series(series_title)
        updated = False
        for ch in entry.get("chapters", []) or []:
            folder = ch.get("folder") or ""
            if folder:
                imported = _count_images(folder) > 0
                translated = _has_ml_state(folder)
                if ch.get("imported") != imported or ch.get("translated") != translated:
                    ch["imported"] = imported
                    ch["translated"] = translated
                    ch["missing_raw"] = not imported
                    ch["needs_sync"] = not imported
                    updated = True
            if ch.get("source") and ch.get("source_id") and not ch.get("chapter_memory_key"):
                ch["chapter_memory_key"] = _stable_chapter_memory_key(
                    str(ch.get("source")),
                    str(entry.get("source_id") or ""),
                    str(ch.get("source_id") or ""),
                    ch.get("episode_no") or "",
                )
                updated = True
            if ch.get("chapter_memory_key") and not ch.get("chapter_memory_fs_key"):
                ch["chapter_memory_fs_key"] = _memory_fs_key(str(ch["chapter_memory_key"]))
                updated = True
        if updated:
            self.save()
        return list(entry.get("chapters", []) or [])

    def current_series_title(self) -> Optional[str]:
        """Return the title of the most recently accessed series (last in list)."""
        if self.series:
            return self.series[-1].get("title")
        return None
