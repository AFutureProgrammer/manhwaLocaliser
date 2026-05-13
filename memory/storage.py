"""
memory/storage.py
─────────────────
JSON persistence helpers for the memory system.

Portable: no dependencies on translator_v14.py or engine.py.
The memory_root and chapter paths are always passed in by the
integration layer so this module has no hard-coded project paths.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


# ── Path helpers ──────────────────────────────────────────────────────────────

# Reserved slug for the global (cross-series) memory store.
GLOBAL_SLUG = "_global"

# Directory name under <series_slug>/ that holds per-chapter TM data.
_CHAPTER_TM_DIRNAME = ".chapter_tm"


def series_slug(series_title: str) -> str:
    """
    Convert a series title to a safe, human-readable filesystem name.

    The special sentinel "_global" passes through unchanged so that
    GlossaryStore("_global") reliably maps to the global directory.

    Examples
    --------
    "_global"                        → "_global"
    "A Knight Living Only for Today" → "a_knight_living_only_for_today"
    "Solo Leveling!!"                → "solo_leveling"
    """
    if series_title == GLOBAL_SLUG:
        return GLOBAL_SLUG
    slug = str(series_title or "").lower().replace(":", "_")
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug or "unnamed_series"


def series_dir(memory_root: str, series_title: str) -> str:
    """
    Return the per-series (or global) memory directory, creating it if absent.

    Layout
    ------
    <memory_root>/
        _global/              ← cross-series rules (empty by default)
            glossary.json
            names.json
            blocked.json
        <series_slug>/        ← per-series rules (always starts empty)
            glossary.json
            names.json
            blocked.json
            .chapter_tm/
                <chapter_id>/
                    tm.json
    """
    d = os.path.join(memory_root, series_slug(series_title))
    os.makedirs(d, exist_ok=True)
    return d


def chapter_tm_dir(memory_root: str, series_title: str, chapter_id: str) -> str:
    """
    Return the per-chapter TM directory, creating it if absent.

    chapter_id is the chapter folder basename (e.g. "chapter_101").
    The directory is nested under the series slug so different series
    never share chapter TM data.

    Layout
    ------
    <memory_root>/<series_slug>/.chapter_tm/<chapter_id>/
        tm.json     ← ChapterTMEntry list
    """
    # Sanitise chapter_id: keep logical ids stable while making Windows-safe paths.
    safe_id = re.sub(r"[:/\\<>|?*\"]", "_", chapter_id.strip()) or "chapter"
    d = os.path.join(
        series_dir(memory_root, series_title),
        _CHAPTER_TM_DIRNAME,
        safe_id,
    )
    os.makedirs(d, exist_ok=True)
    return d


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def load_json(path: str, default: Any = None) -> Any:
    """
    Load JSON from *path*.

    Returns *default* on FileNotFoundError.
    Returns *default* and prints a warning on any other parse/IO error.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except Exception as exc:
        print(f"[memory.storage] load_json failed for {path!r}: {exc}")
        return default


def save_json(path: str, data: Any) -> None:
    """
    Write *data* as formatted JSON to *path*.

    Uses a .tmp intermediate file + os.replace() for a near-atomic write
    (safe against partial writes on crash).  Directory is created if absent.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[memory.storage] save_json failed for {path!r}: {exc}")
