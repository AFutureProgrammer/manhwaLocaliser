"""
memory/name_memory.py
─────────────────────
NameMemory: persistent, series-scoped character name mappings.

Phase 1 scope
─────────────
• load / save (JSON-backed)
• exact_match(kr_text)      — substring search after whitespace normalisation
• to_prompt_block()         — compact constraint string for the translation prompt
• migrate_from_name_map()   — one-time seed from the legacy NAME_MAP dict

Kept separate from GlossaryStore because names need different handling:
  • More OCR aliases (spacing errors in proper nouns are common)
  • May be augmented with honorific context ("한스 도련님")
  • Will later track appearance counts for confidence scoring

Portable: no dependencies on translator_v14.py or engine.py.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List

from .models    import NameEntry, make_id, now_iso
from .normalize import normalize_for_match
from .storage   import series_dir, load_json, save_json

_NAMES_FILENAME = "names.json"

_TRUST_RANK: Dict[str, int] = {
    "manual":   0,
    "approved": 1,
    "imported": 2,
    "machine":  3,
}


class NameMemory:
    """
    Loads and saves NameEntry objects from/to::

        <memory_root>/<series_slug>/names.json

    Only exact-match lookup is implemented in this phase.

    Parameters
    ----------
    memory_root   : str
        Root directory for all series memory.
    series_title  : str
        Human-readable series title used to derive the subdirectory name.
    """

    def __init__(self, memory_root: str, series_title: str) -> None:
        self._series_title = series_title
        self._path = os.path.join(
            series_dir(memory_root, series_title), _NAMES_FILENAME
        )
        self._entries: List[NameEntry] = []
        self.load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load entries from disk.  Silently skips malformed items."""
        missing = not os.path.exists(self._path)
        raw = load_json(self._path, default=[])
        if not isinstance(raw, list):
            raw = []
        loaded: List[NameEntry] = []
        valid_keys = set(NameEntry.__dataclass_fields__)
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                clean = {k: v for k, v in item.items() if k in valid_keys}
                loaded.append(NameEntry(**clean))
            except TypeError as exc:
                print(f"[NameMemory] skipping malformed entry: {exc}")
        self._entries = loaded
        if missing:
            self.save()

    def save(self) -> None:
        """Persist current entries to disk."""
        save_json(self._path, [asdict(e) for e in self._entries])

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, entry: NameEntry) -> None:
        """Add *entry*, replacing any existing entry with the same id."""
        self._entries = [e for e in self._entries if e.id != entry.id]
        self._entries.append(entry)
        self.save()

    def all_entries(self) -> List[NameEntry]:
        """Return a shallow copy of all stored entries."""
        return list(self._entries)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def exact_match(self, kr_text: str) -> List[NameEntry]:
        """
        Return all name entries whose ``kr_name`` (or any ``aliases_kr``)
        appears as a substring of *kr_text* after whitespace normalisation.

        Ordering
        --------
        1. Higher trust first.
        2. Longer ``kr_name`` first (more specific / full names before short ones).

        Examples
        --------
        Input "아이작 도련님이 오셨습니다" →
            • Matches NameEntry(kr_name="아이작") because "아이작" ⊂ normalised input.
            • Also matches an alias entry for "아이작도련님" if one was added.
        """
        norm_input = normalize_for_match(kr_text)
        if not norm_input:
            return []

        hits: List[NameEntry] = []
        seen_ids: set = set()
        for entry in self._entries:
            if entry.id in seen_ids:
                continue
            matched = normalize_for_match(entry.kr_name) in norm_input or any(
                normalize_for_match(alias) in norm_input
                for alias in entry.aliases_kr
            )
            if matched:
                hits.append(entry)
                seen_ids.add(entry.id)

        hits.sort(
            key=lambda e: (_TRUST_RANK.get(e.trust, 9), -len(e.kr_name))
        )
        return hits

    # ── Prompt helpers ────────────────────────────────────────────────────────

    @staticmethod
    def to_prompt_block(entries: List[NameEntry]) -> str:
        """
        Format name hits as compact translation-constraint lines.

        Output example::

            - Always translate "아이작" as "Isaac"
            - Always translate "한스" as "Hans"

        Returns an empty string if *entries* is empty.
        """
        if not entries:
            return ""
        return "\n".join(
            f'- Always translate "{e.kr_name}" as "{e.en_name}"'
            for e in entries
        )

    # ── Migration ─────────────────────────────────────────────────────────────

    def migrate_from_name_map(self, name_map: Dict[str, str]) -> int:
        """
        One-time seed from the legacy ``NAME_MAP`` dict in translator_v14.py.

        • Existing entries are **never overwritten** — idempotent.
        • All migrated entries get ``trust: "imported"``.

        Parameters
        ----------
        name_map : Dict[str, str]
            The ``NAME_MAP`` dict, e.g. ``{"아이작": "Isaac", "한스": "Hans"}``.

        Returns
        -------
        int
            Count of newly created entries (0 on subsequent calls).
        """
        existing_norm = {normalize_for_match(e.kr_name) for e in self._entries}
        added = 0

        for kr_name, en_name in name_map.items():
            if normalize_for_match(kr_name) in existing_norm:
                continue

            entry = NameEntry(
                id                 = make_id(),
                kr_name            = kr_name,
                en_name            = en_name,
                aliases_kr         = [],
                trust              = "imported",
                scope              = f"series:{self._series_title}",
                note               = "Auto-migrated from NAME_MAP.",
                created_at         = now_iso(),
                updated_at         = now_iso(),
            )
            self._entries.append(entry)
            existing_norm.add(normalize_for_match(kr_name))
            added += 1

        if added:
            self.save()
        return added
