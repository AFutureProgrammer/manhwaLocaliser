"""
memory/blocked.py
─────────────────
BlockedMappingStore: persistent negative mappings (Phase 6).

A blocked mapping says: "when the Korean source contains source_kr,
the English output must NOT contain blocked_en."

Used in two ways:
1. Post-translation validation (consistency.py calls check_blocked_output).
2. Retrieval suppression (retrieval.py filters out TM hits that match a
   blocked (kr, en) pair before injecting them into prompts).

Scope
─────
Blocked mappings follow the same global / series separation as the rest of
the memory system.  engine.py loads both and merges them the same way it
merges glossary and name stores.

File: <memory_root>/<series_slug>/blocked.json
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import List

from .models    import BlockedMappingEntry, make_id, now_iso
from .normalize import normalize_for_match
from .storage   import series_dir, load_json, save_json

_BLOCKED_FILENAME = "blocked.json"


class BlockedMappingStore:
    """
    Loads and saves BlockedMappingEntry objects.

    Lookup is substring-based (same strategy as GlossaryStore):
    both source_kr and blocked_en are checked case-insensitively after
    whitespace normalisation.
    """

    def __init__(self, memory_root: str, series_title: str) -> None:
        self._path = os.path.join(
            series_dir(memory_root, series_title), _BLOCKED_FILENAME
        )
        self._entries: List[BlockedMappingEntry] = []
        self.load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> None:
        missing = not os.path.exists(self._path)
        raw = load_json(self._path, default=[])
        if not isinstance(raw, list):
            raw = []
        loaded: List[BlockedMappingEntry] = []
        valid_keys = set(BlockedMappingEntry.__dataclass_fields__)
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                clean = {k: v for k, v in item.items() if k in valid_keys}
                loaded.append(BlockedMappingEntry(**clean))
            except TypeError as exc:
                print(f"[BlockedMappingStore] skipping malformed entry: {exc}")
        self._entries = loaded
        if missing:
            self.save()

    def save(self) -> None:
        save_json(self._path, [asdict(e) for e in self._entries])

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, source_kr: str, blocked_en: str, scope: str, reason: str = "") -> BlockedMappingEntry:
        """
        Add a blocked mapping.  Deduplicates by (source_kr, blocked_en) pair.
        Returns the existing entry if already present.
        """
        norm_kr = normalize_for_match(source_kr)
        norm_en = blocked_en.strip().lower()
        for e in self._entries:
            if normalize_for_match(e.source_kr) == norm_kr and e.blocked_en.lower() == norm_en:
                return e
        entry = BlockedMappingEntry(
            id         = make_id(),
            source_kr  = source_kr,
            blocked_en = blocked_en,
            scope      = scope,
            reason     = reason,
            created_at = now_iso(),
        )
        self._entries.append(entry)
        self.save()
        return entry

    def remove(self, entry_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != entry_id]
        if len(self._entries) < before:
            self.save()
            return True
        return False

    def all_entries(self) -> List[BlockedMappingEntry]:
        return list(self._entries)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def matches(self, kr_text: str, en_text: str) -> List[BlockedMappingEntry]:
        """
        Return all blocked entries where:
          • source_kr is a substring of kr_text (normalised)
          • blocked_en is a substring of en_text (case-insensitive)

        An entry matching on BOTH conditions means the output is violating
        a negative rule.
        """
        norm_kr = normalize_for_match(kr_text)
        en_lower = en_text.lower()
        return [
            e for e in self._entries
            if normalize_for_match(e.source_kr) in norm_kr
            and e.blocked_en.lower() in en_lower
        ]

    def suppresses_tm(self, kr_text: str, en_text: str) -> bool:
        """
        Return True if any blocked mapping makes this (kr, en) pair unsafe
        to retrieve into a prompt — i.e. the blocked_en appears in en_text
        while source_kr appears in kr_text.

        Used by retrieval.py to filter TM candidates before prompt injection.
        """
        return bool(self.matches(kr_text, en_text))
