"""
memory/chapter_tm.py
─────────────────────
ChapterTM: chapter-local provisional translation memory.

Phase 1  — write-only storage of machine translations.
Phase 5  — status lifecycle: pending → reviewed / approved / rejected.
           Prompt retrieval remains disabled for now; approved entries are
           available for explicit future review/promote flows only.
Phase 6  — flagging: store() accepts a flagged kwarg; flag() method added.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Iterable, List, Optional, Tuple

from .models  import ChapterTMEntry, make_id, now_iso
from .storage import chapter_tm_dir, load_json, save_json

_TM_FILENAME = "tm.json"

# Statuses considered "human-reviewed in some form"
_TERMINAL_STATUSES = {"approved", "rejected"}


class ChapterTM:
    """
    Stores ChapterTMEntry objects for one chapter.

    Read path: pending machine entries are never prompt constraints. Approved,
    non-flagged, non-promoted entries are exposed only for explicit future
    review/promote flows; retrieval.py currently ignores ChapterTM.
    """

    def __init__(self, memory_root: str, series_title: str, chapter_id: str) -> None:
        self._chapter_id = chapter_id
        self._path = os.path.join(
            chapter_tm_dir(memory_root, series_title, chapter_id),
            _TM_FILENAME,
        )
        self._entries: List[ChapterTMEntry] = []
        self.load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> None:
        missing = not os.path.exists(self._path)
        raw = load_json(self._path, default=[])
        if not isinstance(raw, list):
            raw = []
        loaded: List[ChapterTMEntry] = []
        valid_keys = set(ChapterTMEntry.__dataclass_fields__)
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                clean = {k: v for k, v in item.items() if k in valid_keys}
                loaded.append(ChapterTMEntry(**clean))
            except TypeError as exc:
                print(f"[ChapterTM] skipping malformed entry: {exc}")
        self._entries = loaded
        if missing:
            self.save()

    def save(self) -> None:
        save_json(self._path, [asdict(e) for e in self._entries])

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        kr_text:    str,
        en_text:    str,
        page_idx:   int,
        region_idx: int,
        flagged:    bool = False,
    ) -> Optional[str]:
        """
        Persist a machine-generated translation.

        Returns the new or existing entry's id, or None if the text was empty.
        New entries always set trust="machine", status="pending".
        flagged=True marks the entry as bad immediately (heuristic caller
        detected a problem before storing).

        Idempotency key:
            chapter_id + page_idx + region_idx + stripped kr_text + stripped en_text

        Exact duplicates keep their existing status/trust/promoted_to/created_at.
        If a duplicate arrives with flagged=True, the existing entry is marked
        flagged and saved.
        """
        kr = kr_text.strip()
        en = en_text.strip()
        if not kr or not en:
            return None

        for existing in self._entries:
            if self._same_store_key(existing, page_idx, region_idx, kr, en):
                if flagged and not existing.flagged:
                    existing.flagged = True
                    self.save()
                return existing.id

        entry = ChapterTMEntry(
            id         = make_id(),
            kr_text    = kr,
            en_text    = en,
            trust      = "machine",
            chapter_id = self._chapter_id,
            page_idx   = page_idx,
            region_idx = region_idx,
            status     = "pending",
            flagged    = flagged,
            promoted_to = None,
            created_at = now_iso(),
        )
        self._entries.append(entry)
        self.save()
        return entry.id

    def store_batch(
        self,
        entries: Iterable[Tuple[str, str, int, int, bool]],
        chapter_dir: str = "",
    ) -> int:
        """
        Compatibility helper for older engine glue.

        Stores (kr_text, en_text, page_idx, region_idx, flagged) tuples and
        returns the number of newly-created entries. Exact duplicates are not
        counted, though a duplicate may still update flagged=True.
        """
        before_ids = {e.id for e in self._entries}
        for kr_text, en_text, page_idx, region_idx, flagged in entries:
            self.store(kr_text, en_text, page_idx, region_idx, flagged=flagged)
        after_ids = {e.id for e in self._entries}
        return len(after_ids - before_ids)

    def _same_store_key(
        self,
        entry: ChapterTMEntry,
        page_idx: int,
        region_idx: int,
        kr_text: str,
        en_text: str,
    ) -> bool:
        return (
            entry.chapter_id == self._chapter_id
            and entry.page_idx == page_idx
            and entry.region_idx == region_idx
            and entry.kr_text.strip() == kr_text
            and entry.en_text.strip() == en_text
        )

    # ── Status mutations (Phase 5) ────────────────────────────────────────────

    def _get(self, entry_id: str) -> Optional[ChapterTMEntry]:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def approve(self, entry_id: str) -> bool:
        """Mark entry as approved.  Returns False if not found."""
        e = self._get(entry_id)
        if e is None:
            return False
        e.status  = "approved"
        e.flagged = False   # explicit approval clears a heuristic flag
        self.save()
        return True

    def reject(self, entry_id: str) -> bool:
        """Mark entry as rejected.  Rejected entries are never retrieved."""
        e = self._get(entry_id)
        if e is None:
            return False
        e.status = "rejected"
        self.save()
        return True

    def mark_reviewed(self, entry_id: str) -> bool:
        """Mark as reviewed without approving (intermediate state)."""
        e = self._get(entry_id)
        if e is None or e.status in _TERMINAL_STATUSES:
            return False
        e.status = "reviewed"
        self.save()
        return True

    def flag(self, entry_id: str, reason: str = "") -> bool:
        """Heuristically flag an entry as bad.  Does not change status."""
        e = self._get(entry_id)
        if e is None:
            return False
        e.flagged = True
        self.save()
        return True

    def mark_promoted(self, entry_id: str, scope: str) -> bool:
        """Mark as promoted to 'series' or 'global'.  Removes from retrieval."""
        if scope not in ("series", "global"):
            raise ValueError(f"Invalid promotion scope: {scope!r}")
        e = self._get(entry_id)
        if e is None:
            return False
        e.promoted_to = scope
        self.save()
        return True

    def update_translation(self, entry_id: str, new_en: str) -> bool:
        """
        Replace the English text (e.g. after human edit).
        Resets status to 'reviewed' so it requires re-approval for retrieval.
        Clears heuristic flag since text has changed.
        """
        e = self._get(entry_id)
        if e is None or not new_en.strip():
            return False
        e.en_text = new_en.strip()
        e.status  = "reviewed"
        e.flagged = False
        self.save()
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def all_entries(self) -> List[ChapterTMEntry]:
        return list(self._entries)

    def total_count(self) -> int:
        return len(self._entries)

    def provisional_count(self) -> int:
        return sum(
            1 for e in self._entries
            if e.trust == "machine" and e.status in ("pending", "reviewed")
        )

    def entries_for_page(self, page_idx: int) -> List[ChapterTMEntry]:
        return [e for e in self._entries if e.page_idx == page_idx]

    def retrievable_entries(self) -> List[ChapterTMEntry]:
        """
        Entries eligible for explicit future review/reuse flows.

        Rules (conservative by design):
        • status == "approved"
        • flagged == False
        • promoted_to is None  (promoted entries live in the series store now)
        """
        return [
            e for e in self._entries
            if e.status == "approved"
            and not e.flagged
            and e.promoted_to is None
        ]

    def pending_review(self) -> List[ChapterTMEntry]:
        """All entries that haven't been approved or rejected yet."""
        return [e for e in self._entries if e.status in ("pending", "reviewed")]
