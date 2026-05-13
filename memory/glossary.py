"""
memory/glossary.py
──────────────────
GlossaryStore: persistent, series-scoped glossary rules.

Phase 1 scope
─────────────
• load / save (JSON-backed)
• exact_match(kr_text)  — substring search after whitespace normalisation
• to_prompt_block()     — compact constraint string for the translation prompt
• migrate_from_anchors() — one-time seed from the legacy GLOSSARY_ANCHORS dict

Not yet implemented (future phases):
• fuzzy / edit-distance matching
• blocked-mapping entries
• trust-decay or auto-promotion

Portable: no dependencies on translator_v14.py or engine.py.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List

from .models    import GlossaryEntry, make_id, now_iso
from .normalize import normalize_for_match
from .storage   import series_dir, load_json, save_json

_GLOSSARY_FILENAME = "glossary.json"

# Rank order for sorting hits: lower number = higher priority.
_TRUST_RANK: Dict[str, int] = {
    "manual":   0,
    "approved": 1,
    "imported": 2,
    "machine":  3,
}


class GlossaryStore:
    """
    Loads and saves GlossaryEntry objects from/to::

        <memory_root>/<series_slug>/glossary.json

    Only exact-match lookup is implemented in this phase.

    Parameters
    ----------
    memory_root   : str
        Root directory for all series memory (supplied by the engine glue,
        typically ``<project_root>/series_memory``).
    series_title  : str
        Human-readable series title used to derive the subdirectory name.
    """

    def __init__(self, memory_root: str, series_title: str) -> None:
        self._series_title = series_title
        self._path = os.path.join(
            series_dir(memory_root, series_title), _GLOSSARY_FILENAME
        )
        self._entries: List[GlossaryEntry] = []
        self.load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load entries from disk.  Silently skips malformed items."""
        missing = not os.path.exists(self._path)
        raw = load_json(self._path, default=[])
        if not isinstance(raw, list):
            raw = []
        loaded: List[GlossaryEntry] = []
        valid_keys = set(GlossaryEntry.__dataclass_fields__)
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                # Strip unknown keys for forward-compatibility
                clean = {k: v for k, v in item.items() if k in valid_keys}
                loaded.append(GlossaryEntry(**clean))
            except TypeError as exc:
                print(f"[GlossaryStore] skipping malformed entry: {exc}")
        self._entries = loaded
        if missing:
            self.save()

    def save(self) -> None:
        """Persist current entries to disk."""
        save_json(self._path, [asdict(e) for e in self._entries])

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, entry: GlossaryEntry) -> None:
        """Add *entry*, replacing any existing entry with the same id."""
        self._entries = [e for e in self._entries if e.id != entry.id]
        self._entries.append(entry)
        self.save()

    def all_entries(self) -> List[GlossaryEntry]:
        """Return a shallow copy of all stored entries."""
        return list(self._entries)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def exact_match(self, kr_text: str) -> List[GlossaryEntry]:
        """
        Return all glossary entries whose ``source_kr`` (or any ``aliases_kr``)
        appears as a substring of *kr_text* after whitespace normalisation.

        Ordering
        --------
        1. Higher trust first (manual > approved > imported > machine).
        2. Longer ``source_kr`` first (more specific terms take priority when
           one term is a substring of another).

        This is the same behaviour as the existing ``apply_glossary_anchors``
        check in translator_v14.py but now returns structured objects instead
        of just logging.

        Known limitation (Phase 1)
        --------------------------
        Substring matching means a very short glossary key (e.g. "학") could
        spuriously match inside a longer unrelated word (e.g. "입학").  The
        GLOSSARY_ANCHORS terms migrated from translator_v14.py are long enough
        that false positives are unlikely in practice.  Proper word-boundary
        matching will be added in a future phase.
        """
        norm_input = normalize_for_match(kr_text)
        if not norm_input:
            return []

        hits: List[GlossaryEntry] = []
        seen_ids: set = set()
        for entry in self._entries:
            if entry.id in seen_ids:
                continue
            matched = normalize_for_match(entry.source_kr) in norm_input or any(
                normalize_for_match(alias) in norm_input
                for alias in entry.aliases_kr
            )
            if matched:
                hits.append(entry)
                seen_ids.add(entry.id)

        hits.sort(
            key=lambda e: (_TRUST_RANK.get(e.trust, 9), -len(e.source_kr))
        )
        return hits

    # ── Prompt helpers ────────────────────────────────────────────────────────

    @staticmethod
    def to_prompt_block(entries: List[GlossaryEntry]) -> str:
        """
        Format a list of glossary hits as compact translation-constraint lines.

        Output example::

            - Always translate "도련님" as "Young Master"
            - Always translate "왕실" as "royal"

        Returns an empty string if *entries* is empty so the caller can do a
        simple ``if constraint_block:`` guard before inserting it.
        """
        if not entries:
            return ""
        return "\n".join(
            f'- Always translate "{e.source_kr}" as "{e.target_en}"'
            for e in entries
        )

    # ── Migration ─────────────────────────────────────────────────────────────

    def migrate_from_anchors(
        self, glossary_anchors: Dict[str, List[str]]
    ) -> int:
        """
        One-time seed from the legacy ``GLOSSARY_ANCHORS`` dict in
        translator_v14.py.

        Design notes
        ------------
        • The *first* element of each alternatives list becomes ``target_en``;
          the rest go into ``alternatives_en`` for drift detection only.
        • This means the prompt constraint ("always translate X as Y") uses
          only the first alternative.  If that choice is poor for a given term
          (e.g. "run" for "도망"), edit ``glossary.json`` manually and set a
          better ``target_en``.
        • Existing entries are **never overwritten** — migration is idempotent.
        • All migrated entries get ``trust: "imported"`` and are clearly
          annotated in their ``note`` field.

        Parameters
        ----------
        glossary_anchors : Dict[str, List[str]]
            The ``GLOSSARY_ANCHORS`` dict from translator_v14.py.

        Returns
        -------
        int
            Count of newly created entries (0 on subsequent calls).
        """
        existing_norm = {normalize_for_match(e.source_kr) for e in self._entries}
        added = 0

        for kr_term, en_alts in glossary_anchors.items():
            if not en_alts:
                continue
            if normalize_for_match(kr_term) in existing_norm:
                continue

            entry = GlossaryEntry(
                id              = make_id(),
                source_kr       = kr_term,
                target_en       = en_alts[0],
                alternatives_en = list(en_alts[1:]),
                aliases_kr      = [],
                trust           = "imported",
                scope           = f"series:{self._series_title}",
                note            = (
                    "Auto-migrated from GLOSSARY_ANCHORS. "
                    f"All alternatives: {', '.join(en_alts)}. "
                    "Review target_en — it is the first listed alternative and "
                    "may not be the best choice for every context."
                ),
                created_at      = now_iso(),
                updated_at      = now_iso(),
            )
            self._entries.append(entry)
            existing_norm.add(normalize_for_match(kr_term))
            added += 1

        if added:
            self.save()
        return added
