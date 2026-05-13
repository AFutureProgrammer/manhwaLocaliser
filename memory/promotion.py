"""
memory/promotion.py
───────────────────
Phase 5 — Promotion / Approval Workflow.

Moves ChapterTMEntry objects from provisional ("machine") state into
higher-trust stores.  All promotion requires explicit human action routed
through the engine API — nothing auto-promotes silently.

Promotion paths
───────────────
                 ChapterTM (machine/reviewed/approved)
                       │
          ─────────────┼──────────────
          │                          │
    promote_to_series()        promote_to_global()
          │                          │
    series GlossaryStore       global GlossaryStore
    or series NameMemory        or global NameMemory
    (trust="approved")          (trust="approved")

The source ChapterTMEntry is marked promoted_to="series"|"global" so it
is excluded from future ChapterTM retrieval (the promoted store has it now).

Convenience functions
─────────────────────
approve_entry()           — approve without promoting (stays in ChapterTM)
reject_entry()            — reject with optional blocked-mapping creation
promote_entry_to_series() — promote to series GlossaryStore / NameMemory
promote_entry_to_global() — promote to global GlossaryStore / NameMemory

All functions are pure logic; they accept already-loaded store objects and
return (bool_success, message) so the engine layer can log or surface errors.

Portable: no imports from translator_v14.py or engine.py.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .chapter_tm  import ChapterTM
from .glossary    import GlossaryStore
from .name_memory import NameMemory
from .blocked     import BlockedMappingStore
from .models      import GlossaryEntry, NameEntry, make_id, now_iso


# ── Simple approval / rejection (no store promotion) ─────────────────────────

def approve_entry(tm: ChapterTM, entry_id: str) -> Tuple[bool, str]:
    """
    Mark a ChapterTMEntry as approved so it becomes eligible for TM retrieval.
    Does NOT promote it to series or global memory.
    """
    if tm.approve(entry_id):
        return True, f"Entry {entry_id} approved."
    return False, f"Entry {entry_id} not found."


def reject_entry(
    tm:               ChapterTM,
    entry_id:         str,
    blocked_store:    Optional[BlockedMappingStore] = None,
    series_scope:     str = "",
    reason:           str = "",
) -> Tuple[bool, str]:
    """
    Reject a ChapterTMEntry.

    If *blocked_store* is provided and the entry has non-empty kr/en text,
    a BlockedMappingEntry is automatically created to prevent the same
    (kr, en) pair from being retrieved in the future — this is the Phase 6
    poisoning safeguard for explicit human rejections.

    Parameters
    ──────────
    tm            : the ChapterTM that owns the entry
    entry_id      : id of the entry to reject
    blocked_store : series-scoped BlockedMappingStore (optional but recommended)
    series_scope  : scope string for the blocked entry, e.g. "series:MyTitle"
    reason        : human-readable reason for the rejection
    """
    entry = tm._get(entry_id)
    if entry is None:
        return False, f"Entry {entry_id} not found."

    tm.reject(entry_id)

    if blocked_store and entry.kr_text.strip() and entry.en_text.strip():
        blocked_store.add(
            source_kr  = entry.kr_text,
            blocked_en = entry.en_text,
            scope      = series_scope or "series:unknown",
            reason     = reason or f"Rejected TM entry {entry_id}",
        )
        return True, (
            f"Entry {entry_id} rejected and blocked mapping created "
            f"for '{entry.en_text[:40]}'."
        )

    return True, f"Entry {entry_id} rejected."


def mark_reviewed(tm: ChapterTM, entry_id: str) -> Tuple[bool, str]:
    """Mark a pending entry as reviewed (human has seen it; awaiting approval)."""
    if tm.mark_reviewed(entry_id):
        return True, f"Entry {entry_id} marked reviewed."
    return False, f"Entry {entry_id} not found or already in terminal state."


# ── Promotion to series memory ────────────────────────────────────────────────

def promote_entry_to_series(
    tm:             ChapterTM,
    entry_id:       str,
    glossary_store: GlossaryStore,
    name_store:     NameMemory,
    series_title:   str,
    *,
    as_name:        bool = False,
    kr_canonical:   Optional[str] = None,
    en_canonical:   Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Promote a ChapterTMEntry to the series-scoped GlossaryStore or NameMemory.

    Parameters
    ──────────
    as_name       : if True, promote as a NameEntry; otherwise GlossaryEntry.
    kr_canonical  : override source Korean text (defaults to entry.kr_text).
    en_canonical  : override English text (defaults to entry.en_text).

    The entry must be approved before promotion is allowed.  This prevents
    accidental promotion of pending machine output.

    The source entry is marked promoted_to="series" so ChapterTM.retrievable()
    excludes it (the series store has it now, avoiding duplicates).
    """
    entry = tm._get(entry_id)
    if entry is None:
        return False, f"Entry {entry_id} not found."
    if entry.status != "approved":
        return False, (
            f"Entry {entry_id} must be approved before promotion "
            f"(current status: {entry.status!r})."
        )

    kr = (kr_canonical or entry.kr_text).strip()
    en = (en_canonical or entry.en_text).strip()
    scope = f"series:{series_title}"

    if as_name:
        new_entry = NameEntry(
            id         = make_id(),
            kr_name    = kr,
            en_name    = en,
            aliases_kr = [],
            trust      = "approved",
            scope      = scope,
            note       = f"Promoted from ChapterTM entry {entry_id}.",
            created_at = now_iso(),
            updated_at = now_iso(),
        )
        name_store.add(new_entry)
    else:
        new_entry = GlossaryEntry(
            id              = make_id(),
            source_kr       = kr,
            target_en       = en,
            alternatives_en = [],
            aliases_kr      = [],
            trust           = "approved",
            scope           = scope,
            note            = f"Promoted from ChapterTM entry {entry_id}.",
            created_at      = now_iso(),
            updated_at      = now_iso(),
        )
        glossary_store.add(new_entry)

    tm.mark_promoted(entry_id, "series")
    kind = "name" if as_name else "glossary"
    return True, f"Entry {entry_id} promoted to series {kind} store."


def promote_entry_to_global(
    tm:             ChapterTM,
    entry_id:       str,
    glossary_store: GlossaryStore,
    name_store:     NameMemory,
    *,
    as_name:        bool = False,
    kr_canonical:   Optional[str] = None,
    en_canonical:   Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Promote a ChapterTMEntry to the global GlossaryStore or NameMemory.

    Same as promote_entry_to_series() but targets the global scope.
    Use sparingly — only for truly language-wide terms.
    """
    entry = tm._get(entry_id)
    if entry is None:
        return False, f"Entry {entry_id} not found."
    if entry.status != "approved":
        return False, (
            f"Entry {entry_id} must be approved before global promotion "
            f"(current status: {entry.status!r})."
        )

    kr = (kr_canonical or entry.kr_text).strip()
    en = (en_canonical or entry.en_text).strip()

    if as_name:
        new_entry = NameEntry(
            id         = make_id(),
            kr_name    = kr,
            en_name    = en,
            aliases_kr = [],
            trust      = "approved",
            scope      = "global",
            note       = f"Promoted from ChapterTM entry {entry_id}.",
            created_at = now_iso(),
            updated_at = now_iso(),
        )
        name_store.add(new_entry)
    else:
        new_entry = GlossaryEntry(
            id              = make_id(),
            source_kr       = kr,
            target_en       = en,
            alternatives_en = [],
            aliases_kr      = [],
            trust           = "approved",
            scope           = "global",
            note            = f"Promoted from ChapterTM entry {entry_id}.",
            created_at      = now_iso(),
            updated_at      = now_iso(),
        )
        glossary_store.add(new_entry)

    tm.mark_promoted(entry_id, "global")
    kind = "name" if as_name else "glossary"
    return True, f"Entry {entry_id} promoted to global {kind} store."
