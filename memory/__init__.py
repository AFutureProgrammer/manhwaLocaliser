"""
memory
──────
Portable translation memory + glossary consistency core.

Scope separation
────────────────
Global memory   — cross-series rules stored in <memory_root>/_global/
                  Empty by default.  Add entries here only for things that
                  truly apply to every manhwa you translate (e.g. universal
                  romanisation conventions).

Series memory   — per-series rules in <memory_root>/<series_slug>/
                  Character names and series-specific terms live here.
                  A new series ALWAYS starts with empty stores.
                  Legacy NAME_MAP / GLOSSARY_ANCHORS are NEVER applied
                  automatically — call engine.migrate_legacy_series() once,
                  explicitly, for the legacy project only.

Chapter memory  — per-chapter provisional machine-generated TM in
                  <memory_root>/<series_slug>/.chapter_tm/<chapter_id>/tm.json
                  Write-only for prompts in the current implementation.
                  Entries remain pending machine output until explicitly
                  reviewed/promoted by a human.

Public API
──────────
GlossaryStore        — persistent glossary rules (exact-match lookup)
NameMemory           — persistent character name mappings (exact-match)
ChapterTM            — chapter-local provisional translation memory
BlockedMappingStore  — negative mapping rules (Phase 6)

retrieve_batch       — build bounded prompt context for a batch of lines
check_name_drift     — post-translation name consistency check
check_glossary_drift — post-translation glossary consistency check
check_blocked_output — post-translation blocked-mapping violation check

approve_entry            — approve a ChapterTM entry (enables retrieval)
reject_entry             — reject + optionally auto-block the (kr, en) pair
mark_reviewed            — mark entry as seen but not yet approved
promote_entry_to_series  — move approved entry to series GlossaryStore/NameMemory
promote_entry_to_global  — move approved entry to global GlossaryStore/NameMemory

Data-classes
────────────
GlossaryEntry, NameEntry, ChapterTMEntry, BlockedMappingEntry,
RetrievalResult, ConsistencyWarning

Migration helpers (opt-in only, legacy project only)
────────────────────────────────────────────────────
GlossaryStore.migrate_from_anchors(GLOSSARY_ANCHORS)
NameMemory.migrate_from_name_map(NAME_MAP)

These are NOT called automatically.  Route through
engine.migrate_legacy_series(series_title) for legacy projects only.

Disk layout
───────────
<memory_root>/
    _global/
        glossary.json       ← global GlossaryEntry list  (empty by default)
        names.json          ← global NameEntry list       (empty by default)
        blocked.json        ← global BlockedMappingEntry list
    <series_slug>/
        glossary.json       ← series GlossaryEntry list  (empty by default)
        names.json          ← series NameEntry list       (empty by default)
        blocked.json        ← series BlockedMappingEntry list
        .chapter_tm/
            <chapter_id>/
                tm.json     ← ChapterTMEntry list (pending → approved/rejected)
"""
from .glossary      import GlossaryStore
from .name_memory   import NameMemory
from .chapter_tm    import ChapterTM
from .blocked       import BlockedMappingStore
from .retrieval     import retrieve_batch
from .consistency   import check_name_drift, check_glossary_drift, check_blocked_output
from .promotion     import (
    approve_entry,
    reject_entry,
    mark_reviewed,
    promote_entry_to_series,
    promote_entry_to_global,
)
from .models        import (
    GlossaryEntry,
    NameEntry,
    ChapterTMEntry,
    BlockedMappingEntry,
    RetrievalResult,
    ConsistencyWarning,
)

__all__ = [
    # Stores
    "GlossaryStore",
    "NameMemory",
    "ChapterTM",
    "BlockedMappingStore",
    # Retrieval
    "retrieve_batch",
    # Consistency checks
    "check_name_drift",
    "check_glossary_drift",
    "check_blocked_output",
    # Promotion / approval workflow
    "approve_entry",
    "reject_entry",
    "mark_reviewed",
    "promote_entry_to_series",
    "promote_entry_to_global",
    # Data-classes
    "GlossaryEntry",
    "NameEntry",
    "ChapterTMEntry",
    "BlockedMappingEntry",
    "RetrievalResult",
    "ConsistencyWarning",
]
