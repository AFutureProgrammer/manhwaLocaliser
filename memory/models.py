"""
memory/models.py
────────────────
Portable, JSON-serialisable data-classes for the translation memory system.

No imports from translator_v14.py or engine.py.
Safe to copy into any future backend unchanged.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_id() -> str:
    """Generate a short unique ID (8 hex chars)."""
    return str(uuid.uuid4())[:8]


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Trust level ───────────────────────────────────────────────────────────────

# Represented as a string literal so it survives JSON round-trips without
# an enum registry.  Rank order (best → worst): manual > approved > imported > machine.
TrustLevel = Literal["manual", "approved", "imported", "machine"]


# ── Core data-classes ─────────────────────────────────────────────────────────

@dataclass
class GlossaryEntry:
    """
    A single glossary rule: 'always translate source_kr as target_en'.

    target_en       — the primary, preferred English rendering.
    alternatives_en — other acceptable renderings used *only* in drift
                      detection (not sent to the model as a constraint).
    aliases_kr      — known OCR variant forms of source_kr (e.g. spacing errors).
    scope           — "global" | "series:<title>"
    """
    id:               str
    source_kr:        str
    target_en:        str
    alternatives_en:  List[str]
    aliases_kr:       List[str]
    trust:            TrustLevel
    scope:            str
    note:             str           = ""
    approved_by:      Optional[str] = None
    created_at:       str           = field(default_factory=now_iso)
    updated_at:       str           = field(default_factory=now_iso)


@dataclass
class NameEntry:
    """A character name mapping: 'always translate kr_name as en_name'.

    scope — "global" | "series:<title>"
            Character names are almost always series-scoped.
            A name is only global if it is a common honorific or term that
            applies universally across every manhwa you translate (rare).
    """
    id:                 str
    kr_name:            str
    en_name:            str
    aliases_kr:         List[str]
    trust:              TrustLevel
    scope:              str
    appearances:        int           = 0
    first_seen_chapter: str           = ""
    note:               str           = ""
    approved_by:        Optional[str] = None
    created_at:         str           = field(default_factory=now_iso)
    updated_at:         str           = field(default_factory=now_iso)


@dataclass
class ChapterTMEntry:
    """
    A single chapter-local provisional translation memory entry.

    Lifecycle
    ---------
    status: "pending" → "reviewed" → "approved" | "rejected"

    Created after each page translation for every non-empty region.
    Always starts as trust="machine", status="pending".

    Stored in:
        <memory_root>/<series_slug>/.chapter_tm/<chapter_id>/tm.json

    Entries are not retrieved into prompts in the current implementation.
    ChapterTM.retrievable_entries() remains available for explicit future
    review flows, but prompt construction keeps TM examples out for now.

    Fields
    ------
    chapter_id    — the chapter folder basename used to scope this store.
    status        — lifecycle state (see above).
    flagged       — heuristic or human-marked bad translation; excluded from
                    retrieval even if approved.
    promoted_to   — None | "series" | "global"; promoted entries are excluded
                    from retrieval (they live in the higher-trust store now).
    """
    id:           str
    kr_text:      str
    en_text:      str
    trust:        TrustLevel          # always "machine" at creation
    chapter_id:   str
    page_idx:     int
    region_idx:   int
    status:       str                 # "pending" | "reviewed" | "approved" | "rejected"
    flagged:      bool                = False
    promoted_to:  Optional[str]       = None   # None | "series" | "global"
    created_at:   str                 = field(default_factory=now_iso)


@dataclass
class BlockedMappingEntry:
    """
    A negative rule: when source_kr appears in the Korean input, the English
    output must NOT contain blocked_en.

    Used in two ways:
    1. Post-translation validation — check_blocked_output() fires a
       ConsistencyWarning when this rule is violated.
    2. TM retrieval suppression — retrieval.py filters out TM candidates
       whose (kr, en) pair would violate a blocked mapping.

    scope — "global" | "series:<title>"
    """
    id:         str
    source_kr:  str
    blocked_en: str
    scope:      str
    reason:     str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class RetrievalResult:
    """
    Output of retrieval.retrieve() for a single source line.

    constraint_block — formatted [CONSTRAINTS] section ready to inject into
                       the translation prompt (empty string when no hits).
    tm_examples      — formatted [TRANSLATION EXAMPLES] section (empty when
                       no approved TM hits survive the budget cap).
    glossary_hits    — GlossaryEntry objects that matched; used by the caller
                       for post-translation consistency checks.
    name_hits        — NameEntry objects that matched; same purpose.
    """
    constraint_block: str
    tm_examples:      str
    glossary_hits:    List[GlossaryEntry]
    name_hits:        List[NameEntry]


@dataclass
class ConsistencyWarning:
    """
    A structured warning produced by the post-translation consistency checker.

    The integration layer in engine.py converts these into the engine's
    issues-dict format for the UI.

    severity        — "critical" when a hard rule is violated; "warn" otherwise.
    warning_type    — machine-readable tag for the specific check that fired.
    expected        — the English term that was required/expected but absent
                      (or, for blocked_output, the term that was forbidden but
                      appeared).
    memory_source   — which store the triggering entry came from.
    memory_id       — ID of the specific entry.
    trust_of_source — trust level of that entry (for UI badge colouring).
    """
    severity:        Literal["critical", "warn", "info"]
    warning_type:    Literal["name_drift", "glossary_drift", "blocked_output"]
    page_idx:        int
    region_idx:      int
    kr_text:         str
    en_text:         str
    expected:        str
    memory_source:   Literal["glossary", "name", "blocked"]
    memory_id:       str
    trust_of_source: str
