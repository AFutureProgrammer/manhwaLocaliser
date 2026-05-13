"""
memory/retrieval.py
───────────────────
Phase 4 — Retrieval Engine.

Produces a bounded, prompt-safe RetrievalResult for a source line or batch.

Priority order (strict, non-overridable)
─────────────────────────────────────────
1. Exact name matches       (hard constraint — never skipped)
2. Exact glossary matches   (hard constraint — never skipped)

Rules
─────
• Exact glossary/name constraints are ALWAYS included regardless of budget.
  They are short and deterministic; truncating them would break correctness.
• Series memory is checked before global memory; duplicate global entries are
  suppressed by the series entry.
• Chapter TM is write-only for now.  Machine-only ("pending" / "reviewed") TM
  entries are NEVER retrieved, and no TM examples are injected into prompts.
• Fuzzy TM retrieval is intentionally disabled for prompt construction.

Prompt layout produced
──────────────────────
    [CONSTRAINTS]
    - Always translate "도련님" as "Young Master"
    - Always translate "아이작" as "Isaac"

Portable: no imports from translator_v14.py or engine.py.
"""
from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Sequence

from .models      import (
    GlossaryEntry,
    NameEntry,
    ChapterTMEntry,
    BlockedMappingEntry,
    RetrievalResult,
)
from .normalize   import normalize_for_match

# ── Budget constants ──────────────────────────────────────────────────────────

# Kept for API compatibility. TM examples are not injected into prompts.
TM_EXAMPLES_HARD_CAP: int = 4

# Kept for API compatibility. Fuzzy TM retrieval is disabled for prompts.
FUZZY_THRESHOLD: float = 0.72

# Maximum characters in a TM example line (kr + en together).
# Prevents very long lines from consuming disproportionate prompt space.
TM_LINE_MAX_CHARS: int = 180


# ── Trust rank ────────────────────────────────────────────────────────────────

_TRUST_RANK: Dict[str, int] = {
    "manual":   0,
    "approved": 1,
    "imported": 2,
    "machine":  3,
}


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve(
    kr_text:        str,
    global_glossary: List[GlossaryEntry],
    series_glossary: List[GlossaryEntry],
    global_names:    List[NameEntry],
    series_names:    List[NameEntry],
    chapter_tm:      List[ChapterTMEntry],         # pre-filtered by ChapterTM.retrievable_entries()
    blocked:         List[BlockedMappingEntry],    # merged global + series blocked entries
    *,
    tm_cap:          int   = TM_EXAMPLES_HARD_CAP,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> RetrievalResult:
    """
    Build a RetrievalResult for *kr_text*.

    Parameters
    ──────────
    All list parameters are pre-loaded by the engine; retrieval.py does not
    touch disk itself.

    chapter_tm and blocked are accepted for API compatibility, but chapter TM
    is not used for prompt construction in this implementation.
    """
    norm_input = normalize_for_match(kr_text)

    # ── Step 1: exact glossary hits ───────────────────────────────────────────
    g_hits = _exact_glossary_hits(norm_input, global_glossary, series_glossary)

    # ── Step 2: exact name hits ───────────────────────────────────────────────
    n_hits = _exact_name_hits(norm_input, global_names, series_names)

    # ── Assemble prompt blocks ────────────────────────────────────────────────
    constraint_block = _build_constraint_block(g_hits, n_hits)

    return RetrievalResult(
        constraint_block = constraint_block,
        tm_examples      = "",
        glossary_hits    = g_hits,
        name_hits        = n_hits,
    )


def retrieve_batch(
    kr_texts:        Sequence[str],
    global_glossary: List[GlossaryEntry],
    series_glossary: List[GlossaryEntry],
    global_names:    List[NameEntry],
    series_names:    List[NameEntry],
    chapter_tm:      List[ChapterTMEntry],
    blocked:         List[BlockedMappingEntry],
    *,
    tm_cap:          int   = TM_EXAMPLES_HARD_CAP,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> "BatchRetrievalResult":
    """
    Build a single merged RetrievalResult covering a batch of source lines.

    Merges glossary/name hits across all lines (deduped by id).
    Chapter TM is deliberately ignored so prompts contain only compact exact
    name/glossary constraints.

    Returns a BatchRetrievalResult with per-line hit lists for consistency
    checking plus a single merged constraint_block / tm_examples for the
    batch prompt.
    """
    per_line_g: List[List[GlossaryEntry]] = []
    per_line_n: List[List[NameEntry]]     = []
    all_g:      Dict[str, GlossaryEntry]  = {}
    all_n:      Dict[str, NameEntry]      = {}

    for kr in kr_texts:
        norm = normalize_for_match(kr)
        g = _exact_glossary_hits(norm, global_glossary, series_glossary)
        n = _exact_name_hits(norm, global_names, series_names)
        per_line_g.append(g)
        per_line_n.append(n)
        for e in g:
            all_g.setdefault(normalize_for_match(e.source_kr), e)
        for e in n:
            all_n.setdefault(normalize_for_match(e.kr_name), e)

    merged_g = sorted(all_g.values(), key=lambda e: (_TRUST_RANK.get(e.trust, 9), -len(e.source_kr)))
    merged_n = sorted(all_n.values(), key=lambda e: (_TRUST_RANK.get(e.trust, 9), -len(e.kr_name)))

    return BatchRetrievalResult(
        constraint_block  = _build_constraint_block(merged_g, merged_n),
        tm_examples       = "",
        glossary_hits     = merged_g,
        name_hits         = merged_n,
        per_line_glossary = per_line_g,
        per_line_names    = per_line_n,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _exact_glossary_hits(
    norm_input:      str,
    global_glossary: List[GlossaryEntry],
    series_glossary: List[GlossaryEntry],
) -> List[GlossaryEntry]:
    """
    Exact substring hits from series first, then global.
    Deduped by normalised source term so series entries override duplicates.
    Sorted: trust asc, length desc.
    """
    seen: set = set()
    hits: List[GlossaryEntry] = []
    for entry in (*series_glossary, *global_glossary):
        key = normalize_for_match(entry.source_kr)
        if not key or key in seen:
            continue
        if _glossary_matches(norm_input, entry):
            hits.append(entry)
            seen.add(key)
    hits.sort(key=lambda e: (_TRUST_RANK.get(e.trust, 9), -len(e.source_kr)))
    return hits


def _glossary_matches(norm_input: str, entry: GlossaryEntry) -> bool:
    norm_src = normalize_for_match(entry.source_kr)
    if norm_src and norm_src in norm_input:
        return True
    return any(
        normalize_for_match(a) in norm_input
        for a in entry.aliases_kr
        if a
    )


def _exact_name_hits(
    norm_input:   str,
    global_names: List[NameEntry],
    series_names: List[NameEntry],
) -> List[NameEntry]:
    seen: set = set()
    hits: List[NameEntry] = []
    for entry in (*series_names, *global_names):
        key = normalize_for_match(entry.kr_name)
        if not key or key in seen:
            continue
        if _name_matches(norm_input, entry):
            hits.append(entry)
            seen.add(key)
    hits.sort(key=lambda e: (_TRUST_RANK.get(e.trust, 9), -len(e.kr_name)))
    return hits


def _name_matches(norm_input: str, entry: NameEntry) -> bool:
    norm_name = normalize_for_match(entry.kr_name)
    if norm_name and norm_name in norm_input:
        return True
    return any(
        normalize_for_match(a) in norm_input
        for a in entry.aliases_kr
        if a
    )


def _tm_hits(
    kr_text:        str,
    norm_input:     str,
    chapter_tm:     List[ChapterTMEntry],
    blocked:        List[BlockedMappingEntry],
    cap:            int,
    fuzzy_threshold: float,
) -> List[ChapterTMEntry]:
    """
    Collect TM hits: exact matches first, then fuzzy, up to cap.

    Safety filters applied before any entry is considered:
    • Entry must have status="approved", flagged=False, promoted_to=None.
      (caller passes retrievable_entries() so this is already guaranteed,
      but we double-check as a defence-in-depth measure.)
    • Entry must not be suppressed by a blocked mapping.
    • Entry en_text must not be empty or very short (noise filter).
    • Entry text length must not exceed TM_LINE_MAX_CHARS (prompt bloat guard).
    """
    exact:  List[ChapterTMEntry] = []
    fuzzy:  List[ChapterTMEntry] = []
    seen:   set                  = set()

    for entry in chapter_tm:
        # Defence-in-depth guard (should already be filtered by caller)
        if entry.status != "approved" or entry.flagged or entry.promoted_to:
            continue
        if not entry.en_text.strip() or len(entry.en_text.strip()) < 2:
            continue
        total_len = len(entry.kr_text) + len(entry.en_text)
        if total_len > TM_LINE_MAX_CHARS:
            continue
        # Blocked-mapping suppression
        if _is_suppressed(entry.kr_text, entry.en_text, blocked):
            continue
        if entry.id in seen:
            continue

        norm_entry = normalize_for_match(entry.kr_text)
        if norm_entry == norm_input:
            exact.append(entry)
            seen.add(entry.id)
        else:
            fuzzy.append(entry)

    result = list(exact)

    # Fuzzy: rank by similarity ratio, apply threshold, fill up to cap
    if len(result) < cap and fuzzy:
        scored = _score_fuzzy(norm_input, fuzzy, fuzzy_threshold)
        for entry, _ in scored:
            if len(result) >= cap:
                break
            if entry.id not in seen:
                result.append(entry)
                seen.add(entry.id)

    return result[:cap]


def _is_suppressed(
    kr_text: str,
    en_text: str,
    blocked: List[BlockedMappingEntry],
) -> bool:
    """Return True if any blocked mapping fires on this (kr, en) pair."""
    norm_kr  = normalize_for_match(kr_text)
    en_lower = en_text.lower()
    return any(
        normalize_for_match(e.source_kr) in norm_kr
        and e.blocked_en.lower() in en_lower
        for e in blocked
    )


def _score_fuzzy(
    norm_input: str,
    candidates: List[ChapterTMEntry],
    threshold:  float,
) -> List[tuple]:
    """
    Score each candidate with SequenceMatcher ratio.
    Returns sorted list of (entry, ratio) above threshold.
    """
    scored = []
    matcher = difflib.SequenceMatcher(None, norm_input, "")
    for entry in candidates:
        norm_c = normalize_for_match(entry.kr_text)
        matcher.set_seq2(norm_c)
        ratio = matcher.ratio()
        if ratio >= threshold:
            scored.append((entry, ratio))
    scored.sort(key=lambda x: -x[1])
    return scored


# ── Prompt block builders ─────────────────────────────────────────────────────

def _build_constraint_block(
    g_hits: List[GlossaryEntry],
    n_hits: List[NameEntry],
) -> str:
    """
    Build the hard-constraint block.

    Names are listed before glossary terms so that character-specific rules
    (which carry more context) appear first.

    Returns "" when both lists are empty.
    """
    lines: List[str] = []
    for e in n_hits:
        lines.append(f'- Always translate "{e.kr_name}" as "{e.en_name}"')
    for e in g_hits:
        lines.append(f'- Always translate "{e.source_kr}" as "{e.target_en}"')
    if not lines:
        return ""
    return "[CONSTRAINTS]\n" + "\n".join(lines)


def _build_tm_block(tm_hits: List[ChapterTMEntry]) -> str:
    """
    Build the TM examples block.

    Clearly labelled as 'reference only' so the model understands these are
    examples, not rules to copy verbatim.

    Returns "" when tm_hits is empty.
    """
    if not tm_hits:
        return ""
    lines = ["[TRANSLATION EXAMPLES — reference only, not rules]"]
    for e in tm_hits:
        lines.append(f"KR: {e.kr_text.strip()}")
        lines.append(f"EN: {e.en_text.strip()}")
    return "\n".join(lines)


# ── Batch result ──────────────────────────────────────────────────────────────

class BatchRetrievalResult:
    """
    Result of retrieve_batch().

    Carries a single merged prompt-ready block for the batch plus
    per-line hit lists needed for post-translation consistency checks.
    """
    __slots__ = (
        "constraint_block",
        "tm_examples",
        "glossary_hits",
        "name_hits",
        "per_line_glossary",
        "per_line_names",
    )

    def __init__(
        self,
        constraint_block:  str,
        tm_examples:       str,
        glossary_hits:     List[GlossaryEntry],
        name_hits:         List[NameEntry],
        per_line_glossary: List[List[GlossaryEntry]],
        per_line_names:    List[List[NameEntry]],
    ) -> None:
        self.constraint_block  = constraint_block
        self.tm_examples       = tm_examples
        self.glossary_hits     = glossary_hits
        self.name_hits         = name_hits
        self.per_line_glossary = per_line_glossary
        self.per_line_names    = per_line_names
