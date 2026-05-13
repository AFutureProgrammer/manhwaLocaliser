"""
memory/consistency.py
─────────────────────
Post-translation consistency checks.

All functions here are pure Python — no Ollama calls, no model inference.
They run *after* the model has returned a translation and check whether the
output respects the constraints encoded in the memory stores.

Checks
──────
• check_name_drift()      — did the expected English name appear in the output?
• check_glossary_drift()  — did the required English term appear in the output?
• check_blocked_output()  — did a forbidden English term appear in the output?

Portable: no dependencies on translator_v14.py or engine.py.
"""
from __future__ import annotations

from typing import List

from .models import (
    BlockedMappingEntry,
    ConsistencyWarning,
    GlossaryEntry,
    NameEntry,
)


def check_name_drift(
    kr_text:    str,
    en_text:    str,
    name_hits:  List[NameEntry],
    page_idx:   int,
    region_idx: int,
) -> List[ConsistencyWarning]:
    """
    For each NameEntry that matched the Korean source, verify that the expected
    English name appears (case-insensitively) in the English output.

    A warning is emitted per violated entry, not per line.  If the same line
    contains two character names and both are missing, two warnings are returned.

    Parameters
    ----------
    kr_text    : the Korean source text for this region
    en_text    : the model's English translation for this region
    name_hits  : the NameEntry objects returned by NameMemory.exact_match(kr_text)
    page_idx   : 0-based page index (for tracing)
    region_idx : 0-based region index on the page (for tracing)

    Returns
    -------
    List[ConsistencyWarning]
        Empty list when all expected names are present.
    """
    warnings: List[ConsistencyWarning] = []
    en_lower = en_text.lower()

    for entry in name_hits:
        if entry.en_name.lower() not in en_lower:
            warnings.append(
                ConsistencyWarning(
                    severity        = "critical",
                    warning_type    = "name_drift",
                    page_idx        = page_idx,
                    region_idx      = region_idx,
                    kr_text         = kr_text,
                    en_text         = en_text,
                    expected        = entry.en_name,
                    memory_source   = "name",
                    memory_id       = entry.id,
                    trust_of_source = entry.trust,
                )
            )
    return warnings


def check_glossary_drift(
    kr_text:       str,
    en_text:       str,
    glossary_hits: List[GlossaryEntry],
    page_idx:      int,
    region_idx:    int,
) -> List[ConsistencyWarning]:
    """
    For each GlossaryEntry that matched the Korean source, verify that
    ``target_en`` *or* any ``alternatives_en`` appear in the English output.

    All alternatives are acceptable so that an entry like::

        target_en       = "Young Master"
        alternatives_en = ["master"]

    passes when the model outputs either "Young Master" or "master".

    Parameters
    ----------
    kr_text       : the Korean source text
    en_text       : the model's English translation
    glossary_hits : GlossaryEntry objects from GlossaryStore.exact_match(kr_text)
    page_idx      : 0-based page index
    region_idx    : 0-based region index

    Returns
    -------
    List[ConsistencyWarning]
        Empty list when all expected terms are present.
    """
    warnings: List[ConsistencyWarning] = []
    en_lower = en_text.lower()

    for entry in glossary_hits:
        acceptable = [entry.target_en.lower()] + [
            alt.lower() for alt in entry.alternatives_en
        ]
        if not any(term in en_lower for term in acceptable if term):
            warnings.append(
                ConsistencyWarning(
                    severity        = "critical",
                    warning_type    = "glossary_drift",
                    page_idx        = page_idx,
                    region_idx      = region_idx,
                    kr_text         = kr_text,
                    en_text         = en_text,
                    expected        = entry.target_en,
                    memory_source   = "glossary",
                    memory_id       = entry.id,
                    trust_of_source = entry.trust,
                )
            )
    return warnings


def check_blocked_output(
    kr_text:      str,
    en_text:      str,
    blocked_hits: List[BlockedMappingEntry],
    page_idx:     int,
    region_idx:   int,
) -> List[ConsistencyWarning]:
    """
    For each BlockedMappingEntry whose source_kr matched the Korean input,
    verify that ``blocked_en`` does NOT appear in the English output.

    A warning is emitted for each violated rule.  The warning severity is
    "critical" because a blocked mapping is an explicit human rejection —
    the engine's UI layer renders these as errors ("err"), not warnings.

    Parameters
    ----------
    kr_text      : the Korean source text for this region
    en_text      : the model's English translation for this region
    blocked_hits : BlockedMappingEntry objects that fired for this (kr, en) pair.
                   The caller (engine._post_translate) supplies these from
                   BlockedMappingStore.matches(kr_text, en_text) — i.e. only
                   entries where BOTH source_kr and blocked_en match.
    page_idx     : 0-based page index (for tracing)
    region_idx   : 0-based region index on the page (for tracing)

    Returns
    -------
    List[ConsistencyWarning]
        Empty list when no blocked mappings are violated.
    """
    warnings: List[ConsistencyWarning] = []
    en_lower = en_text.lower()

    for entry in blocked_hits:
        # Double-check: blocked_en must actually appear in the output.
        # (Caller already filters via matches(), but this is defence-in-depth.)
        if entry.blocked_en.lower() in en_lower:
            warnings.append(
                ConsistencyWarning(
                    severity        = "critical",
                    warning_type    = "blocked_output",
                    page_idx        = page_idx,
                    region_idx      = region_idx,
                    kr_text         = kr_text,
                    en_text         = en_text,
                    # expected = the term that was forbidden but appeared
                    expected        = entry.blocked_en,
                    memory_source   = "blocked",
                    memory_id       = entry.id,
                    trust_of_source = "manual",
                )
            )
    return warnings
