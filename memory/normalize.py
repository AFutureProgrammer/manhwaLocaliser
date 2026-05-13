"""
memory/normalize.py
───────────────────
Shared Korean text normalisation for memory lookups.

Intentionally kept separate from the normalisation in translator_v14.py
so this package has zero imports from the main codebase.

Portable: no dependencies on translator_v14.py or engine.py.
"""
from __future__ import annotations

import re


def normalize_for_match(text: str) -> str:
    """
    Aggressively normalise Korean text for substring-based memory lookup.

    Strategy: remove *all* whitespace so that OCR-introduced spaces inside
    a word (e.g. "한 스" → "한스", "도련 님" → "도련님") do not break a match
    against a clean dictionary key.

    This mirrors what translator_v14.normalize_ocr_korean() does at a coarser
    level, but is intentionally self-contained so the memory package remains
    portable to a future backend.

    Note
    ────
    We do NOT case-fold here because Korean is case-insensitive by nature.
    Case folding (.lower()) is applied separately on the *English* side
    inside the consistency checker.
    """
    return re.sub(r"\s+", "", text.strip())
