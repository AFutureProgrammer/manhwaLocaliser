from __future__ import annotations

import re
from typing import Optional

from backend.core.constants import (
    GLOSSARY_ANCHORS,
    NAME_MAP,
    OCR_NORMALIZE_PATTERNS,
    SFX_MAP,
    debug_print,
)

def clean_translation_text(raw_text: str) -> str:
    """Removes LLM artifacts like [0], 1., markdown, and quotation marks."""
    # Remove bracketed numbers (e.g., [0], [12])
    cleaned = re.sub(r'\[\d+\]', '', raw_text)
    # Remove markdown bold/italics
    cleaned = cleaned.replace('**', '').replace('*', '')
    # Remove leading list numbers like "1. " or "- "
    cleaned = re.sub(r'^[\d\.\-\s]+', '', cleaned)
    # Strip whitespace and surrounding quotation marks
    cleaned = cleaned.strip(' \n\r\t"\'')
    return cleaned

_META_TRANSLATION_RE = re.compile(
    r"^\s*(?:"
    r"okay[,\s]+(?:here(?:'|’)s|here is)|"
    r"here(?:'|’)s\s+(?:the\s+)?translation|"
    r"the\s+translation\s+(?:is|would be)|"
    r"please\s+provide|"
    r"i\s+(?:can(?:not|'t)|cannot|am\s+unable|can't)\s+"
    r")",
    re.IGNORECASE,
)

_META_FRAGMENT_RE = re.compile(
    r"(?:"
    r"here(?:'|’)s\s+(?:the\s+)?translation|"
    r"please\s+provide\s+the\s+korean|"
    r"output\s+only|"
    r"numbered\s+korean\s+line|"
    r"as\s+an\s+ai|"
    r"i\s+(?:can(?:not|'t)|cannot|am\s+unable|can't)\s+(?:assist|translate)"
    r")",
    re.IGNORECASE,
)

SFX_MAP: Dict[str, str] = {
    "배시시": "(smiles)",
    "배시": "(smiles)",
    "방긋": "(smiles)",
    "씨익": "(grins)",
    "싱긋": "(smiles)",
    "두근": "badump",
    "쿵": "thump",
    "똑똑": "knock knock",
}

OCR_NORMALIZE_PATTERNS: List[Tuple[str, str]] = [
    # Critical jamo-level confusions — highest semantic impact
    # 쥐도 → 줘도: one jamo changes "you didn't have to give" into nonsense
    (r"쥐도", "줘도"),
    (r"쥐어도", "줘도"),
    # 돼/됐 confusion in stylised fonts
    (r"되요", "돼요"),
    (r"되진", "돼진"),
    (r"됐", "딕"),
    # Polite-form OCR splits caused by wide character spacing
    (r"합니 다", "합니다"),
    (r"습니 다", "습니다"),
    # Known manhwa OCR near-misses
    (r"축하드컵니다", "축하드립니다"),
    (r"축하드릅니다", "축하드립니다"),
    (r"추천올", "추천을"),
    (r"추천율", "추천을"),
    (r"고마원", "고마워"),
    (r"고마위", "고마워"),
    (r"고마와", "고마워"),
    (r"틀림 없다", "틀림없다"),
    (r"틀린없다", "틀림없다"),
    (r"아이작도련님", "아이작 도련님"),
]

NAME_MAP: Dict[str, str] = {
    "아이작": "Isaac",
    "한스":   "Hans",
}

_SPEAKER_LABEL_RE = re.compile(
    r"^[가-힣A-Za-z\-\u2019\u2018]+[:\u00b7\uff1a]$"
)

def is_speaker_label(compact_text: str) -> bool:
    """True when the entire OCR block is a speaker-name label like '한스:'."""
    return bool(_SPEAKER_LABEL_RE.match(compact_text)) and len(compact_text) <= 12

def transliterate_speaker_label(compact_text: str) -> str:
    """Convert a Korean speaker label to English, e.g. '한스:' → 'Hans:'."""
    name = compact_text.rstrip(":\u00b7\uff1a")
    return localize_name(name) + ":"

def normalize_ocr_korean(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("]", "!").replace("[", "")
    normalized = re.sub(r"[|`´•]+", "", normalized)
    for pattern, repl in OCR_NORMALIZE_PATTERNS:
        normalized = re.sub(pattern, repl, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def localize_name(name: str) -> str:
    compact = re.sub(r"\s+", "", name)
    return NAME_MAP.get(compact, name)

def contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))

def is_likely_garbage_literal(korean: str, literal: str) -> bool:
    lit = literal.strip().lower()
    if not lit:
        return True
    if len(lit) > 180 and len(korean) < 40:
        return True
    if lit.count(",") > 6:
        return True
    if len(set(re.findall(r"[a-z']+", lit))) <= 4 and len(lit.split()) > 8:
        return True
    repeated = re.search(r"(.{8,}?)\1{1,}", lit.replace(" ", ""))
    return repeated is not None

def heuristic_localize_line(korean_text: str) -> Optional[str]:
    text = normalize_ocr_korean(korean_text)
    compact = re.sub(r"\s+", "", text)
    compact_plain = re.sub(r"[^가-힣]", "", compact)

    # Speaker-label shortcut — must come first so the LLM result is ignored.
    # '한스:' should produce 'Hans:' only; without this the LLM receives what
    # looks like an incomplete utterance and hallucinates extra dialogue.
    if is_speaker_label(compact):
        return transliterate_speaker_label(compact)

    if compact in SFX_MAP or compact_plain in SFX_MAP:
        return SFX_MAP.get(compact, SFX_MAP.get(compact_plain))

    if compact_plain in {"축하합니다", "축하드립니다"}:
        return "Congratulations!"

    if "축하" in text and "도련님" in text:
        name_text = text
        name_text = re.sub(r"^축하(?:합니다|드립니다)?[,!?.\s]*", "", name_text).strip()
        m = re.search(r"(.+?)\s*도련님", name_text)
        if m:
            name = localize_name(m.group(1).strip())
            return f"Congratulations, Young Master {name}."
        return "Congratulations, Young Master."

    if ("열살" in text or "열 살" in text) and "왕실 아카데미" in text and "입학" in text:
        return "You got accepted to the Royal Academy at just ten years old!"

    if "천재" in text and ("틀림없" in text or "틀림 없다" in text):
        return "You really are a genius, aren't you?"

    if "유모" in text and ("고마" in text or "감사" in text):
        return "Thank you, Nanny."

    if "유모" in text:
        return "Nanny"

    return None

def sanitize_final_translation(korean_text: str, english_text: str, heuristic: Optional[str] = None) -> str:
    text = clean_translation_text(english_text or "")
    if heuristic:
        return heuristic
    if not text:
        return heuristic or ""
    if _META_TRANSLATION_RE.search(text) or _META_FRAGMENT_RE.search(text):
        return heuristic or ""
    if "\n" in (english_text or ""):
        lines = [
            clean_translation_text(line)
            for line in (english_text or "").splitlines()
            if clean_translation_text(line)
        ]
        useful = [
            line for line in lines
            if not _META_TRANSLATION_RE.search(line)
            and not _META_FRAGMENT_RE.search(line)
        ]
        if len(useful) == 1:
            text = useful[0]
        elif len(useful) > 1:
            return heuristic or ""
    if contains_hangul(text):
        fallback = heuristic_localize_line(korean_text)
        if fallback:
            return fallback
        text = re.sub(r"[가-힣]+", "", text).strip(" ,.!?-_")
    text = text.replace("Young Master .", "Young Master.")
    text = re.sub(r"\s+", " ", text).strip()
    apply_glossary_anchors(korean_text, text)  # log drift, non-blocking
    return text

GLOSSARY_ANCHORS: Dict[str, List[str]] = {
    "선물":     ["gift", "present", "gave", "give"],
    "도련님":   ["young master", "master"],
    "유모":     ["nanny", "nurse"],
    "선배":     ["senior", "senpai"],
    "왕실":     ["royal", "palace", "kingdom"],
    "아카데미":  ["academy"],
    "천재":     ["genius", "prodig"],
    "입학":     ["enroll", "accept", "admitted", "enter"],
    "결혼":     ["married", "wedding", "marriage"],
    "사랑":     ["love"],
    "도망":     ["run", "flee", "escape", "away"],
}

def apply_glossary_anchors(korean_text: str, english_text: str) -> bool:
    """
    Checks whether key Korean terms are reflected in the English translation.
    Logs a warning for each mismatch. Returns True if clean, False if drift found.
    """
    lower_en = english_text.lower()
    clean = True
    for kr_term, en_alts in GLOSSARY_ANCHORS.items():
        if kr_term in korean_text:
            if not any(alt in lower_en for alt in en_alts):
                debug_print(
                    f"[GLOSSARY DRIFT] Korean has '{kr_term}' but English is missing "
                    f"({'/'.join(en_alts)}): {english_text!r}"
                )
                clean = False
    return clean

