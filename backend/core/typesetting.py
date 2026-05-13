from __future__ import annotations

from PIL import ImageFont

def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    if not text:
        return 0
    if hasattr(font, "getlength"):
        return int(round(font.getlength(text)))
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0])

def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Greedy word wrap with explicit-newline preservation."""
    out: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            out.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _text_width(font, candidate) <= max_width:
                current = candidate
            else:
                out.append(current)
                current = word
        out.append(current)
    return out


