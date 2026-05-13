from __future__ import annotations

from typing import Dict, List, Tuple

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        import time
        ts = time.strftime("%H:%M:%S")
        print(f"[DEBUG {ts}]", *args, **kwargs, flush=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 180
MODEL_CONFIG_FILE = "model_config.json"
COMIC_FONTS_DIR = "C:\\Users\\zaina\\Downloads\\ManhwaTranslator\\comic_fonts"
NLLB_MODEL_DIR = "nllb-200-ct2"
KR_SFX_MAP = "{'배시시': '(smiling)', '똑똑': 'knock knock', '두근': 'badump', '씨익': '(grins)'}"

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
    (r"쥐도", "줘도"),
    (r"쥐어도", "줘도"),
    (r"되요", "돼요"),
    (r"되진", "돼진"),
    (r"됐", "딕"),
    (r"합니 다", "합니다"),
    (r"습니 다", "습니다"),
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
    "한스": "Hans",
}

GLOSSARY_ANCHORS: Dict[str, List[str]] = {
    "선물": ["gift", "present", "gave", "give"],
    "도련님": ["young master", "master"],
    "유모": ["nanny", "nurse"],
    "선배": ["senior", "senpai"],
    "왕실": ["royal", "palace", "kingdom"],
    "아카데미": ["academy"],
    "천재": ["genius", "prodig"],
    "입학": ["enroll", "accept", "admitted", "enter"],
    "결혼": ["married", "wedding", "marriage"],
    "사랑": ["love"],
    "도망": ["run", "flee", "escape", "away"],
}
