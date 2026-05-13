"""
backend/core/sources/__init__.py
─────────────────────────────────
Provider registry.

Usage::

    from backend.core.sources import get_provider, list_providers

    p = get_provider("naver-comic")
    cards = p.search_series("어떤")

To disable a provider entirely, set the corresponding env var, e.g.::

    MANHWA_NAVER_DISABLED=1

or simply delete its module file.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from backend.core.sources.base import SourceProvider

_REGISTRY: Dict[str, SourceProvider] = {}
_LOADED = False


def _load_providers() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    # ── Naver ──────────────────────────────────────────────────────────────
    if not os.environ.get("MANHWA_NAVER_DISABLED", "").strip():
        try:
            from backend.core.sources.naver import NaverComicProvider
            _REGISTRY["naver-comic"] = NaverComicProvider()
            print("[sources] naver-comic provider loaded")
        except Exception as exc:
            print(f"[sources] naver-comic provider failed to load: {exc}")

    # ── Future providers registered here ───────────────────────────────────


def get_provider(name: str) -> Optional[SourceProvider]:
    """Return the provider for *name*, or None if not registered."""
    _load_providers()
    return _REGISTRY.get(name)


def list_providers() -> list[str]:
    """Return a sorted list of registered provider names."""
    _load_providers()
    return sorted(_REGISTRY.keys())
