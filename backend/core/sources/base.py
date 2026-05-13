"""
backend/core/sources/base.py
────────────────────────────
Abstract base class for series source providers.

All provider implementations must inherit from SourceProvider and
implement the abstract methods below.  A provider that cannot fulfil a
request must raise NotImplementedError (for unimplemented operations)
or return an appropriate empty/error value.

Providers are stateless: they receive all context via method arguments
and return plain dicts/lists that are JSON-serialisable.

Import isolation
────────────────
Each provider file (naver.py, etc.) is imported lazily via get_provider()
so that heavyweight or optional dependencies do not break the main process
when a particular provider is unused.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SourceProvider(ABC):
    """
    Abstract base for all series source providers.

    All returned dicts use snake_case keys and contain only JSON-safe types.
    """

    # ── Identity ────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider slug, e.g. 'naver-comic'."""
        ...

    # ── URL / ID parsing ────────────────────────────────────────────────────

    def parse_series_ref(self, input_str: str) -> Dict[str, Any]:
        """
        Parse a user-supplied string (URL or numeric ID) into a canonical
        reference dict.

        Returns::

            {
                "source_id": str,        # canonical numeric/slug ID
                "source_url": str,       # canonical URL if derivable
                "ok": bool,
                "error": str,            # non-empty on failure
            }
        """
        return {"ok": False, "error": "parse_series_ref not implemented", "source_id": "", "source_url": ""}

    # ── Discovery ───────────────────────────────────────────────────────────

    def search_series(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the source for series matching *query*.

        Returns a list of card dicts::

            {
                "source_id": str,
                "title_ko":  str,
                "title_en":  str,        # "" if unknown
                "thumbnail_url": str,    # "" if unavailable
                "chapter_count": int,    # -1 if unknown
                "source_url": str,
            }

        An empty list is a valid "no results" response.
        Implementations that do not support live search should return [].
        """
        return []

    # ── Metadata ────────────────────────────────────────────────────────────

    def get_series_metadata(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch full series metadata.

        Returns::

            {
                "ok": bool,
                "error": str,
                "source_id": str,
                "title_ko": str,
                "title_en": str,
                "synopsis_ko": str,
                "synopsis_en": str,
                "thumbnail_url": str,
                "chapter_count": int,
                "source_url": str,
                "source_metadata": dict,   # provider-specific extras
            }
        """
        return {"ok": False, "error": "get_series_metadata not implemented"}

    # ── Chapter list ─────────────────────────────────────────────────────────

    def get_chapter_list(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Fetch the chapter index for the given series.

        Returns a list of chapter dicts::

            {
                "source_id":     str,
                "episode_no":    int,        # 1-based episode number
                "title_ko":      str,
                "title_en":      str,
                "source_url":    str,
                "thumbnail_url": str,
                "page_count":    int,        # -1 if unknown
                "folder":        str,        # local folder mapping or ""
            }
        """
        return []

    # ── Raw image sync ───────────────────────────────────────────────────────

    def sync_chapter_images(
        self,
        source_id: str,
        chapter_source_id: str,
        dest_folder: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Download / copy raw images for one chapter into *dest_folder*.

        Returns::

            {
                "ok": bool,
                "error": str,
                "pages_synced": int,
                "dest_folder": str,
            }
        """
        return {"ok": False, "error": "sync_chapter_images not implemented", "pages_synced": 0, "dest_folder": dest_folder}
