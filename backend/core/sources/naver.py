"""
backend/core/sources/naver.py
──────────────────────────────
Naver Comic source provider.

First-pass implementation: manifest-based only.
Live network fetch is NOT implemented in this pass.
A naver_manifest.json file placed alongside series_db.json (project root)
is the primary data source.

naver_manifest.json schema
──────────────────────────
{
  "series": [
    {
      "source_id":     "12345",
      "title_ko":      "웹툰 제목",
      "title_en":      "Webtoon Title",
      "synopsis_ko":   "시놉시스...",
      "synopsis_en":   "Synopsis...",
      "thumbnail_url": "https://...",       // optional
      "thumbnail_path":"./thumbs/12345.jpg",// optional local path
      "source_url":    "https://comic.naver.com/webtoon/list?titleId=12345",
      "chapters": [
        {
          "source_id":     "1",             // episode/chapter ID
          "episode_no":    1,
          "title_ko":      "1화",
          "title_en":      "Chapter 1",
          "source_url":    "https://...",
          "thumbnail_url": "",
          "thumbnail_path":"",
          "page_count":    -1,
          "folder":        "./raw/12345/001"  // optional local image folder
        }
      ]
    }
  ]
}

To disable the Naver provider entirely, set the environment variable::

    MANHWA_NAVER_DISABLED=1

or simply remove naver.py — the registry in __init__.py will gracefully
fall back to an empty provider list.
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import re
import shutil
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

from backend.core.sources.base import SourceProvider

# ── URL patterns ─────────────────────────────────────────────────────────────

_NAVER_TITLE_ID_RE = re.compile(r"titleId=(\d+)", re.IGNORECASE)
_NUMERIC_ID_RE = re.compile(r"^\d+$")

_NAVER_URL_TEMPLATE = "https://comic.naver.com/webtoon/list?titleId={title_id}"
_NAVER_CHAPTER_URL_TEMPLATE = "https://comic.naver.com/webtoon/detail?titleId={title_id}&no={episode_no}"
_NAVER_SEARCH_URL = "https://comic.naver.com/api/search/all"
_NAVER_INFO_URL = "https://comic.naver.com/api/article/list/info"
_NAVER_ARTICLE_LIST_URL = "https://comic.naver.com/api/article/list"
_NAVER_DETAIL_URL = "https://comic.naver.com/webtoon/detail"
_RAW_AUTH_ERROR = "Raw image sync requires manifest/local folder import or official provider configuration."
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ManhwaLocaliser/1.0",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Referer": "https://comic.naver.com/",
}


def _canonical_url(title_id: str) -> str:
    return _NAVER_URL_TEMPLATE.format(title_id=title_id)


def _canonical_chapter_url(title_id: str, episode_no: int) -> str:
    return _NAVER_CHAPTER_URL_TEMPLATE.format(title_id=title_id, episode_no=episode_no)


def _memory_key(title_id: str) -> str:
    return f"source:naver-comic:{title_id}"


def _chapter_memory_key(title_id: str, chapter_id: str) -> str:
    return f"source:naver-comic:{title_id}:{chapter_id}"


def _memory_fs_key(value: str) -> str:
    slug = str(value or "").lower().replace(":", "_")
    slug = re.sub(r"[^a-z0-9_\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug or "unnamed_series"


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.get(url, params=params or {}, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _http_get_text(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    resp = requests.get(url, params=params or {}, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _is_sample_entry(entry: Dict[str, Any]) -> bool:
    return str(entry.get("source_id") or "") == "12345" and "예시" in str(entry.get("title_ko") or "")


def _card_from_manifest(entry: Dict[str, Any]) -> Dict[str, Any]:
    source_id = str(entry.get("source_id") or "")
    chapters = entry.get("chapters") or []
    card = {
        "source": "naver-comic",
        "source_id": source_id,
        "title_ko": str(entry.get("title_ko") or ""),
        "title_en": str(entry.get("title_en") or ""),
        "thumbnail_url": str(entry.get("thumbnail_url") or ""),
        "thumbnail_path": str(entry.get("thumbnail_path") or ""),
        "chapter_count": len(chapters),
        "source_url": str(entry.get("source_url") or _canonical_url(source_id)),
        "memory_key": _memory_key(source_id),
        "memory_fs_key": _memory_fs_key(_memory_key(source_id)),
        "sync_status": "sample_manifest" if _is_sample_entry(entry) else "manifest",
        "source_metadata": {k: v for k, v in entry.items() if k != "chapters"},
    }
    return card


def _card_from_live(item: Dict[str, Any]) -> Dict[str, Any]:
    source_id = str(item.get("titleId") or item.get("source_id") or "")
    return {
        "source": "naver-comic",
        "source_id": source_id,
        "title_ko": str(item.get("titleName") or item.get("title_ko") or ""),
        "title_en": str(item.get("title_en") or ""),
        "thumbnail_url": str(item.get("thumbnailUrl") or item.get("thumbnail_url") or ""),
        "thumbnail_path": "",
        "chapter_count": int(item.get("articleTotalCount") or item.get("chapter_count") or -1),
        "source_url": _canonical_url(source_id),
        "memory_key": _memory_key(source_id),
        "memory_fs_key": _memory_fs_key(_memory_key(source_id)),
        "sync_status": "live_public",
        "source_metadata": item,
    }


# ── Manifest loader ───────────────────────────────────────────────────────────

def _find_manifest() -> Optional[pathlib.Path]:
    """Search common locations for naver_manifest.json."""
    candidates = [
        pathlib.Path.cwd() / "naver_manifest.json",
        pathlib.Path(__file__).resolve().parent.parent.parent.parent / "naver_manifest.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_manifest() -> List[Dict[str, Any]]:
    path = _find_manifest()
    if path is None:
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("series", []) if isinstance(data, dict) else []
    except Exception as exc:
        print(f"[naver] manifest load error: {exc}")
        return []


# ── Provider class ────────────────────────────────────────────────────────────

class NaverComicProvider(SourceProvider):
    """
    Naver Comic source provider — manifest-based first pass.

    Live network fetch is stubbed; all data comes from naver_manifest.json.
    """

    @property
    def name(self) -> str:
        return "naver-comic"

    def _mode(self) -> str:
        mode = os.environ.get("MANHWA_NAVER_MODE", "").strip().lower()
        return mode if mode in {"manifest", "live_public", "disabled"} else ""

    def _manifest_entry(self, source_id: str) -> Optional[Dict[str, Any]]:
        for entry in _load_manifest():
            if str(entry.get("source_id") or "") == str(source_id):
                return entry
        return None

    # ── URL / ID parsing ──────────────────────────────────────────────────────

    def parse_series_ref(self, input_str: str) -> Dict[str, Any]:
        """
        Accept:
          - numeric ID: "12345"
          - comic.naver.com/webtoon/list?titleId=12345
          - m.comic.naver.com/webtoon/list?titleId=12345
          - any URL containing titleId=
        """
        s = str(input_str or "").strip()
        if not s:
            return {"ok": False, "error": "empty input", "source_id": "", "source_url": ""}

        # Numeric ID
        if _NUMERIC_ID_RE.match(s):
            return {
                "ok": True,
                "error": "",
                "source": self.name,
                "source_id": s,
                "source_url": _canonical_url(s),
                "memory_key": _memory_key(s),
                "memory_fs_key": _memory_fs_key(_memory_key(s)),
            }

        # URL with titleId=
        m = _NAVER_TITLE_ID_RE.search(s)
        if m:
            title_id = m.group(1)
            return {
                "ok": True,
                "error": "",
                "source": self.name,
                "source_id": title_id,
                "source_url": _canonical_url(title_id),
                "memory_key": _memory_key(title_id),
                "memory_fs_key": _memory_fs_key(_memory_key(title_id)),
            }

        return {
            "ok": False,
            "error": f"Cannot parse Naver series ref: {s!r}",
            "source_id": "",
            "source_url": "",
        }

    # ── Discovery ─────────────────────────────────────────────────────────────

    def search_series(self, query: str) -> List[Dict[str, Any]]:
        """
        Search manifest entries first, then Naver's public search when needed.
        Numeric titleIds and URLs are resolved directly.
        """
        if self._mode() == "disabled":
            return []
        raw_query = str(query or "").strip()
        live_requested = raw_query.lower().startswith("live:")
        if live_requested:
            raw_query = raw_query[5:].strip()

        parsed = self.parse_series_ref(raw_query)
        if parsed.get("ok"):
            meta = self.get_series_metadata(str(parsed["source_id"]))
            if meta.get("ok"):
                return [_card_from_live({
                    "titleId": meta.get("source_id"),
                    "titleName": meta.get("title_ko"),
                    "thumbnailUrl": meta.get("thumbnail_url"),
                    "articleTotalCount": meta.get("chapter_count", -1),
                    "source_metadata": meta.get("source_metadata", {}),
                })]
            return [{
                "source": self.name,
                "source_id": str(parsed["source_id"]),
                "title_ko": "",
                "title_en": "",
                "thumbnail_url": "",
                "thumbnail_path": "",
                "chapter_count": -1,
                "source_url": str(parsed["source_url"]),
                "memory_key": parsed.get("memory_key", ""),
                "memory_fs_key": parsed.get("memory_fs_key", ""),
                "sync_status": "id_only",
                "source_metadata": {"error": meta.get("error", "")},
            }]

        query_lc = raw_query.lower()
        results: List[Dict[str, Any]] = []
        if self._mode() != "live_public":
            for entry in _load_manifest():
                ko = str(entry.get("title_ko") or "")
                en = str(entry.get("title_en") or "")
                if query_lc and query_lc not in ko.lower() and query_lc not in en.lower():
                    continue
                results.append(_card_from_manifest(entry))
        if results and not live_requested:
            return results
        if self._mode() == "manifest" or not raw_query:
            return results

        try:
            data = _http_get_json(_NAVER_SEARCH_URL, {"keyword": raw_query})
            live_items = (data.get("searchWebtoonResult") or {}).get("searchViewList") or []
            seen = {r["source_id"] for r in results}
            for item in live_items:
                card = _card_from_live(item)
                if card["source_id"] and card["source_id"] not in seen:
                    results.append(card)
                    seen.add(card["source_id"])
        except Exception as exc:
            print(f"[naver] live search failed: {exc}")
        return results

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_series_metadata(self, source_id: str) -> Dict[str, Any]:
        source_id = str(source_id)
        entry = self._manifest_entry(source_id)
        if entry and self._mode() != "live_public":
            chapters = entry.get("chapters") or []
            return {
                "ok": True,
                "error": "",
                "source": self.name,
                "source_id": str(entry.get("source_id") or source_id),
                "title_ko": str(entry.get("title_ko") or ""),
                "title_en": str(entry.get("title_en") or ""),
                "synopsis_ko": str(entry.get("synopsis_ko") or ""),
                "synopsis_en": str(entry.get("synopsis_en") or ""),
                "thumbnail_url": str(entry.get("thumbnail_url") or ""),
                "thumbnail_path": str(entry.get("thumbnail_path") or ""),
                "source_url": str(entry.get("source_url") or _canonical_url(source_id)),
                "chapter_count": len(chapters),
                "memory_key": _memory_key(source_id),
                "memory_fs_key": _memory_fs_key(_memory_key(source_id)),
                "sync_status": "sample_manifest" if _is_sample_entry(entry) else "manifest",
                "source_metadata": {k: v for k, v in entry.items() if k != "chapters"},
            }
        if self._mode() != "manifest":
            try:
                data = _http_get_json(_NAVER_INFO_URL, {"titleId": source_id})
                title = str(data.get("titleName") or "")
                return {
                    "ok": True,
                    "error": "",
                    "source": self.name,
                    "source_id": source_id,
                    "title_ko": title,
                    "title_en": "",
                    "synopsis_ko": str(data.get("synopsis") or ""),
                    "synopsis_en": "",
                    "thumbnail_url": str(data.get("thumbnailUrl") or data.get("posterThumbnailUrl") or ""),
                    "thumbnail_path": "",
                    "source_url": _canonical_url(source_id),
                    "chapter_count": int(data.get("totalCount") or data.get("articleTotalCount") or -1),
                    "memory_key": _memory_key(source_id),
                    "memory_fs_key": _memory_fs_key(_memory_key(source_id)),
                    "sync_status": "live_public",
                    "source_metadata": data,
                }
            except Exception as exc:
                print(f"[naver] live metadata failed: {exc}")
        if entry:
            chapters = entry.get("chapters") or []
            return {
                "ok": True,
                "error": "",
                "source": self.name,
                "source_id": source_id,
                "title_ko": str(entry.get("title_ko") or ""),
                "title_en": str(entry.get("title_en") or ""),
                "synopsis_ko": str(entry.get("synopsis_ko") or ""),
                "synopsis_en": str(entry.get("synopsis_en") or ""),
                "thumbnail_url": str(entry.get("thumbnail_url") or ""),
                "thumbnail_path": str(entry.get("thumbnail_path") or ""),
                "source_url": str(entry.get("source_url") or _canonical_url(source_id)),
                "chapter_count": len(chapters),
                "memory_key": _memory_key(source_id),
                "memory_fs_key": _memory_fs_key(_memory_key(source_id)),
                "sync_status": "manifest",
                "source_metadata": {k: v for k, v in entry.items() if k != "chapters"},
            }
        return {
            "ok":          False,
            "error":       f"Naver metadata unavailable for titleId {source_id!r}. Try naver_manifest.json or official provider configuration.",
            "source":      self.name,
            "source_id":   source_id,
            "source_url":  _canonical_url(source_id),
            "memory_key":  _memory_key(source_id),
            "memory_fs_key": _memory_fs_key(_memory_key(source_id)),
            "title_ko":    "",
            "title_en":    "",
            "synopsis_ko": "",
            "synopsis_en": "",
        }

    # ── Chapter list ──────────────────────────────────────────────────────────

    def get_chapter_list(self, source_id: str) -> List[Dict[str, Any]]:
        source_id = str(source_id)
        entry = self._manifest_entry(source_id)
        if entry and self._mode() != "live_public":
            return self._chapter_list_from_manifest(entry, source_id)
        if self._mode() != "manifest":
            try:
                out: List[Dict[str, Any]] = []
                seen_chapters = set()
                page = 1
                total = None
                while True:
                    data = _http_get_json(_NAVER_ARTICLE_LIST_URL, {"titleId": source_id, "page": page})
                    items = data.get("articleList") or []
                    if total is None:
                        total = int(data.get("totalCount") or len(items) or 0)
                    if not items:
                        break
                    for item in items:
                        ch_src_id = str(item.get("no") or item.get("volumeNo") or "")
                        if ch_src_id in seen_chapters:
                            continue
                        seen_chapters.add(ch_src_id)
                        ep = int(item.get("volumeNo") or item.get("no") or 0)
                        out.append({
                            "source": self.name,
                            "source_id": ch_src_id,
                            "episode_no": ep,
                            "title_ko": str(item.get("subtitle") or ""),
                            "title_en": "",
                            "source_url": _canonical_chapter_url(source_id, int(item.get("no") or ep or 0)),
                            "thumbnail_url": str(item.get("thumbnailUrl") or ""),
                            "thumbnail_path": "",
                            "page_count": -1,
                            "folder": "",
                            "indexed": True,
                            "imported": False,
                            "missing_raw": True,
                            "needs_sync": True,
                            "chapter_memory_key": _chapter_memory_key(source_id, ch_src_id or f"episode-{ep}"),
                            "chapter_memory_fs_key": _memory_fs_key(_chapter_memory_key(source_id, ch_src_id or f"episode-{ep}")),
                        })
                    if len(out) >= total or page >= 20:
                        break
                    page += 1
                if out:
                    out.sort(key=lambda ch: int(ch.get("episode_no") or 0))
                    return out
            except Exception as exc:
                print(f"[naver] live chapter list failed: {exc}")
        if entry:
            return self._chapter_list_from_manifest(entry, source_id)
        return []

    def _chapter_list_from_manifest(self, entry: Dict[str, Any], source_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        title_id = str(source_id)
        for ch in entry.get("chapters") or []:
            ep = int(ch.get("episode_no") or 0)
            ch_src_id = str(ch.get("source_id") or ep)
            ch_url = str(ch.get("source_url") or "")
            if not ch_url and ep:
                ch_url = _canonical_chapter_url(title_id, ep)
            cmk = _chapter_memory_key(title_id, ch_src_id or f"episode-{ep}")
            out.append({
                "source": self.name,
                "source_id": ch_src_id,
                "episode_no": ep,
                "title_ko": str(ch.get("title_ko") or ""),
                "title_en": str(ch.get("title_en") or ""),
                "source_url": ch_url,
                "thumbnail_url": str(ch.get("thumbnail_url") or ""),
                "thumbnail_path": str(ch.get("thumbnail_path") or ""),
                "page_count": int(ch.get("page_count") or -1),
                "folder": str(ch.get("folder") or ""),
                "indexed": True,
                "imported": False,
                "missing_raw": True,
                "needs_sync": True,
                "chapter_memory_key": cmk,
                "chapter_memory_fs_key": _memory_fs_key(cmk),
            })
        return out

    # ── Raw image sync ────────────────────────────────────────────────────────

    def sync_chapter_images(
        self,
        source_id: str,
        chapter_source_id: str,
        dest_folder: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Copy manifest/local images first. If no local mapping exists, try
        public/free image URLs from the Naver detail page. Auth-only content
        returns a clear configuration error.
        """
        chapter_folder = ""
        for entry in _load_manifest():
            if str(entry.get("source_id") or "") == str(source_id):
                for ch in entry.get("chapters") or []:
                    if str(ch.get("source_id") or ch.get("episode_no") or "") == str(chapter_source_id):
                        chapter_folder = str(ch.get("folder") or "")
                        break

        if chapter_folder:
            return self._copy_manifest_images(chapter_folder, dest_folder, overwrite)
        if self._mode() == "manifest":
            return {"ok": False, "error": _RAW_AUTH_ERROR, "pages_synced": 0, "dest_folder": dest_folder}
        try:
            return self._sync_public_detail_images(str(source_id), str(chapter_source_id), dest_folder, overwrite)
        except Exception as exc:
            print(f"[naver] live raw sync failed: {exc}")
            return {"ok": False, "error": _RAW_AUTH_ERROR, "pages_synced": 0, "dest_folder": dest_folder}

    def _copy_manifest_images(self, chapter_folder: str, dest_folder: str, overwrite: bool) -> Dict[str, Any]:
        src = pathlib.Path(chapter_folder)
        if not src.is_dir():
            return {"ok": False, "error": f"Manifest folder does not exist: {chapter_folder!r}", "pages_synced": 0, "dest_folder": dest_folder}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        images = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in image_exts)
        if not images:
            return {"ok": False, "error": f"No image files found in {chapter_folder!r}", "pages_synced": 0, "dest_folder": dest_folder}
        dest = pathlib.Path(dest_folder)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img in images:
            target = dest / img.name
            if target.exists() and not overwrite:
                copied += 1
                continue
            shutil.copy2(str(img), str(target))
            copied += 1
        print(f"[naver] manifest image sync: {copied} images -> {dest_folder!r}")
        return {"ok": True, "error": "", "pages_synced": copied, "dest_folder": str(dest)}

    def _sync_public_detail_images(self, source_id: str, chapter_source_id: str, dest_folder: str, overwrite: bool) -> Dict[str, Any]:
        text = _http_get_text(_NAVER_DETAIL_URL, {"titleId": source_id, "no": chapter_source_id})
        pattern = rf"https://image-comic\.pstatic\.net/webtoon/{re.escape(source_id)}/{re.escape(chapter_source_id)}/[^\"'<> ]+"
        urls = []
        seen = set()
        for match in re.findall(pattern, text):
            url = html.unescape(match)
            if "thumbnail" in url.lower():
                continue
            if url not in seen:
                urls.append(url)
                seen.add(url)
        if not urls:
            return {"ok": False, "error": _RAW_AUTH_ERROR, "pages_synced": 0, "dest_folder": dest_folder}
        dest = pathlib.Path(dest_folder)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for idx, url in enumerate(urls, start=1):
            suffix = pathlib.Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
            target = dest / f"{idx:04d}{suffix}"
            if target.exists() and not overwrite:
                copied += 1
                continue
            referer = f"https://comic.naver.com/webtoon/detail?titleId={source_id}&no={chapter_source_id}"
            resp = requests.get(url, headers={**_HEADERS, "Referer": referer}, timeout=30)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "image" not in ctype.lower():
                return {"ok": False, "error": _RAW_AUTH_ERROR, "pages_synced": copied, "dest_folder": str(dest)}
            with open(target, "wb") as f:
                f.write(resp.content)
            copied += 1
        print(f"[naver] public image sync: {copied} images -> {dest_folder!r}")
        return {"ok": True, "error": "", "pages_synced": copied, "dest_folder": str(dest)}
