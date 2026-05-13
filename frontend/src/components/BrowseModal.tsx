// frontend/src/components/BrowseModal.tsx
// ─────────────────────────────────────────────────────────────────────────────
// "Browse series" modal — search provider cards and select one.
// Import and render in App.tsx when browsing is triggered.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useEffect, useRef, useCallback } from "react";
import api from "../api";
import type { BrowseCard, SeriesSummary } from "../types";

interface Props {
  sources: string[];
  onSelect: (seriesTitle: string, source: string, sourceId: string, card: BrowseCard) => void;
  onClose: () => void;
}

// ── ThumbnailCell ─────────────────────────────────────────────────────────────
// Renders a browse card thumbnail.  Priority:
//   1. thumbnail_url directly (works when no hotlink protection)
//   2. Backend-proxied b64 (bypasses Naver Referer requirement)
//   3. thumbnail_path via backend b64 proxy (local file)
//   4. "No image" text fallback

const ThumbnailCell: React.FC<{ card: BrowseCard }> = ({ card }) => {
  // If only a local path (no remote URL) go straight to proxy
  const directUrl = card.thumbnail_url || "";
  const localPath = card.thumbnail_path || "";

  const [src, setSrc] = useState<string>(directUrl);
  const [failed, setFailed] = useState(false);
  const [proxying, setProxying] = useState(false);
  const fetchedRef = useRef(false);

  // Reset when card changes
  useEffect(() => {
    const initial = card.thumbnail_url || "";
    setSrc(initial);
    setFailed(false);
    setProxying(false);
    fetchedRef.current = false;
    // If there's no direct URL but there is a local path, proxy immediately
    if (!initial && localPath) {
      fetchProxy(initial, localPath);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [card.thumbnail_url, card.thumbnail_path]);

  const fetchProxy = useCallback(async (url: string, path: string) => {
    if (fetchedRef.current || proxying) return;
    if (!url && !path) { setFailed(true); return; }
    fetchedRef.current = true;
    setProxying(true);
    try {
      const res = await api.getThumbnailB64(url, path);
      if (res.ok && (res as { b64?: string }).b64) {
        setSrc((res as { b64: string }).b64);
        setFailed(false);
      } else {
        setFailed(true);
      }
    } catch {
      setFailed(true);
    } finally {
      setProxying(false);
    }
  }, [proxying]);

  const handleError = useCallback(() => {
    // Direct load failed — try backend proxy
    fetchProxy(directUrl, localPath);
  }, [directUrl, localPath, fetchProxy]);

  if (failed || (!src && !proxying)) {
    return <div style={thumbPlaceholder}>No image</div>;
  }
  if (proxying && !src) {
    return <div style={{ ...thumbPlaceholder, fontSize: 12, color: "#666" }}>…</div>;
  }
  return (
    <img
      src={src}
      alt=""
      style={thumb}
      onError={handleError}
      loading="lazy"
    />
  );
};

export const BrowseModal: React.FC<Props> = ({ sources, onSelect, onClose }) => {
  const [source, setSource] = useState<string>(sources[0] ?? "naver-comic");
  const [query, setQuery] = useState("");
  const [cards, setCards] = useState<BrowseCard[]>([]);
  const [syncedCards, setSyncedCards] = useState<BrowseCard[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [savingId, setSavingId] = useState<string | null>(null);
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.getSeriesList().then((res) => {
      if (cancelled || !res.ok) return;
      const synced = (res.series ?? [])
        .filter((s: SeriesSummary) => s.source && s.source !== "local" && s.source_id)
        .map((s: SeriesSummary): BrowseCard => ({
          source: s.source ?? "naver-comic",
          source_id: s.source_id ?? "",
          memory_key: s.memory_key,
          memory_fs_key: s.memory_fs_key,
          sync_status: s.sync_status || "synced",
          title_ko: s.title_ko || s.title || "",
          title_en: s.title_en || s.title || "",
          thumbnail_url: s.thumbnail_url || s.thumbnail_path || "",
          thumbnail_path: s.thumbnail_path,
          chapter_count: s.chapter_count ?? 0,
          source_url: s.source_url || "",
        }));
      setSyncedCards(synced);
      setSavedIds(new Set(synced.map((s) => s.source_id)));
    });
    return () => { cancelled = true; };
  }, []);

  const doSearch = async () => {
    if (!query.trim()) {
      setCards([]);
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await api.browseSourceSeries(source, query);
      if (res.ok) {
        setCards(res.cards);
      } else {
        setError(res.error ?? "Search failed.");
        setCards([]);
      }
    } catch (e) {
      setError(String(e));
      setCards([]);
    } finally {
      setLoading(false);
    }
  };

  const showingSynced = !query.trim() && !loading && !error;
  const displayCards = showingSynced ? syncedCards : cards;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") doSearch();
    if (e.key === "Escape") onClose();
  };

  const handleSelect = async (card: BrowseCard) => {
    if (savingId === card.source_id) return;

    // Derive a local series title (prefer English, fall back to Korean)
    const seriesTitle = (card.title_en || card.title_ko || card.source_id).trim();

    setSavingId(card.source_id);
    try {
      const res = await api.selectBrowseSeries(seriesTitle, card.source, card.source_id, card);
      if (res.ok) {
        setSavedIds((prev) => new Set(prev).add(card.source_id));
        onSelect(seriesTitle, card.source, card.source_id, card);
      } else {
        alert(`Failed to save series: ${res.error}`);
      }
    } catch (e) {
      alert(`Error: ${e}`);
    } finally {
      setSavingId(null);
    }
  };

  return (
    <div
      style={overlay}
      onClick={(e) => e.target === e.currentTarget && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div style={modal}>
        {/* Header */}
        <div style={header}>
          <span style={{ fontWeight: 700, fontSize: 16 }}>Browse Naver</span>
          <button style={closeBtn} onClick={onClose} title="Close">x</button>
        </div>

        {/* Controls */}
        <div style={controls}>
          <select
            value={source}
            onChange={(e) => setSource(e.target.value)}
            style={select}
          >
            {sources.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <input
            ref={inputRef}
            style={searchInput}
            placeholder="Search Naver by title, titleId, or URL"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button style={searchBtn} onClick={doSearch} disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {error && <div style={errorBox}>{error}</div>}
        {!loading && displayCards.length === 0 && !error && !query.trim() && (
          <div style={empty}>Search Naver by title, titleId, or URL.</div>
        )}

        {/* Results */}
        {!loading && cards.length === 0 && !error && query.trim() && (
          <div style={empty}>
            No results found. Try a Naver titleId/URL or configure <code>naver_manifest.json</code>.
          </div>
        )}

        {showingSynced && displayCards.length > 0 && (
          <div style={subhead}>Synced source series</div>
        )}

        <div style={grid}>
          {displayCards.map((card) => {
            const isSaving = savingId === card.source_id;
            const isSaved = savedIds.has(card.source_id);
            return (
              <div
                key={card.source_id}
                style={{ ...cardStyle, ...(isSaved ? savedCard : {}) }}
                onClick={() => !isSaving && handleSelect(card)}
                title={isSaved ? "Saved — click to re-select" : "Click to save this series"}
              >
                <ThumbnailCell card={card} />
                <div style={cardBody}>
                  <div style={cardTitle}>{card.title_ko || card.title_en || card.source_id}</div>
                  {card.title_ko && card.title_en && (
                    <div style={cardSub}>{card.title_en}</div>
                  )}
                  <div style={cardMeta}>
                    {card.source} · titleId {card.source_id}
                    {card.chapter_count >= 0 ? ` · ${card.chapter_count} ch` : ""}
                  </div>
                  {card.sync_status === "sample_manifest" && (
                    <div style={sampleBadge}>Sample manifest entry</div>
                  )}
                  <div style={cardActions}>
                    <button style={cardBtn} onClick={(e) => { e.stopPropagation(); handleSelect(card); }} disabled={isSaving}>
                    {isSaving ? "Saving..." : isSaved ? "Open / refresh index" : "Select / Add series"}
                    </button>
                    {card.source_url && (
                      <a style={sourceLink} href={card.source_url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                        View source
                      </a>
                    )}
                  </div>
                  {isSaved && !isSaving && <div style={savedBadge}>Saved</div>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────────

const overlay: React.CSSProperties = {
  position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center",
  zIndex: 1000,
};
const modal: React.CSSProperties = {
  background: "#1e1e2e", borderRadius: 12, padding: 24,
  width: "min(860px, 95vw)", maxHeight: "80vh",
  display: "flex", flexDirection: "column", gap: 12,
  boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
};
const header: React.CSSProperties = {
  display: "flex", alignItems: "center", justifyContent: "space-between",
};
const closeBtn: React.CSSProperties = {
  background: "none", border: "none", color: "#aaa",
  fontSize: 18, cursor: "pointer", padding: "2px 6px",
};
const controls: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center" };
const select: React.CSSProperties = {
  background: "#2a2a3e", border: "1px solid #444", borderRadius: 6,
  color: "#ddd", padding: "6px 10px", fontSize: 13,
};
const searchInput: React.CSSProperties = {
  flex: 1, background: "#2a2a3e", border: "1px solid #444", borderRadius: 6,
  color: "#ddd", padding: "6px 10px", fontSize: 13,
};
const searchBtn: React.CSSProperties = {
  background: "#6c63ff", border: "none", borderRadius: 6,
  color: "#fff", padding: "6px 16px", cursor: "pointer", fontSize: 13,
};
const errorBox: React.CSSProperties = {
  background: "#3a1a1a", color: "#f88", borderRadius: 6, padding: "8px 12px", fontSize: 13,
};
const empty: React.CSSProperties = { color: "#777", fontSize: 13, padding: "12px 0" };
const subhead: React.CSSProperties = { color: "#aaa", fontSize: 12, fontWeight: 700, paddingTop: 2 };
const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
  gap: 12, overflowY: "auto", paddingRight: 4,
};
const cardStyle: React.CSSProperties = {
  background: "#2a2a3e", borderRadius: 8, overflow: "hidden",
  cursor: "pointer", transition: "transform 0.15s, box-shadow 0.15s",
  border: "2px solid transparent",
};
const savedCard: React.CSSProperties = { border: "2px solid #6c63ff" };
const thumb: React.CSSProperties = {
  width: "100%", aspectRatio: "2 / 3", maxHeight: 230,
  objectFit: "contain", display: "block", background: "#151525",
};
const thumbPlaceholder: React.CSSProperties = {
  width: "100%", aspectRatio: "2 / 3", display: "flex",
  alignItems: "center", justifyContent: "center",
  fontSize: 36, background: "#1a1a2e",
};
const cardBody: React.CSSProperties = { padding: "8px 10px" };
const cardTitle: React.CSSProperties = {
  fontSize: 13, fontWeight: 600, color: "#eee",
  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
};
const cardSub: React.CSSProperties = {
  fontSize: 11, color: "#aaa",
  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
};
const cardMeta: React.CSSProperties = { fontSize: 11, color: "#888", marginTop: 3 };
const savedBadge: React.CSSProperties = { fontSize: 11, color: "#6c63ff", marginTop: 4 };
const sampleBadge: React.CSSProperties = { fontSize: 11, color: "#f0b36a", marginTop: 4 };
const cardActions: React.CSSProperties = { display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 8 };
const cardBtn: React.CSSProperties = {
  background: "#6c63ff", border: "none", borderRadius: 5, color: "#fff",
  padding: "4px 8px", cursor: "pointer", fontSize: 11,
};
const sourceLink: React.CSSProperties = { color: "#aaa", fontSize: 11, textDecoration: "none" };
