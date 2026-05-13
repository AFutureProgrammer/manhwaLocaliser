// frontend/src/components/SeriesDetailPanel.tsx
// ─────────────────────────────────────────────────────────────────────────────
// Series detail panel: metadata editor + sync controls + chapter list.
//
// Integration (App.tsx):
//   1. Import SeriesDetailPanel
//   2. Track `selectedSeriesTitle: string | null` in App state
//   3. Render <SeriesDetailPanel seriesTitle={selectedSeriesTitle} ... />
//      alongside your existing chapter viewer
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useEffect, useCallback } from "react";
import api from "../api";
import type { Bootstrap, SeriesDetail, SourceChapter, SeriesStats } from "../types";

interface Props {
  seriesTitle: string;
  onOpenChapter?: (folder: string) => void;  // existing import_chapter hook
  onBootstrap?: (bootstrap: Bootstrap) => void;
  onBrowseClick?: () => void;                // open BrowseModal
  onDeleteSeries?: (title: string) => void;  // danger zone
  onSourceChange?: () => void;               // refresh parent
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtBytes(bytes: number): string {
  if (bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function chapterLabel(ch: SourceChapter): string {
  if (ch.title_en) return ch.title_en;
  if (ch.title_ko) return ch.title_ko;
  if (ch.episode_no) return `Episode ${ch.episode_no}`;
  return ch.source_id ?? ch.name ?? "Chapter";
}

function chapterStatusBadge(ch: SourceChapter): { label: string; color: string } {
  if (ch.translated) return { label: "translated", color: "#6c63ff" };
  if (ch.imported)   return { label: "imported",   color: "#4caf50" };
  if (ch.missing_raw) return { label: "missing raw", color: "#b46a3a" };
  if (ch.indexed)    return { label: "indexed",    color: "#888" };
  return { label: "?", color: "#555" };
}

function chapterBadges(ch: SourceChapter): Array<{ label: string; color: string }> {
  const badges = [];
  if (ch.indexed) badges.push({ label: "Indexed", color: "#777" });
  if (ch.missing_raw) badges.push({ label: "Missing raw", color: "#b46a3a" });
  if (ch.imported) badges.push({ label: "Imported", color: "#4caf50" });
  if (ch.translated) badges.push({ label: "Translated", color: "#6c63ff" });
  if (ch.needs_sync) badges.push({ label: "Needs sync", color: "#a44" });
  return badges;
}

// ── Component ─────────────────────────────────────────────────────────────────

export const SeriesDetailPanel: React.FC<Props> = ({
  seriesTitle,
  onOpenChapter,
  onBootstrap,
  onBrowseClick,
  onDeleteSeries,
  onSourceChange,
}) => {
  const [detail, setDetail] = useState<SeriesDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [editing, setEditing] = useState<Partial<SeriesDetail>>({});
  const [dirty, setDirty] = useState(false);
  const [importingChId, setImportingChId] = useState<string | null>(null);
  const [showDanger, setShowDanger] = useState(false);
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem("ml.sourceSyncCollapsed") === "1"; }
    catch { return false; }
  });

  useEffect(() => {
    try { localStorage.setItem("ml.sourceSyncCollapsed", collapsed ? "1" : "0"); }
    catch { /* noop */ }
  }, [collapsed]);

  const reload = useCallback(async () => {
    const res = await api.getSeriesDetail(seriesTitle);
    if (res.ok && res.detail) {
      setDetail(res.detail);
      setEditing({});
      setDirty(false);
    } else {
      setStatus(`Error: ${res.error}`);
    }
  }, [seriesTitle]);

  useEffect(() => {
    reload();
  }, [reload]);

  // Merge editing changes
  const merged: SeriesDetail | null = detail
    ? { ...detail, ...editing }
    : null;

  const set = (key: keyof SeriesDetail, value: unknown) => {
    setEditing((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  // ── Actions ──────────────────────────────────────────────────────────────

  const handleSave = async () => {
    if (!dirty) return;
    setBusy(true);
    setStatus("Saving…");
    const res = await api.updateSeriesMetadata(seriesTitle, editing);
    setStatus(res.ok ? "Saved." : `Error: ${res.error}`);
    setBusy(false);
    if (res.ok) { setDirty(false); onSourceChange?.(); }
  };

  const handleSyncIndex = async () => {
    setBusy(true);
    setStatus("Syncing chapter index…");
    const src = merged?.source ?? "";
    const sid = merged?.source_id ?? "";
    const res = await api.syncSeriesMetadata(seriesTitle, src, sid);
    setStatus(res.ok ? `Index synced — ${(res as any).chapter_count ?? "?"} chapters.` : `Error: ${res.error}`);
    setBusy(false);
    if (res.ok) { await reload(); onSourceChange?.(); }
  };

  const handleSyncMissing = async () => {
    setBusy(true);
    setStatus("Syncing missing chapters…");
    const res = await api.syncSeriesChapters(seriesTitle, "missing");
    setStatus(res.ok
      ? `Done — ${res.synced ?? 0}/${res.total ?? 0} chapters synced.`
      : `Error: ${res.error}`);
    setBusy(false);
    if (res.ok) await reload();
  };

  const handleSyncAll = async () => {
    const stats = detail?.stats;
    const count = stats?.indexed ?? detail?.chapters.length ?? 0;
    if (!window.confirm(
      `This may import many raw images and use significant disk space. Continue?\n\nChapters: ${count}`
    )) return;
    setBusy(true);
    setStatus("Syncing all chapters…");
    const res = await api.syncSeriesChapters(seriesTitle, "all");
    setStatus(res.ok
      ? `Done — ${res.synced ?? 0}/${res.total ?? 0} chapters synced.`
      : `Error: ${res.error}`);
    setBusy(false);
    if (res.ok) await reload();
  };

  const handleSyncThumbs = async () => {
    setBusy(true);
    setStatus("Syncing thumbnails…");
    const res = await api.syncMissingThumbnails(seriesTitle);
    setStatus(res.ok ? "Thumbnails synced." : `Note: ${res.error}`);
    setBusy(false);
  };

  const handleTranslateMeta = async () => {
    setBusy(true);
    setStatus("Translating metadata…");
    const res = await api.translateSeriesMetadata(seriesTitle);
    setStatus(res.ok ? "Metadata translated." : `Error: ${res.error}`);
    setBusy(false);
    if (res.ok) await reload();
  };

  const handleRetranslate = async () => {
    if (!window.confirm("Re-translate ALL chapters? This cannot be undone.")) return;
    setBusy(true);
    const res = await api.retranslateSeries(seriesTitle);
    setStatus(res.ok ? "Re-translation complete." : `Note: ${res.error}`);
    setBusy(false);
  };

  const handleImportChapter = async (ch: SourceChapter) => {
    const chId = ch.source_id ?? "";
    if (!chId) {
      // Local chapter — open directly via existing workflow
      const folder = ch.folder ?? ch.chapter_folder ?? "";
      if (folder && onOpenChapter) onOpenChapter(folder);
      return;
    }
    setImportingChId(chId);
    setStatus(`Importing chapter ${chapterLabel(ch)}…`);
    const res = await api.importSourceChapter(seriesTitle, chId);
    setStatus(res.ok ? "Opened source chapter." : `Error: ${res.error}`);
    setImportingChId(null);
    if (res.ok) {
      const bootstrap = res.bootstrap ?? (Array.isArray((res as Partial<Bootstrap>).pages) ? res as unknown as Bootstrap : null);
      if (bootstrap) onBootstrap?.(bootstrap);
      await reload();
      onSourceChange?.();
    }
  };

  const handleSyncChapter = async (ch: SourceChapter) => {
    const chId = ch.source_id ?? "";
    if (!chId) return;
    setImportingChId(chId);
    setStatus(`Syncing chapter ${chapterLabel(ch)}...`);
    const res = await api.syncSourceChapter(seriesTitle, chId);
    setStatus(res.ok ? `Chapter synced (${res.pages_synced ?? 0} pages).` : `Error: ${res.error}`);
    setImportingChId(null);
    if (res.ok) await reload();
  };

  const handleDeleteSeries = async () => {
    if (!merged) return;
    const ok = window.confirm(
      `Delete this series from the app?\n\nThis removes the SeriesDB entry only. Local chapter folders and memory will be preserved.`
    );
    if (!ok) return;
    setBusy(true);
    setStatus("Removing from library...");
    const res = await api.deleteSeries(seriesTitle, merged.source ?? "", merged.source_id ?? "", false);
    setBusy(false);
    if (!res.ok) {
      setStatus(`Error: ${res.error}`);
      return;
    }
    setStatus("Removed from library. Local files and memory were preserved.");
    onDeleteSeries?.(seriesTitle);
    onSourceChange?.();
  };

  // ── Render ────────────────────────────────────────────────────────────────

  if (!merged) {
    return <div style={panel}><div style={statusLine}>Loading {seriesTitle}…</div></div>;
  }

  const stats: SeriesStats | undefined = merged.stats;
  const isRemote = merged.source && merged.source !== "local";
  const displayTitle = merged.title_en || merged.title_ko || seriesTitle;

  if (collapsed) {
    return (
      <div style={panel}>
        <div style={sectionHead}>
          <button style={collapseBtn} onClick={() => setCollapsed(false)} title="Expand Source Sync">▶</button>
          <span style={{ fontWeight: 700, fontSize: 13 }}>Source Sync</span>
          {isRemote && <span style={sourceBadge}>{merged.source}</span>}
          {status && <span style={statusPill}>{status}</span>}
        </div>
        <div style={compactSummary}>
          <div style={{ fontWeight: 700, color: "#eee", overflowWrap: "anywhere" }}>{displayTitle}</div>
          <div style={{ color: "#888", fontSize: 11 }}>
            Indexed {stats?.indexed ?? 0} · Imported {stats?.imported ?? 0} · Translated {stats?.translated ?? 0}
          </div>
          {onBrowseClick && <Btn label="Browse Naver" onClick={onBrowseClick} disabled={busy} />}
        </div>
      </div>
    );
  }

  return (
    <div style={panel}>
      {/* ── Title bar ─────────────────────────────────────────────────────── */}
      <div style={sectionHead}>
        <button style={collapseBtn} onClick={() => setCollapsed(true)} title="Collapse Source Sync">▼</button>
        <span style={{ fontWeight: 700, fontSize: 13 }}>Source Sync</span>
        {isRemote && <span style={sourceBadge}>{merged.source}</span>}
        {status && <span style={statusPill}>{status}</span>}
      </div>
      <div style={{ fontWeight: 700, fontSize: 12, color: "#eee", overflowWrap: "anywhere" }}>{seriesTitle}</div>

      {/* ── Metadata fields ───────────────────────────────────────────────── */}
      <div style={grid2}>
        <Field label="Title (EN)" value={merged.title_en ?? ""} onChange={(v) => set("title_en", v)} />
        <Field label="Title (KO)" value={merged.title_ko ?? ""} onChange={(v) => set("title_ko", v)} />
      </div>
      <Field label="Synopsis (EN)" value={merged.synopsis_en ?? ""} onChange={(v) => set("synopsis_en", v)} multiline />
      <Field label="Synopsis (KO)" value={merged.synopsis_ko ?? ""} onChange={(v) => set("synopsis_ko", v)} multiline />
      <div style={grid2}>
        <label style={fieldLabel}>
          Source
          <select
            style={input}
            value={merged.source ?? "local"}
            onChange={(e) => set("source", e.target.value)}
          >
            <option value="local">local</option>
            <option value="naver-comic">naver-comic</option>
          </select>
        </label>
        <Field label="Source ID" value={merged.source_id ?? ""} onChange={(v) => set("source_id", v)} placeholder="e.g. 12345" />
      </div>
      <Field label="Source URL" value={merged.source_url ?? ""} onChange={(v) => set("source_url", v)} placeholder="https://comic.naver.com/…" />
      {isRemote && (
        <div style={debugBlock}>
          <div>memory_key: {merged.memory_key || "source key pending"}</div>
          {merged.memory_fs_key && <div>memory_fs_key: {merged.memory_fs_key}</div>}
        </div>
      )}

      {/* ── Sync status line ──────────────────────────────────────────────── */}
      {isRemote && (
        <div style={syncLine}>
          <span>Last synced: {merged.last_synced_at || "never"}</span>
          {merged.sync_status && <span style={syncBadge(merged.sync_status)}>{merged.sync_status}</span>}
        </div>
      )}

      {/* ── Stats ─────────────────────────────────────────────────────────── */}
      {stats && (
        <div style={statsRow}>
          <Stat label="Indexed" value={stats.indexed} />
          <Stat label="Imported" value={stats.imported} />
          <Stat label="Translated" value={stats.translated} />
          <Stat label="Missing raw" value={stats.missing_raw} warn={stats.missing_raw > 0} />
          <Stat label="Storage" value={fmtBytes(stats.estimated_bytes)} />
        </div>
      )}

      {/* ── Action buttons ────────────────────────────────────────────────── */}
      <div style={btnRow}>
        <Btn label="Save" onClick={handleSave} disabled={!dirty || busy} primary />
        {onBrowseClick && <Btn label="Browse Naver" onClick={onBrowseClick} disabled={busy} />}
      </div>
      {isRemote && (
        <div style={btnRow}>
          <Btn label="Sync chapter index" onClick={handleSyncIndex} disabled={busy} />
          <Btn label="Sync missing only" onClick={handleSyncMissing} disabled={busy} />
          <Btn label="Sync all chapters" onClick={handleSyncAll} disabled={busy} warn />
          <Btn label="Sync thumbnails" onClick={handleSyncThumbs} disabled={busy} />
        </div>
      )}
      <div style={btnRow}>
        <Btn label="Translate metadata" onClick={handleTranslateMeta} disabled={busy} />
        <Btn label="Re-translate all" onClick={handleRetranslate} disabled={busy} warn />
      </div>

      {/* ── Chapter list ──────────────────────────────────────────────────── */}
      <div style={sectionHead} onClick={() => {}}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>
          Source chapters ({merged.chapters?.length ?? 0})
        </span>
      </div>
      <div style={chapterList}>
        {(merged.chapters ?? []).map((ch, i) => {
          const chId = ch.source_id ?? "";
          const isImporting = importingChId === chId;
          const label = chapterLabel(ch);
          const canImport = ch.source_id || ch.folder || ch.chapter_folder;
          const primaryBadge = chapterStatusBadge(ch);
          return (
            <div
              key={chId || i}
              style={{ ...chRow, cursor: canImport && !isImporting && !busy ? "pointer" : "default" }}
              onClick={() => {
                if (canImport && !isImporting && !busy) handleImportChapter(ch);
              }}
              title={canImport ? "Import/Open this source chapter" : undefined}
            >
              <div style={chInfo}>
                <span style={chLabel}>{label}</span>
                {ch.episode_no ? <span style={chEp}>Ep {ch.episode_no}</span> : null}
                {ch.source_id ? <span style={chEp}>ID {ch.source_id}</span> : null}
                <span style={{ ...chBadge, background: primaryBadge.color }}>{primaryBadge.label}</span>
                {ch.page_count && ch.page_count > 0 ? (
                  <span style={chPages}>{ch.page_count}p</span>
                ) : null}
                <div style={badgeWrap}>
                  {chapterBadges(ch).map(b => (
                    <span key={b.label} style={{ ...chBadge, background: b.color }}>{b.label}</span>
                  ))}
                </div>
              </div>
              <div style={chActions}>
                {ch.source_id && (
                  <button
                    style={smallBtn}
                    disabled={isImporting || busy || ch.imported}
                    onClick={(e) => { e.stopPropagation(); handleSyncChapter(ch); }}
                    title="Sync this chapter's raw images without opening it"
                  >
                    Sync
                  </button>
                )}
                {canImport && (
                  <button
                    style={smallBtn}
                    disabled={isImporting || busy}
                    onClick={(e) => { e.stopPropagation(); handleImportChapter(ch); }}
                  >
                    {isImporting ? "…" : ch.imported ? "Open" : "Import/Open"}
                  </button>
                )}
                <button
                  style={smallBtn}
                  disabled
                  title="Translate missing chapters is not implemented yet"
                  onClick={(e) => e.stopPropagation()}
                >
                  Translate missing
                </button>
                {ch.source_url && (
                  <a
                    href={ch.source_url}
                    target="_blank"
                  rel="noreferrer"
                  style={linkBtn}
                  title="Open source page"
                  onClick={(e) => e.stopPropagation()}
                >
                    ↗
                  </a>
                )}
              </div>
            </div>
          );
        })}
        {(!merged.chapters || merged.chapters.length === 0) && (
          <div style={{ color: "#666", fontSize: 12, padding: "8px 0" }}>
            No chapters. Run "Sync chapter index" to fetch the chapter list.
          </div>
        )}
      </div>

      {/* ── Danger zone ───────────────────────────────────────────────────── */}
      {onDeleteSeries && <div style={dangerZone}>
        <button style={dangerToggle} onClick={() => setShowDanger((v) => !v)}>
          {showDanger ? "▲" : "▼"} Danger zone
        </button>
        {showDanger && (
          <button
            style={deleteBtn}
            onClick={handleDeleteSeries}
          >
            Remove from library
          </button>
        )}
      </div>}
    </div>
  );
};

// ── Sub-components ────────────────────────────────────────────────────────────

const Field: React.FC<{
  label: string;
  value: string;
  onChange: (v: string) => void;
  multiline?: boolean;
  placeholder?: string;
}> = ({ label, value, onChange, multiline, placeholder }) => (
  <label style={fieldLabel}>
    {label}
    {multiline ? (
      <textarea
        style={{ ...input, height: 60, resize: "vertical" }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    ) : (
      <input
        style={input}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    )}
  </label>
);

const Btn: React.FC<{
  label: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  warn?: boolean;
}> = ({ label, onClick, disabled, primary, warn }) => (
  <button
    style={{
      ...btn,
      ...(primary ? primaryBtn : {}),
      ...(warn ? warnBtn : {}),
      ...(disabled ? disabledBtn : {}),
    }}
    onClick={onClick}
    disabled={disabled}
  >
    {label}
  </button>
);

const Stat: React.FC<{ label: string; value: number | string; warn?: boolean }> = ({
  label, value, warn,
}) => (
  <div style={statBox}>
    <div style={{ ...statValue, color: warn ? "#f88" : "#eee" }}>{value}</div>
    <div style={statLabel}>{label}</div>
  </div>
);

// ── Styles ────────────────────────────────────────────────────────────────────

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 10,
  padding: 16, fontSize: 13,
  background: "#1a1a2e", color: "#ddd",
  position: "relative",
};
const sectionHead: React.CSSProperties = {
  display: "flex", alignItems: "center", gap: 8, paddingBottom: 4,
  borderBottom: "1px solid #333",
};
const collapseBtn: React.CSSProperties = {
  background: "none", border: "1px solid #3a3a4e", borderRadius: 4,
  color: "#aaa", cursor: "pointer", fontSize: 10, lineHeight: 1,
  padding: "2px 5px",
};
const compactSummary: React.CSSProperties = {
  display: "grid", gap: 6,
};
const sourceBadge: React.CSSProperties = {
  fontSize: 10, color: "#4ec9b4", border: "1px solid #315d55",
  borderRadius: 8, padding: "1px 6px", fontFamily: "monospace",
};
const statusLine: React.CSSProperties = { color: "#aaa", fontSize: 12 };
const statusPill: React.CSSProperties = {
  fontSize: 11, color: "#aaa", background: "#2a2a3e",
  padding: "2px 8px", borderRadius: 10,
};
const grid2: React.CSSProperties = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 };
const fieldLabel: React.CSSProperties = { display: "flex", flexDirection: "column", gap: 3, fontSize: 12, color: "#999" };
const input: React.CSSProperties = {
  background: "#2a2a3e", border: "1px solid #444", borderRadius: 6,
  color: "#eee", padding: "5px 8px", fontSize: 13, width: "100%",
  boxSizing: "border-box",
};
const syncLine: React.CSSProperties = {
  display: "flex", gap: 8, alignItems: "center", fontSize: 12, color: "#888",
};
const debugBlock: React.CSSProperties = {
  background: "#111120", border: "1px solid #303045", borderRadius: 4,
  color: "#777", fontFamily: "monospace", fontSize: 10, padding: "5px 6px",
  overflowWrap: "anywhere",
};
const syncBadge = (status: string): React.CSSProperties => ({
  padding: "1px 8px", borderRadius: 10, fontSize: 11,
  background: status === "ok" ? "#1a3a1a" : "#3a1a1a",
  color: status === "ok" ? "#4f4" : "#f84",
});
const statsRow: React.CSSProperties = {
  display: "flex", gap: 8, flexWrap: "wrap",
};
const statBox: React.CSSProperties = {
  flex: "1 0 70px", background: "#2a2a3e", borderRadius: 6,
  padding: "6px 10px", textAlign: "center",
};
const statValue: React.CSSProperties = { fontWeight: 700, fontSize: 16 };
const statLabel: React.CSSProperties = { fontSize: 10, color: "#777", marginTop: 2 };
const btnRow: React.CSSProperties = { display: "flex", gap: 6, flexWrap: "wrap" };
const btn: React.CSSProperties = {
  padding: "5px 12px", borderRadius: 6, border: "1px solid #444",
  background: "#2a2a3e", color: "#ccc", cursor: "pointer", fontSize: 12,
};
const primaryBtn: React.CSSProperties = { background: "#6c63ff", border: "none", color: "#fff" };
const warnBtn: React.CSSProperties = { background: "#3a2010", border: "1px solid #a44", color: "#f84" };
const disabledBtn: React.CSSProperties = { opacity: 0.4, cursor: "not-allowed" };
const chapterList: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 4,
};
const chRow: React.CSSProperties = {
  display: "flex", alignItems: "flex-start", justifyContent: "space-between",
  padding: "5px 8px", background: "#2a2a3e", borderRadius: 6,
};
const chInfo: React.CSSProperties = { display: "flex", alignItems: "center", gap: 6, minWidth: 0, flexWrap: "wrap" };
const chLabel: React.CSSProperties = {
  fontWeight: 500, color: "#ddd", whiteSpace: "nowrap",
  overflow: "hidden", textOverflow: "ellipsis", maxWidth: 160,
};
const chEp: React.CSSProperties = { fontSize: 10, color: "#888", flexShrink: 0 };
const chBadge: React.CSSProperties = {
  fontSize: 9, padding: "1px 5px", borderRadius: 8, color: "#fff", flexShrink: 0,
};
const badgeWrap: React.CSSProperties = { display: "flex", gap: 3, flexWrap: "wrap", width: "100%" };
const chPages: React.CSSProperties = { fontSize: 10, color: "#666", flexShrink: 0 };
const chActions: React.CSSProperties = { display: "flex", gap: 4, flexShrink: 0 };
const smallBtn: React.CSSProperties = {
  fontSize: 11, padding: "2px 8px", borderRadius: 5,
  border: "1px solid #555", background: "#333", color: "#ccc", cursor: "pointer",
};
const linkBtn: React.CSSProperties = {
  fontSize: 12, color: "#6c63ff", textDecoration: "none", padding: "2px 4px",
};
const dangerZone: React.CSSProperties = {
  marginTop: 8, paddingTop: 8, borderTop: "1px solid #2a1010",
  display: "flex", flexDirection: "column", gap: 8,
};
const dangerToggle: React.CSSProperties = {
  background: "none", border: "none", color: "#a44",
  cursor: "pointer", fontSize: 11, textAlign: "left", padding: 0,
};
const deleteBtn: React.CSSProperties = {
  background: "#3a1010", border: "1px solid #a44", color: "#f44",
  borderRadius: 6, padding: "5px 14px", cursor: "pointer", fontSize: 12,
};
