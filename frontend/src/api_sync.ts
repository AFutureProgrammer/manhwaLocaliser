// frontend/src/api_sync.ts
// ─────────────────────────────────────────────────────────────────────────────
// PATCH: Merge these exports into frontend/src/api.ts
//
// The existing api.ts file presumably exposes a `callApi` or `window.pywebview`
// wrapper.  Replace the placeholder `callApi` below with whatever pattern
// your existing api.ts uses.
//
// Example: if your api.ts does
//   export const importChapter = (folder: string) =>
//     window.pywebview.api.import_chapter(folder);
// then follow the same pattern here.
// ─────────────────────────────────────────────────────────────────────────────

import type {
  BrowseCard,
  SeriesDetail,
  SeriesSummary,
  SyncResult,
} from "./types_sync";

// ── Internal helper (replace with your actual bridge) ─────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function api<T = unknown>(method: string, ...args: unknown[]): Promise<T> {
  // pywebview bridge — adjust if your project uses a different pattern
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bridge = (window as any).pywebview?.api;
  if (!bridge) return Promise.reject(new Error("pywebview API not ready"));
  const fn = bridge[method];
  if (typeof fn !== "function") return Promise.reject(new Error(`No API method: ${method}`));
  return fn(...args);
}

// ── Source / Browse ───────────────────────────────────────────────────────────

export const listSources = (): Promise<{ ok: boolean; sources: string[] }> =>
  api("list_sources");

export const browseSourceSeries = (
  source: string,
  query = ""
): Promise<{ ok: boolean; error?: string; cards: BrowseCard[] }> =>
  api("browse_source_series", source, query);

/** Save a browsed series: metadata + chapter index ONLY. No image import. */
export const selectBrowseSeries = (
  seriesTitle: string,
  source: string,
  sourceId: string,
  card: BrowseCard
): Promise<SyncResult> =>
  api("select_browse_series", seriesTitle, source, sourceId, card);

// ── Series metadata ────────────────────────────────────────────────────────────

export const getSeriesList = (): Promise<{
  ok: boolean;
  series: SeriesSummary[];
}> => api("get_series_list");

export const getSeriesDetail = (
  seriesTitle: string
): Promise<{ ok: boolean; detail?: SeriesDetail; error?: string }> =>
  api("get_series_detail", seriesTitle);

export const updateSeriesMetadata = (
  seriesTitle: string,
  updates: Partial<SeriesDetail>
): Promise<SyncResult> => api("update_series_metadata", seriesTitle, updates);

/** Re-fetch metadata + chapter index from provider. Metadata-only, no images. */
export const syncSeriesMetadata = (
  seriesTitle: string,
  source?: string,
  sourceId?: string
): Promise<SyncResult> =>
  api("sync_series_metadata", seriesTitle, source ?? "", sourceId ?? "");

// ── Chapter sync ───────────────────────────────────────────────────────────────

/**
 * Import raw images for chapters.
 * mode="missing" — only chapters with missing_raw=true
 * mode="all"     — every chapter (warn user before calling!)
 */
export const syncSeriesChapters = (
  seriesTitle: string,
  mode: "missing" | "all" = "missing"
): Promise<SyncResult & { synced?: number; total?: number; errors?: string[] }> =>
  api("sync_series_chapters", seriesTitle, mode);

export const syncSourceChapter = (
  seriesTitle: string,
  chapterSourceId: string
): Promise<SyncResult & { pages_synced?: number; folder?: string; skipped?: boolean }> =>
  api("sync_source_chapter", seriesTitle, chapterSourceId);

/** Import + open one chapter (lazy import single chapter). */
export const importSourceChapter = (
  seriesTitle: string,
  chapterSourceId: string
): Promise<SyncResult & { bootstrap?: unknown; folder?: string; page_count?: number; opened?: boolean }> =>
  api("import_source_chapter", seriesTitle, chapterSourceId);

export const deleteSeries = (
  seriesTitle: string,
  source = "",
  sourceId = "",
  deleteFiles = false
): Promise<SyncResult> => api("delete_series", seriesTitle, source, sourceId, deleteFiles);

export const syncMissingThumbnails = (
  seriesTitle: string
): Promise<SyncResult> => api("sync_missing_thumbnails", seriesTitle);

// ── Translation ────────────────────────────────────────────────────────────────

export const translateSeriesMetadata = (
  seriesTitle: string
): Promise<SyncResult & { translated?: Record<string, string> }> =>
  api("translate_series_metadata", seriesTitle);

export const retranslateSeries = (seriesTitle: string): Promise<SyncResult> =>
  api("retranslate_series", seriesTitle);
