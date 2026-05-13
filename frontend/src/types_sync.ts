// frontend/src/types_sync.ts
// ─────────────────────────────────────────────────────────────────────────────
// PATCH: Append these type definitions to frontend/src/types.ts
// (or import from this file wherever needed).
// ─────────────────────────────────────────────────────────────────────────────

/** Registered source provider slugs */
export type SourceSlug = "local" | "naver-comic" | string;

/** Lightweight card returned by browse_source_series */
export interface BrowseCard {
  source: SourceSlug;
  source_id: string;
  memory_key?: string;
  memory_fs_key?: string;
  sync_status?: string;
  source_metadata?: Record<string, unknown>;
  title_ko: string;
  title_en: string;
  thumbnail_url: string;
  thumbnail_path?: string;
  chapter_count: number;
  source_url: string;
}

/** Chapter status flags */
export interface ChapterStatus {
  indexed: boolean;
  imported: boolean;
  translated: boolean;
  missing_raw: boolean;
  needs_sync: boolean;
}

/** Extended chapter record (includes legacy local-only fields) */
export interface SourceChapter extends ChapterStatus {
  // Identity
  source?: SourceSlug;
  source_id?: string;
  episode_no?: number;
  chapter_memory_key?: string;
  chapter_memory_fs_key?: string;
  // Titles
  title_en?: string;
  title_ko?: string;
  // Link
  source_url?: string;
  thumbnail_url?: string;
  thumbnail_path?: string;
  // Local
  folder?: string;
  page_count?: number;
  // Local legacy (used by existing workflow)
  name?: string;
  chapter_folder?: string;
  last_synced_at?: string;
}

/** Storage stats for a series */
export interface SeriesStats {
  indexed: number;
  imported: number;
  translated: number;
  missing_raw: number;
  estimated_bytes: number;
}

/** Full series detail */
export interface SeriesDetail {
  title: string;
  title_en?: string;
  title_ko?: string;
  synopsis_en?: string;
  synopsis_ko?: string;
  source?: SourceSlug;
  source_id?: string;
  memory_key?: string;
  memory_fs_key?: string;
  memory_aliases?: string[];
  source_url?: string;
  thumbnail_url?: string;
  thumbnail_path?: string;
  last_synced_at?: string;
  sync_status?: string;
  chapters: SourceChapter[];
  stats?: SeriesStats;
}

/** Lightweight series summary (used in sidebar / series list) */
export interface SeriesSummary {
  title: string;
  title_en?: string;
  title_ko?: string;
  source?: SourceSlug;
  source_id?: string;
  memory_key?: string;
  memory_fs_key?: string;
  memory_aliases?: string[];
  source_url?: string;
  thumbnail_url?: string;
  thumbnail_path?: string;
  chapter_count: number;
  last_synced_at?: string;
  sync_status?: string;
}

/** Generic ok/error response */
export interface SyncResult {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}
