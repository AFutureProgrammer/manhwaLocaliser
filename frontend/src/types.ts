// frontend/src/types.ts
// All shapes mirror the dicts returned by backend/engine.py get_bootstrap()

export interface Layer {
  id:     string;
  type:   "region" | "text" | "cleanup";
  name:   string;
  vis:    boolean;
  locked: boolean;
}

export interface Region {
  id:      string;   // "r1", "r2"…
  label:   string;   // "R-01"
  idx:     number;   // 0-based index into the regions array
  x:       number;
  y:       number;
  w:       number;
  h:       number;
  src:     string;   // Korean source text
  tl:      string;   // English translation
  conf:    number;   // 0-100
  font:    string;
  size:    number;
  leading: number;
  fg:      string;   // hex color
  bg:      string;   // hex color
  outline?: string;  // hex color
  outline_width?: number;
  shadow?: string;   // hex color
  shadow_on?: boolean;
  align:   "left" | "center" | "right";
  visible: boolean;
  locked:  boolean;
  detector_source: "ocr" | "yolo" | "manual" | string;
  manually_adjusted: boolean;
  /** Pass 4: role string from the backend (dialog/sfx/sound/shout/narration/…). */
  role?: string;
  /**
   * Pass 4: true when the region is class-SFX and the master
   * `process_sfx_regions` toggle is OFF. The overlay greys these out and the
   * pipeline skips them (no OCR / translate / cleanup / typeset / preview).
   */
  pipeline_disabled?: boolean;
  // ── Pass 6: per-stage confidence + status ─────────────────────────
  /** Detector (YOLO) confidence in [0,1]. Fallbacks to EasyOCR score for legacy OCR-detected blocks. */
  detector_confidence?: number;
  /** Qwen-VL / EasyOCR text-confidence in [0,1]. 0 until OCR has run. */
  ocr_confidence?: number;
  yolo_kind?:      string | null;
  yolo_class_id?:  number | null;
  yolo_train_class_id?: number | null;
  cleanup_tier?:   number;
  cleanup_status?: string;
  cleanup_reason?: string;
  cleanup_patch?: CleanupPatchInfo | null;
  cleanup_override?: CleanupOverride;
  cleanup_container_confidence?: number;
  cleanup_safe_rect_confidence?: number;
  detector_text_bbox?: number[] | null;
  bbox_override?: number[] | null;
  cleanup_container_bbox?: number[] | null;
  container_bbox?: number[] | null;
  cross_page?: boolean;
  cross_page_group_id?: string | null;
  cross_page_pages?: number[];
  composite_bbox?: number[] | null;
  page_local_bboxes?: Record<string, number[]>;
  typeset_status?: string;
  typeset_reason?: string;
  /** Derived: "ok" | "flagged" | "skipped_sfx" | "pending". No fake numeric confidence. */
  translation_status?: "ok" | "flagged" | "skipped_sfx" | "pending" | string;
  memory_hits: MemoryHit[];
  layers:  Layer[];
}

export interface CleanupPatchInfo {
  page_idx?: number;
  region_id?: string;
  region_idx?: number;
  bbox?: number[];
  strategy?: string;
  backend?: string;
  inpaint_method?: string;
  candidate_id?: string;
  mask_hash?: string;
  created_at?: string;
  review_required?: boolean;
  cleanup_status?: string;
  cleanup_reason?: string;
  rerun?: boolean;
  manual_mask_used?: boolean;
  grouped_inpaint?: boolean;
  group_id?: string;
  group_region_ids?: string[];
  group_backend?: string;
  group_reason?: string;
  fallback_error?: string;
}

export interface CleanupPreviewResponse extends ApiResponse {
  b64?: string | null;
  bbox?: number[];
  mask_b64?: string | null;
  mask_bbox?: number[];
  manual_mask_used?: boolean;
  grouped_inpaint?: boolean;
  group_region_ids?: string[];
  group_backend?: string;
  group_reason?: string;
  fallback_error?: string;
  plan?: Record<string, unknown>;
  debug?: Record<string, unknown>;
}

export interface Sam2MaskResponse extends ApiResponse {
  status?: string;
  mask_b64?: string | null;
  bbox?: number[] | null;
  confidence?: number;
  reason?: string;
}

export interface CleanupDebugMask {
  b64?: string | null;
  bbox?: number[] | null;
  available?: boolean;
}

export interface CleanupDebugResponse extends ApiResponse {
  analysis?: CleanupQaAnalysis;
  boxes?: {
    detector_text_bbox?: number[] | null;
    editable_bbox?: number[] | null;
    container_bbox?: number[] | null;
    patch_bbox?: number[] | null;
  };
  labels?: Record<string, string>;
  masks?: {
    text_mask?: CleanupDebugMask;
    cleanup_mask?: CleanupDebugMask;
    halo_mask?: CleanupDebugMask;
    manual_mask?: CleanupDebugMask;
    grouped_mask?: CleanupDebugMask;
  };
}

export interface CleanupQaAnalysis {
  page_index?: number;
  region_id?: string;
  region_label?: string;
  region_type?: string;
  effective_cleanup_action?: string;
  effective_cleanup_mode?: string;
  cleanup_status?: string;
  cleanup_reason?: string;
  skip_reason?: string;
  background_model?: string;
  container_confidence?: number;
  text_mask_confidence?: number;
  mask_container_ratio?: number;
  mask_region_ratio?: number;
  mask_area?: number;
  border_touch_ratio?: number;
  border_collision_bbox_source?: string;
  rectangularity?: number;
  cleanup_mask_rejected?: boolean;
  cleanup_mask_rejection_reason?: string;
  selected_text_mask_candidate_source?: string;
  solid_fill_eligible?: boolean | null;
  halo_mask_used?: boolean | null;
  residual_retry_used?: boolean | null;
  grouped_fallback_used?: boolean | null;
  selected_cleanup_candidate?: string;
  last_patch_status?: string;
  last_patch_reason?: string;
  bbox?: number[] | null;
  detector_text_bbox?: number[] | null;
  container_bbox?: number[] | null;
  cleanup_override?: Record<string, unknown>;
}

export interface CleanupCandidateScores {
  score?: number;
  residual_dark_pixels?: number;
  residual_light_pixels?: number;
  residual_text_pixels?: number;
  residual_edge_energy?: number;
  color_distance_to_sampled_bg?: number;
  seam_score?: number;
  blur_local_variance_loss?: number;
  texture_variance_loss?: number;
  mask_area?: number;
  mask_container_ratio?: number;
}

export interface CleanupCandidate {
  candidate_id: string;
  label: string;
  backend?: string;
  strategy?: string;
  method?: string;
  b64?: string | null;
  bbox?: number[];
  scores?: CleanupCandidateScores;
  warnings?: string[];
  reasons?: string[];
  review_required?: boolean;
  is_available?: boolean;
  unavailable_reason?: string;
  manual_mask_used?: boolean;
  grouped_inpaint?: boolean;
}

export interface CleanupCandidateCompareResponse extends ApiResponse {
  candidates?: CleanupCandidate[];
  recommended_candidate_id?: string;
}

export interface CleanupOverride {
  cleanup_override_mode?: string | null;
  cleanup_region_class?: string | null;
  cleanup_halo_max_px?: number | null;
  cleanup_residual_retry_enabled?: boolean | null;
  cleanup_residual_retry_dilate_px?: number | null;
  cleanup_min_container_confidence?: number | null;
  cleanup_max_mask_container_ratio?: number | null;
  cleanup_max_mask_region_ratio?: number | null;
  cleanup_max_border_touch_ratio?: number | null;
  cleanup_max_rectangularity?: number | null;
  cleanup_allow_low_confidence?: boolean | null;
  cleanup_allow_texture_inpaint?: boolean | null;
  cleanup_allow_translucent_caption?: boolean | null;
}

export interface MemoryHit {
  type:       "name" | "glossary";
  kr:         string;
  en:         string;
  trust:      string;
  matched_by: string;
  scope:      string;
}

export interface NameMemoryEntry {
  id:                 string;
  kr_name:            string;
  en_name:            string;
  aliases_kr:         string[];
  trust:              string;
  scope:              string;
  appearances:        number;
  first_seen_chapter: string;
  note:               string;
  approved_by?:       string | null;
  created_at:         string;
  updated_at:         string;
}

export interface GlossaryEntry {
  id:               string;
  source_kr:        string;
  target_en:        string;
  alternatives_en:  string[];
  aliases_kr:       string[];
  trust:            string;
  scope:            string;
  note:             string;
  approved_by?:     string | null;
  created_at:       string;
  updated_at:       string;
}

export interface SeriesMemory {
  available:     boolean;
  error?:        string;
  series_title:  string;
  memory_key?:   string;
  memory_fs_key?: string;
  memory_aliases?: string[];
  names:         NameMemoryEntry[];
  glossary:      GlossaryEntry[];
}

export interface PageSummary {
  id:      string;
  idx:     number;
  regions: number;
  dirty?:  boolean;
  /**
   * Pass 2: server-authored monotonically-increasing render counter.
   * Bumped on every cleaned_cv/typeset_pil mutation. Used in the
   * frontend PAGE_IMAGE_CACHE key so freshly produced images show up
   * immediately after Run Page without a client-side race.
   */
  render_version?: number;
  // 5-element array: [detect, ocr, translate, cleanup, typeset]
  status:  Array<"done" | "pend" | "active">;
}

export interface Chapter {
  id:       string;
  title:    string;
  pages:    number;
  progress: number;   // 0-100
  status:   string;
}

export interface Series {
  id:       string;
  title:    string;
  subtitle: string;
  source?:  SourceSlug;
  source_id?: string;
  thumbnail_url?: string;
  thumbnail_path?: string;
  lang:     string;
  chapters: number;
  color:    string;
}

export interface Issue {
  id:     string;
  sev:    "err" | "warn" | "info";
  msg:    string;
  region: string | null;
  page:   number;
}

export interface MetaSettings {
  /** Pass 4: master SFX toggle — OFF hides & skips SFX regions end-to-end. */
  process_sfx_regions?:  boolean;
  ocr_backend?:          "cascade" | "qwen_vl" | "paddleocr" | "easyocr" | string;
  qwen_ocr_model?:       string;
  paddleocr_service_url?: string;
  paddleocr_lang?:       string;
  ocr_vlm_fallback_confidence?: number | string;
  ocr_cache_enabled?:    boolean;
  /** Pass 8: "ollama" | "deepseek". */
  translation_provider?: string;
  deepseek_model?:       string;
  /** True iff the env var named by `deepseek_api_key_env` is populated. */
  deepseek_configured?:  boolean;
  cleanup_mode?:         string;
  auto_clean_sfx?:       boolean;
  auto_typeset_sfx?:     boolean;
  auto_clean_text_over_art?: boolean;
  cleanup_allow_sfx_cleanup?: boolean;
  cleanup_allow_text_over_art?: boolean;
  cleanup_solid_bubble_fill_enabled?: boolean;
  cleanup_solid_bubble_min_container_confidence?: number | string;
  cleanup_solid_bubble_max_mask_container_ratio?: number | string;
  cleanup_solid_bubble_max_rectangularity?: number | string;
  cleanup_halo_mask_enabled?: boolean;
  cleanup_halo_max_px?: number | string;
  cleanup_residual_retry_enabled?: boolean;
  cleanup_residual_retry_dilate_px?: number | string;
  cleanup_allow_grouped_inpaint?: boolean;
  cleanup_manual_review_only?: boolean;
  cleanup_min_container_confidence?: number | string;
  cleanup_max_mask_container_ratio?: number | string;
  cleanup_max_mask_region_ratio?: number | string;
  cleanup_max_border_touch_ratio?: number | string;
  cleanup_max_rectangularity?: number | string;
  cleanup_allow_translucent_caption?: boolean;
  cleanup_allow_texture_inpaint?: boolean;
  cleanup_risky_action?: string;
  cleanup_fallback_backend?: string;
  cleanup_verbose_logs?: boolean;
  cleanup_show_diagnostics?: boolean;
  sam2_enabled?: boolean;
  sam2_load_mode?: "startup" | "lazy" | string;
  sam2_required?: boolean;
  sam2_backend_url?: string;
  sam2_timeout_sec?: number | string;
  sam2_model_path?: string;
  sam2_checkpoint_path?: string;
  sam2_device?: string;
  sam2_mask_mode?: "manual_only" | "cleanup_assist" | "container_assist" | string;
  sam2_status?: Record<string, unknown>;
}

export interface Meta {
  activeSeriesId:  string | null;
  activeChapterId: string | null;
  activePageIdx:   number;
  busy:            boolean;
  status:          string;
  chapterDir:      string;
  totalPages:      number;
  chapterProgress: number;
  runAllCheckpoint?: {
    active?: boolean;
    phase?: string;
    page_idx?: number;
    page_total?: number;
    message?: string;
  };
  memoryStats?:    Record<string, unknown>;
  pendingReviewCount?: number;
  settings?:       MetaSettings;
}

export interface Bootstrap {
  series:   Series[];
  chapters: Record<string, Chapter[]>;
  pages:    PageSummary[];
  regions:  Region[];
  issues:   Issue[];
  memory:   SeriesMemory;
  meta:     Meta;
}

// API response wrapper — every api.* call returns this
export interface ApiResponse {
  ok:     boolean;
  error?: string;
  cancelled?: boolean;
}

export interface RegionPreviewSprite extends ApiResponse {
  b64: string | null;
  x: number;
  y: number;
  w: number;
  h: number;
  bbox?: [number, number, number, number];
  resolved_font_size?: number;
  line_count?: number;
  overflow?: boolean;
  font?: string;
  role?: string;
  fg?: string;
  outline?: string;
  outline_width?: number;
  shadow?: string;
  shadow_on?: boolean;
  align?: Region["align"];
}

export interface FontOptionsResponse extends ApiResponse {
  roles: string[];
  fonts: string[];
}

export type BootstrapResponse = ApiResponse & Bootstrap;

export type SourceSlug = "local" | "naver-comic" | string;

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

export interface SourceChapter {
  indexed: boolean;
  imported: boolean;
  translated: boolean;
  missing_raw: boolean;
  needs_sync: boolean;
  source?: SourceSlug;
  source_id?: string;
  episode_no?: number;
  chapter_memory_key?: string;
  chapter_memory_fs_key?: string;
  title_en?: string;
  title_ko?: string;
  source_url?: string;
  thumbnail_url?: string;
  thumbnail_path?: string;
  folder?: string;
  page_count?: number;
  name?: string;
  chapter_folder?: string;
  last_synced_at?: string;
}

export interface SeriesStats {
  indexed: number;
  imported: number;
  translated: number;
  missing_raw: number;
  estimated_bytes: number;
}

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

export interface SyncResult {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}

// Progress event pushed from Python via window.dispatchEvent
export interface ProgressEvent {
  message: string;
  current: number;
  total:   number;
  running?: boolean;
  job?: "run_page" | "run_all" | string;
  stage?: "detect" | "ocr" | "translate" | "cleanup" | "typeset" | string;
  page_idx?: number;
  page_total?: number;
  region_idx?: number | null;
  region_total?: number | null;
  percent?: number;
  updated_pages?: number[];
  error?: string;
}
