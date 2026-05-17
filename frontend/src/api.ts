/**
 * frontend/src/api.ts
 * ───────────────────
 * Thin wrapper around window.pywebview.api.*
 *
 * Design rules:
 *   1. This is the ONLY file that references window.pywebview.
 *   2. Every call returns a typed ApiResponse — callers never need try/catch.
 *   3. If pywebview isn't available (dev without launcher), a stub is used
 *      so Vite previews still work against the DEFAULT_MOCK.
 *
 * Usage in components:
 *   import api from "@/api";
 *   const resp = await api.runStep("detect");
 *   if (!resp.ok) { showError(resp.error); return; }
 *   setBootstrap(resp);
 */

import type {
  Bootstrap,
  BootstrapResponse,
  ApiResponse,
  FontOptionsResponse,
  ProgressEvent,
  RegionPreviewSprite,
  CleanupPreviewResponse,
  Sam2MaskResponse,
  CleanupDebugResponse,
  CleanupCandidateCompareResponse,
  SeriesMemory,
  BrowseCard,
  SeriesDetail,
  SeriesSummary,
  SyncResult,
} from "./types";

// The pywebview JS bridge.  pywebview injects this before the page scripts run.
type PywebviewBridge = {
  api: {
    get_bootstrap:        ()                              => Promise<BootstrapResponse>;
    open_chapter_folder:  ()                              => Promise<BootstrapResponse & { cancelled: boolean }>;
    import_chapter:       (folder: string)                => Promise<BootstrapResponse>;
    go_to_page:           (idx: number)                   => Promise<BootstrapResponse>;
    run_step:             (step: string)                  => Promise<BootstrapResponse>;
    run_all:              ()                              => Promise<BootstrapResponse>;
    continue_run_all:     ()                              => Promise<BootstrapResponse>;
    run_current_page:     ()                              => Promise<BootstrapResponse>;
    detect_all:           ()                              => Promise<BootstrapResponse>;
    export_project:       (dir?: string)                  => Promise<ApiResponse & { export_dir: string }>;
    export_yolo_finetune_dataset: ()                      => Promise<ApiResponse & { dataset_dir?: string; manifest?: string; pages?: number }>;
    set_yolo_train_class: (idx: number, classId: number, pageIdx?: number | null) => Promise<BootstrapResponse>;
    train_yolo_detector:  ()                              => Promise<ApiResponse & { status?: string; dataset_dir?: string; status_path?: string; log?: string; running?: boolean; pid?: number; pages?: number }>;
    get_yolo_training_status: ()                          => Promise<ApiResponse & { status?: string; running?: boolean; dataset_dir?: string; status_path?: string; log?: string; onnx?: string; error?: string }>;
    reveal_export_folder: ()                              => Promise<ApiResponse>;
    get_page_image:       (idx: number, mode?: "best" | "raw" | "cleaned" | "typeset") => Promise<ApiResponse & { b64: string | null }>;
    update_region:        (idx: number, field: string, value: unknown, pageIdx?: number | null) => Promise<BootstrapResponse>;
    update_region_bbox:   (idx: number, x: number, y: number, w: number, h: number, pageIdx?: number | null, editPageIdx?: number | null) => Promise<BootstrapResponse>;
    get_region_preview_sprite: (idx: number, draft?: Record<string, unknown>, pageIdx?: number | null) => Promise<RegionPreviewSprite>;
    list_fonts:           ()                              => Promise<FontOptionsResponse>;
    add_region:           (x: number, y: number, w: number, h: number, text?: string, pageIdx?: number | null) => Promise<BootstrapResponse>;
    delete_region:        (idx: number, yoloRejectReason?: string, pageIdx?: number | null) => Promise<BootstrapResponse>;
    ocr_region:           (idx: number, pageIdx?: number | null) => Promise<BootstrapResponse>;
    translate_region:     (idx: number, pageIdx?: number | null) => Promise<BootstrapResponse>;
    preview_region_cleanup: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<CleanupPreviewResponse>;
    propose_cleanup_mask_sam2: (idx: number, prompt?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<Sam2MaskResponse>;
    get_sam2_status:     (load?: boolean)                 => Promise<ApiResponse & { status?: string; loaded?: boolean }>;
    get_region_cleanup_debug: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<CleanupDebugResponse>;
    record_mask_qa_label: (idx: number, label: string, notes?: string, pageIdx?: number | null) => Promise<ApiResponse & { labels_path?: string; label?: string }>;
    export_mask_qa_dataset: () => Promise<ApiResponse & { dataset_dir?: string; labels_path?: string; manifest?: string; records?: number }>;
    train_mask_qa_model: () => Promise<ApiResponse & { model_path?: string; log?: string; records?: number }>;
    compare_region_cleanup_candidates: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<CleanupCandidateCompareResponse>;
    apply_region_cleanup_candidate: (idx: number, candidateId: string, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<BootstrapResponse>;
    apply_region_cleanup: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<BootstrapResponse>;
    rerun_region_cleanup: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) => Promise<BootstrapResponse>;
    delete_region_cleanup: (idx: number, pageIdx?: number | null) => Promise<BootstrapResponse>;
    undo:                 ()                              => Promise<BootstrapResponse>;
    list_series_memory:   ()                              => Promise<ApiResponse & { memory: SeriesMemory }>;
    add_series_name:      (krName: string, enName: string, aliasesKr?: string[], note?: string) => Promise<BootstrapResponse>;
    update_series_name:   (id: string, fields: Record<string, unknown>) => Promise<BootstrapResponse>;
    delete_series_name:   (id: string)                    => Promise<BootstrapResponse>;
    add_series_glossary:  (sourceKr: string, targetEn: string, alternativesEn?: string[], aliasesKr?: string[], note?: string) => Promise<BootstrapResponse>;
    update_series_glossary: (id: string, fields: Record<string, unknown>) => Promise<BootstrapResponse>;
    delete_series_glossary: (id: string)                  => Promise<BootstrapResponse>;
    list_sources:         ()                              => Promise<{ ok: boolean; sources: string[]; error?: string }>;
    browse_source_series: (source: string, query?: string) => Promise<{ ok: boolean; error?: string; cards: BrowseCard[] }>;
    select_browse_series: (seriesTitle: string, source: string, sourceId: string, card: BrowseCard) => Promise<SyncResult>;
    get_series_list:      ()                              => Promise<{ ok: boolean; error?: string; series: SeriesSummary[] }>;
    get_series_detail:    (seriesTitle: string)           => Promise<{ ok: boolean; error?: string; detail?: SeriesDetail }>;
    update_series_metadata: (seriesTitle: string, updates: Partial<SeriesDetail>) => Promise<SyncResult>;
    sync_series_metadata: (seriesTitle: string, source?: string, sourceId?: string) => Promise<SyncResult>;
    sync_series_chapters: (seriesTitle: string, mode?: "missing" | "all") => Promise<SyncResult & { synced?: number; total?: number; errors?: string[] }>;
    sync_source_chapter: (seriesTitle: string, chapterSourceId: string) => Promise<SyncResult & { pages_synced?: number; folder?: string; skipped?: boolean }>;
    import_source_chapter: (seriesTitle: string, chapterSourceId: string) => Promise<SyncResult & { bootstrap?: Bootstrap; folder?: string; page_count?: number; opened?: boolean }>;
    delete_series:       (seriesTitle: string, source?: string, sourceId?: string, deleteFiles?: boolean) => Promise<SyncResult>;
    sync_missing_thumbnails: (seriesTitle: string)        => Promise<SyncResult>;
    get_thumbnail_b64:   (url: string, path?: string)     => Promise<ApiResponse & { b64?: string }>;
    translate_series_metadata: (seriesTitle: string)      => Promise<SyncResult & { translated?: Record<string, string> }>;
    retranslate_series:   (seriesTitle: string)           => Promise<SyncResult>;
    get_model_config:     ()                              => Promise<ApiResponse & { config: Record<string, string> }>;
    update_model_config:  (updates: Record<string, string>) => Promise<ApiResponse & { config: Record<string, string> }>;
  };
};

declare global {
  interface Window {
    pywebview?: PywebviewBridge;
  }
}

// ── Dev-mode stub (used when running Vite without the launcher) ──────────────
// Returns sensible empty state so the UI renders.

const EMPTY_BOOTSTRAP: Bootstrap = {
  series:   [],
  chapters: {},
  pages:    [],
  regions:  [],
  issues:   [{ id: "no-page", sev: "info", msg: "Open a chapter to get started.", region: null, page: 0 }],
  memory:   { available: true, series_title: "Local Series", names: [], glossary: [] },
  meta: {
    activeSeriesId:  null,
    activeChapterId: null,
    activePageIdx:   0,
    busy:            false,
    status:          "No backend (dev stub)",
    chapterDir:      "",
    totalPages:      0,
    chapterProgress: 0,
  },
};

function makeStub(): PywebviewBridge["api"] {
  const noop = async (): Promise<BootstrapResponse> => ({
    ok: true,
    ...EMPTY_BOOTSTRAP,
  });
  return {
    get_bootstrap:        noop,
    open_chapter_folder:  async () => ({ ok: true, cancelled: true, ...EMPTY_BOOTSTRAP }),
    import_chapter:       noop,
    go_to_page:           noop,
    run_step:             noop,
    run_all:              noop,
    continue_run_all:     noop,
    run_current_page:     noop,
    detect_all:           noop,
    export_project:       async () => ({ ok: true, export_dir: "" }),
    export_yolo_finetune_dataset: async () => ({ ok: true, dataset_dir: "", manifest: "", pages: 0 }),
    set_yolo_train_class: noop,
    train_yolo_detector:  async () => ({ ok: false, error: "No backend" }),
    get_yolo_training_status: async () => ({ ok: true, status: "idle", running: false }),
    reveal_export_folder: async () => ({ ok: true }),
    get_page_image:       async () => ({ ok: true, b64: null }),
    update_region:        noop,
    update_region_bbox:   noop,
    get_region_preview_sprite: async () => ({ ok: true, b64: null, x: 0, y: 0, w: 0, h: 0 }),
    list_fonts:           async () => ({ ok: true, roles: ["auto", "dialog", "bold", "thought", "sfx"], fonts: [] }),
    add_region:           noop,
    delete_region:        noop,
    ocr_region:           noop,
    translate_region:     noop,
    preview_region_cleanup: async () => ({ ok: true, b64: null, bbox: [], mask_b64: null, mask_bbox: [], plan: {}, debug: {} }),
    propose_cleanup_mask_sam2: async () => ({ ok: false, status: "disabled", error: "No backend" }),
    get_sam2_status:     async () => ({ ok: true, status: "disabled", loaded: false, error: "No backend" }),
    get_region_cleanup_debug: async () => ({ ok: true, boxes: {}, labels: {}, masks: {} }),
    record_mask_qa_label: async () => ({ ok: false, error: "No backend" }),
    export_mask_qa_dataset: async () => ({ ok: true, dataset_dir: "", labels_path: "", records: 0 }),
    train_mask_qa_model: async () => ({ ok: false, error: "No backend" }),
    compare_region_cleanup_candidates: async () => ({ ok: true, candidates: [], recommended_candidate_id: "" }),
    apply_region_cleanup_candidate: noop,
    apply_region_cleanup: noop,
    rerun_region_cleanup: noop,
    delete_region_cleanup: noop,
    undo:                 noop,
    list_series_memory:   async () => ({ ok: true, memory: EMPTY_BOOTSTRAP.memory }),
    add_series_name:      noop,
    update_series_name:   noop,
    delete_series_name:   noop,
    add_series_glossary:  noop,
    update_series_glossary: noop,
    delete_series_glossary: noop,
    list_sources:         async () => ({ ok: true, sources: ["naver-comic"] }),
    browse_source_series: async () => ({ ok: true, cards: [] }),
    select_browse_series: async () => ({ ok: true }),
    get_series_list:      async () => ({ ok: true, series: [] }),
    get_series_detail:    async () => ({ ok: false, error: "No backend" }),
    update_series_metadata: async () => ({ ok: true }),
    sync_series_metadata: async () => ({ ok: true }),
    sync_series_chapters: async () => ({ ok: true, synced: 0, total: 0, errors: [] }),
    sync_source_chapter: async () => ({ ok: false, error: "No backend" }),
    import_source_chapter: async () => ({ ok: false, error: "No backend" }),
    delete_series:       async () => ({ ok: false, error: "No backend" }),
    sync_missing_thumbnails: async () => ({ ok: false, error: "No backend" }),
    get_thumbnail_b64:   async () => ({ ok: false, error: "No backend" }),
    translate_series_metadata: async () => ({ ok: false, error: "No backend" }),
    retranslate_series:   async () => ({ ok: false, error: "No backend" }),
    get_model_config:     async () => ({ ok: true, config: {} }),
    update_model_config:  async () => ({ ok: true, config: {} }),
  };
}

// ── Resolve the bridge ──────────────────────────────────────────────────────
function getBridge() {
  if (window.pywebview?.api) {
    return window.pywebview.api;
  }
  console.warn(
    "[api] window.pywebview not found — using dev stub. "
    + "Start the app with: python launcher.py --dev"
  );
  return makeStub();
}

// ── Public API object ────────────────────────────────────────────────────────
// We wrap each bridge call to:
//   a) lazily resolve the bridge (pywebview may not inject it at module eval time)
//   b) catch unexpected JS exceptions and normalise them to {ok: false, error}

async function call<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, error: msg } as T;
  }
}

const api = {
  /** Fetch the full app state snapshot. */
  getBootstrap: () =>
    call(() => getBridge().get_bootstrap()),

  /** Show a native folder picker and import the selected chapter. */
  openChapterFolder: () =>
    call(() => getBridge().open_chapter_folder()),

  /** Import a chapter from a known path. */
  importChapter: (folder: string) =>
    call(() => getBridge().import_chapter(folder)),

  /** Navigate to a specific page (0-based). */
  goToPage: (idx: number) =>
    call(() => getBridge().go_to_page(idx)),

  /** Run a single pipeline step on the current page. */
  runStep: (step: "detect" | "ocr" | "translate" | "cleanup" | "typeset") =>
    call(() => getBridge().run_step(step)),

  /** Run the full pipeline on the current page. */
  runAll: () =>
    call(() => getBridge().run_all()),

  /** Continue a previously interrupted Run All checkpoint. */
  continueRunAll: () =>
    call(() => getBridge().continue_run_all()),

  /** Run the full pipeline on the current page only. */
  runCurrentPage: () =>
    call(() => getBridge().run_current_page()),

  /** Run detector only across all loaded pages. */
  detectAll: () =>
    call(() => getBridge().detect_all()),

  /** Export the chapter to disk. */
  exportProject: (dir?: string) =>
    call(() => getBridge().export_project(dir)),

  /** Export current YOLO positives and recorded deletion corrections. */
  exportYoloFinetuneDataset: () =>
    call(() => getBridge().export_yolo_finetune_dataset()),

  setYoloTrainClass: (idx: number, classId: number, pageIdx?: number | null) =>
    call(() => getBridge().set_yolo_train_class(idx, classId, pageIdx ?? null)),

  /** Start background YOLO training from the exported dataset. */
  trainYoloDetector: () =>
    call(() => getBridge().train_yolo_detector()),

  /** Read background YOLO training status. */
  getYoloTrainingStatus: () =>
    call(() => getBridge().get_yolo_training_status()),

  /** Open the export folder in Finder / Explorer. */
  revealExportFolder: () =>
    call(() => getBridge().reveal_export_folder()),

  /**
   * Get the best available image for a page as a base64 PNG string.
   * Use as:  <img src={`data:image/png;base64,${b64}`} />
   */
  getPageImage: (idx: number, mode: "best" | "raw" | "cleaned" | "typeset" = "best") =>
    call(() => getBridge().get_page_image(idx, mode)),

  /** Edit a region field and get back updated bootstrap. */
  updateRegion: (idx: number, field: string, value: unknown, pageIdx?: number | null) =>
    call(() => getBridge().update_region(idx, field, value, pageIdx ?? null)),

  updateRegionBBox: (idx: number, x: number, y: number, w: number, h: number, pageIdx?: number | null, editPageIdx?: number | null) =>
    call(() => getBridge().update_region_bbox(idx, x, y, w, h, pageIdx ?? null, editPageIdx ?? null)),

  getRegionPreviewSprite: (idx: number, draft: Record<string, unknown> = {}, pageIdx?: number | null) =>
    call(() => getBridge().get_region_preview_sprite(idx, draft, pageIdx ?? null)),

  listFonts: () =>
    call(() => getBridge().list_fonts()),

  addRegion: (x: number, y: number, w: number, h: number, text = "", pageIdx?: number | null) =>
    call(() => getBridge().add_region(x, y, w, h, text, pageIdx ?? null)),

  deleteRegion: (idx: number, yoloRejectReason = "", pageIdx?: number | null) =>
    call(() => getBridge().delete_region(idx, yoloRejectReason, pageIdx ?? null)),

  ocrRegion: (idx: number, pageIdx?: number | null) =>
    call(() => getBridge().ocr_region(idx, pageIdx ?? null)),

  translateRegion: (idx: number, pageIdx?: number | null) =>
    call(() => getBridge().translate_region(idx, pageIdx ?? null)),

  previewRegionCleanup: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().preview_region_cleanup(idx, manualMask ?? null, pageIdx ?? null)),

  proposeCleanupMaskSam2: (idx: number, prompt?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().propose_cleanup_mask_sam2(idx, prompt ?? null, pageIdx ?? null)),

  getSam2Status: (load = false) =>
    call(() => getBridge().get_sam2_status(load)),

  getRegionCleanupDebug: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().get_region_cleanup_debug(idx, manualMask ?? null, pageIdx ?? null)),

  recordMaskQaLabel: (idx: number, label: string, notes = "", pageIdx?: number | null) =>
    call(() => getBridge().record_mask_qa_label(idx, label, notes, pageIdx ?? null)),

  exportMaskQaDataset: () =>
    call(() => getBridge().export_mask_qa_dataset()),

  trainMaskQaModel: () =>
    call(() => getBridge().train_mask_qa_model()),

  compareRegionCleanupCandidates: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().compare_region_cleanup_candidates(idx, manualMask ?? null, pageIdx ?? null)),

  applyRegionCleanupCandidate: (idx: number, candidateId: string, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().apply_region_cleanup_candidate(idx, candidateId, manualMask ?? null, pageIdx ?? null)),

  applyRegionCleanup: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().apply_region_cleanup(idx, manualMask ?? null, pageIdx ?? null)),

  rerunRegionCleanup: (idx: number, manualMask?: Record<string, unknown> | null, pageIdx?: number | null) =>
    call(() => getBridge().rerun_region_cleanup(idx, manualMask ?? null, pageIdx ?? null)),

  deleteRegionCleanup: (idx: number, pageIdx?: number | null) =>
    call(() => getBridge().delete_region_cleanup(idx, pageIdx ?? null)),

  undo: () =>
    call(() => getBridge().undo()),

  /** Fetch current series-only name/glossary entries. */
  listSeriesMemory: () =>
    call(() => getBridge().list_series_memory()),

  addSeriesName: (krName: string, enName: string, aliasesKr: string[] = [], note = "") =>
    call(() => getBridge().add_series_name(krName, enName, aliasesKr, note)),

  updateSeriesName: (id: string, fields: Record<string, unknown>) =>
    call(() => getBridge().update_series_name(id, fields)),

  deleteSeriesName: (id: string) =>
    call(() => getBridge().delete_series_name(id)),

  addSeriesGlossary: (
    sourceKr: string,
    targetEn: string,
    alternativesEn: string[] = [],
    aliasesKr: string[] = [],
    note = "",
  ) =>
    call(() => getBridge().add_series_glossary(sourceKr, targetEn, alternativesEn, aliasesKr, note)),

  updateSeriesGlossary: (id: string, fields: Record<string, unknown>) =>
    call(() => getBridge().update_series_glossary(id, fields)),

  deleteSeriesGlossary: (id: string) =>
    call(() => getBridge().delete_series_glossary(id)),

  listSources: () =>
    call(() => getBridge().list_sources()),

  browseSourceSeries: (source: string, query = "") =>
    call(() => getBridge().browse_source_series(source, query)),

  selectBrowseSeries: (seriesTitle: string, source: string, sourceId: string, card: BrowseCard) =>
    call(() => getBridge().select_browse_series(seriesTitle, source, sourceId, card)),

  getSeriesList: () =>
    call(() => getBridge().get_series_list()),

  getSeriesDetail: (seriesTitle: string) =>
    call(() => getBridge().get_series_detail(seriesTitle)),

  updateSeriesMetadata: (seriesTitle: string, updates: Partial<SeriesDetail>) =>
    call(() => getBridge().update_series_metadata(seriesTitle, updates)),

  syncSeriesMetadata: (seriesTitle: string, source = "", sourceId = "") =>
    call(() => getBridge().sync_series_metadata(seriesTitle, source, sourceId)),

  syncSeriesChapters: (seriesTitle: string, mode: "missing" | "all" = "missing") =>
    call(() => getBridge().sync_series_chapters(seriesTitle, mode)),

  syncSourceChapter: (seriesTitle: string, chapterSourceId: string) =>
    call(() => getBridge().sync_source_chapter(seriesTitle, chapterSourceId)),

  importSourceChapter: (seriesTitle: string, chapterSourceId: string) =>
    call(() => getBridge().import_source_chapter(seriesTitle, chapterSourceId)),

  deleteSeries: (seriesTitle: string, source = "", sourceId = "", deleteFiles = false) =>
    call(() => getBridge().delete_series(seriesTitle, source, sourceId, deleteFiles)),

  syncMissingThumbnails: (seriesTitle: string) =>
    call(() => getBridge().sync_missing_thumbnails(seriesTitle)),

  getThumbnailB64: (url: string, path = "") =>
    call(() => getBridge().get_thumbnail_b64(url, path)),

  translateSeriesMetadata: (seriesTitle: string) =>
    call(() => getBridge().translate_series_metadata(seriesTitle)),

  retranslateSeries: (seriesTitle: string) =>
    call(() => getBridge().retranslate_series(seriesTitle)),

  /** Fetch model config (OCR model name, etc.). */
  getModelConfig: () =>
    call(() => getBridge().get_model_config()),

  /** Save new model config values. */
  updateModelConfig: (updates: Record<string, string>) =>
    call(() => getBridge().update_model_config(updates)),
};

export default api;

// ── Event listener helpers ────────────────────────────────────────────────────
// Python pushes progress by calling window.evaluate_js which dispatches
// CustomEvents.  Use these helpers to subscribe.

export function onProgress(
  handler: (e: ProgressEvent) => void
): () => void {
  const listener = (ev: Event) => handler((ev as CustomEvent<ProgressEvent>).detail);
  window.addEventListener("ml:progress", listener);
  return () => window.removeEventListener("ml:progress", listener);
}

export function onBusyChange(
  handler: (busy: boolean) => void
): () => void {
  const listener = (ev: Event) =>
    handler(((ev as CustomEvent<{ busy: boolean }>).detail).busy);
  window.addEventListener("ml:busy", listener);
  return () => window.removeEventListener("ml:busy", listener);
}
