/**
 * frontend/src/App.tsx
 *
 * What changed from ManhwaLocalizer.jsx:
 *  1. Converted to TypeScript with proper types from ./types
 *  2. getBootstrapData() removed — replaced with api.getBootstrap() (async)
 *  3. globalThis.__ML_ACTIONS__ removed — replaced with api.runStep() etc.
 *  4. DEFAULT_MOCK removed — empty bootstrap shown until a chapter is opened
 *  5. Canvas now loads real images via api.getPageImage(idx)
 *  6. Progress events from Python drive the status bar and step indicators
 *  7. "Open chapter" button wired to api.openChapterFolder()
 *  8. All CSS kept identical — only logic changed
 *  9. Status bar shows real live status from Python (no hardcoded 45%)
 * 10. Inspector edits call api.updateRegion() instead of being ignored
 */

import { Fragment, useState, useEffect, useRef, useCallback } from "react";
import type {
  Bootstrap,
  Region,
  Issue,
  ProgressEvent,
  NameMemoryEntry,
  GlossaryEntry,
  RegionPreviewSprite,
  CleanupPreviewResponse,
  CleanupDebugResponse,
  CleanupCandidateCompareResponse,
  CleanupCandidate,
  Sam2MaskResponse,
  BrowseCard,
  Series,
} from "./types";
import api, { onProgress, onBusyChange } from "./api";
import { BrowseModal } from "./components/BrowseModal";
import { SeriesDetailPanel } from "./components/SeriesDetailPanel";

/* ─── Global CSS (identical to the original JSX) ─────────────────────────── */
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Figtree:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-0:  #08080d; --bg-1:  #0c0c14; --bg-2:  #10101a;
    --bg-3:  #151520; --bg-4:  #1a1a28; --bg-5:  #1f1f30;
    --br-0:  #181825; --br-1:  #202035; --br-2:  #2a2a45;
    --accent:     #c9a55a; --accent-dim: rgba(201,165,90,0.14);
    --accent-glo: rgba(201,165,90,0.06);
    --teal:   #4ec9b4; --teal-d: rgba(78,201,180,0.13);
    --red:    #e05a6a; --red-d:  rgba(224,90,106,0.13);
    --grn:    #56c87a; --grn-d:  rgba(86,200,122,0.12);
    --amr:    #d4884a; --amr-d:  rgba(212,136,74,0.13);
    --blue:   #6090e8;
    --t1: #e8e8f6; --t2: #9898c0; --t3: #52527a; --t4: #32324e;
    --r: 3px; --r2: 5px; --r3: 7px;
    --fnt-disp: 'Syne', sans-serif;
    --fnt-mono: 'IBM Plex Mono', monospace;
    --fnt-body: 'Figtree', sans-serif;
    --panel-w-l: 220px; --panel-w-r: 380px;
    --topbar-h: 46px; --statusbar-h: 26px;
  }

  html, body, #root {
    height: 100%; background: var(--bg-0); color: var(--t1);
    font-family: var(--fnt-body); font-size: 13px; line-height: 1.45;
    -webkit-font-smoothing: antialiased; overflow: hidden;
  }

  ::-webkit-scrollbar { width: 3px; height: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--br-1); border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--br-2); }
  ::selection { background: var(--accent-dim); }

  .ml-topbar {
    position: fixed; top: 0; left: 0; right: 0; height: var(--topbar-h);
    background: var(--bg-1); border-bottom: 1px solid var(--br-0);
    display: flex; align-items: center; gap: 0; z-index: 100;
  }
  .ml-logo {
    width: var(--panel-w-l); flex-shrink: 0;
    display: flex; align-items: center; gap: 9px; padding: 0 14px;
    border-right: 1px solid var(--br-0); height: 100%;
  }
  .ml-logo-mark { width: 24px; height: 24px; background: var(--accent); border-radius: 4px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .ml-logo-text { font-family: var(--fnt-disp); font-weight: 700; font-size: 13px; color: var(--t1); letter-spacing: -0.01em; }
  .ml-logo-sub  { font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); letter-spacing: 0.05em; text-transform: uppercase; }

  .ml-pipeline { display: flex; align-items: center; gap: 2px; padding: 0 14px; border-right: 1px solid var(--br-0); height: 100%; }

  .pip-step { display: flex; align-items: center; gap: 5px; padding: 4px 9px; border-radius: var(--r2); border: 1px solid transparent; cursor: pointer; font-family: var(--fnt-mono); font-size: 10px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: var(--t3); transition: all 0.12s; }
  .pip-step:hover { color: var(--t2); background: var(--bg-4); }
  .pip-step.done    { color: var(--teal); }
  .pip-step.done .pip-dot { background: var(--teal); }
  .pip-step.active  { color: var(--accent); background: var(--accent-glo); border-color: var(--br-1); }
  .pip-step.active .pip-dot { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
  .pip-step.running { color: var(--amr); border-color: var(--br-1); animation: pip-pulse 1.5s ease-in-out infinite; }
  .pip-step.running .pip-dot { background: var(--amr); }
  .pip-step.error   { color: var(--red); }
  .pip-step.error .pip-dot { background: var(--red); }
  @keyframes pip-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
  .pip-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--t4); flex-shrink: 0; transition: background 0.12s, box-shadow 0.12s; }
  .pip-arrow { font-family: var(--fnt-mono); font-size: 10px; color: var(--t4); padding: 0 1px; user-select: none; }

  .ml-actions { display: flex; align-items: center; gap: 6px; padding: 0 14px; margin-left: auto; }

  .btn-primary { display: flex; align-items: center; gap: 5px; padding: 5px 12px; background: var(--accent); color: var(--bg-0); border: none; border-radius: var(--r2); font-family: var(--fnt-mono); font-size: 10px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; cursor: pointer; transition: opacity 0.12s; }
  .btn-primary:hover { opacity: 0.85; }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-ghost { display: flex; align-items: center; gap: 5px; padding: 5px 10px; background: transparent; color: var(--t2); border: 1px solid var(--br-1); border-radius: var(--r2); font-family: var(--fnt-mono); font-size: 10px; font-weight: 400; letter-spacing: 0.04em; text-transform: uppercase; cursor: pointer; transition: all 0.12s; }
  .btn-ghost:hover { color: var(--t1); border-color: var(--br-2); background: var(--bg-4); }
  .btn-ghost.active { color: var(--accent); border-color: var(--br-2); background: var(--accent-dim); }
  .btn-ghost:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-icon { width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; background: transparent; border: 1px solid transparent; border-radius: var(--r2); color: var(--t3); cursor: pointer; transition: all 0.12s; flex-shrink: 0; }
  .btn-icon:hover { background: var(--bg-4); color: var(--t2); border-color: var(--br-1); }
  .btn-icon.on { background: var(--accent-dim); color: var(--accent); }

  .ml-body { position: fixed; top: var(--topbar-h); bottom: var(--statusbar-h); left: 0; right: 0; display: flex; }

  .ml-left { width: var(--panel-w-l); flex-shrink: 0; background: var(--bg-1); border-right: 1px solid var(--br-0); display: flex; flex-direction: column; overflow: hidden; transition: width 0.18s ease; position: relative; min-width: 0; }
  .ml-left.collapsed { width: 0; }
  .left-resize-handle { position: absolute; top: 0; right: -3px; bottom: 0; width: 6px; cursor: col-resize; z-index: 60; }
  .left-resize-handle::after { content: ""; position: absolute; top: 0; bottom: 0; left: 2px; width: 1px; background: rgba(255,255,255,0.06); }
  .left-resize-handle:hover::after { background: var(--accent); opacity: 0.8; }

  .ml-canvas-wrap { flex: 1; background: var(--bg-0); display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
  .ml-canvas-toolbar { height: 36px; flex-shrink: 0; background: var(--bg-1); border-bottom: 1px solid var(--br-0); display: flex; align-items: center; padding: 0 10px; gap: 4px; }
  .ml-canvas-area { flex: 1; overflow: auto; display: flex; align-items: center; justify-content: center; position: relative; padding: 32px; }

  .ml-right { width: var(--panel-w-r); flex-shrink: 0; background: var(--bg-1); border-left: 1px solid var(--br-0); display: flex; flex-direction: column; overflow: hidden; transition: width 0.18s ease; position: relative; }
  .ml-right.collapsed { width: 0; }
  .right-resize-handle { position: absolute; top: 0; left: -3px; bottom: 0; width: 7px; cursor: ew-resize; z-index: 60; }
  .right-resize-handle::after { content: ""; position: absolute; top: 0; bottom: 0; left: 3px; width: 1px; background: rgba(255,255,255,0.06); }
  .right-resize-handle:hover::after { background: var(--accent); opacity: 0.8; }

  .ml-statusbar { position: fixed; bottom: 0; left: 0; right: 0; height: var(--statusbar-h); background: var(--bg-1); border-top: 1px solid var(--br-0); display: flex; align-items: center; padding: 0 14px; gap: 20px; z-index: 100; }
  .sb-item { display: flex; align-items: center; gap: 5px; font-family: var(--fnt-mono); font-size: 10px; color: var(--t3); letter-spacing: 0.04em; }
  .sb-item span { color: var(--t2); }
  .sb-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--grn); flex-shrink: 0; }
  .sb-dot.busy { background: var(--amr); animation: pip-pulse 1.2s infinite; }
  .sb-progress { flex: 1; max-width: 80px; height: 2px; background: var(--bg-4); border-radius: 1px; overflow: hidden; }
  .sb-progress-fill { height: 100%; background: var(--accent); border-radius: 1px; transition: width 0.3s; }

  .left-scroll-body { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; overflow-y: auto; overflow-x: hidden; }
  .pnl-section { border-bottom: 1px solid var(--br-0); flex: 0 0 auto; }
  .pnl-header { display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; cursor: pointer; user-select: none; height: 30px; }
  .pnl-header:hover { background: var(--bg-3); }
  .pnl-title { font-family: var(--fnt-mono); font-size: 9.5px; font-weight: 500; letter-spacing: 0.09em; text-transform: uppercase; color: var(--t3); display: flex; align-items: center; gap: 6px; }
  .pnl-title.lit { color: var(--t2); }
  .pnl-body { padding: 8px 0; overflow: hidden; }
  .pnl-body.scroll { overflow-y: auto; }

  .series-row { display: flex; align-items: center; gap: 8px; padding: 5px 12px; cursor: pointer; border-left: 2px solid transparent; transition: all 0.1s; }
  .series-row:hover { background: var(--bg-3); }
  .series-row.active { background: var(--bg-4); border-left-color: var(--accent); }
  .series-thumb { width: 28px; height: 36px; border-radius: var(--r); flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 9px; font-family: var(--fnt-mono); font-weight: 500; overflow: hidden; }
  .series-thumb.has-cover { background: var(--bg-0); border: 1px solid var(--br-1); cursor: zoom-in; }
  .series-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .series-info { min-width: 0; }
  .series-title { font-size: 12px; font-weight: 600; color: var(--t1); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .series-meta { font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); }

  .chapter-row { display: flex; align-items: center; justify-content: space-between; padding: 4px 12px 4px 20px; cursor: pointer; transition: background 0.1s; }
  .chapter-row:hover { background: var(--bg-3); }
  .chapter-row.active { background: var(--bg-4); }
  .chapter-row.active .chapter-name { color: var(--accent); }
  .chapter-name { font-size: 12px; color: var(--t2); }
  .prog-bar { width: 36px; height: 2px; background: var(--bg-5); border-radius: 1px; }
  .prog-fill { height: 100%; border-radius: 1px; background: var(--teal); }

  .page-strip { display: flex; flex-direction: column; gap: 4px; padding: 8px; overflow: visible; flex: 0 0 auto; }
  .page-thumb-item { display: flex; align-items: center; gap: 7px; padding: 3px 4px; border-radius: var(--r2); cursor: pointer; transition: background 0.1s; flex-shrink: 0; }
  .page-thumb-item:hover { background: var(--bg-3); }
  .page-thumb-item.active { background: var(--bg-5); }
  .page-thumb { width: 34px; height: 48px; border-radius: var(--r); flex-shrink: 0; border: 1.5px solid var(--br-1); overflow: hidden; background: var(--bg-0); transition: border-color 0.1s; }
  .page-thumb-item.active .page-thumb { border-color: var(--accent); }
  .page-num { font-family: var(--fnt-mono); font-size: 10px; color: var(--t3); }
  .page-status-dots { display: flex; gap: 2px; margin-top: 2px; }
  .psd { width: 5px; height: 5px; border-radius: 50%; background: var(--bg-5); }
  .psd.on { background: var(--teal); }
  .page-run-badge { margin-top: 3px; font-family: var(--fnt-mono); font-size: 8px; color: var(--amr); text-transform: uppercase; }
  .page-run-badge.error { color: var(--red); }

  .manga-page { background: #f5f4f0; box-shadow: 0 8px 60px rgba(0,0,0,0.7), 0 2px 12px rgba(0,0,0,0.5); position: relative; user-select: none; border-radius: 1px; touch-action: none; }
  .continuous-reader { width: 100%; height: 100%; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; align-items: center; gap: 0; padding: 0 8px 18px; scroll-snap-type: none; }
  .continuous-page { position: relative; background: #f5f4f0; box-shadow: 0 2px 12px rgba(0,0,0,0.42); border: 0; cursor: pointer; scroll-snap-align: none; }
  .continuous-page.active { z-index: 2; outline: 1px solid rgba(201,165,90,0.45); box-shadow: 0 0 0 1px rgba(201,165,90,0.16), 0 2px 12px rgba(0,0,0,0.42); }
  .continuous-page img { display: block; width: 100%; height: auto; }
  .continuous-page .region-overlay { z-index: 4; }
  .continuous-placeholder { width: min(92%, 760px); height: 72vh; display: flex; align-items: center; justify-content: center; background: var(--bg-3); color: var(--t4); font-family: var(--fnt-mono); font-size: 11px; }
  .page-indicator-float { position: sticky; top: 12px; align-self: flex-end; margin-right: 18px; z-index: 15; border: 1px solid var(--br-2); border-radius: 999px; background: rgba(12,12,20,0.9); color: var(--t1); font-family: var(--fnt-mono); font-size: 11px; padding: 6px 10px; cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,0.35); }
  .page-start-marker { position: absolute; top: 6px; left: 6px; z-index: 3; pointer-events: none; border: 1px solid rgba(255,255,255,0.18); border-radius: 999px; background: rgba(12,12,20,0.72); color: rgba(255,255,255,0.86); font-family: var(--fnt-mono); font-size: 10px; padding: 3px 7px; box-shadow: 0 2px 8px rgba(0,0,0,0.25); }
  .region-overlay { position: absolute; cursor: move; border: 1.5px solid rgba(78,201,180,0.6); border-radius: 2px; background: rgba(78,201,180,0.04); transition: background 0.1s, border-color 0.1s, box-shadow 0.1s; touch-action: none; }
  .region-overlay:hover { border-color: var(--teal); background: rgba(78,201,180,0.1); }
  .region-overlay.sel { border-color: var(--accent); background: rgba(201,165,90,0.1); box-shadow: 0 0 0 1px var(--accent-dim); }
  .region-overlay.dragging { cursor: grabbing; transition: none; }
  .alignment-guide { position: absolute; pointer-events: none; z-index: 11; opacity: 0.82; }
  .alignment-guide.v { top: 0; height: 100%; width: 0; border-left: 1px dashed rgba(245,216,76,0.82); }
  .alignment-guide.h { left: 0; width: 100%; height: 0; border-top: 1px dashed rgba(245,216,76,0.82); }
  .alignment-guide.container { border-color: rgba(78,201,180,0.78); }
  .alignment-guide.selected { border-color: rgba(255,255,255,0.58); border-style: dotted; }
  .alignment-guide.near { border-color: rgba(120,160,255,0.58); opacity: 0.56; }
  .debug-box-overlay { position: absolute; border: 2px solid currentColor; background: transparent; pointer-events: none; z-index: 8; box-shadow: 0 0 0 1px rgba(0,0,0,0.38); }
  .debug-box-label { position: absolute; left: -2px; top: -18px; padding: 2px 5px; border-radius: 2px; background: rgba(8,8,13,0.86); color: currentColor; font-family: var(--fnt-mono); font-size: 9px; white-space: nowrap; }
  .debug-mask-overlay { position: absolute; pointer-events: none; z-index: 7; mix-blend-mode: screen; image-rendering: pixelated; }
  .region-label { position: absolute; top: -18px; left: -1px; font-family: var(--fnt-mono); font-size: 9px; font-weight: 500; padding: 2px 5px; border-radius: 3px 3px 3px 0; pointer-events: none; white-space: nowrap; }
  .region-label.teal { background: var(--teal); color: var(--bg-0); }
  .region-label.gold { background: var(--accent); color: var(--bg-0); }
  .resize-handle { position: absolute; width: 10px; height: 10px; border: 1px solid var(--bg-0); background: var(--accent); box-shadow: 0 0 0 1px rgba(201,165,90,0.45); border-radius: 2px; z-index: 2; touch-action: none; }
  .resize-handle.nw { left: -6px; top: -6px; cursor: nwse-resize; }
  .resize-handle.ne { right: -6px; top: -6px; cursor: nesw-resize; }
  .resize-handle.sw { left: -6px; bottom: -6px; cursor: nesw-resize; }
  .resize-handle.se { right: -6px; bottom: -6px; cursor: nwse-resize; }
  .drag-cleaned-patch { position: absolute; overflow: hidden; pointer-events: none; }
  .drag-cleaned-patch img { position: absolute; max-width: none; }
  .translation-bitmap-preview { position: absolute; pointer-events: none; user-select: none; max-width: none; z-index: 1; }
  .translation-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; padding: 5%; pointer-events: none; white-space: pre-wrap; overflow: hidden; overflow-wrap: anywhere; line-height: 1.18; opacity: 0.78; text-shadow: 0 1px 1px rgba(255,255,255,0.8); }
  .canvas-toggle { margin-left: auto; display: flex; align-items: center; gap: 5px; font-family: var(--fnt-mono); font-size: 10px; color: var(--t2); cursor: pointer; }
  .canvas-toggle input { accent-color: var(--accent); }

  .ml-canvas-toolbar .divider-v { width: 1px; height: 18px; background: var(--br-1); flex-shrink: 0; margin: 0 4px; }
  .zoom-display { font-family: var(--fnt-mono); font-size: 10px; color: var(--t2); padding: 0 6px; min-width: 44px; text-align: center; }

  .tab-bar { display: flex; border-bottom: 1px solid var(--br-0); flex-shrink: 0; }
  .tab { flex: 1; padding: 8px 4px; font-family: var(--fnt-mono); font-size: 9.5px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--t3); cursor: pointer; text-align: center; border-bottom: 2px solid transparent; transition: all 0.12s; }
  .tab:hover { color: var(--t2); background: var(--bg-3); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .insp-body { padding: 0; overflow-y: auto; flex: 1; }
  .insp-section { border-bottom: 1px solid var(--br-0); padding: 10px 12px; }
  .insp-section-title { font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
  .editor-section { border-bottom: 1px solid var(--br-0); }
  .editor-section > summary { list-style: none; cursor: pointer; padding: 10px 12px; font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); letter-spacing: 0.08em; text-transform: uppercase; user-select: none; display: flex; align-items: center; justify-content: space-between; }
  .editor-section > summary::-webkit-details-marker { display: none; }
  .editor-section > summary::after { content: "›"; color: var(--t4); transform: rotate(90deg); transition: transform 0.12s; }
  .editor-section:not([open]) > summary::after { transform: rotate(0deg); }
  .editor-section > summary:hover { background: var(--bg-3); color: var(--t2); }
  .editor-section-body { padding: 0 12px 12px; }
  .style-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .style-row .prop-val { width: auto; min-width: 72px; }
  .raw-match-card { border: 1px solid var(--br-1); background: var(--bg-2); border-radius: var(--r2); padding: 8px; display: grid; gap: 7px; }
  .raw-match-summary { color: var(--t1); font-size: 12px; }
  .raw-match-meta { color: var(--t3); font-family: var(--fnt-mono); font-size: 9px; line-height: 1.45; overflow-wrap: anywhere; }
  .topbar-overflow { position: relative; }
  .topbar-overflow summary { list-style: none; }
  .topbar-overflow summary::-webkit-details-marker { display: none; }
  .topbar-menu { position: absolute; right: 0; top: 32px; z-index: 150; min-width: 190px; border: 1px solid var(--br-2); background: var(--bg-2); box-shadow: 0 12px 40px rgba(0,0,0,0.45); padding: 6px; display: grid; gap: 5px; }
  .topbar-menu .btn-ghost { justify-content: flex-start; width: 100%; }
  .prop-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .prop-cell { display: flex; flex-direction: column; gap: 3px; }
  .prop-label { font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); letter-spacing: 0.04em; }
  .prop-val { padding: 4px 7px; background: var(--bg-0); border: 1px solid var(--br-1); border-radius: var(--r); color: var(--t1); font-family: var(--fnt-mono); font-size: 11px; outline: none; width: 100%; transition: border-color 0.12s; }
  .prop-val:focus { border-color: var(--accent); }
  .cleanup-candidate-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(116px, 1fr)); gap: 7px; }
  .cleanup-candidate { border: 1px solid var(--br-1); background: var(--bg-2); border-radius: var(--r2); padding: 6px; cursor: pointer; display: grid; gap: 5px; min-width: 0; }
  .cleanup-candidate:hover { border-color: var(--br-2); background: var(--bg-3); }
  .cleanup-candidate.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent-dim); }
  .cleanup-candidate.off { opacity: 0.55; cursor: not-allowed; }
  .cleanup-candidate img { width: 100%; height: 86px; object-fit: contain; background: var(--bg-0); border: 1px solid var(--br-0); }
  .cleanup-candidate-title { font-family: var(--fnt-mono); font-size: 10px; color: var(--t1); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cleanup-candidate-meta { font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); line-height: 1.35; overflow-wrap: anywhere; }
  .cleanup-candidate-warn { color: var(--amr); font-family: var(--fnt-mono); font-size: 9px; line-height: 1.35; }
  .det-badge { display: inline-flex; align-items: center; height: 18px; padding: 0 6px; border-radius: var(--r); border: 1px solid var(--br-1); color: var(--teal); background: var(--teal-d); font-family: var(--fnt-mono); font-size: 9px; text-transform: uppercase; }
  .det-badge.manual { color: var(--accent); background: var(--accent-dim); }
  .det-badge.yolo { color: var(--blue); background: rgba(96,144,232,0.13); }
  .text-area { width: 100%; padding: 6px 8px; min-height: 60px; background: var(--bg-0); border: 1px solid var(--br-1); border-radius: var(--r2); color: var(--t1); font-family: var(--fnt-body); font-size: 12px; line-height: 1.5; outline: none; resize: vertical; transition: border-color 0.12s; }
  .text-area:focus { border-color: var(--accent); }
  .text-area.source { color: var(--t2); font-size: 11.5px; }

  .layer-row { display: flex; align-items: center; gap: 6px; padding: 5px 12px; cursor: pointer; transition: background 0.1s; }
  .layer-row:hover { background: var(--bg-3); }
  .layer-row.sel { background: var(--bg-4); }
  .layer-row.off { opacity: 0.45; }
  .layer-icon { width: 18px; height: 18px; border-radius: var(--r); display: flex; align-items: center; justify-content: center; font-size: 9px; font-family: var(--fnt-mono); flex-shrink: 0; font-weight: 500; }
  .li-region  { background: var(--blue);   color: var(--bg-0); opacity: 0.85; }
  .li-text    { background: var(--accent);  color: var(--bg-0); opacity: 0.9; }
  .li-cleanup { background: var(--teal);    color: var(--bg-0); opacity: 0.8; }
  .layer-name { font-size: 12px; color: var(--t2); flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .layer-row.sel .layer-name { color: var(--t1); }
  .layer-actions { display: flex; gap: 4px; }
  .layer-action { width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; border: 1px solid var(--br-1); border-radius: var(--r); background: var(--bg-2); color: var(--t3); cursor: pointer; }
  .layer-action.on { color: var(--accent); border-color: var(--br-2); }
  .layer-action:hover { color: var(--t1); }

  .issue-item { padding: 8px 12px; border-left: 2px solid var(--br-1); margin: 2px 6px; border-radius: 0 var(--r) var(--r) 0; cursor: pointer; transition: all 0.1s; }
  .issue-item:hover { background: var(--bg-3); }
  .issue-item.err  { border-left-color: var(--red); }
  .issue-item.warn { border-left-color: var(--amr); }
  .issue-item.info { border-left-color: var(--teal); }
  .issue-msg { font-size: 11.5px; color: var(--t2); line-height: 1.4; }
  .issue-ref { font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); margin-top: 3px; }

  .mem-list { display: flex; flex-direction: column; gap: 6px; padding: 0 10px 10px; }
  .mem-card { border: 1px solid var(--br-1); border-radius: var(--r2); background: var(--bg-2); padding: 7px; }
  .mem-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
  .mem-main { min-width: 0; flex: 1; }
  .mem-term { font-size: 12px; color: var(--t1); overflow-wrap: anywhere; }
  .mem-meta { margin-top: 2px; font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); overflow-wrap: anywhere; }
  .mem-actions { display: flex; gap: 4px; flex-shrink: 0; }
  .mem-mini { border: 1px solid var(--br-1); background: var(--bg-4); color: var(--t2); border-radius: var(--r); font-family: var(--fnt-mono); font-size: 9px; padding: 2px 5px; cursor: pointer; }
  .mem-mini:hover { color: var(--t1); border-color: var(--br-2); }
  .mem-mini.danger:hover { color: var(--red); }
  .mem-form { display: grid; gap: 5px; padding: 8px 10px 10px; border-bottom: 1px solid var(--br-0); }
  .mem-input { width: 100%; background: var(--bg-0); border: 1px solid var(--br-1); border-radius: var(--r); color: var(--t1); font-family: var(--fnt-body); font-size: 11px; padding: 5px 6px; outline: none; }
  .mem-input:focus { border-color: var(--accent); }
  .mem-two { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }

  .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 24px 12px; color: var(--t4); font-size: 11px; text-align: center; }

  .kbd { display: inline-flex; align-items: center; padding: 1px 5px; background: var(--bg-4); border: 1px solid var(--br-2); border-radius: 3px; font-family: var(--fnt-mono); font-size: 9px; color: var(--t3); }
  .fade-in { animation: fadeIn 0.15s ease; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(3px); } to { opacity:1; transform:none; } }
  .no-select { user-select: none; }
  .divider-v { width: 1px; height: 18px; background: var(--br-1); flex-shrink: 0; margin: 0 4px; }

  .open-chapter-prompt { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; height: 100%; color: var(--t3); }
  .open-chapter-prompt h2 { font-family: var(--fnt-disp); font-size: 18px; color: var(--t2); }
  .open-chapter-prompt p  { font-size: 12px; max-width: 260px; text-align: center; line-height: 1.6; }
  .toast { position: fixed; left: 50%; bottom: 38px; transform: translateX(-50%); z-index: 200; padding: 7px 12px; border: 1px solid var(--br-2); border-radius: var(--r2); background: var(--bg-2); color: var(--t1); font-family: var(--fnt-mono); font-size: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.45); }
  .cover-lightbox { position: fixed; inset: 0; z-index: 380; background: rgba(0,0,0,0.72); display: flex; align-items: center; justify-content: center; padding: 32px; }
  .cover-card { width: min(420px, 92vw); background: var(--bg-2); border: 1px solid var(--br-2); border-radius: var(--r3); overflow: hidden; box-shadow: 0 18px 60px rgba(0,0,0,0.55); }
  .cover-card img { width: 100%; max-height: 70vh; object-fit: contain; display: block; background: var(--bg-0); }
  .cover-info { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; }
  .cover-title { font-weight: 700; color: var(--t1); overflow-wrap: anywhere; }
  .cover-source { font-family: var(--fnt-mono); font-size: 10px; color: var(--t3); }
  .modal-backdrop { position: fixed; inset: 0; z-index: 400; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.58); }
  .settings-modal { width: min(620px, calc(100vw - 32px)); max-height: calc(100vh - 64px); overflow: auto; background: var(--bg-1); border: 1px solid var(--br-2); box-shadow: 0 18px 80px rgba(0,0,0,0.55); }
  .settings-head { height: 42px; display: flex; align-items: center; justify-content: space-between; padding: 0 14px; border-bottom: 1px solid var(--br-0); }
  .settings-title { font-family: var(--fnt-mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--t2); }
  .settings-body { padding: 14px; display: grid; gap: 14px; }
  .settings-grid { display: grid; grid-template-columns: 150px 1fr; gap: 8px 10px; align-items: center; }
  .settings-grid label, .settings-note { font-family: var(--fnt-mono); font-size: 10px; color: var(--t3); }
  .settings-input { background: var(--bg-3); color: var(--t1); border: 1px solid var(--br-1); padding: 6px 8px; font: 11px var(--fnt-mono); min-width: 0; }
  .settings-warning { color: var(--amr); font-family: var(--fnt-mono); font-size: 10px; }
`;

/* ── Icons ─────────────────────────────────────────────────────────────────── */
const ICONS: Record<string, string> = {
  chevDown:  "M6 9l6 6 6-6",
  chevRight: "M9 18l6-6-6-6",
  eye:       "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
  lock:      "M19 11H5a2 2 0 00-2 2v7a2 2 0 002 2h14a2 2 0 002-2v-7a2 2 0 00-2-2zM7 11V7a5 5 0 0110 0v4",
  zoomIn:    "M11 8v6M8 11h6M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z",
  zoomOut:   "M8 11h6M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z",
  fit:       "M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3",
  run:       "M5 3l14 9-14 9V3z",
  export:    "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12",
  folder:    "M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z",
  collapse:  "M4 6h16M4 12h16M4 18h16",
  prev:      "M15 18l-6-6 6-6",
  next:      "M9 18l6-6-6-6",
  issues:    "M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01",
  insp:      "M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18",
  layers:    "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
};

const Svg = ({ icon, size = 14 }: { icon: string; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d={ICONS[icon]} />
  </svg>
);

const PIPELINE_STEPS = ["Detect", "OCR", "Translate", "Cleanup", "Typeset"] as const;
type StepState = "idle" | "done" | "active" | "running" | "error";
type ImageMode = "best" | "raw" | "cleaned" | "typeset";
type PageSelectMode = "immediate" | "debounced" | "local";
type RegionBBox = Pick<Region, "x" | "y" | "w" | "h">;
type RegionDraft = Partial<Pick<Region, "x" | "y" | "w" | "h" | "src" | "tl" | "font" | "size" | "align" | "fg" | "bg" | "outline" | "outline_width" | "shadow" | "shadow_on" | "shadow_offset_x" | "shadow_offset_y" | "shadow_opacity" | "shadow_blur" | "glow" | "glow_on" | "glow_radius" | "glow_intensity" | "reflection_on" | "reflection_opacity" | "reflection_offset" | "reflection_blur" | "reflection_fade" | "gradient_on" | "gradient_start" | "gradient_end" | "gradient_angle" | "rotation_angle" | "visible" | "locked">>;
type CleanupMaskPayload = { b64: string; bbox: number[] } | null;
type Sam2UiMode = "cleanup" | "container" | "protect";
type Sam2MergeMode = "replace" | "add" | "subtract";
type DebugOverlayKey =
  | "yoloBox"
  | "editableBox"
  | "containerBox"
  | "textMask"
  | "cleanupMask"
  | "haloMask"
  | "manualMask"
  | "patchBox"
  | "groupedMask";
type DebugOverlayToggles = Record<DebugOverlayKey, boolean>;
type FontOptions = { roles: string[]; fonts: string[] };
type ResizeHandle = "nw" | "ne" | "sw" | "se";
const DEBUG_OVERLAY_DEFAULTS: DebugOverlayToggles = {
  yoloBox: false,
  editableBox: false,
  containerBox: false,
  textMask: false,
  cleanupMask: false,
  haloMask: false,
  manualMask: false,
  patchBox: false,
  groupedMask: false,
};
const PAGE_IMAGE_CACHE = new Map<string, string>();
const invalidatePageImageCache = (chapterDir: string, idx: number) => {
  const prefixes = [`${chapterDir}:page:${idx}:`, `${chapterDir}:thumb:${idx}:`];
  for (const key of Array.from(PAGE_IMAGE_CACHE.keys())) {
    if (prefixes.some(prefix => key.startsWith(prefix))) PAGE_IMAGE_CACHE.delete(key);
  }
};
type FrozenPreviewStyle = {
  styleKey: string;
  color: string;
  fontFamily: string;
  fontSize: number;
  textAlign: Region["align"];
  textShadow: string;
};
type BackendPreviewSprite = RegionPreviewSprite & {
  styleKey: string;
  sourceBox: RegionBBox;
  reason: string;
};
 const MODE_BADGE_LABELS: Record<ImageMode, string> = {
   best: "BEST",
   raw: "RAW",
   cleaned: "CLEANED",
   typeset: "TYPESET",
 };

 function ModeBadge({ mode }: { mode: ImageMode }) {
   return (
     <span
       title={`Base layer: ${MODE_BADGE_LABELS[mode]}`}
       style={{
         fontSize: 10,
         fontWeight: 700,
         letterSpacing: "0.08em",
         padding: "2px 6px",
         borderRadius: 999,
         border: "1px solid var(--line)",
         color: mode === "raw" ? "var(--amr)" : "var(--t2)",
         background: "var(--panel)",
         userSelect: "none",
         flexShrink: 0,
       }}
     >
       {MODE_BADGE_LABELS[mode]}
     </span>
   );
 }
const EDITOR_DEBUG = (() => {
  try { return localStorage.getItem("manhwa.renderDebug") === "1"; }
  catch { return false; }
})();
const renderDebug = (action: string, payload: Record<string, unknown>) => {
  if (EDITOR_DEBUG) console.debug("[renderDebug]", action, payload);
};
const DEFAULT_FONT_ROLES = ["auto", "dialog", "bold", "thought", "sfx"];
// Role labels emitted by the backend when no explicit font is set.
// These must NEVER be forwarded to CSS font-family — the browser will
// silently not find them and fall back to Comic Sans.
const ROLE_FONT_NAMES = new Set(["auto", "dialog", "dialogue", "bold", "thought", "sfx"]);
const regionStyleDebug = (r: Region | (Region & RegionDraft)) => ({
  bbox: { x: r.x, y: r.y, w: r.w, h: r.h },
  font: r.font,
  size: r.size,
  fg: r.fg,
  bg: r.bg,
  outline: r.outline,
  outline_width: r.outline_width,
  shadow: r.shadow,
  shadow_on: r.shadow_on,
  shadow_offset_x: r.shadow_offset_x,
  shadow_offset_y: r.shadow_offset_y,
  shadow_opacity: r.shadow_opacity,
  shadow_blur: r.shadow_blur,
  glow: r.glow,
  glow_on: r.glow_on,
  glow_radius: r.glow_radius,
  glow_intensity: r.glow_intensity,
  reflection_on: r.reflection_on,
  reflection_opacity: r.reflection_opacity,
  reflection_offset: r.reflection_offset,
  reflection_blur: r.reflection_blur,
  reflection_fade: r.reflection_fade,
  gradient_on: r.gradient_on,
  gradient_start: r.gradient_start,
  gradient_end: r.gradient_end,
  gradient_angle: r.gradient_angle,
  rotation_angle: r.rotation_angle,
  align: r.align,
  detector_source: r.detector_source,
});
const regionVisualStyleKey = (r: Region | (Region & RegionDraft)) => JSON.stringify({
  tl: r.tl,
  font: r.font,
  size: r.size,
  align: r.align,
  fg: r.fg,
  // Pass 3: include all inputs that feed the backend's effective_style()
  // resolution. Previously `bg` and `role` were omitted, which caused the
  // preview sprite cache to serve stale renders until the user manually
  // toggled the font field.
  bg: r.bg,
  role: (r as any).role ?? "",
  outline: r.outline,
  outline_width: r.outline_width,
  shadow: r.shadow,
  shadow_on: r.shadow_on,
  shadow_offset_x: r.shadow_offset_x,
  shadow_offset_y: r.shadow_offset_y,
  shadow_opacity: r.shadow_opacity,
  shadow_blur: r.shadow_blur,
  glow: r.glow,
  glow_on: r.glow_on,
  glow_radius: r.glow_radius,
  glow_intensity: r.glow_intensity,
  reflection_on: r.reflection_on,
  reflection_opacity: r.reflection_opacity,
  reflection_offset: r.reflection_offset,
  reflection_blur: r.reflection_blur,
  reflection_fade: r.reflection_fade,
  gradient_on: r.gradient_on,
  gradient_start: r.gradient_start,
  gradient_end: r.gradient_end,
  gradient_angle: r.gradient_angle,
  rotation_angle: r.rotation_angle,
});
const regionPreviewSpriteKey = (r: Region | (Region & RegionDraft)) => JSON.stringify({
  visual: regionVisualStyleKey(r),
  bbox: { x: r.x, y: r.y, w: r.w, h: r.h },
  visible: r.visible,
});
const previewReason = (draft?: RegionDraft) => {
  if (!draft) return "dirty_page";
  if (draft.tl !== undefined) return "translation_change";
  if (draft.x !== undefined || draft.y !== undefined || draft.w !== undefined || draft.h !== undefined) return "bbox_commit";
  if (
    draft.font !== undefined || draft.size !== undefined || draft.align !== undefined ||
    draft.fg !== undefined || draft.outline !== undefined ||
    draft.outline_width !== undefined || draft.shadow !== undefined || draft.shadow_on !== undefined ||
    draft.shadow_offset_x !== undefined || draft.shadow_offset_y !== undefined || draft.shadow_opacity !== undefined || draft.shadow_blur !== undefined ||
    draft.glow !== undefined || draft.glow_on !== undefined || draft.glow_radius !== undefined || draft.glow_intensity !== undefined ||
    draft.reflection_on !== undefined || draft.reflection_opacity !== undefined || draft.reflection_offset !== undefined ||
    draft.reflection_blur !== undefined || draft.reflection_fade !== undefined ||
    draft.gradient_on !== undefined || draft.gradient_start !== undefined || draft.gradient_end !== undefined || draft.gradient_angle !== undefined ||
    draft.rotation_angle !== undefined
  ) return "style_change";
  return "local_change";
};
const isBboxOnlyDraft = (draft?: RegionDraft) => {
  if (!draft) return false;
  const keys = Object.keys(draft);
  return keys.length > 0 && keys.every(k => k === "x" || k === "y" || k === "w" || k === "h");
};
const clampNumber = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const validBoxArray = (box?: number[] | null): RegionBBox | null => {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const [x, y, w, h] = box.map(Number);
  if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
  return { x, y, w, h };
};
const containerBoxForRegion = (region: Region): RegionBBox => (
  validBoxArray(region.cleanup_container_bbox) ??
  validBoxArray(region.container_bbox) ??
  { x: region.x, y: region.y, w: region.w, h: region.h }
);
const boxCenter = (box: RegionBBox) => ({ x: box.x + box.w / 2, y: box.y + box.h / 2 });
const clampBboxToSize = (bbox: RegionBBox, size: { w: number; h: number }): RegionBBox => {
  const pageW = Math.max(1, Number(size.w) || 1);
  const pageH = Math.max(1, Number(size.h) || 1);
  const minW = Math.min(8, pageW);
  const minH = Math.min(8, pageH);
  const w = Math.min(Math.max(Math.round(bbox.w), minW), pageW);
  const h = Math.min(Math.max(Math.round(bbox.h), minH), pageH);
  return {
    x: Math.min(Math.max(Math.round(bbox.x), 0), Math.max(0, pageW - w)),
    y: Math.min(Math.max(Math.round(bbox.y), 0), Math.max(0, pageH - h)),
    w,
    h,
  };
};
const isCrossPageSecondary = (r: Region) => Boolean((r as any).cross_page_secondary);
const regionOwnerPage = (r: Region, fallbackPage: number) =>
  typeof r.source_page_idx === "number" ? r.source_page_idx : fallbackPage;
const regionSegmentForPage = (r: Region, pageIdx: number): Region => {
  const local = r.page_local_bboxes?.[String(pageIdx)];
  if (r.cross_page && local?.length === 4) {
    const [x, y, w, h] = local.map(Number);
    if ([x, y, w, h].every(Number.isFinite) && w > 0 && h > 0) {
      return { ...r, x, y, w, h };
    }
  }
  return r;
};
const editorRegionId = (r: Region, pageIdx: number) => {
  const ownerPage = regionOwnerPage(r, pageIdx);
  const ownerRegionIdx = typeof r.primary_region_idx === "number" ? r.primary_region_idx : r.idx;
  return `p${ownerPage}-r${ownerRegionIdx + 1}`;
};
const editorRegionsForPage = (data: Bootstrap, pageIdx: number): Region[] => {
  const byPageMap = data.regionsByPage as unknown as Record<string, Region[]> | undefined;
  const byPage = byPageMap?.[String(pageIdx)];
  const source = byPage ?? (pageIdx === data.meta.activePageIdx ? data.regions : []);
  return source
    .filter(r => !isCrossPageSecondary(r))
    .map(r => {
      const segmented = regionSegmentForPage(r, pageIdx);
      return { ...segmented, id: editorRegionId(segmented, pageIdx), display_page_idx: pageIdx };
    });
};

/* ── Empty bootstrap ─────────────────────────────────────────────────────── */
const EMPTY_BOOTSTRAP: Bootstrap = {
  series: [], chapters: {}, pages: [], regions: [], issues: [],
  memory: { available: true, series_title: "Local Series", names: [], glossary: [] },
  meta: { activeSeriesId: null, activeChapterId: null, activePageIdx: 0,
          busy: false, status: "Ready", chapterDir: "", totalPages: 0, chapterProgress: 0 },
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  TOP BAR                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */
const TopBar = ({
  pipeState, onStep, onRunAll, onContinueRunAll, onRunPage, onDetectAll, onExportYoloDataset, onTrainYolo, onUndo, onExport, onOpenChapter, onToggleLeft, onToggleRight,
  onBrowse, onSettings, leftOpen, rightOpen, busy, canRun, canContinueRunAll, continueTitle,
}: {
  pipeState:     StepState[];
  onStep:        (i: number) => void;
  onRunAll:      () => void;
  onContinueRunAll: () => void;
  onRunPage:     () => void;
  onDetectAll:   () => void;
  onExportYoloDataset: () => void;
  onTrainYolo:   () => void;
  onUndo:        () => void;
  onExport:      () => void;
  onOpenChapter: () => void;
  onBrowse:      () => void;
  onToggleLeft:  () => void;
  onToggleRight: () => void;
  onSettings:    () => void;
  leftOpen:      boolean;
  rightOpen:     boolean;
  busy:          boolean;
  canRun:        boolean;
  canContinueRunAll: boolean;
  continueTitle: string;
}) => (
  <div className="ml-topbar no-select">
    <div className="ml-logo">
      <div className="ml-logo-mark">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#0c0c14" strokeWidth="2.5" strokeLinecap="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      </div>
      <div>
        <div className="ml-logo-text">Manhwa</div>
        <div className="ml-logo-sub">Localizer</div>
      </div>
    </div>

    <div className="ml-pipeline">
      {PIPELINE_STEPS.map((name, i) => (
        <Fragment key={name}>
          {i > 0 && <span key={`a${i}`} className="pip-arrow">›</span>}
          <button
            className={`pip-step ${pipeState[i] ?? "idle"}`}
            onClick={() => !busy && canRun && onStep(i)}
            disabled={busy || !canRun}
            title={canRun ? name : "Open or import a chapter before running the pipeline."}
          >
            <div className="pip-dot" />
            {name}
          </button>
        </Fragment>
      ))}
    </div>

    <div className="ml-actions">
      <button className="btn-ghost" onClick={onOpenChapter} disabled={busy}>
        <Svg icon="folder" size={12} /> Open Chapter
      </button>
      <button className="btn-ghost" onClick={onRunPage} disabled={busy || !canRun} title={canRun ? "Run the selected page" : "Open or import a chapter before running the pipeline."}>
        <Svg icon="run" size={12} /> Run Page
      </button>
      <button className="btn-primary" onClick={onRunAll} disabled={busy || !canRun} title={canRun ? "Run all loaded pages" : "Open or import a chapter before running the pipeline."}>
        <Svg icon="run" size={12} /> Run All
      </button>
      <button className="btn-ghost" onClick={onExport} disabled={busy}>
        <Svg icon="export" size={12} /> Export
      </button>
      <details className="topbar-overflow">
        <summary className="btn-ghost">More</summary>
        <div className="topbar-menu">
          <button className="btn-ghost" onClick={onBrowse} disabled={busy}>Browse Source</button>
          {canContinueRunAll && <button className="btn-ghost" onClick={onContinueRunAll} disabled={busy || !canRun} title={continueTitle}>Continue Run All</button>}
          <button className="btn-ghost" onClick={onDetectAll} disabled={busy || !canRun} title={canRun ? "Detect boxes on every page only" : "Open or import a chapter first."}>Detect All</button>
          <button className="btn-ghost" onClick={onExportYoloDataset} disabled={busy || !canRun} title="Export YOLO-format images, labels, and deletion corrections">YOLO Data</button>
          <button className="btn-ghost" onClick={onTrainYolo} disabled={busy || !canRun} title="Train a YOLO checkpoint from the current fine-tune dataset">Train YOLO</button>
          <button className="btn-ghost" onClick={onUndo} disabled={busy}>Undo</button>
          <button className="btn-ghost" onClick={onSettings} disabled={busy}>Settings</button>
        </div>
      </details>
      <div className="divider-v" />
      <button className={`btn-icon ${leftOpen ? "on" : ""}`} onClick={onToggleLeft} title="Toggle left panel">
        <Svg icon="collapse" size={13} />
      </button>
      <button className={`btn-icon ${rightOpen ? "on" : ""}`} onClick={onToggleRight} title="Toggle right panel">
        <Svg icon="insp" size={13} />
      </button>
    </div>
  </div>
);

const SettingsModal = ({
  config, onChange, onSave, onClose,
}: {
  config: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
  onSave: () => void;
  onClose: () => void;
}) => {
  const update = (key: string, value: string) => onChange({ ...config, [key]: value });
  const checked = (key: string, fallback = false) => {
    const raw = config[key];
    if (raw === undefined || raw === null || raw === "") return fallback;
    return String(raw).toLowerCase() === "true";
  };
  const valueOf = (key: string, fallback: string) => config[key] ?? fallback;
  const debugFlag = (key: string) => {
    try { return localStorage.getItem(key) === "1"; } catch { return false; }
  };
  const setDebugFlag = (key: string, value: boolean) => {
    try { localStorage.setItem(key, value ? "1" : "0"); } catch { /* noop */ }
  };
  const yoloMissing = (config.detector_backend || "ocr") === "yolo" && !(config.yolo_model_path || "").trim();
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="settings-modal" onMouseDown={e => e.stopPropagation()}>
        <div className="settings-head">
          <div className="settings-title">Settings</div>
          <button className="btn-icon" onClick={onClose} title="Close">×</button>
        </div>
        <div className="settings-body">
          <div className="insp-section-title">Models</div>
          <div className="settings-grid">
            <label>Detector backend</label>
            <select className="settings-input" value={config.detector_backend || "ocr"} onChange={e => update("detector_backend", e.target.value)}>
              <option value="ocr">OCR</option>
              <option value="yolo">YOLO</option>
            </select>
            <label>OCR model</label>
            <input className="settings-input" value={config.ocr_model || ""} onChange={e => update("ocr_model", e.target.value)} />
            <label>OCR backend</label>
            <select className="settings-input" value={valueOf("ocr_backend", "cascade")} onChange={e => update("ocr_backend", e.target.value)}>
              <option value="cascade">Fast cascade</option>
              <option value="paddleocr">PaddleOCR only</option>
              <option value="qwen_vl">Qwen-VL only</option>
              <option value="easyocr">EasyOCR</option>
            </select>
            <label>Qwen OCR model</label>
            <input className="settings-input" value={valueOf("qwen_ocr_model", config.ocr_model || "")} onChange={e => update("qwen_ocr_model", e.target.value)} />
            <label>PaddleOCR service URL</label>
            <input className="settings-input" value={valueOf("paddleocr_service_url", "")} onChange={e => update("paddleocr_service_url", e.target.value)} placeholder="http://127.0.0.1:8899/ocr" />
            <label>PaddleOCR language</label>
            <input className="settings-input" value={valueOf("paddleocr_lang", "korean")} onChange={e => update("paddleocr_lang", e.target.value)} />
            <label>VLM fallback conf</label>
            <input className="settings-input" type="number" step="0.05" min="0" max="1" value={valueOf("ocr_vlm_fallback_confidence", "0.70")} onChange={e => update("ocr_vlm_fallback_confidence", e.target.value)} />
            <label>OCR cache</label>
            <input type="checkbox" checked={checked("ocr_cache_enabled", true)} onChange={e => update("ocr_cache_enabled", String(e.target.checked))} />
            <label>Translate model</label>
            <input className="settings-input" value={config.translate_model || ""} onChange={e => update("translate_model", e.target.value)} />
            <label>Vision model</label>
            <input className="settings-input" value={config.vision_model || ""} onChange={e => update("vision_model", e.target.value)} />
            <label>Polisher model</label>
            <input className="settings-input" value={config.polisher_model || ""} onChange={e => update("polisher_model", e.target.value)} />
            <label>keep_alive</label>
            <input className="settings-input" value={config.keep_alive || ""} onChange={e => update("keep_alive", e.target.value)} />
            <label>YOLO model path</label>
            <input className="settings-input" value={config.yolo_model_path || ""} onChange={e => update("yolo_model_path", e.target.value)} />
            <label>YOLO train base</label>
            <input className="settings-input" value={valueOf("yolo_training_base_model", "yolov8n.pt")} onChange={e => update("yolo_training_base_model", e.target.value)} />
            <label>YOLO train epochs</label>
            <input className="settings-input" type="number" min="1" max="1000" value={valueOf("yolo_training_epochs", "30")} onChange={e => update("yolo_training_epochs", e.target.value)} />
            <label>YOLO train image size</label>
            <input className="settings-input" type="number" min="320" max="2048" value={valueOf("yolo_training_imgsz", "640")} onChange={e => update("yolo_training_imgsz", e.target.value)} />
            <label>YOLO train batch</label>
            <input className="settings-input" type="number" min="1" max="128" value={valueOf("yolo_training_batch", "8")} onChange={e => update("yolo_training_batch", e.target.value)} />
            <label>YOLO train device</label>
            <input className="settings-input" value={valueOf("yolo_training_device", "")} onChange={e => update("yolo_training_device", e.target.value)} placeholder="auto, cpu, 0" />
            <label>Klein backend URL</label>
            <input className="settings-input" value={config.klein_backend_url || ""} onChange={e => update("klein_backend_url", e.target.value)} />
          </div>
          {yoloMissing && <div className="settings-warning">YOLO is selected but no model path is configured; detection will fall back to OCR.</div>}
          <div className="insp-section-title">SAM2 Mask Assist</div>
          <div className="settings-grid">
            <label>Enable SAM2</label>
            <input type="checkbox" checked={checked("sam2_enabled", false)} onChange={e => update("sam2_enabled", String(e.target.checked))} />
            <label>Load mode</label>
            <select className="settings-input" value={valueOf("sam2_load_mode", "lazy")} onChange={e => update("sam2_load_mode", e.target.value)}>
              <option value="lazy">lazy</option>
              <option value="startup">startup</option>
            </select>
            <label>Require SAM2</label>
            <input type="checkbox" checked={checked("sam2_required", false)} onChange={e => update("sam2_required", String(e.target.checked))} />
            <label>Backend URL</label>
            <input className="settings-input" value={valueOf("sam2_backend_url", "")} onChange={e => update("sam2_backend_url", e.target.value)} placeholder="http://127.0.0.1:8765/propose_cleanup_mask" />
            <label>Timeout sec</label>
            <input className="settings-input" type="number" min="1" max="300" value={valueOf("sam2_timeout_sec", "30")} onChange={e => update("sam2_timeout_sec", e.target.value)} />
            <label>Mask mode</label>
            <select className="settings-input" value={valueOf("sam2_mask_mode", "manual_only")} onChange={e => update("sam2_mask_mode", e.target.value)}>
              <option value="manual_only">Manual only</option>
              <option value="cleanup_assist">Cleanup assist</option>
              <option value="auto_cleanup">Auto cleanup</option>
              <option value="container_assist">Container/protect assist</option>
            </select>
            <label>Model path</label>
            <input className="settings-input" value={valueOf("sam2_model_path", "")} onChange={e => update("sam2_model_path", e.target.value)} placeholder="external/sam2" />
            <label>Checkpoint path</label>
            <input className="settings-input" value={valueOf("sam2_checkpoint_path", "")} onChange={e => update("sam2_checkpoint_path", e.target.value)} placeholder="external/sam2_checkpoints/sam2.1_hiera_tiny.pt" />
            <label>Device</label>
            <select className="settings-input" value={valueOf("sam2_device", "auto")} onChange={e => update("sam2_device", e.target.value)}>
              <option value="auto">auto</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
              <option value="mps">mps</option>
            </select>
          </div>
          <div className="settings-note">SAM2 proposes masks. With cleanup mask backend set to Auto or SAM2, automatic cleanup can use those masks before the normal patch flow.</div>
          <div className="insp-section-title">Pipeline</div>
          <div className="settings-grid">
            <label>Process SFX regions</label>
            {/* Pass 4: master SFX toggle — OFF by default. When OFF, SFX/shout
                regions are persisted but hidden and skipped by all pipeline
                stages (OCR, translate, cleanup, typeset). Existing SFX regions
                reappear if turned back ON without re-running detect. */}
            <input
              type="checkbox"
              checked={String(config.process_sfx_regions) === "true"}
              onChange={e => update("process_sfx_regions", String(e.target.checked))}
              title="When off (default), SFX/shout regions are skipped by the pipeline and hidden from the overlay. Existing SFX regions are preserved and reappear if re-enabled."
            />
          </div>
          <div className="insp-section-title">Cleanup</div>
          <div className="settings-grid">
            <label>Cleanup preset</label>
            <select
              className="settings-input"
              value={checked("cleanup_manual_review_only") ? "manual" : valueOf("cleanup_mode", "balanced")}
              onChange={e => {
                update("cleanup_mode", e.target.value === "manual" ? "conservative" : e.target.value);
                onChange({
                  ...config,
                  cleanup_mode: e.target.value === "manual" ? "conservative" : e.target.value,
                  cleanup_manual_review_only: String(e.target.value === "manual"),
                });
              }}
            >
              <option value="conservative">Conservative</option>
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
              <option value="manual">Manual / Review only</option>
            </select>
            <label>Mask backend</label>
            <select className="settings-input" value={valueOf("cleanup_mask_backend", "auto")} onChange={e => update("cleanup_mask_backend", e.target.value)}>
              <option value="auto">Auto (CV + SAM2)</option>
              <option value="cv">CV only</option>
              <option value="sam2">SAM2</option>
            </select>
            <label>Solid bubble fill</label>
            <input type="checkbox" checked={checked("cleanup_solid_bubble_fill_enabled", true)} onChange={e => update("cleanup_solid_bubble_fill_enabled", String(e.target.checked))} title="Best for white/solid bubbles" />
            <label>Solid min container conf</label>
            <input className="settings-input" type="number" step="0.05" min="0" max="1" value={valueOf("cleanup_solid_bubble_min_container_confidence", "0.6")} onChange={e => update("cleanup_solid_bubble_min_container_confidence", e.target.value)} />
            <label>Solid max mask/container</label>
            <input className="settings-input" type="number" step="0.01" min="0" max="1" value={valueOf("cleanup_solid_bubble_max_mask_container_ratio", "0.15")} onChange={e => update("cleanup_solid_bubble_max_mask_container_ratio", e.target.value)} />
            <label>Solid max rectangularity</label>
            <input className="settings-input" type="number" step="0.01" min="0" max="1" value={valueOf("cleanup_solid_bubble_max_rectangularity", "0.45")} onChange={e => update("cleanup_solid_bubble_max_rectangularity", e.target.value)} />
            <label>Edge cleanup</label>
            <input type="checkbox" checked={checked("cleanup_halo_mask_enabled", true)} onChange={e => update("cleanup_halo_mask_enabled", String(e.target.checked))} title="Expands the cleanup mask over faint text edges" />
            <label>Edge cleanup px</label>
            <input className="settings-input" type="number" min="0" max="8" value={valueOf("cleanup_halo_max_px", "2")} onChange={e => update("cleanup_halo_max_px", e.target.value)} />
            <label>Complete outlines</label>
            <input type="checkbox" checked={checked("cleanup_contrast_mask_completion_enabled", true)} onChange={e => update("cleanup_contrast_mask_completion_enabled", String(e.target.checked))} title="Expands flat-bubble masks over missed high-contrast outlines and shadows" />
            <label>Outline radius</label>
            <input className="settings-input" type="number" min="4" max="32" value={valueOf("cleanup_contrast_mask_completion_radius", "18")} onChange={e => update("cleanup_contrast_mask_completion_radius", e.target.value)} />
            <label>Leftover text retry</label>
            <input type="checkbox" checked={checked("cleanup_residual_retry_enabled", true)} onChange={e => update("cleanup_residual_retry_enabled", String(e.target.checked))} title="Tries once more with a slightly larger mask when text remains" />
            <label>Retry dilation px</label>
            <input className="settings-input" type="number" min="0" max="8" value={valueOf("cleanup_residual_retry_dilate_px", "1")} onChange={e => update("cleanup_residual_retry_dilate_px", e.target.value)} />
            <label>Force cleanup</label>
            <input type="checkbox" checked={checked("cleanup_force_enabled", false)} onChange={e => update("cleanup_force_enabled", String(e.target.checked))} title="Always attempts cleanup, falling back to bbox masks and bypassing normal skip/review gates" />
            <label>Show cleanup status</label>
            <input type="checkbox" checked={checked("cleanup_status_enabled", true)} onChange={e => update("cleanup_status_enabled", String(e.target.checked))} title="When off, cleanup still runs but cautious/review status labels are hidden" />
            <label>Grouped inpaint</label>
            <input type="checkbox" checked={checked("cleanup_allow_grouped_inpaint", false)} onChange={e => update("cleanup_allow_grouped_inpaint", String(e.target.checked))} />
            <label>Fallback backend</label>
            <select className="settings-input" value={valueOf("cleanup_fallback_backend", "telea")} onChange={e => update("cleanup_fallback_backend", e.target.value)}>
              <option value="telea">TELEA</option>
              <option value="ns">OpenCV NS</option>
              <option value="iopaint">IOPaint</option>
            </select>
          </div>
          <div className="settings-note">Solid fill is best for white or flat colored bubbles. Edge cleanup and leftover text retry help remove faint glyph edges.</div>
          <div className="insp-section-title">Skip / Safety Rules</div>
          <div className="settings-grid">
            <label>Allow SFX cleanup</label>
            <input type="checkbox" checked={checked("cleanup_allow_sfx_cleanup", false)} onChange={e => update("cleanup_allow_sfx_cleanup", String(e.target.checked))} title="Risky: can damage drawn SFX art" />
            <label>Allow text-over-art</label>
            <input type="checkbox" checked={checked("cleanup_allow_text_over_art", false)} onChange={e => update("cleanup_allow_text_over_art", String(e.target.checked))} title="Risky: can damage page art" />
            <label>Min container confidence</label>
            <input className="settings-input" type="number" step="0.05" min="0" max="1" value={valueOf("cleanup_min_container_confidence", "0")} onChange={e => update("cleanup_min_container_confidence", e.target.value)} />
            <label>Max mask/container</label>
            <input className="settings-input" type="number" step="0.01" min="0" max="2" value={valueOf("cleanup_max_mask_container_ratio", "0.5")} onChange={e => update("cleanup_max_mask_container_ratio", e.target.value)} />
            <label>Max mask/region</label>
            <input className="settings-input" type="number" step="0.01" min="0" max="2" value={valueOf("cleanup_max_mask_region_ratio", "0.28")} onChange={e => update("cleanup_max_mask_region_ratio", e.target.value)} />
            <label>Max border touch</label>
            <input className="settings-input" type="number" step="0.01" min="0" max="1" value={valueOf("cleanup_max_border_touch_ratio", "0.35")} onChange={e => update("cleanup_max_border_touch_ratio", e.target.value)} />
            <label>Max rectangularity</label>
            <input className="settings-input" type="number" step="0.01" min="0" max="1" value={valueOf("cleanup_max_rectangularity", "0.88")} onChange={e => update("cleanup_max_rectangularity", e.target.value)} />
            <label>Translucent captions</label>
            <input type="checkbox" checked={checked("cleanup_allow_translucent_caption", false)} onChange={e => update("cleanup_allow_translucent_caption", String(e.target.checked))} />
            <label>Texture/halftone fallback</label>
            <input type="checkbox" checked={checked("cleanup_allow_texture_inpaint", false)} onChange={e => update("cleanup_allow_texture_inpaint", String(e.target.checked))} />
            <label>Risky result action</label>
            <select className="settings-input" value={valueOf("cleanup_risky_action", "skip")} onChange={e => update("cleanup_risky_action", e.target.value)}>
              <option value="skip">Skip</option>
              <option value="review">Mark review</option>
              <option value="attempt">Attempt anyway</option>
            </select>
            <label>Verbose cleanup logs</label>
            <input type="checkbox" checked={checked("cleanup_verbose_logs", false)} onChange={e => update("cleanup_verbose_logs", String(e.target.checked))} />
            <label>Cleanup diagnostics</label>
            <input type="checkbox" checked={checked("cleanup_show_diagnostics", false)} onChange={e => update("cleanup_show_diagnostics", String(e.target.checked))} />
          </div>
          <div className="settings-warning">SFX and text-over-art cleanup can damage artwork. Keep them off unless you explicitly want cleanup attempted.</div>
          <div className="insp-section-title">Debug</div>
          <div className="settings-grid">
            <label>renderDebug</label>
            <input type="checkbox" defaultChecked={debugFlag("manhwa.renderDebug")} onChange={e => setDebugFlag("manhwa.renderDebug", e.target.checked)} />
            <label>Pointer logs</label>
            <input type="checkbox" defaultChecked={debugFlag("manhwa.pointerDebug")} onChange={e => setDebugFlag("manhwa.pointerDebug", e.target.checked)} />
            <label>Preview sprite logs</label>
            <input type="checkbox" defaultChecked={debugFlag("manhwa.previewSpriteDebug")} onChange={e => setDebugFlag("manhwa.previewSpriteDebug", e.target.checked)} />
          </div>
          <div className="settings-note">Model and detector changes are saved to the existing backend config. Heavy model reloads may require restart.</div>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
            <button className="btn-ghost" onClick={onClose}>Cancel</button>
            <button className="btn-primary" onClick={onSave}>Save</button>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  LEFT PANEL                                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
const PageThumb = ({ idx, chapterDir, pageRevision }: { idx: number; chapterDir: string; pageRevision: string }) => {
  const cacheKey = `${chapterDir}:thumb:${idx}:raw:${pageRevision}`;
  const [src, setSrc] = useState<string | null>(() => PAGE_IMAGE_CACHE.get(cacheKey) ?? null);

  useEffect(() => {
    if (!chapterDir) {
      setSrc(null);
      return;
    }
    const cached = PAGE_IMAGE_CACHE.get(cacheKey);
    if (cached) {
      setSrc(cached);
      return;
    }
    let cancelled = false;
    api.getPageImage(idx, "raw").then(resp => {
      if (cancelled) return;
      const next = resp.ok && resp.b64 ? `data:image/png;base64,${resp.b64}` : null;
      if (next) PAGE_IMAGE_CACHE.set(cacheKey, next);
      setSrc(next);
    }).catch(() => {
      if (!cancelled) setSrc(null);
    });
    return () => { cancelled = true; };
  }, [idx, chapterDir, cacheKey]);

  return (
    <div className="page-thumb">
      {src && <img src={src} alt={`Page ${idx + 1}`} loading="lazy" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} draggable={false} />}
    </div>
  );
};

const seriesStableKey = (series?: Series | null) => {
  if (!series) return "";
  const source = series.source || series.subtitle || "local";
  const sourceId = series.source_id || "";
  return source !== "local" && sourceId ? `${source}:${sourceId}` : `local:${series.title}`;
};

const SeriesThumb = ({
  series,
  onPreview,
}: {
  series: Series;
  onPreview?: (src: string, title: string, source: string) => void;
}) => {
  const directSrc = series.thumbnail_url || series.thumbnail_path || "";
  const [src, setSrc] = useState(directSrc);
  const [broken, setBroken] = useState(false);
  const proxiedRef = useRef(false);

  useEffect(() => {
    const next = series.thumbnail_url || series.thumbnail_path || "";
    setSrc(next);
    setBroken(false);
    proxiedRef.current = false;
  }, [series.thumbnail_url, series.thumbnail_path]);

  const handleError = useCallback(async () => {
    if (proxiedRef.current) { setBroken(true); return; }
    proxiedRef.current = true;
    const url = series.thumbnail_url || "";
    const path = series.thumbnail_path || "";
    if (!url && !path) { setBroken(true); return; }
    try {
      const res = await api.getThumbnailB64(url, path);
      if (res.ok && (res as { b64?: string }).b64) {
        setSrc((res as { b64: string }).b64);
      } else {
        setBroken(true);
      }
    } catch {
      setBroken(true);
    }
  }, [series.thumbnail_url, series.thumbnail_path]);

  const canShow = Boolean(src) && !broken;
  return (
    <div
      className={`series-thumb ${canShow ? "has-cover" : ""}`}
      style={{ background: canShow ? undefined : series.color + "22", color: series.color }}
      onClick={(e) => {
        if (!canShow) return;
        e.stopPropagation();
        onPreview?.(src, series.title, series.source || series.subtitle || "local");
      }}
      title={canShow ? "Preview cover" : undefined}
    >
      {canShow ? (
        <img src={src} alt={`${series.title} cover`} onError={handleError} draggable={false} />
      ) : (
        series.title.slice(0, 2)
      )}
    </div>
  );
};

const CoverLightbox = ({
  src,
  title,
  source,
  onClose,
}: {
  src: string;
  title: string;
  source: string;
  onClose: () => void;
}) => {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="cover-lightbox" onMouseDown={onClose}>
      <div className="cover-card" onMouseDown={e => e.stopPropagation()}>
        <img src={src} alt={`${title} cover`} />
        <div className="cover-info">
          <div>
            <div className="cover-title">{title}</div>
            <div className="cover-source">{source}</div>
          </div>
          <button className="btn-ghost" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

const LeftPanel = ({
  data, activeSeries, setActiveSeries, activeChapter, setActiveChapter,
  activePage, onPageSelect, onSeriesSelect, selectedSeriesTitle,
  onBrowseSource, onOpenFolder, onBootstrap, onSourceChange, onDeleteSeries, onCoverPreview, detailRefreshKey,
  pageVersions, pipelineProgress,
}: {
  data:             Bootstrap;
  activeSeries:     string;
  setActiveSeries:  (id: string) => void;
  activeChapter:    string;
  setActiveChapter: (id: string) => void;
  activePage:       number;
  onPageSelect:     (idx: number, mode?: PageSelectMode) => void;
  onSeriesSelect?:  (series: Series) => void;
  selectedSeriesTitle?: string | null;
  onBrowseSource: () => void;
  onOpenFolder: (folder: string) => void;
  onBootstrap: (b: Bootstrap) => void;
  onSourceChange: () => void;
  onDeleteSeries: (title: string) => void;
  onCoverPreview: (src: string, title: string, source: string) => void;
  detailRefreshKey: number;
  pageVersions: Record<number, number>;
  pipelineProgress: ProgressEvent | null;
}) => {
  const [seriesOpen, setSeriesOpen] = useState(true);
  const [chapterOpen, setChapterOpen] = useState(true);
  const [pagesOpen, setPagesOpen] = useState(true);

  const chapters = data.chapters[activeSeries] ?? [];
  const pages    = data.pages;
  const selectedSeries = data.series.find(s => s.id === activeSeries);
  const selectedIsSource = !!selectedSeries?.subtitle && selectedSeries.subtitle !== "local";
  const processingPageIdx = typeof pipelineProgress?.page_idx === "number" ? pipelineProgress.page_idx : null;

  return (
    <div className="left-scroll-body">
      {/* Series */}
      <div className="pnl-section">
        <div className="pnl-header" onClick={() => setSeriesOpen(o => !o)}>
          <span className={`pnl-title ${data.series.length > 0 ? "lit" : ""}`}>Series</span>
          <Svg icon={seriesOpen ? "chevDown" : "chevRight"} size={11} />
        </div>
        {seriesOpen && (
          <div className="pnl-body">
            {data.series.length === 0 ? (
              <div className="empty-state">No series yet</div>
            ) : data.series.map(s => (
              <div key={s.id} className={`series-row ${activeSeries === s.id ? "active" : ""}`}
                onClick={() => { setActiveSeries(s.id); onSeriesSelect?.(s); }}>
                <SeriesThumb series={s} onPreview={onCoverPreview} />
                <div className="series-info">
                  <div className="series-title">{s.title}</div>
                  <div className="series-meta">
                    {s.chapters} ch · {s.lang}
                    {s.subtitle && s.subtitle !== "local" ? ` · ${s.subtitle}` : ""}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Source Sync / Series details */}
      {selectedSeriesTitle && (
        <div className="pnl-section" style={{ minHeight: 0 }}>
          <SeriesDetailPanel
            key={`${selectedSeriesTitle}:${detailRefreshKey}`}
            seriesTitle={selectedSeriesTitle}
            onOpenChapter={onOpenFolder}
            onBootstrap={onBootstrap}
            onBrowseClick={onBrowseSource}
            onSourceChange={onSourceChange}
            onDeleteSeries={onDeleteSeries}
          />
        </div>
      )}

      {/* Chapters */}
      {!selectedIsSource && <div className="pnl-section">
        <div className="pnl-header" onClick={() => setChapterOpen(o => !o)}>
          <span className={`pnl-title ${chapters.length > 0 ? "lit" : ""}`}>Local chapters</span>
          <Svg icon={chapterOpen ? "chevDown" : "chevRight"} size={11} />
        </div>
        {chapterOpen && (
          <div className="pnl-body">
            {chapters.length === 0 ? (
              <div className="empty-state">No chapters loaded</div>
            ) : chapters.map(ch => (
              <div key={ch.id} className={`chapter-row ${activeChapter === ch.id ? "active" : ""}`}
                onClick={() => setActiveChapter(ch.id)}>
                <span className="chapter-name">{ch.title}</span>
                <div className="chapter-prog">
                  <div className="prog-bar">
                    <div className="prog-fill" style={{ width: `${ch.progress}%` }} />
                  </div>
                  <span style={{ fontFamily: "var(--fnt-mono)", fontSize: 9, color: "var(--t3)" }}>
                    {ch.progress}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>}

      {/* Pages */}
      <div className="pnl-section" style={{ display: "flex", flexDirection: "column", flex: "0 0 auto", minHeight: 0 }}>
        <div className="pnl-header" onClick={() => setPagesOpen(o => !o)}>
          <span className={`pnl-title ${pages.length > 0 ? "lit" : ""}`}>
            Pages {pages.length > 0 ? `(${pages.length})` : ""}
          </span>
          <Svg icon={pagesOpen ? "chevDown" : "chevRight"} size={11} />
        </div>
        {pagesOpen && (
          <div className="page-strip">
            {pages.length === 0 ? (
              <div className="empty-state">Open a chapter to see pages</div>
            ) : pages.map((pg) => (
              <div key={pg.id} className={`page-thumb-item ${activePage === pg.idx ? "active" : ""}`}
                onClick={() => onPageSelect(pg.idx)}>
                <PageThumb idx={pg.idx} chapterDir={data.meta.chapterDir} pageRevision={`rv${pg.render_version ?? 0}:${pageVersions[pg.idx] ?? 0}:${pg.status.join("")}:${pg.dirty ? 1 : 0}`} />
                <div>
                  <div className="page-num">Pg {pg.idx + 1}</div>
                  <div className="page-status-dots">
                    {pg.status.map((s, i) => (
                      <div key={i} className={`psd ${s === "done" ? "on" : ""}`} />
                    ))}
                  </div>
                  {processingPageIdx === pg.idx && (
                    <div className={`page-run-badge ${pipelineProgress?.error ? "error" : ""}`}>
                      {pipelineProgress?.error ? "error" : (pipelineProgress?.stage ?? "processing")}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  CANVAS AREA — loads real images from the backend                           */
/* ─────────────────────────────────────────────────────────────────────────── */
const ContinuousPage = ({
  idx,
  active,
  chapterDir,
  imageMode,
  pageRevision,
  regions,
  selectedRegion,
  regionDrafts,
  showEnglishOverlay,
  showOverlayBoxes,
  zoom,
  showMarker,
  totalPages,
  onSelect,
  onRegionSelect,
  onPreviewRegion,
  onCommitBBox,
}: {
  idx: number;
  active: boolean;
  chapterDir: string;
  imageMode: ImageMode;
  pageRevision: string;
  regions: Region[];
  selectedRegion: Region | null;
  regionDrafts: Record<string, RegionDraft>;
  showEnglishOverlay: boolean;
  showOverlayBoxes: boolean;
  zoom: number;
  showMarker: boolean;
  totalPages: number;
  onSelect: (idx: number, mode?: PageSelectMode) => void;
  onRegionSelect: (region: Region) => void;
  onPreviewRegion: (regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => void;
  onCommitBBox: (region: Region, bbox: RegionBBox) => Promise<boolean>;
}) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [dragBox, setDragBox] = useState<{ id: string; bbox: RegionBBox } | null>(null);
  const [previewSprites, setPreviewSprites] = useState<Record<string, BackendPreviewSprite>>({});
  const previewSpritesRef = useRef<Record<string, BackendPreviewSprite>>({});
  const pendingSpriteKeysRef = useRef<Set<string>>(new Set());
  const dragRef = useRef<{
    pointerId: number;
    mode: "move" | ResizeHandle;
    region: Region;
    startPoint: { x: number; y: number };
    startBox: RegionBBox;
    currentBox: RegionBBox;
  } | null>(null);
  const [nearView, setNearView] = useState(idx < 2);
  const cacheKey = `${chapterDir}:page:${idx}:${imageMode}:${pageRevision}`;
  const rawCacheKey = `${chapterDir}:page:${idx}:raw:${pageRevision}`;
  const [src, setSrc] = useState<string | null>(() => PAGE_IMAGE_CACHE.get(cacheKey) ?? PAGE_IMAGE_CACHE.get(rawCacheKey) ?? null);

  useEffect(() => {
    previewSpritesRef.current = previewSprites;
  }, [previewSprites]);

  useEffect(() => {
    const node = ref.current;
    if (!node || nearView) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some(entry => entry.isIntersecting)) {
        setNearView(true);
        observer.disconnect();
      }
    }, { rootMargin: "450px 0px" });
    observer.observe(node);
    return () => observer.disconnect();
  }, [nearView]);

  useEffect(() => {
    if (!chapterDir || !nearView) return;
    const cached = PAGE_IMAGE_CACHE.get(cacheKey);
    if (cached) {
      setSrc(cached);
      return;
    }
    let cancelled = false;
    api.getPageImage(idx, imageMode).then(async resp => {
      if (cancelled) return;
      if (resp.ok && resp.b64) {
        const next = `data:image/png;base64,${resp.b64}`;
        PAGE_IMAGE_CACHE.set(cacheKey, next);
        setSrc(next);
        return;
      }
      const rawCached = PAGE_IMAGE_CACHE.get(rawCacheKey);
      if (rawCached) {
        setSrc(rawCached);
        return;
      }
      const raw = await api.getPageImage(idx, "raw");
      if (!cancelled) {
        const next = raw.ok && raw.b64 ? `data:image/png;base64,${raw.b64}` : null;
        if (next) PAGE_IMAGE_CACHE.set(rawCacheKey, next);
        setSrc(next);
      }
    });
    return () => { cancelled = true; };
  }, [chapterDir, idx, imageMode, nearView, cacheKey, rawCacheKey]);

  useEffect(() => {
    if (!active || !chapterDir || !showOverlayBoxes || !showEnglishOverlay || naturalSize.w <= 0 || naturalSize.h <= 0) return;
    if (dragRef.current) return;
    let cancelled = false;
    regions.forEach(region => {
      if (!region.visible || !region.tl) return;
      if (regionOwnerPage(region, idx) !== idx) return;
      if (selectedRegion?.id !== region.id && !regionDrafts[region.id]) return;
      const draft = {
        ...(regionDrafts[region.id] ?? {}),
      };
      const merged = { ...region, ...draft } as Region;
      const styleKey = regionPreviewSpriteKey(merged);
      const slot = `${idx}:${region.id}`;
      const requestKey = `${slot}:${styleKey}`;
      const cached = previewSpritesRef.current[slot];
      if (cached?.styleKey === styleKey && cached.b64) return;
      if (pendingSpriteKeysRef.current.has(requestKey)) return;
      pendingSpriteKeysRef.current.add(requestKey);
      api.getRegionPreviewSprite(region.idx, draft, idx).then(resp => {
        pendingSpriteKeysRef.current.delete(requestKey);
        if (cancelled || !showOverlayBoxes || !showEnglishOverlay || !resp.ok || !resp.b64) return;
        setPreviewSprites(prev => ({
          ...prev,
          [slot]: {
            ...resp,
            styleKey,
            reason: "continuous_overlay",
            sourceBox: { x: merged.x, y: merged.y, w: merged.w, h: merged.h },
          },
        }));
      }).catch(() => {
        pendingSpriteKeysRef.current.delete(requestKey);
        // keep the lightweight CSS fallback if sprite generation fails
      });
    });
    return () => { cancelled = true; };
  }, [active, chapterDir, showOverlayBoxes, showEnglishOverlay, naturalSize.w, naturalSize.h, idx, regions, regionDrafts, selectedRegion?.id]);

  const clampContinuousBox = (box: RegionBBox): RegionBBox => {
    const minW = 8;
    const minH = 8;
    const pageW = naturalSize.w || minW;
    const pageH = naturalSize.h || minH;
    const crossPages = 1 + (idx > 0 ? 1 : 0) + (idx < totalPages - 1 ? 1 : 0);
    const w = Math.min(Math.max(Math.round(box.w), minW), pageW);
    const h = Math.min(Math.max(Math.round(box.h), minH), pageH * crossPages);
    const minY = idx > 0 ? -pageH : 0;
    const maxY = idx < totalPages - 1 ? pageH - minH : Math.max(0, pageH - h);
    let y = Math.min(Math.max(Math.round(box.y), minY), maxY);
    if (y + h < minH) y = minH - h;
    if (y > pageH - minH) y = pageH - minH;
    return {
      x: Math.min(Math.max(Math.round(box.x), 0), Math.max(0, pageW - w)),
      y,
      w,
      h,
    };
  };

  const resizeContinuousBox = (handle: ResizeHandle, start: RegionBBox, dx: number, dy: number): RegionBBox => {
    let left = start.x;
    let top = start.y;
    let right = start.x + start.w;
    let bottom = start.y + start.h;
    if (handle.includes("w")) left += dx;
    if (handle.includes("e")) right += dx;
    if (handle.includes("n")) top += dy;
    if (handle.includes("s")) bottom += dy;
    if (right < left + 8) {
      if (handle.includes("w")) left = right - 8;
      else right = left + 8;
    }
    if (bottom < top + 8) {
      if (handle.includes("n")) top = bottom - 8;
      else bottom = top + 8;
    }
    return clampContinuousBox({ x: left, y: top, w: right - left, h: bottom - top });
  };

  const startContinuousDrag = (e: React.PointerEvent<HTMLDivElement>, region: Region, mode: "move" | ResizeHandle) => {
    if (region.locked || naturalSize.w <= 0 || naturalSize.h <= 0) {
      onRegionSelect(region);
      return;
    }
    e.stopPropagation();
    e.preventDefault();
    onSelect(idx, "local");
    onRegionSelect(region);
    const draft = regionDrafts[region.id] ?? {};
    const box = clampContinuousBox({
      x: draft.x ?? region.x,
      y: draft.y ?? region.y,
      w: draft.w ?? region.w,
      h: draft.h ?? region.h,
    });
    dragRef.current = {
      pointerId: e.pointerId,
      mode,
      region,
      startPoint: { x: e.clientX, y: e.clientY },
      startBox: box,
      currentBox: box,
    };
    setDragBox({ id: region.id, bbox: box });
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const onContinuousPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || naturalSize.w <= 0 || naturalSize.h <= 0) return;
    e.preventDefault();
    const rect = ref.current?.getBoundingClientRect();
    if (!rect?.width) return;
    const scale = naturalSize.w / rect.width;
    const dx = (e.clientX - drag.startPoint.x) * scale;
    const dy = (e.clientY - drag.startPoint.y) * scale;
    const next = drag.mode === "move"
      ? clampContinuousBox({ ...drag.startBox, x: drag.startBox.x + dx, y: drag.startBox.y + dy })
      : resizeContinuousBox(drag.mode, drag.startBox, dx, dy);
    drag.currentBox = next;
    setDragBox({ id: drag.region.id, bbox: next });
  };

  const endContinuousDrag = async (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    e.preventDefault();
    dragRef.current = null;
    try {
      e.currentTarget.releasePointerCapture?.(drag.pointerId);
    } catch {
      // pointer capture may already be released
    }
    const finalBox = clampContinuousBox(drag.currentBox);
    setDragBox(null);
    onPreviewRegion(drag.region.id, finalBox, { requestSprite: false, reason: "continuous_bbox_commit" });
    await onCommitBBox(drag.region, finalBox);
  };

  const continuousPreviewFontSize = (r: Region) => Math.max(8, Math.min(42, Math.round((r.size || Math.max(10, r.h * 0.2)) * 0.72)));

  return (
    <div
      ref={ref}
      id={`continuous-page-${idx}`}
      data-page-idx={idx}
      className={`continuous-page ${active ? "active" : ""}`}
      style={{ width: `min(96%, ${Math.max(280, Math.round(760 * zoom / 100))}px)` }}
      onPointerMove={onContinuousPointerMove}
      onPointerUp={endContinuousDrag}
      onPointerCancel={endContinuousDrag}
      onClick={() => onSelect(idx)}
    >
      {showMarker && <div className="page-start-marker">{idx + 1} / {totalPages}</div>}
      {src ? (
        <>
          <img
            src={src}
            alt={`Page ${idx + 1}`}
            draggable={false}
            onLoad={e => setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          />
          {showOverlayBoxes && naturalSize.w > 0 && naturalSize.h > 0 && regions.map(region => {
            const draft = regionDrafts[region.id] ?? {};
            const r = {
              ...region,
              ...draft,
              ...(dragBox?.id === region.id ? dragBox.bbox : {}),
            } as Region;
            if (!r.visible && selectedRegion?.id !== r.id) return null;
            // Pass 4: hide SFX regions entirely when the master toggle is OFF.
            if (r.pipeline_disabled && selectedRegion?.id !== r.id) return null;
            const isSelected = selectedRegion?.id === r.id;
            const sprite = previewSprites[`${idx}:${region.id}`];
            const isLiveEdit = isSelected || Boolean(regionDrafts[region.id]) || dragBox?.id === region.id;
            const showLiveText = showOverlayBoxes && showEnglishOverlay && isLiveEdit && r.visible && Boolean(r.tl);
            return (
              <div
                key={r.id}
                className={`region-overlay ${isSelected ? "sel" : ""}`}
                style={{
                  left: `${(r.x / naturalSize.w) * 100}%`,
                  top: `${(r.y / naturalSize.h) * 100}%`,
                  width: `${(r.w / naturalSize.w) * 100}%`,
                  height: `${(r.h / naturalSize.h) * 100}%`,
                }}
                onClick={e => {
                  e.stopPropagation();
                  onRegionSelect(region);
                }}
                onPointerDown={e => startContinuousDrag(e, region, "move")}
              >
                <div className={`region-label ${isSelected ? "gold" : "teal"}`}>
                  {r.label}{r.locked ? " · LOCK" : ""}
                </div>
                {showLiveText && sprite?.b64 ? (
                  <img
                    className="translation-bitmap-preview"
                    src={`data:image/png;base64,${sprite.b64}`}
                    alt=""
                    draggable={false}
                    style={{
                      left: `${((sprite.x - sprite.sourceBox.x) / sprite.sourceBox.w) * 100}%`,
                      top: `${((sprite.y - sprite.sourceBox.y) / sprite.sourceBox.h) * 100}%`,
                      width: `${(sprite.w / sprite.sourceBox.w) * 100}%`,
                      height: `${(sprite.h / sprite.sourceBox.h) * 100}%`,
                    }}
                  />
                ) : showLiveText && (
                  <div
                    className="translation-overlay"
                    style={{
                      color: r.fg || "#111111",
                      fontFamily: r.font && !ROLE_FONT_NAMES.has(r.font.toLowerCase()) ? `"${r.font}", "Comic Sans MS", "Segoe UI", sans-serif` : `"Comic Sans MS", "Segoe UI", sans-serif`,
                      fontSize: continuousPreviewFontSize(r),
                      textAlign: r.align,
                      textShadow: `${r.outline || "#ffffff"} 0 0 ${Math.max(1, r.outline_width || 1)}px`,
                      transform: `rotate(${Number(r.rotation_angle ?? 0)}deg)`,
                      transformOrigin: "center center",
                    }}
                  >
                    {r.tl}
                  </div>
                )}
                {isSelected && (["nw", "ne", "sw", "se"] as const).map(handle => (
                  <div
                    key={handle}
                    className={`resize-handle ${handle}`}
                    onPointerDown={e => startContinuousDrag(e, region, handle)}
                  />
                ))}
              </div>
            );
          })}
        </>
      ) : (
        <div className="continuous-placeholder">Page {idx + 1}</div>
      )}
    </div>
  );
};

const ContinuousReader = ({
  data,
  activePage,
  selectedRegion,
  regionDrafts,
  showEnglishOverlay,
  showOverlayBoxes,
  imageMode,
  pageVersions,
  zoom,
  showIndicator,
  showMarkers,
  scrollTarget,
  onPageSelect,
  onRegionSelect,
  onPreviewRegion,
  onCommitBBox,
}: {
  data: Bootstrap;
  activePage: number;
  selectedRegion: Region | null;
  regionDrafts: Record<string, RegionDraft>;
  showEnglishOverlay: boolean;
  showOverlayBoxes: boolean;
  imageMode: ImageMode;
  pageVersions: Record<number, number>;
  zoom: number;
  showIndicator: boolean;
  showMarkers: boolean;
  scrollTarget: number | null;
  onPageSelect: (idx: number, mode?: PageSelectMode) => void;
  onRegionSelect: (region: Region) => void;
  onPreviewRegion: (regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => void;
  onCommitBBox: (region: Region, bbox: RegionBBox) => Promise<boolean>;
}) => {
  const readerRef = useRef<HTMLDivElement | null>(null);
  const [visiblePage, setVisiblePage] = useState(activePage);
  const scrollRafRef = useRef<number | null>(null);

  useEffect(() => {
    setVisiblePage(activePage);
  }, [activePage]);

  useEffect(() => {
    if (scrollTarget === null) return;
    document.getElementById(`continuous-page-${scrollTarget}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [scrollTarget]);

  useEffect(() => () => {
    if (scrollRafRef.current !== null) window.cancelAnimationFrame(scrollRafRef.current);
  }, []);

  const updateVisiblePage = useCallback(() => {
    const readerRect = readerRef.current?.getBoundingClientRect();
    const center = readerRect ? (readerRect.top + readerRect.bottom) / 2 : window.innerHeight / 2;
    let bestIdx = activePage;
    let bestDist = Number.POSITIVE_INFINITY;
    document.querySelectorAll<HTMLElement>(".continuous-page[data-page-idx]").forEach(node => {
      const rect = node.getBoundingClientRect();
      const dist = Math.abs((rect.top + rect.bottom) / 2 - center);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = Number(node.dataset.pageIdx ?? activePage);
      }
    });
    if (bestIdx !== visiblePage) setVisiblePage(bestIdx);
    if (bestIdx !== activePage) onPageSelect(bestIdx, "debounced");
  }, [activePage, onPageSelect, visiblePage]);

  const onReaderScroll = () => {
    if (scrollRafRef.current !== null) return;
    scrollRafRef.current = window.requestAnimationFrame(() => {
      scrollRafRef.current = null;
      updateVisiblePage();
    });
  };

  const jumpToPage = () => {
    const raw = window.prompt("Go to page", String(visiblePage + 1));
    if (!raw) return;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    const idx = clampNumber(parsed - 1, 0, Math.max(0, data.pages.length - 1));
    onPageSelect(idx);
  };

  return (
    <div ref={readerRef} className="continuous-reader" onScroll={onReaderScroll}>
      {showIndicator && (
        <button className="page-indicator-float" onClick={jumpToPage}>
          Page {visiblePage + 1} / {data.pages.length}
        </button>
      )}
      {data.pages.map(pg => (
        <ContinuousPage
          key={pg.id}
          idx={pg.idx}
          active={activePage === pg.idx}
          chapterDir={data.meta.chapterDir}
          imageMode={imageMode}
          pageRevision={`rv${pg.render_version ?? 0}:${pageVersions[pg.idx] ?? 0}:${pg.status.join("")}:${pg.dirty ? 1 : 0}`}
          regions={editorRegionsForPage(data, pg.idx)}
          selectedRegion={selectedRegion}
          regionDrafts={regionDrafts}
          showEnglishOverlay={showEnglishOverlay}
          showOverlayBoxes={showOverlayBoxes}
          zoom={zoom}
          showMarker={showMarkers}
          totalPages={data.pages.length}
          onSelect={(idx) => onPageSelect(idx, "local")}
          onRegionSelect={onRegionSelect}
          onPreviewRegion={onPreviewRegion}
          onCommitBBox={onCommitBBox}
        />
      ))}
    </div>
  );
};

const CanvasArea = ({
  data, activePage, selectedRegion, setSelectedRegion, zoom, setZoom,
  regionDrafts, onPreviewRegion, onCommitBBox, showEnglishOverlay, setShowEnglishOverlay,
  readerMode, setReaderMode, showPageIndicator, setShowPageIndicator, scrollTarget, onPageSelect,
  imageMode, setImageMode, pageVersions, cleanupDebug, debugOverlays, onPageSizeChange,
}: {
  data:             Bootstrap;
  activePage:       number;
  selectedRegion:   Region | null;
  setSelectedRegion:(r: Region | null) => void;
  zoom:             number;
  setZoom:          (fn: (z: number) => number) => void;
  regionDrafts:     Record<string, RegionDraft>;
  onPreviewRegion:  (regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => void;
  onCommitBBox:     (region: Region, bbox: RegionBBox) => Promise<boolean>;
  showEnglishOverlay: boolean;
  setShowEnglishOverlay: (v: boolean) => void;
  readerMode:       "single" | "continuous";
  setReaderMode:    (mode: "single" | "continuous") => void;
  showPageIndicator: boolean;
  setShowPageIndicator: (v: boolean) => void;
  scrollTarget:      number | null;
  onPageSelect:     (idx: number, mode?: PageSelectMode) => void;
  imageMode:         ImageMode;
  setImageMode:      (mode: ImageMode) => void;
  pageVersions:      Record<number, number>;
  cleanupDebug:      CleanupDebugResponse | null;
  debugOverlays:     DebugOverlayToggles;
  onPageSizeChange:  (pageIdx: number, size: { w: number; h: number }) => void;
}) => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [cleanedImageSrc, setCleanedImageSrc] = useState<string | null>(null);
  const [showPageMarkers, setShowPageMarkers] = useState(true);
  const [showAlignmentGuides, setShowAlignmentGuides] = useState<boolean>(() => {
    try { return localStorage.getItem("ml.alignmentGuides") === "1"; } catch { return false; }
  });
  const [snapAlignment, setSnapAlignment] = useState<boolean>(() => {
    try { return localStorage.getItem("ml.snapAlignment") === "1"; } catch { return false; }
  });
  /**
   * Pass 5: UI-only toggle that hides region rectangles / labels / resize
   * handles while keeping the rendered translated text, cleaned image, and
   * export output unchanged. Deliberately NOT tied to `region.visible`
   * (that field still means "exclude from typeset + export").
   */
  const [showOverlayBoxes, setShowOverlayBoxes] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem("ml.showOverlayBoxes");
      return v === null ? true : v === "1";
    } catch { return true; }
  });
  useEffect(() => {
    try { localStorage.setItem("ml.showOverlayBoxes", showOverlayBoxes ? "1" : "0"); } catch {}
  }, [showOverlayBoxes]);
  useEffect(() => {
    try { localStorage.setItem("ml.alignmentGuides", showAlignmentGuides ? "1" : "0"); } catch {}
  }, [showAlignmentGuides]);
  useEffect(() => {
    try { localStorage.setItem("ml.snapAlignment", snapAlignment ? "1" : "0"); } catch {}
  }, [snapAlignment]);
  useEffect(() => {
    const toggleGuides = () => setShowAlignmentGuides(v => !v);
    const toggleSnap = () => setSnapAlignment(v => !v);
    window.addEventListener("ml:toggle-alignment-guides", toggleGuides);
    window.addEventListener("ml:toggle-alignment-snap", toggleSnap);
    return () => {
      window.removeEventListener("ml:toggle-alignment-guides", toggleGuides);
      window.removeEventListener("ml:toggle-alignment-snap", toggleSnap);
    };
  }, []);
  const [imageError, setImageError] = useState("");
  const [imageSize, setImageSize] = useState({ w: 460, h: 660 });
  const [dragPreview, setDragPreview] = useState<{ id: string; bbox: RegionBBox } | null>(null);
  const [previewStyleSnapshots, setPreviewStyleSnapshots] = useState<Record<string, FrozenPreviewStyle>>({});
  const [backendPreviewSprites, setBackendPreviewSprites] = useState<Record<string, BackendPreviewSprite>>({});
  const [cssPreviewFallbacks, setCssPreviewFallbacks] = useState<Record<string, string>>({});
  const previewTimers = useRef<Record<string, number>>({});
  const previewCallCounts = useRef<Record<string, number>>({});
  const bboxCommitCount = useRef(0);
  const activePageGuardRef = useRef(activePage);
  const prevKey = useRef<string>("");
  const cleanedKey = useRef<string>("");
  const imageRef = useRef<HTMLImageElement | null>(null);
  const pageRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    mode: "move" | ResizeHandle;
    region: Region;
    startPoint: { x: number; y: number };
    startBox: RegionBBox;
    currentBox: RegionBBox;
    visual: FrozenPreviewStyle;
    moves: number;
    committed: boolean;
  } | null>(null);
  const activePageState = data.pages[activePage];
  const isTypesetDone = activePageState?.status?.[4] === "done";
  const isRenderDirty = Boolean(activePageState?.dirty);
  const hasRegionDrafts = Object.keys(regionDrafts).length > 0;
  const displayImageMode: ImageMode = imageMode;
  const pageStatusKey = `rv${activePageState?.render_version ?? 0}:${pageVersions[activePage] ?? 0}:${activePageState?.status?.join("") ?? ""}:${activePageState?.dirty ? 1 : 0}`;
  const spriteSlot = (regionId: string) => `${activePage}:${regionId}`;
  const previewSpriteMatches = (sprite: BackendPreviewSprite | undefined, region: Region & RegionDraft) =>
    Boolean(sprite?.b64 && sprite.styleKey === regionPreviewSpriteKey({ ...region, ...sprite.sourceBox }));

  useEffect(() => {
    activePageGuardRef.current = activePage;
  }, [activePage]);

  // Load image whenever the page changes
  useEffect(() => {
    // Only fetch if a chapter is loaded
    if (!data.meta.chapterDir) {
      setImageSrc(null);
      setImageError("");
      prevKey.current = "";
      return;
    }
    const fetchKey = `${data.meta.chapterDir}:${activePage}:${pageStatusKey}:${displayImageMode}`;
    if (prevKey.current === fetchKey) return;
    prevKey.current = fetchKey;
    setImageError("");

    let cancelled = false;
    api.getPageImage(activePage, displayImageMode).then(async resp => {
      if (cancelled || prevKey.current !== fetchKey) return;
      if (resp.ok && resp.b64) {
        setImageSrc(`data:image/png;base64,${resp.b64}`);
        return;
      }
      const rawResp = await api.getPageImage(activePage, "raw");
      if (cancelled || prevKey.current !== fetchKey) return;
      if (rawResp.ok && rawResp.b64) {
        setImageSrc(`data:image/png;base64,${rawResp.b64}`);
        return;
      }
      setImageError(rawResp.ok ? "Page image unavailable." : (rawResp.error || "Page image failed to load."));
      setImageSrc(null);
    }).catch(err => {
      if (!cancelled) {
        setImageSrc(null);
        setImageError(err instanceof Error ? err.message : String(err));
      }
    });
    return () => { cancelled = true; };
  }, [activePage, data.meta.chapterDir, pageStatusKey, displayImageMode]);

  useEffect(() => {
    if (!data.meta.chapterDir || displayImageMode !== "best" || !isTypesetDone) {
      setCleanedImageSrc(null);
      cleanedKey.current = "";
      return;
    }
    const fetchKey = `${data.meta.chapterDir}:${activePage}:${pageStatusKey}:cleaned`;
    if (cleanedKey.current === fetchKey) return;
    cleanedKey.current = fetchKey;

    let cancelled = false;
    api.getPageImage(activePage, "cleaned").then(resp => {
      if (cancelled || cleanedKey.current !== fetchKey) return;
      setCleanedImageSrc(resp.ok && resp.b64 ? `data:image/png;base64,${resp.b64}` : null);
    }).catch(() => {
      if (!cancelled) setCleanedImageSrc(null);
    });
    return () => { cancelled = true; };
  }, [activePage, data.meta.chapterDir, pageStatusKey, displayImageMode, isTypesetDone]);

  useEffect(() => {
    if (displayImageMode === "best" && isTypesetDone && !isRenderDirty && !dragRef.current) {
      setBackendPreviewSprites({});
      setCssPreviewFallbacks({});
    }
  }, [activePage, pageStatusKey, displayImageMode, isTypesetDone, isRenderDirty]);

  useEffect(() => {
    setBackendPreviewSprites({});
    setCssPreviewFallbacks({});
    setDragPreview(null);
  }, [activePage, pageStatusKey, showOverlayBoxes, selectedRegion?.id]);

  const onImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const next = { w: img.naturalWidth, h: img.naturalHeight };
    setImageSize(next);
    onPageSizeChange(activePage, next);
  };

  const scale = zoom / 100;
  const dispW = Math.round(imageSize.w * scale);
  const dispH = Math.round(imageSize.h * scale);

  const clampBbox = (bbox: RegionBBox): RegionBBox => {
    const minW = Math.min(8, imageSize.w);
    const minH = Math.min(8, imageSize.h);
    const w = Math.min(Math.max(Math.round(bbox.w), minW), imageSize.w);
    const h = Math.min(Math.max(Math.round(bbox.h), minH), imageSize.h);
    return {
      x: Math.min(Math.max(Math.round(bbox.x), 0), Math.max(0, imageSize.w - w)),
      y: Math.min(Math.max(Math.round(bbox.y), 0), Math.max(0, imageSize.h - h)),
      w,
      h,
    };
  };

  const pointFromEvent = (e: React.PointerEvent) => {
    const rect = pageRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: Math.min(Math.max((e.clientX - rect.left) / scale, 0), imageSize.w),
      y: Math.min(Math.max((e.clientY - rect.top) / scale, 0), imageSize.h),
    };
  };

  const resizeBox = (handle: ResizeHandle, start: RegionBBox, dx: number, dy: number): RegionBBox => {
    const minW = Math.min(8, imageSize.w);
    const minH = Math.min(8, imageSize.h);
    let left = start.x;
    let top = start.y;
    let right = start.x + start.w;
    let bottom = start.y + start.h;

    if (handle.includes("w")) left = Math.min(left + dx, right - minW);
    if (handle.includes("e")) right = Math.max(right + dx, left + minW);
    if (handle.includes("n")) top = Math.min(top + dy, bottom - minH);
    if (handle.includes("s")) bottom = Math.max(bottom + dy, top + minH);

    left = Math.max(0, left);
    top = Math.max(0, top);
    right = Math.min(imageSize.w, right);
    bottom = Math.min(imageSize.h, bottom);

    if (right - left < minW) {
      if (handle.includes("w")) left = Math.max(0, right - minW);
      else right = Math.min(imageSize.w, left + minW);
    }
    if (bottom - top < minH) {
      if (handle.includes("n")) top = Math.max(0, bottom - minH);
      else bottom = Math.min(imageSize.h, top + minH);
    }

    return clampBbox({ x: left, y: top, w: right - left, h: bottom - top });
  };

  const snapBoxToGuides = (box: RegionBBox, region: Region, snapActive: boolean): RegionBBox => {
    if (!snapActive || imageSize.w <= 0 || imageSize.h <= 0) return box;
    const threshold = 6 / Math.max(0.2, scale);
    const center = boxCenter(box);
    const containerCenter = boxCenter(containerBoxForRegion(region));
    const xGuides = [imageSize.w / 2, containerCenter.x];
    const yGuides = [imageSize.h / 2, containerCenter.y];
    regions.forEach(other => {
      if (other.id === region.id || !other.visible) return;
      xGuides.push(other.x, other.x + other.w / 2, other.x + other.w);
      yGuides.push(other.y, other.y + other.h / 2, other.y + other.h);
    });
    let next = { ...box };
    const nearestX = xGuides.reduce<{ value: number; dist: number } | null>((best, guide) => {
      const dist = Math.abs(center.x - guide);
      return dist <= threshold && (!best || dist < best.dist) ? { value: guide, dist } : best;
    }, null);
    const nearestY = yGuides.reduce<{ value: number; dist: number } | null>((best, guide) => {
      const dist = Math.abs(center.y - guide);
      return dist <= threshold && (!best || dist < best.dist) ? { value: guide, dist } : best;
    }, null);
    if (nearestX) next.x += nearestX.value - center.x;
    if (nearestY) next.y += nearestY.value - center.y;
    return clampBbox(next);
  };

  const startDrag = (e: React.PointerEvent<HTMLDivElement>, region: Region, mode: "move" | ResizeHandle) => {
    e.preventDefault();
    e.stopPropagation();
    renderDebug("select.dragStart", {
      page: activePage,
      region: region.idx,
      action: mode,
      baseLayer: displayImageMode,
      dirty: isRenderDirty,
      style: regionStyleDebug(region),
    });
    if (region.locked) {
      setSelectedRegion(region);
      return;
    }
    const draft = regionDrafts[region.id] ?? {};
    const box = clampBbox({
      x: draft.x ?? region.x,
      y: draft.y ?? region.y,
      w: draft.w ?? region.w,
      h: draft.h ?? region.h,
    });
    const point = pointFromEvent(e);
    setSelectedRegion({ ...region, ...box });
    const visual = {
      styleKey: regionVisualStyleKey(region),
      color: region.fg || "#111111",
      fontFamily: previewFontFamily(region),
      fontSize: overlayFontSize({ ...region, ...box }),
      textAlign: region.align,
      textShadow: previewTextShadow(region),
    };
    dragRef.current = {
      pointerId: e.pointerId,
      mode,
      region,
      startPoint: point,
      startBox: box,
      currentBox: box,
      visual,
      moves: 0,
      committed: false,
    };
    setPreviewStyleSnapshots(prev => ({ ...prev, [`${activePage}:${region.id}`]: visual }));
    renderDebug("drag.pointerdown", { page: activePage, region: region.idx, mode, counts: { down: 1, move: 0, up: 0 } });
    pageRef.current?.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    const point = pointFromEvent(e);
    const dx = point.x - drag.startPoint.x;
    const dy = point.y - drag.startPoint.y;
    const next = drag.mode === "move"
      ? clampBbox({ ...drag.startBox, x: drag.startBox.x + dx, y: drag.startBox.y + dy })
      : resizeBox(drag.mode, drag.startBox, dx, dy);
    const snapped = drag.mode === "move" ? snapBoxToGuides(next, drag.region, snapAlignment || e.altKey) : next;
    drag.currentBox = snapped;
    drag.moves += 1;
    renderDebug("bbox.preview", {
      page: activePage,
      region: drag.region.idx,
      action: drag.mode,
      moves: drag.moves,
      baseLayer: displayImageMode,
      dirty: isRenderDirty,
      bbox: snapped,
      style: regionStyleDebug(drag.region),
    });
    setDragPreview({ id: drag.region.id, bbox: snapped });
  };

  const onPointerUp = async (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    if (drag.committed) return;
    drag.committed = true;
    const finalBox = drag.currentBox;
    dragRef.current = null;
    setDragPreview(null);
    pageRef.current?.releasePointerCapture(e.pointerId);
    renderDebug("drag.pointerup", {
      page: activePage,
      region: drag.region.idx,
      moves: drag.moves,
      counts: { down: 1, move: drag.moves, up: 1 },
      backendBBoxCallsExpected: 1,
      finalBox,
    });
    const unchanged =
      finalBox.x === drag.startBox.x && finalBox.y === drag.startBox.y &&
      finalBox.w === drag.startBox.w && finalBox.h === drag.startBox.h;
    if (unchanged) {
      renderDebug("drag.pointerup.noop", { page: activePage, region: drag.region.idx, didCommit: false, finalBox });
      return;
    }
    onPreviewRegion(drag.region.id, finalBox);
    bboxCommitCount.current += 1;
    renderDebug("api.updateRegionBBox.call", {
      page: activePage,
      region: drag.region.idx,
      count: bboxCommitCount.current,
      from: "pointerup",
      bbox: finalBox,
    });
    const didCommit = await onCommitBBox(drag.region, finalBox);
    renderDebug("drag.pointerup.commitResult", { page: activePage, region: drag.region.idx, didCommit, finalBox });
    if (didCommit) {
      requestPreviewSprite({ ...drag.region, ...finalBox }, finalBox, "bbox_commit", 0);
    }
  };

  const onPointerCancel = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    dragRef.current = null;
    setDragPreview(null);
    pageRef.current?.releasePointerCapture(e.pointerId);
    renderDebug("drag.pointercancel", {
      page: activePage,
      region: drag.region.idx,
      moves: drag.moves,
      backendBBoxCallsExpected: 0,
    });
  };

  const regions = editorRegionsForPage(data, activePage).map(r => ({
    ...r,
    ...(regionDrafts[r.id] ?? {}),
    ...(dragPreview?.id === r.id ? dragPreview.bbox : {}),
  }));
  const requestPreviewSprite = (region: Region | (Region & RegionDraft), draft: RegionDraft, reason: string, debounceMs = 0) => {
    const activeForPreview = selectedRegion?.id === region.id || Boolean(regionDrafts[region.id]) || dragRef.current?.region.id === region.id;
    if (!data.meta.chapterDir || !showOverlayBoxes || !showEnglishOverlay || !activeForPreview || !region.visible || !region.tl || dragRef.current?.region.id === region.id) return;
    const slot = spriteSlot(region.id);
    if (previewTimers.current[slot]) window.clearTimeout(previewTimers.current[slot]);
    previewTimers.current[slot] = window.setTimeout(() => {
      const requestPage = activePage;
      const ownerPage = regionOwnerPage(region as Region, requestPage);
      previewCallCounts.current[reason] = (previewCallCounts.current[reason] ?? 0) + 1;
      renderDebug("preview.sprite.request", {
        page: requestPage,
        ownerPage,
        region: region.idx,
        reason,
        count: previewCallCounts.current[reason],
        key: regionPreviewSpriteKey(region),
      });
      const requestStyleKey = regionPreviewSpriteKey(region);
      api.getRegionPreviewSprite(region.idx, draft, ownerPage).then(resp => {
        const latest = editorRegionsForPage(data, requestPage).find(r => r.id === region.id);
        if (requestPage !== activePageGuardRef.current || !showOverlayBoxes || !latest || regionPreviewSpriteKey({ ...latest, ...(regionDrafts[region.id] ?? {}) }) !== requestStyleKey) {
          return;
        }
        if (resp.ok && resp.b64) {
          renderDebug("preview.sprite.backend", {
            page: requestPage,
            ownerPage,
            region: region.idx,
            reason,
            font: resp.font,
            resolved_font_size: resp.resolved_font_size,
            fg: resp.fg,
            outline: resp.outline,
            shadow: resp.shadow,
            sprite: { x: resp.x, y: resp.y, w: resp.w, h: resp.h },
          });
          setBackendPreviewSprites(prev => ({
            ...prev,
            [slot]: {
              ...resp,
              styleKey: requestStyleKey,
              sourceBox: { x: region.x, y: region.y, w: region.w, h: region.h },
              reason,
            },
          }));
          setCssPreviewFallbacks(prev => {
            if (!prev[slot]) return prev;
            const next = { ...prev };
            delete next[slot];
            return next;
          });
        } else {
          renderDebug("preview.sprite.fallback", {
            page: requestPage,
            ownerPage,
            region: region.idx,
            reason,
            path: "css",
            error: resp.error,
          });
          setBackendPreviewSprites(prev => {
            const next = { ...prev };
            delete next[slot];
            return next;
          });
          setCssPreviewFallbacks(prev => ({ ...prev, [slot]: reason }));
        }
      });
    }, debounceMs);
  };

  const activeBackendSprites = showOverlayBoxes ? regions.reduce<BackendPreviewSprite[]>((acc, r) => {
    const sprite = backendPreviewSprites[spriteSlot(r.id)];
    if (previewSpriteMatches(sprite, r) && (selectedRegion?.id === r.id || dragPreview?.id === r.id || regionDrafts[r.id])) {
      acc.push(sprite);
    }
    return acc;
  }, []) : [];
  const hasBitmapPreview = activeBackendSprites.length > 0;
  const cleanedPatchBoxes = activeBackendSprites.map(sprite => {
      const x1 = Math.min(sprite.sourceBox.x, sprite.x);
      const y1 = Math.min(sprite.sourceBox.y, sprite.y);
      const x2 = Math.max(sprite.sourceBox.x + sprite.sourceBox.w, sprite.x + sprite.w);
      const y2 = Math.max(sprite.sourceBox.y + sprite.sourceBox.h, sprite.y + sprite.h);
      return clampBbox({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
  });
  const selectedLiveRegion = selectedRegion ? regions.find(r => r.id === selectedRegion.id) ?? null : null;
  const guidesVisible = showOverlayBoxes && imageSize.w > 0 && imageSize.h > 0 && Boolean(selectedLiveRegion) && (showAlignmentGuides || Boolean(dragPreview));
  const alignmentGuides = guidesVisible && selectedLiveRegion ? (() => {
    const region = selectedLiveRegion;
    const center = boxCenter(region);
    const container = containerBoxForRegion(region);
    const containerCenter = boxCenter(container);
    const guides: Array<{ id: string; axis: "v" | "h"; value: number; kind: "page" | "container" | "selected" | "near" }> = [
      { id: "page-v", axis: "v", value: imageSize.w / 2, kind: "page" },
      { id: "page-h", axis: "h", value: imageSize.h / 2, kind: "page" },
      { id: "container-v", axis: "v", value: containerCenter.x, kind: "container" },
      { id: "container-h", axis: "h", value: containerCenter.y, kind: "container" },
      { id: "selected-v", axis: "v", value: center.x, kind: "selected" },
      { id: "selected-h", axis: "h", value: center.y, kind: "selected" },
    ];
    regions.forEach(other => {
      if (other.id === region.id || !other.visible) return;
      guides.push({ id: `${other.id}-left`, axis: "v", value: other.x, kind: "near" });
      guides.push({ id: `${other.id}-cx`, axis: "v", value: other.x + other.w / 2, kind: "near" });
      guides.push({ id: `${other.id}-right`, axis: "v", value: other.x + other.w, kind: "near" });
      guides.push({ id: `${other.id}-top`, axis: "h", value: other.y, kind: "near" });
      guides.push({ id: `${other.id}-cy`, axis: "h", value: other.y + other.h / 2, kind: "near" });
      guides.push({ id: `${other.id}-bottom`, axis: "h", value: other.y + other.h, kind: "near" });
    });
    return guides;
  })() : [];

  useEffect(() => {
    const onSpriteRequest = (ev: Event) => {
      const detail = (ev as CustomEvent<{ page: number; regionId: string; reason: string; draft?: RegionDraft }>).detail;
      if (!detail || detail.page !== activePage) return;
      const region = editorRegionsForPage(data, detail.page).find(r => r.id === detail.regionId);
      if (!region) return;
      const draft = detail.draft ?? regionDrafts[region.id] ?? {};
      requestPreviewSprite({ ...region, ...draft }, draft, detail.reason, 160);
    };
    window.addEventListener("ml:preview-sprite-request", onSpriteRequest);
    return () => window.removeEventListener("ml:preview-sprite-request", onSpriteRequest);
  }, [activePage, data, regionDrafts, data.meta.chapterDir]);

  useEffect(() => {
    if (!data.meta.chapterDir || !showOverlayBoxes || !showEnglishOverlay || dragRef.current) return;
    regions.forEach(region => {
      if (region.visible && region.tl && (selectedRegion?.id === region.id || regionDrafts[region.id])) {
        requestPreviewSprite(region, {}, "overlay_restore", 0);
      }
    });
  }, [activePage, pageStatusKey, data.meta.chapterDir, imageMode, showEnglishOverlay, showOverlayBoxes, selectedRegion?.id, regionDrafts]);

  const overlayFontSize = (r: Region) => {
    if (r.size && r.size > 0) return Math.max(8, Math.min(72, r.size)) * scale;
    const textLen = Math.max(1, (r.tl || "").length);
    const base = Math.min(34, Math.max(10, Math.sqrt((r.w * r.h) / textLen) * 0.58));
    return base * scale;
  };
  const previewFontFamily = (r: Region) => {
    const stored = (r.font || "").trim();
    // Guard: backend sends bubble_role ("dialog", "bold", "thought", "sfx") as
    // the font field when no explicit font has been chosen.  These are NOT CSS
    // font-family names — passing them through makes the browser silently skip
    // them and land on Comic Sans.  Treat any known role label as "auto".
    const isRealFont = stored && !ROLE_FONT_NAMES.has(stored.toLowerCase());
    return isRealFont
      ? `"${stored}", "Comic Sans MS", "Segoe UI", sans-serif`
      : `"Comic Sans MS", "Segoe UI", sans-serif`;
  };
  const previewTextShadow = (r: Region) => {
    const outline = r.outline || r.bg || "#ffffff";
    const width = Math.max(0, Math.min(4, Number(r.outline_width ?? 1)));
    const stroke = width > 0
      ? [
          `${-width}px 0 ${outline}`, `${width}px 0 ${outline}`,
          `0 ${-width}px ${outline}`, `0 ${width}px ${outline}`,
          `${-width}px ${-width}px ${outline}`, `${width}px ${-width}px ${outline}`,
          `${-width}px ${width}px ${outline}`, `${width}px ${width}px ${outline}`,
        ]
      : [];
    if (r.shadow_on) {
      const sx = Number(r.shadow_offset_x ?? 1);
      const sy = Number(r.shadow_offset_y ?? 2);
      const blur = Math.max(0, Number(r.shadow_blur ?? 2));
      stroke.push(`${sx}px ${sy}px ${blur}px ${r.shadow || "rgba(0,0,0,0.45)"}`);
    }
    if (r.glow_on) {
      const radius = Math.max(1, Number(r.glow_radius ?? 4));
      stroke.push(`0 0 ${radius}px ${r.glow || "rgba(255,255,255,0.6)"}`);
    }
    return stroke.length ? stroke.join(", ") : "none";
  };

  useEffect(() => {
    renderDebug("canvas.base", {
      page: activePage,
      baseLayer: displayImageMode,
      selectedLayer: imageMode,
      reason: displayImageMode === "best"
        ? (isTypesetDone && !isRenderDirty ? "best-typeset" : "best-available")
        : (!showEnglishOverlay ? "preview-overlay-raw-display" : "explicit"),
      dirty: isRenderDirty,
      drafts: Object.keys(regionDrafts),
    });
  }, [activePage, imageMode, displayImageMode, showEnglishOverlay, hasRegionDrafts, isRenderDirty, isTypesetDone, regionDrafts]);

  useEffect(() => {
    renderDebug("canvas.preview", {
      page: activePage,
      baseLayer: displayImageMode,
      selectedLayer: imageMode,
      dirty: isRenderDirty,
      showEnglishOverlay,
      regions: regions.map(r => ({
        region: r.idx,
        shown: Boolean(showEnglishOverlay && (displayImageMode === "cleaned" || displayImageMode === "raw" || !isTypesetDone || Boolean(regionDrafts[r.id]) || dragPreview?.id === r.id) && r.visible && r.tl),
        reason: !showEnglishOverlay ? "overlay-disabled"
          : !(displayImageMode === "cleaned" || displayImageMode === "raw" || !isTypesetDone || Boolean(regionDrafts[r.id]) || dragPreview?.id === r.id) ? "final-typeset-clean"
          : !r.visible ? "region-hidden"
          : !r.tl ? "no-translation"
          : "dirty-or-not-typeset",
        style: regionStyleDebug(r),
      })),
    });
  }, [activePage, imageMode, displayImageMode, isRenderDirty, showEnglishOverlay, isTypesetDone, regionDrafts, dragPreview, regions]);

  const renderDebugBox = (key: DebugOverlayKey, bbox: number[] | null | undefined, color: string, label: string) => {
    if (!debugOverlays[key] || !bbox || bbox.length !== 4) return null;
    const [x, y, w, h] = bbox.map(Number);
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
    return (
      <div
        key={key}
        className="debug-box-overlay"
        style={{ left: x * scale, top: y * scale, width: w * scale, height: h * scale, color }}
      >
        <div className="debug-box-label">{label}</div>
      </div>
    );
  };
  const renderDebugMask = (key: DebugOverlayKey, maskKey: keyof NonNullable<CleanupDebugResponse["masks"]>, color: string) => {
    const mask = cleanupDebug?.masks?.[maskKey];
    const bbox = mask?.bbox;
    if (!debugOverlays[key] || !mask?.available || !mask.b64 || !bbox || bbox.length !== 4) return null;
    const [x, y, w, h] = bbox.map(Number);
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
    const maskUrl = `url(data:image/png;base64,${mask.b64})`;
    return (
      <div
        key={key}
        className="debug-mask-overlay"
        style={{
          left: x * scale,
          top: y * scale,
          width: w * scale,
          height: h * scale,
          opacity: 0.5,
          background: color,
          WebkitMaskImage: maskUrl,
          maskImage: maskUrl,
          WebkitMaskSize: "100% 100%",
          maskSize: "100% 100%",
        }}
      />
    );
  };

  return (
    <div className="ml-canvas-wrap">
      {/* Toolbar */}
      <div className="ml-canvas-toolbar no-select">
        <button className="btn-icon" onClick={() => setZoom(z => Math.max(z - 15, 20))} title="Zoom out"><Svg icon="zoomOut" size={13} /></button>
        <span className="zoom-display">{zoom}%</span>
        <button className="btn-icon" onClick={() => setZoom(z => Math.min(z + 15, 300))} title="Zoom in"><Svg icon="zoomIn" size={13} /></button>
        <button className="btn-icon" onClick={() => setZoom(() => 85)} title="Reset zoom"><Svg icon="fit" size={13} /></button>
        <div className="divider-v" />
        <span style={{ fontFamily: "var(--fnt-mono)", fontSize: 10, color: "var(--t3)" }}>
          {imageSize.w} × {imageSize.h}px
        </span>
        {regions.length > 0 && (
          <span style={{ marginLeft: 8, fontFamily: "var(--fnt-mono)", fontSize: 10, color: "var(--teal)" }}>
            {regions.length} region{regions.length !== 1 ? "s" : ""}
          </span>
        )}
        <div className="divider-v" />
        {(["best", "raw", "cleaned", "typeset"] as const).map(mode => (
          <button
            key={mode}
            className={`btn-ghost ${imageMode === mode ? "active" : ""}`}
            style={{ padding: "4px 7px", fontSize: 10, textTransform: "capitalize" }}
            onClick={() => setImageMode(mode)}
            title={`Show ${mode} base image`}
          >
            {mode}
          </button>
        ))}
        <ModeBadge mode={displayImageMode} />
        <div className="divider-v" />
        <button
          className={`btn-ghost ${readerMode === "single" ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setReaderMode("single")}
          title="Single page reader"
        >
          Single
        </button>
        <button
          className={`btn-ghost ${readerMode === "continuous" ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setReaderMode("continuous")}
          title="Continuous vertical reader"
        >
          Scroll
        </button>
        <button
          className={`btn-ghost ${showPageIndicator ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setShowPageIndicator(!showPageIndicator)}
          title="Toggle floating current-page indicator"
        >
          Page
        </button>
        <button
          className={`btn-ghost ${showOverlayBoxes ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setShowOverlayBoxes(!showOverlayBoxes)}
          title="Show/hide editor overlays: boxes, labels, handles, debug boxes, and live preview text. Does not affect the base image or exported output."
        >
          Boxes
        </button>
        <button
          className={`btn-ghost ${showAlignmentGuides ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setShowAlignmentGuides(!showAlignmentGuides)}
          title="Show page, container, selected box, and nearby region alignment guides"
        >
          Guides
        </button>
        <button
          className={`btn-ghost ${snapAlignment ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setSnapAlignment(!snapAlignment)}
          title="Snap selected box center to nearby guides while dragging. Hold Alt to snap temporarily."
        >
          Snap
        </button>
        <button
          className={`btn-ghost ${showPageMarkers ? "active" : ""}`}
          style={{ padding: "4px 7px", fontSize: 10 }}
          onClick={() => setShowPageMarkers(!showPageMarkers)}
          title="Toggle per-page start markers"
        >
          Markers
        </button>
        <label className="canvas-toggle" title="Show or hide the live English overlay">
          <input
            type="checkbox"
            checked={showEnglishOverlay}
            onChange={e => {
              renderDebug("preview_overlay", { visible: e.target.checked, base_mode: imageMode, page: activePage });
              setShowEnglishOverlay(e.target.checked);
            }}
          />
          Preview overlay
        </label>
      </div>

      {/* Canvas */}
      <div className="ml-canvas-area">
        {!data.meta.chapterDir ? (
          /* No chapter — show prompt */
          <div className="open-chapter-prompt">
            <Svg icon="folder" size={36} />
            <h2>No chapter open</h2>
            <p>Click "Open Chapter" in the toolbar to import a folder of manga/manhwa images.</p>
          </div>
        ) : readerMode === "continuous" ? (
          <ContinuousReader
            data={data}
            activePage={activePage}
            selectedRegion={selectedRegion}
            regionDrafts={regionDrafts}
            showEnglishOverlay={showEnglishOverlay}
            showOverlayBoxes={showOverlayBoxes}
            imageMode={displayImageMode}
            pageVersions={pageVersions}
            zoom={zoom}
            showIndicator={showPageIndicator}
            showMarkers={showPageMarkers}
            scrollTarget={scrollTarget}
            onPageSelect={onPageSelect}
            onRegionSelect={setSelectedRegion}
            onPreviewRegion={onPreviewRegion}
            onCommitBBox={onCommitBBox}
          />
        ) : (
          <div
            ref={pageRef}
            className="manga-page"
            style={{ width: dispW, height: dispH }}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerCancel}
          >
            {imageSrc ? (
              <img
                ref={imageRef}
                src={imageSrc}
                onLoad={onImgLoad}
                style={{ width: dispW, height: dispH, display: "block" }}
                alt={`Page ${activePage + 1}`}
                draggable={false}
              />
            ) : (
              /* Placeholder while loading */
              <div style={{
                width: dispW, height: dispH, background: "var(--bg-3)",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "var(--t4)", fontFamily: "var(--fnt-mono)", fontSize: 11,
              }}>
                {imageError || "Loading…"}
              </div>
            )}
            {showEnglishOverlay && hasBitmapPreview && cleanedImageSrc && displayImageMode === "best" && cleanedPatchBoxes.map((box, i) => (
              <div
                key={`${box.x}:${box.y}:${i}`}
                className="drag-cleaned-patch"
                style={{
                  left:   box.x * scale,
                  top:    box.y * scale,
                  width:  box.w * scale,
                  height: box.h * scale,
                }}
              >
                <img
                  src={cleanedImageSrc}
                  alt=""
                  draggable={false}
                  style={{
                    left:   -box.x * scale,
                    top:    -box.y * scale,
                    width:  dispW,
                    height: dispH,
                  }}
                />
              </div>
            ))}

            {showOverlayBoxes && cleanupDebug && (
              <>
                {renderDebugMask("textMask", "text_mask", "rgba(64,130,255,0.74)")}
                {renderDebugMask("cleanupMask", "cleanup_mask", "rgba(255,0,220,0.74)")}
                {renderDebugMask("haloMask", "halo_mask", "rgba(180,120,255,0.72)")}
                {renderDebugMask("manualMask", "manual_mask", "rgba(0,220,110,0.72)")}
                {renderDebugMask("groupedMask", "grouped_mask", "rgba(255,0,220,0.56)")}
                {renderDebugBox("yoloBox", cleanupDebug.boxes?.detector_text_bbox, "#38e8ff", `YOLO text${cleanupDebug.labels?.detector ? ` · ${cleanupDebug.labels.detector}` : ""}`)}
                {renderDebugBox("editableBox", cleanupDebug.boxes?.editable_bbox, "#f5d84c", "Editable region")}
                {renderDebugBox("containerBox", cleanupDebug.boxes?.container_bbox, "#ff5c5c", `Container${cleanupDebug.labels?.container ? ` · ${cleanupDebug.labels.container}` : ""}`)}
                {renderDebugBox("patchBox", cleanupDebug.boxes?.patch_bbox, "#ff9f32", `Patch${cleanupDebug.labels?.patch ? ` · ${cleanupDebug.labels.patch}` : ""}`)}
              </>
            )}

            {alignmentGuides.map(guide => (
              <div
                key={guide.id}
                className={`alignment-guide ${guide.axis} ${guide.kind}`}
                style={guide.axis === "v" ? { left: guide.value * scale } : { top: guide.value * scale }}
              />
            ))}

            {/* Region overlays (positioned in original image coordinates, displayed at current zoom) */}
            {imageSrc && showOverlayBoxes && regions.map(r => {
              if (!r.visible && selectedRegion?.id !== r.id) return null;
              // Pass 4: hide SFX regions entirely when the master toggle is OFF.
              if (r.pipeline_disabled && selectedRegion?.id !== r.id) return null;
              const rScale = dispW / imageSize.w;
              const isSelected = selectedRegion?.id === r.id;
              const isDragging = dragRef.current?.region.id === r.id;
              const dragMode = isDragging ? dragRef.current?.mode : null;
              const isResizeDragging = Boolean(isDragging && dragMode !== "move");
              const hasLocalPreview = Boolean(regionDrafts[r.id]) || dragPreview?.id === r.id;
              const snapshot = previewStyleSnapshots[`${activePage}:${r.id}`];
              const frozenVisual = isDragging
                ? dragRef.current?.visual
                : hasLocalPreview && snapshot?.styleKey === regionVisualStyleKey(r) ? snapshot : null;
              const backendSprite = backendPreviewSprites[spriteSlot(r.id)];
              const isLiveEdit = isSelected || isDragging || hasLocalPreview;
              const wantsPreviewText = displayImageMode !== "typeset" && isLiveEdit;
              const liveBackendSprite = wantsPreviewText && previewSpriteMatches(backendSprite, r)
                ? backendSprite
                : null;
              const expectsBackendSprite = r.visible && Boolean(r.tl) && wantsPreviewText;
              const allowCssFallback = !expectsBackendSprite || Boolean(cssPreviewFallbacks[spriteSlot(r.id)]);
              const showChrome = showOverlayBoxes;
              const liveBox = dragPreview?.id === r.id ? dragPreview.bbox : r;
              return (
                <div
                  key={r.id}
                  className={`region-overlay ${isSelected ? "sel" : ""} ${isDragging ? "dragging" : ""}`}
                  style={{
                    left:   liveBox.x * rScale,
                    top:    liveBox.y * rScale,
                    width:  liveBox.w * rScale,
                    height: liveBox.h * rScale,
                    // Pass 5: make border + background invisible when chrome is hidden.
                    ...(showChrome ? {} : { border: "none", background: "transparent", outline: "none" }),
                  }}
                  onPointerDown={e => startDrag(e, r, "move")}
                >
                  {showChrome && (
                    <div
                      className={`region-label ${isSelected ? "gold" : "teal"}`}
                      title={[
                        `detector: ${((r.detector_confidence ?? 0) * 100).toFixed(0)}%${r.yolo_kind ? ` (${r.yolo_kind})` : ""}`,
                        `ocr:      ${((r.ocr_confidence ?? 0) * 100).toFixed(0)}%`,
                        r.cleanup_status ? `cleanup:  ${r.cleanup_status}${r.cleanup_reason ? ` · ${r.cleanup_reason}` : ""}` : undefined,
                        r.typeset_status ? `typeset:  ${r.typeset_status}${r.typeset_reason ? ` · ${r.typeset_reason}` : ""}` : undefined,
                        r.translation_status ? `translation: ${r.translation_status}` : undefined,
                      ].filter(Boolean).join("\n")}
                    >
                      {r.label}
                      {typeof r.detector_confidence === "number" && r.detector_confidence > 0
                        ? ` · YOLO ${Math.round(r.detector_confidence * 100)}%`
                        : ""}
                      {r.locked ? " · LOCK" : ""}
                    </div>
                  )}
                  {showEnglishOverlay && liveBackendSprite && (
                    <img
                      className="translation-bitmap-preview"
                      src={`data:image/png;base64,${liveBackendSprite.b64}`}
                      alt=""
                      draggable={false}
                      style={{
                        left:   (liveBackendSprite.x - liveBackendSprite.sourceBox.x) * rScale,
                        top:    (liveBackendSprite.y - liveBackendSprite.sourceBox.y) * rScale,
                        width:  liveBackendSprite.w * rScale,
                        height: liveBackendSprite.h * rScale,
                      }}
                    />
                  )}
                  {!liveBackendSprite && !isResizeDragging && allowCssFallback && showEnglishOverlay && wantsPreviewText && r.visible && r.tl && (
                    <div
                      className="translation-overlay"
                      style={{
                        color: frozenVisual?.color ?? r.fg ?? "#111111",
                        fontFamily: frozenVisual?.fontFamily ?? previewFontFamily(r),
                        fontSize: frozenVisual?.fontSize ?? overlayFontSize(r),
                        textAlign: frozenVisual?.textAlign ?? r.align,
                        textShadow: frozenVisual?.textShadow ?? previewTextShadow(r),
                        transform: `rotate(${Number(r.rotation_angle ?? 0)}deg)`,
                        transformOrigin: "center center",
                      }}
                    >
                      {r.tl}
                    </div>
                  )}
                  {isSelected && showChrome && (["nw", "ne", "sw", "se"] as const).map(handle => (
                    <div
                      key={handle}
                      className={`resize-handle ${handle}`}
                      onPointerDown={e => startDrag(e, r, handle)}
                    />
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  INSPECTOR TAB                                                              */
/* ─────────────────────────────────────────────────────────────────────────── */
const InspectorTab = ({
  region, regions, pageIndex, pageSize, fontOptions, onUpdateRegion, onPreviewRegion, onCommitBBox, onAddRegion, onDeleteRegion, onOcrRegion, onTranslateRegion,
  onSetYoloTrainClass,
  cleanupPreview, cleanupDebug, debugOverlays, onDebugOverlayChange,
  cleanupCandidates, selectedCandidateId, onSelectCleanupCandidate,
  sam2Settings, onPreviewCleanup, onApplyCleanup, onRerunCleanup, onDeleteCleanup, onSuggestSam2Mask, onRefreshCleanupDebug,
  onRecordMaskQaLabel, onTrainMaskQaModel,
  onCompareCleanupCandidates, onApplyCleanupCandidate, onUseCleanupCandidatePreview, cleanupCompareLoading,
}: {
  region:         Region | null;
  regions:        Region[];
  pageIndex:      number;
  pageSize:       { w: number; h: number } | null;
  fontOptions:    FontOptions;
  onUpdateRegion: (idx: number, field: string, value: unknown) => void;
  onPreviewRegion: (regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => void;
  onCommitBBox:   (region: Region, bbox: RegionBBox) => Promise<boolean>;
  onAddRegion:    () => void;
  onDeleteRegion: (idx: number, yoloRejectReason?: string) => void;
  onSetYoloTrainClass: (region: Region, classId: number) => void;
  onOcrRegion:    (idx: number) => void;
  onTranslateRegion: (idx: number) => void;
  cleanupPreview: (CleanupPreviewResponse & { regionId: string }) | null;
  cleanupDebug: (CleanupDebugResponse & { regionId: string }) | null;
  debugOverlays: DebugOverlayToggles;
  onDebugOverlayChange: (key: DebugOverlayKey, value: boolean) => void;
  cleanupCandidates: (CleanupCandidateCompareResponse & { regionId: string }) | null;
  selectedCandidateId: string;
  onSelectCleanupCandidate: (candidateId: string) => void;
  sam2Settings?: Bootstrap["meta"]["settings"];
  onPreviewCleanup: (region: Region, manualMask?: CleanupMaskPayload) => Promise<CleanupPreviewResponse | null>;
  onApplyCleanup: (idx: number, manualMask?: CleanupMaskPayload) => void;
  onRerunCleanup: (idx: number, manualMask?: CleanupMaskPayload) => void;
  onDeleteCleanup: (idx: number) => void;
  onSuggestSam2Mask: (region: Region, prompt: Record<string, unknown>) => Promise<Sam2MaskResponse>;
  onRefreshCleanupDebug: (region: Region, manualMask?: CleanupMaskPayload) => Promise<CleanupDebugResponse | null>;
  onRecordMaskQaLabel: (region: Region, label: string) => void;
  onTrainMaskQaModel: () => void;
  onCompareCleanupCandidates: (region: Region, manualMask?: CleanupMaskPayload) => Promise<CleanupCandidateCompareResponse | null>;
  onApplyCleanupCandidate: (idx: number, candidateId: string, manualMask?: CleanupMaskPayload) => void;
  onUseCleanupCandidatePreview: (region: Region, candidate: CleanupCandidate) => void;
  cleanupCompareLoading: boolean;
}) => {
  const [tlDraft, setTlDraft] = useState("");
  const [srcDraft, setSrcDraft] = useState("");
  const [bboxDraft, setBboxDraft] = useState({ x: 0, y: 0, w: 1, h: 1 });
  const [showCleanupMask, setShowCleanupMask] = useState(false);
  const [sam2Mode, setSam2Mode] = useState<Sam2UiMode>("cleanup");
  const [sam2Merge, setSam2Merge] = useState<Sam2MergeMode>("replace");
  const [sam2Status, setSam2Status] = useState("");
  const [brushMode, setBrushMode] = useState<"add" | "erase">("add");
  const [brushSize, setBrushSize] = useState(18);
  const commitTimers = useRef<Record<string, number>>({});
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskDrawingRef = useRef(false);
  const roleOptions = fontOptions.roles.length ? fontOptions.roles : DEFAULT_FONT_ROLES;
  const selectedFont = region?.font || "auto";
  const hasSelectedFont = roleOptions.includes(selectedFont) || fontOptions.fonts.includes(selectedFont);
  useEffect(() => { setTlDraft(region?.tl ?? ""); }, [region]);
  useEffect(() => { setSrcDraft(region?.src ?? ""); }, [region]);
  useEffect(() => {
    if (region) setBboxDraft({ x: region.x, y: region.y, w: region.w, h: region.h });
  }, [region]);
  useEffect(() => {
    if (!region || cleanupPreview?.regionId !== region.id || !cleanupPreview.mask_b64) return;
    const canvas = maskCanvasRef.current;
    const bbox = cleanupPreview.mask_bbox ?? [];
    if (!canvas || bbox.length !== 4) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = Math.max(1, Number(bbox[2]) || img.width || 1);
      canvas.height = Math.max(1, Number(bbox[3]) || img.height || 1);
      canvas.dataset.bbox = JSON.stringify(bbox);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `data:image/png;base64,${cleanupPreview.mask_b64}`;
  }, [cleanupPreview, region]);

  const scheduleFieldCommit = (field: string, value: unknown, delay = 350) => {
    if (!region) return;
    const key = `${region.id}:${field}`;
    if (commitTimers.current[key]) window.clearTimeout(commitTimers.current[key]);
    commitTimers.current[key] = window.setTimeout(() => {
      onUpdateRegion(region.idx, field, value);
      delete commitTimers.current[key];
    }, delay);
  };

  const flushFieldCommit = (field: string, value: unknown) => {
    if (!region) return;
    const key = `${region.id}:${field}`;
    if (commitTimers.current[key]) {
      window.clearTimeout(commitTimers.current[key]);
      delete commitTimers.current[key];
    }
    onUpdateRegion(region.idx, field, value);
  };

  const updateBboxDraft = (key: keyof RegionBBox, rawValue: string) => {
    if (!region) return;
    const nextValue = Number(rawValue);
    if (!Number.isFinite(nextValue)) return;
    setBboxDraft(prev => {
      const next = {
        ...prev,
        [key]: key === "w" || key === "h" ? Math.max(8, nextValue) : nextValue,
      };
      onPreviewRegion(region.id, next, { requestSprite: true, reason: "bbox_field_preview" });
      return next;
    });
  };

  const commitBboxDraft = () => {
    if (!region) return;
    onCommitBBox(region, bboxDraft);
  };
  const applyTransformBox = (next: RegionBBox, reason: string) => {
    if (!region) return;
    const clean = pageSize ? clampBboxToSize(next, pageSize) : {
      x: Math.round(next.x),
      y: Math.round(next.y),
      w: Math.max(8, Math.round(next.w)),
      h: Math.max(8, Math.round(next.h)),
    };
    setBboxDraft(clean);
    onPreviewRegion(region.id, clean, { requestSprite: true, reason });
    void onCommitBBox(region, clean);
  };
  const centerRegion = (scope: "bubble" | "page", axis: "x" | "y" | "both") => {
    if (!region) return;
    const target = scope === "bubble"
      ? containerBoxForRegion(region)
      : pageSize
        ? { x: 0, y: 0, w: pageSize.w, h: pageSize.h }
        : null;
    if (!target) return;
    const targetCenter = boxCenter(target);
    const next = { ...bboxDraft };
    if (axis === "x" || axis === "both") next.x = targetCenter.x - next.w / 2;
    if (axis === "y" || axis === "both") next.y = targetCenter.y - next.h / 2;
    applyTransformBox(next, `${scope}_center`);
  };
  const cleanupOverride = region?.cleanup_override ?? {};
  const cleanupValue = (key: string, fallback = "") => {
    const raw = (cleanupOverride as Record<string, unknown>)[key];
    return raw === undefined || raw === null ? fallback : String(raw);
  };
  const cleanupChecked = (key: string) => Boolean((cleanupOverride as Record<string, unknown>)[key]);
  const updateCleanupOverride = (patch: Record<string, unknown>) => {
    if (!region) return;
    onUpdateRegion(region.idx, "cleanup_override", { ...cleanupOverride, ...patch });
  };
  const currentManualMask = (): CleanupMaskPayload => {
    const canvas = maskCanvasRef.current;
    let bbox = cleanupPreview?.mask_bbox;
    try {
      if (canvas?.dataset.bbox) bbox = JSON.parse(canvas.dataset.bbox);
    } catch { /* noop */ }
    if (!canvas || !bbox || bbox.length !== 4) return null;
    return {
      b64: canvas.toDataURL("image/png").split(",")[1] ?? "",
      bbox,
    };
  };
  const ensureGeneratedMask = async () => {
    if (!region) return;
    setShowCleanupMask(true);
    await onPreviewCleanup(region);
  };
  const previewEditedMask = async () => {
    if (!region) return;
    setShowCleanupMask(true);
    await onPreviewCleanup(region, currentManualMask());
  };
  const compareCleanupCandidates = async () => {
    if (!region) return;
    await onCompareCleanupCandidates(region, currentManualMask());
  };
  const activeCleanupCandidates = cleanupCandidates?.regionId === region?.id ? cleanupCandidates : null;
  const selectedCandidate = activeCleanupCandidates
    ? (activeCleanupCandidates.candidates ?? []).find(c => c.candidate_id === selectedCandidateId)
    : undefined;
  const candidateScoreLabel = (score?: number) => Number.isFinite(Number(score)) ? Number(score).toFixed(1) : "n/a";
  const loadMaskImage = (b64: string) => new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = `data:image/png;base64,${b64}`;
  });
  const waitForMaskCanvas = () => new Promise<HTMLCanvasElement | null>((resolve) => {
    let tries = 0;
    const tick = () => {
      if (maskCanvasRef.current || tries > 8) {
        resolve(maskCanvasRef.current);
        return;
      }
      tries += 1;
      window.setTimeout(tick, 30);
    };
    tick();
  });
  const mergeSuggestedMask = async (suggestion: Sam2MaskResponse, mergeMode: Sam2MergeMode) => {
    const canvas = maskCanvasRef.current ?? await waitForMaskCanvas();
    const nextBbox = suggestion.bbox ?? [];
    if (!canvas || !suggestion.mask_b64 || nextBbox.length !== 4) return;
    const nextImg = await loadMaskImage(suggestion.mask_b64);
    const current = currentManualMask();
    const currentBbox = current?.bbox ?? nextBbox;
    const x1 = Math.min(Number(currentBbox[0]), Number(nextBbox[0]));
    const y1 = Math.min(Number(currentBbox[1]), Number(nextBbox[1]));
    const x2 = Math.max(Number(currentBbox[0]) + Number(currentBbox[2]), Number(nextBbox[0]) + Number(nextBbox[2]));
    const y2 = Math.max(Number(currentBbox[1]) + Number(currentBbox[3]), Number(nextBbox[1]) + Number(nextBbox[3]));
    const union = [x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1)];
    const merged = document.createElement("canvas");
    merged.width = union[2];
    merged.height = union[3];
    const ctx = merged.getContext("2d");
    if (!ctx) return;
    if (mergeMode !== "replace" && current?.b64 && currentBbox.length === 4) {
      const currentImg = await loadMaskImage(current.b64);
      ctx.drawImage(currentImg, Number(currentBbox[0]) - union[0], Number(currentBbox[1]) - union[1], Number(currentBbox[2]), Number(currentBbox[3]));
    }
    ctx.globalCompositeOperation = mergeMode === "subtract" ? "destination-out" : "source-over";
    ctx.drawImage(nextImg, Number(nextBbox[0]) - union[0], Number(nextBbox[1]) - union[1], Number(nextBbox[2]), Number(nextBbox[3]));
    ctx.globalCompositeOperation = "source-over";
    canvas.width = union[2];
    canvas.height = union[3];
    canvas.dataset.bbox = JSON.stringify(union);
    const out = canvas.getContext("2d");
    out?.clearRect(0, 0, canvas.width, canvas.height);
    out?.drawImage(merged, 0, 0);
  };
  const suggestSam2Mask = async () => {
    if (!region) return;
    setShowCleanupMask(true);
    if (!cleanupPreview?.mask_b64) await onPreviewCleanup(region);
    const suggestion = await onSuggestSam2Mask(region, {
      bbox: [region.x, region.y, region.w, region.h],
      mode: sam2Mode,
      current_manual_mask: currentManualMask(),
    });
    setSam2Status(suggestion.ok
      ? `${suggestion.status ?? "ok"}${suggestion.confidence ? ` · ${Math.round(suggestion.confidence * 100)}%` : ""}`
      : (suggestion.error ?? suggestion.status ?? "SAM2 unavailable"));
    if (suggestion.ok) {
      await mergeSuggestedMask(suggestion, sam2Merge);
      await onPreviewCleanup(region, currentManualMask());
      if (debugOverlays.manualMask || debugOverlays.cleanupMask) await onRefreshCleanupDebug(region, currentManualMask());
    }
  };
  const clearMaskCanvas = () => {
    const canvas = maskCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (region && debugOverlays.manualMask) onRefreshCleanupDebug(region, currentManualMask());
  };
  const brushMask = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = maskCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / Math.max(1, rect.width)) * canvas.width;
    const y = ((e.clientY - rect.top) / Math.max(1, rect.height)) * canvas.height;
    ctx.save();
    ctx.globalCompositeOperation = brushMode === "add" ? "source-over" : "destination-out";
    ctx.fillStyle = "rgba(255,255,255,1)";
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };
  const handleBboxKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!region) return;
    if (e.key === "Enter") commitBboxDraft();
    if (e.key === "Escape") {
      const original = { x: region.x, y: region.y, w: region.w, h: region.h };
      setBboxDraft(original);
      onPreviewRegion(region.id, original, { requestSprite: true, reason: "bbox_field_revert" });
    }
  };
  const sam2BackendUrl = String(sam2Settings?.sam2_backend_url ?? "").trim();
  const sam2StatusInfo = sam2Settings?.sam2_status ?? {};
  const sam2BackendMode = sam2BackendUrl ? "service" : "embedded";
  const settingBool = (value: unknown) => {
    if (typeof value === "boolean") return value;
    return String(value ?? "").trim().toLowerCase() === "true";
  };
  const sam2StatusLabel = String(
    sam2StatusInfo.status
      ?? (settingBool(sam2Settings?.sam2_enabled) ? (sam2BackendUrl ? "service" : "available") : "disabled")
  );
  const sam2StatusError = String(sam2StatusInfo.error ?? "");
  const sam2Enabled = settingBool(sam2Settings?.sam2_enabled);
  const sam2DisabledReason = !sam2Enabled
    ? "SAM2 disabled in settings"
    : sam2Settings?.sam2_mask_mode === "manual_only"
      ? "SAM2 mask mode is manual_only"
      : "";
  const debugItems: Array<[DebugOverlayKey, string, boolean]> = [
    ["yoloBox", "YOLO text box", Boolean(region?.detector_text_bbox)],
    ["editableBox", "Editable region box", Boolean(region)],
    ["containerBox", "Container box", Boolean(region?.cleanup_container_bbox || region?.container_bbox)],
    ["textMask", "Text mask", Boolean(cleanupDebug?.masks?.text_mask?.available)],
    ["cleanupMask", "Cleanup mask", Boolean(cleanupDebug?.masks?.cleanup_mask?.available)],
    ["haloMask", "Text edge cleanup mask", Boolean(cleanupDebug?.masks?.halo_mask?.available)],
    ["manualMask", "Manual mask edits", Boolean(currentManualMask())],
    ["patchBox", "Applied cleanup patch", Boolean(region?.cleanup_patch?.bbox)],
    ["groupedMask", "Grouped inpaint mask", Boolean(cleanupDebug?.masks?.grouped_mask?.available)],
  ];
  const qa = cleanupDebug?.regionId === region?.id ? cleanupDebug?.analysis : undefined;
  const qaValue = (value: unknown) => {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(value > 1 ? 2 : 3) : "—";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  };
  const qaBool = (value: unknown) => value === null || value === undefined ? "unknown" : qaValue(value);
  const qaBox = (box?: number[] | null) => Array.isArray(box) && box.length === 4 ? box.map(v => Math.round(Number(v))).join(", ") : "—";
  const overrideSummary = (() => {
    const override = qa?.cleanup_override ?? region?.cleanup_override ?? {};
    const entries = Object.entries(override).filter(([, value]) => value !== null && value !== undefined && value !== "" && value !== false);
    return entries.length ? entries.map(([key, value]) => `${key}=${String(value)}`).join("; ") : "—";
  })();
  const candidateWarnings = selectedCandidate?.warnings?.filter(Boolean) ?? [];
  const copyCleanupQaSummary = () => {
    const lines = [
      "Cleanup QA summary",
      `page_index: ${qaValue(qa?.page_index ?? pageIndex)}`,
      `region_id: ${qaValue(qa?.region_id ?? region?.id)}`,
      `region_label: ${qaValue(qa?.region_label ?? region?.label)}`,
      `region_type: ${qaValue(qa?.region_type ?? region?.role)}`,
      `bbox: ${qaBox(qa?.bbox ?? (region ? [region.x, region.y, region.w, region.h] : null))}`,
      `detector_text_bbox: ${qaBox(qa?.detector_text_bbox ?? region?.detector_text_bbox)}`,
      `container_bbox: ${qaBox(qa?.container_bbox ?? region?.cleanup_container_bbox ?? region?.container_bbox)}`,
      `selected_candidate: ${qaValue(selectedCandidate?.candidate_id ?? qa?.selected_cleanup_candidate ?? region?.cleanup_patch?.candidate_id)}`,
      `candidate_warnings: ${candidateWarnings.length ? candidateWarnings.join("; ") : "—"}`,
      `cleanup_status: ${qaValue(qa?.cleanup_status ?? region?.cleanup_status)}`,
      `cleanup_reason: ${qaValue(qa?.cleanup_reason ?? region?.cleanup_reason ?? qa?.skip_reason)}`,
      `background_model: ${qaValue(qa?.background_model)}`,
      `container_confidence: ${qaValue(qa?.container_confidence ?? region?.cleanup_container_confidence)}`,
      `text_mask_confidence: ${qaValue(qa?.text_mask_confidence)}`,
      `mask_container_ratio: ${qaValue(qa?.mask_container_ratio)}`,
      `mask_region_ratio: ${qaValue(qa?.mask_region_ratio)}`,
      `mask_area: ${qaValue(qa?.mask_area)}`,
      `border_touch_ratio: ${qaValue(qa?.border_touch_ratio)}`,
      `border_collision_bbox_source: ${qaValue(qa?.border_collision_bbox_source)}`,
      `rectangularity: ${qaValue(qa?.rectangularity)}`,
      `cleanup_mask_rejected: ${qaBool(qa?.cleanup_mask_rejected)}`,
      `cleanup_mask_rejection_reason: ${qaValue(qa?.cleanup_mask_rejection_reason)}`,
      `selected_candidate_source: ${qaValue(qa?.selected_text_mask_candidate_source)}`,
      `solid_fill_eligible: ${qaBool(qa?.solid_fill_eligible)}`,
      `halo_mask_used: ${qaBool(qa?.halo_mask_used)}`,
      `residual_retry_used: ${qaBool(qa?.residual_retry_used)}`,
      `grouped_fallback_used: ${qaBool(qa?.grouped_fallback_used ?? region?.cleanup_patch?.grouped_inpaint)}`,
      `last_patch_status: ${qaValue(qa?.last_patch_status ?? region?.cleanup_patch?.cleanup_status)}`,
      `last_patch_reason: ${qaValue(qa?.last_patch_reason ?? region?.cleanup_patch?.cleanup_reason ?? region?.cleanup_patch?.fallback_error)}`,
      `cleanup_override: ${overrideSummary}`,
    ];
    void navigator.clipboard?.writeText(lines.join("\n"));
  };
  const rawMatch = region?.raw_style_match ?? {};
  const rawStatus = String(rawMatch.status ?? "fallback");
  const rawIgnored = Array.isArray(rawMatch.ignored) ? rawMatch.ignored : [];
  const rawDowngrades = Array.isArray(rawMatch.downgrade_reasons) ? rawMatch.downgrade_reasons : [];
  const rawMatched = Array.isArray(rawMatch.matched) ? rawMatch.matched : [];
  const rawConfidence = rawMatch.confidence && typeof rawMatch.confidence === "object" ? rawMatch.confidence as Record<string, string> : {};
  const previewAndCommit = (patch: RegionDraft, field: string, value: unknown) => {
    if (!region) return;
    onPreviewRegion(region.id, patch, { requestSprite: true, reason: "style_change" });
    scheduleFieldCommit(field, value);
  };

  useEffect(() => {
    if (region) void onRefreshCleanupDebug(region);
  }, [region?.id, onRefreshCleanupDebug]);

  if (!region) {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div>Select a region to inspect</div>
        <button className="btn-ghost" onClick={onAddRegion}>Add Region</button>
      </div>
    );
  }

  return (
    <div className="insp-body">
      <details className="editor-section" open>
        <summary>Text</summary>
        <div className="editor-section-body">
        <div className="insp-section-title">Source OCR</div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
          <span className={`det-badge ${region.detector_source}`}>{region.detector_source || "ocr"}</span>
          <button className="btn-ghost" onClick={() => onOcrRegion(region.idx)}>OCR Selected</button>
        </div>
        <textarea
          className="text-area source"
          value={srcDraft}
          rows={3}
          disabled={region.locked}
          onChange={e => {
            setSrcDraft(e.target.value);
            onPreviewRegion(region.id, { src: e.target.value });
          }}
          onBlur={() => onUpdateRegion(region.idx, "text", srcDraft)}
        />
        <div className="insp-section-title" style={{ marginTop: 10 }}>Translation</div>
        <textarea
          className="text-area"
          value={tlDraft}
          rows={3}
          disabled={region.locked}
          onChange={e => {
            setTlDraft(e.target.value);
            onPreviewRegion(region.id, { tl: e.target.value });
          }}
          onBlur={() => onUpdateRegion(region.idx, "translation", tlDraft)}
        />
        <button className="btn-ghost" style={{ marginTop: 8 }} onClick={() => onTranslateRegion(region.idx)}>
          Translate Selected
        </button>
        </div>
      </details>

      <details className="editor-section" open>
        <summary>RAW Match</summary>
        <div className="editor-section-body">
          <div className="raw-match-card">
            <div className={`det-badge ${rawStatus === "high" ? "" : "manual"}`}>{rawStatus}</div>
            <div className="raw-match-summary">{String(rawMatch.summary ?? "No RAW style match yet; readable defaults are active.")}</div>
            <div className="raw-match-meta">
              matched: {rawMatched.length ? rawMatched.join(", ") : "none"}<br />
              confidence: {Object.keys(rawConfidence).length ? Object.entries(rawConfidence).map(([k, v]) => `${k} ${v}`).join(" · ") : "not measured"}<br />
              downgrades: {rawDowngrades.length ? rawDowngrades.join(" · ") : "none"}<br />
              ignored: {rawIgnored.length ? rawIgnored.join(" · ") : "none"}<br />
              auto: {rawMatch.auto_applied ? "applied" : "not applied"}<br />
              source: {region.style_source ?? "auto"}
            </div>
            <div className="style-row">
              <button className="btn-ghost" onClick={() => onUpdateRegion(region.idx, "rematch_raw_style", true)}>Re-match RAW Style</button>
              <button className="btn-ghost" onClick={() => onUpdateRegion(region.idx, "apply_raw_match", true)}>Apply RAW Match</button>
              <button className="btn-ghost" onClick={() => onUpdateRegion(region.idx, "reset_style", true)}>Reset to Auto</button>
            </div>
            <div className="style-row">
              <button className="btn-ghost" onClick={() => window.dispatchEvent(new CustomEvent("ml:set-image-mode", { detail: "raw" }))}>RAW</button>
              <button className="btn-ghost" onClick={() => window.dispatchEvent(new CustomEvent("ml:set-image-mode", { detail: "cleaned" }))}>Cleaned</button>
              <button className="btn-ghost" onClick={() => window.dispatchEvent(new CustomEvent("ml:set-image-mode", { detail: "typeset" }))}>Typeset</button>
            </div>
          </div>
        </div>
      </details>

      <details className="editor-section">
        <summary>Transform</summary>
        <div className="editor-section-body">
        <div className="prop-grid">
          <div className="prop-cell">
            <span className="prop-label">X</span>
            <input className="prop-val" type="number" disabled={region.locked} value={bboxDraft.x} onChange={e => updateBboxDraft("x", e.target.value)} onBlur={commitBboxDraft} onKeyDown={handleBboxKey} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Y</span>
            <input className="prop-val" type="number" disabled={region.locked} value={bboxDraft.y} onChange={e => updateBboxDraft("y", e.target.value)} onBlur={commitBboxDraft} onKeyDown={handleBboxKey} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">W</span>
            <input className="prop-val" type="number" disabled={region.locked} min={8} value={bboxDraft.w} onChange={e => updateBboxDraft("w", e.target.value)} onBlur={commitBboxDraft} onKeyDown={handleBboxKey} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">H</span>
            <input className="prop-val" type="number" disabled={region.locked} min={8} value={bboxDraft.h} onChange={e => updateBboxDraft("h", e.target.value)} onBlur={commitBboxDraft} onKeyDown={handleBboxKey} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Rotation</span>
            <input className="prop-val" type="number" disabled={region.locked} min={0} max={359} value={Math.round(Number(region.rotation_angle ?? 0))} onChange={e => previewAndCommit({ rotation_angle: Number(e.target.value) }, "rotation_angle", Number(e.target.value))} />
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <button className="btn-ghost" disabled={region.locked} onClick={commitBboxDraft}>
            Apply Box
          </button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => {
            const next = (Number(region.rotation_angle ?? 0) - 5 + 360) % 360;
            previewAndCommit({ rotation_angle: next }, "rotation_angle", next);
          }}>-5°</button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => {
            const next = (Number(region.rotation_angle ?? 0) + 5) % 360;
            previewAndCommit({ rotation_angle: next }, "rotation_angle", next);
          }}>+5°</button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => previewAndCommit({ rotation_angle: 0 }, "rotation_angle", 0)}>Reset Rotation</button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => onUpdateRegion(region.idx, "reset_transform", true)}>Reset Transform</button>
        </div>
        <div className="style-row" style={{ marginTop: 8 }}>
          <button className="btn-ghost" disabled={region.locked} onClick={() => centerRegion("bubble", "x")}>Bubble H</button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => centerRegion("bubble", "y")}>Bubble V</button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => centerRegion("bubble", "both")}>Bubble Center</button>
        </div>
        <div className="style-row" style={{ marginTop: 6 }}>
          <button className="btn-ghost" disabled={region.locked || !pageSize} onClick={() => centerRegion("page", "x")}>Page H</button>
          <button className="btn-ghost" disabled={region.locked || !pageSize} onClick={() => centerRegion("page", "y")}>Page V</button>
          <button className="btn-ghost" disabled={region.locked || !pageSize} onClick={() => centerRegion("page", "both")}>Page Center</button>
        </div>
        <div className="style-row" style={{ marginTop: 6 }}>
          <button className="btn-ghost" onClick={() => window.dispatchEvent(new CustomEvent("ml:toggle-alignment-guides"))}>Guides</button>
          <button className="btn-ghost" onClick={() => window.dispatchEvent(new CustomEvent("ml:toggle-alignment-snap"))}>Snap</button>
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <button className="btn-ghost" onClick={onAddRegion}>Add Region</button>
          <button className="btn-ghost" disabled={region.locked} onClick={() => onDeleteRegion(region.idx)}>Delete</button>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
          {[
            ["not_sfx", "Not SFX"],
            ["not_caption", "Not Caption"],
            ["not_bubble", "Not Bubble"],
            ["wrongly_detected_art", "Art"],
            ["exclamation_questionmark", "!?"],
          ].map(([reason, label]) => (
            <button
              key={reason}
              className="btn-ghost"
              disabled={region.locked}
              title={`Delete and record YOLO correction: ${label}`}
              onClick={() => onDeleteRegion(region.idx, reason)}
            >
              {label}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
          {[
            [0, "Train Dialogue"],
            [1, "Train Caption"],
            [2, "Train SFX"],
            [3, "Train Shout"],
          ].map(([classId, label]) => (
            <button
              key={classId}
              className="btn-ghost"
              disabled={region.locked}
              title={`Use this box as a positive YOLO ${label} label`}
              onClick={() => onSetYoloTrainClass(region, Number(classId))}
            >
              {label}
            </button>
          ))}
        </div>
        </div>
      </details>

      <details className="editor-section" open>
        <summary>Typography</summary>
        <div className="editor-section-body">
        <div className="prop-grid">
          <div className="prop-cell">
            <span className="prop-label">Conf</span>
            <span className="prop-val">{region.conf}%</span>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Font</span>
            <select
              className="prop-val"
              disabled={region.locked}
              value={selectedFont}
              onChange={e => {
                onPreviewRegion(region.id, { font: e.target.value }, { requestSprite: true, reason: "style_change" });
                scheduleFieldCommit("font_name", e.target.value);
              }}
              onBlur={e => flushFieldCommit("font_name", e.target.value)}
            >
              {!hasSelectedFont && selectedFont && <option value={selectedFont}>{selectedFont}</option>}
              <optgroup label="Roles">
                {roleOptions.map(role => <option key={`role:${role}`} value={role}>{role}</option>)}
              </optgroup>
              {fontOptions.fonts.length > 0 && (
                <optgroup label="Fonts">
                  {fontOptions.fonts.map(font => <option key={`font:${font}`} value={font}>{font}</option>)}
                </optgroup>
              )}
            </select>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Size</span>
            <input className="prop-val" type="number" disabled={region.locked} min={0} max={96} value={region.size || 0} onChange={e => { const value = Number(e.target.value); onPreviewRegion(region.id, { size: value }); scheduleFieldCommit("font_size", value); }} onBlur={e => flushFieldCommit("font_size", Number(e.target.value))} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Align</span>
            <select className="prop-val" disabled={region.locked} value={region.align} onChange={e => { onPreviewRegion(region.id, { align: e.target.value as Region["align"] }); onUpdateRegion(region.idx, "align", e.target.value); }}>
              <option value="left">left</option>
              <option value="center">center</option>
              <option value="right">right</option>
            </select>
          </div>
          <div className="prop-cell">
            <span className="prop-label">State</span>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="checkbox" checked={region.visible} onChange={e => { onPreviewRegion(region.id, { visible: e.target.checked }); onUpdateRegion(region.idx, "visible", e.target.checked); }} />
              Visible
            </label>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Lock</span>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="checkbox" checked={region.locked} onChange={e => { onPreviewRegion(region.id, { locked: e.target.checked }); onUpdateRegion(region.idx, "locked", e.target.checked); }} />
              Locked
            </label>
          </div>
        </div>
        </div>
      </details>

      <details className="editor-section" open>
        <summary>Fill</summary>
        <div className="editor-section-body">
          <div className="style-row">
            <input type="color" disabled={region.locked} value={region.fg} onChange={e => previewAndCommit({ fg: e.target.value }, "fg_color", e.target.value)} onBlur={e => flushFieldCommit("fg_color", e.target.value)} />
            <span className="prop-label">Solid</span>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="checkbox" checked={Boolean(region.gradient_on)} onChange={e => previewAndCommit({ gradient_on: e.target.checked }, "gradient_on", e.target.checked)} />
              Gradient
            </label>
          </div>
          {region.gradient_on && (
            <div className="prop-grid" style={{ marginTop: 8 }}>
              <div className="prop-cell"><span className="prop-label">Start</span><input className="prop-val" type="color" value={region.gradient_start ?? region.fg} onChange={e => previewAndCommit({ gradient_start: e.target.value }, "gradient_start", e.target.value)} /></div>
              <div className="prop-cell"><span className="prop-label">End</span><input className="prop-val" type="color" value={region.gradient_end ?? region.fg} onChange={e => previewAndCommit({ gradient_end: e.target.value }, "gradient_end", e.target.value)} /></div>
              <div className="prop-cell"><span className="prop-label">Angle</span><input className="prop-val" type="number" min={0} max={359} value={region.gradient_angle ?? 90} onChange={e => previewAndCommit({ gradient_angle: Number(e.target.value) }, "gradient_angle", Number(e.target.value))} /></div>
            </div>
          )}
        </div>
      </details>

      <details className="editor-section">
        <summary>Stroke</summary>
        <div className="editor-section-body">
          <div className="prop-grid">
            <div className="prop-cell"><span className="prop-label">Color</span><input className="prop-val" type="color" value={region.outline ?? "#ffffff"} onChange={e => previewAndCommit({ outline: e.target.value }, "outline_color", e.target.value)} /></div>
            <div className="prop-cell"><span className="prop-label">Width</span><input className="prop-val" type="number" min={0} max={8} value={region.outline_width ?? 1} onChange={e => previewAndCommit({ outline_width: Number(e.target.value) }, "outline_width", Number(e.target.value))} /></div>
          </div>
        </div>
      </details>

      <details className="editor-section">
        <summary>Effects</summary>
        <div className="editor-section-body">
          <div className="prop-grid">
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}><input type="checkbox" checked={Boolean(region.shadow_on)} onChange={e => previewAndCommit({ shadow_on: e.target.checked }, "shadow_on", e.target.checked)} />Shadow</label>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}><input type="checkbox" checked={Boolean(region.glow_on)} onChange={e => previewAndCommit({ glow_on: e.target.checked }, "glow_on", e.target.checked)} />Glow</label>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}><input type="checkbox" checked={Boolean(region.reflection_on)} onChange={e => previewAndCommit({ reflection_on: e.target.checked }, "reflection_on", e.target.checked)} />Reflection</label>
            <div className="prop-cell"><span className="prop-label">Shadow color</span><input className="prop-val" type="color" value={region.shadow ?? "#000000"} onChange={e => previewAndCommit({ shadow: e.target.value }, "shadow_color", e.target.value)} /></div>
            <div className="prop-cell"><span className="prop-label">Glow color</span><input className="prop-val" type="color" value={region.glow ?? "#ffffff"} onChange={e => previewAndCommit({ glow: e.target.value }, "glow_color", e.target.value)} /></div>
            <div className="prop-cell"><span className="prop-label">Shadow X</span><input className="prop-val" type="number" min={-24} max={24} value={region.shadow_offset_x ?? 1} onChange={e => previewAndCommit({ shadow_offset_x: Number(e.target.value) }, "shadow_offset_x", Number(e.target.value))} /></div>
            <div className="prop-cell"><span className="prop-label">Shadow Y</span><input className="prop-val" type="number" min={-24} max={24} value={region.shadow_offset_y ?? 2} onChange={e => previewAndCommit({ shadow_offset_y: Number(e.target.value) }, "shadow_offset_y", Number(e.target.value))} /></div>
            <div className="prop-cell"><span className="prop-label">Shadow opacity</span><input className="prop-val" type="number" min={0} max={1} step={0.05} value={region.shadow_opacity ?? 0.55} onChange={e => previewAndCommit({ shadow_opacity: Number(e.target.value) }, "shadow_opacity", Number(e.target.value))} /></div>
            <div className="prop-cell"><span className="prop-label">Shadow blur</span><input className="prop-val" type="number" min={0} max={32} step={0.5} value={region.shadow_blur ?? 0} onChange={e => previewAndCommit({ shadow_blur: Number(e.target.value) }, "shadow_blur", Number(e.target.value))} /></div>
            <div className="prop-cell"><span className="prop-label">Glow radius</span><input className="prop-val" type="number" min={0} max={32} value={region.glow_radius ?? 4} onChange={e => previewAndCommit({ glow_radius: Number(e.target.value) }, "glow_radius", Number(e.target.value))} /></div>
            <div className="prop-cell"><span className="prop-label">Glow intensity</span><input className="prop-val" type="number" min={0} max={1} step={0.05} value={region.glow_intensity ?? 0.45} onChange={e => previewAndCommit({ glow_intensity: Number(e.target.value) }, "glow_intensity", Number(e.target.value))} /></div>
            {region.reflection_on && (
              <>
                <div className="prop-cell"><span className="prop-label">Reflect opacity</span><input className="prop-val" type="number" min={0} max={1} step={0.05} value={region.reflection_opacity ?? 0.32} onChange={e => previewAndCommit({ reflection_opacity: Number(e.target.value) }, "reflection_opacity", Number(e.target.value))} /></div>
                <div className="prop-cell"><span className="prop-label">Reflect offset</span><input className="prop-val" type="number" min={-64} max={128} value={region.reflection_offset ?? 4} onChange={e => previewAndCommit({ reflection_offset: Number(e.target.value) }, "reflection_offset", Number(e.target.value))} /></div>
                <div className="prop-cell"><span className="prop-label">Reflect blur</span><input className="prop-val" type="number" min={0} max={32} step={0.5} value={region.reflection_blur ?? 1.5} onChange={e => previewAndCommit({ reflection_blur: Number(e.target.value) }, "reflection_blur", Number(e.target.value))} /></div>
                <div className="prop-cell"><span className="prop-label">Reflect fade</span><input className="prop-val" type="number" min={0} max={1} step={0.05} value={region.reflection_fade ?? 0.78} onChange={e => previewAndCommit({ reflection_fade: Number(e.target.value) }, "reflection_fade", Number(e.target.value))} /></div>
              </>
            )}
          </div>
        </div>
      </details>

      <details className="editor-section">
        <summary>Presets</summary>
        <div className="editor-section-body">
          <div className="style-row">
            {[
              ["dialog_light", "Dialogue"],
              ["thought_soft", "Thought"],
              ["narration", "Narration"],
              ["shout", "Shout"],
              ["sfx_impact", "SFX"],
              ["sfx_color", "SFX Glow"],
              ["reset_auto", "Reset Auto"],
            ].map(([key, label]) => <button key={key} className="btn-ghost" onClick={() => onUpdateRegion(region.idx, "style_preset", key)}>{label}</button>)}
          </div>
        </div>
      </details>

      <details className="editor-section">
        <summary>Style Tools</summary>
        <div className="editor-section-body">
          <div className="style-row">
            <button className="btn-ghost" onClick={() => localStorage.setItem("ml.styleClipboard", JSON.stringify({ fg: region.fg, outline: region.outline, outline_width: region.outline_width, shadow: region.shadow, shadow_on: region.shadow_on, shadow_offset_x: region.shadow_offset_x, shadow_offset_y: region.shadow_offset_y, shadow_opacity: region.shadow_opacity, shadow_blur: region.shadow_blur, gradient_on: region.gradient_on, gradient_start: region.gradient_start, gradient_end: region.gradient_end, gradient_angle: region.gradient_angle, glow: region.glow, glow_on: region.glow_on, glow_radius: region.glow_radius, glow_intensity: region.glow_intensity, reflection_on: region.reflection_on, reflection_opacity: region.reflection_opacity, reflection_offset: region.reflection_offset, reflection_blur: region.reflection_blur, reflection_fade: region.reflection_fade }))}>Copy Style</button>
            <button className="btn-ghost" onClick={() => {
              try {
                const style = JSON.parse(localStorage.getItem("ml.styleClipboard") || "{}");
                Object.entries(style).forEach(([key, val]) => {
                  const fieldMap: Record<string, string> = { fg: "fg_color", outline: "outline_color", shadow: "shadow_color", glow: "glow_color" };
                  onUpdateRegion(region.idx, fieldMap[key] ?? key, val);
                });
              } catch { /* noop */ }
            }}>Paste Style</button>
            <button className="btn-ghost" onClick={() => {
              const style = { fg: region.fg, outline: region.outline, outline_width: region.outline_width, shadow: region.shadow, shadow_on: region.shadow_on, shadow_offset_x: region.shadow_offset_x, shadow_offset_y: region.shadow_offset_y, shadow_opacity: region.shadow_opacity, shadow_blur: region.shadow_blur, gradient_on: region.gradient_on, gradient_start: region.gradient_start, gradient_end: region.gradient_end, gradient_angle: region.gradient_angle, glow: region.glow, glow_on: region.glow_on, glow_radius: region.glow_radius, glow_intensity: region.glow_intensity, reflection_on: region.reflection_on, reflection_opacity: region.reflection_opacity, reflection_offset: region.reflection_offset, reflection_blur: region.reflection_blur, reflection_fade: region.reflection_fade };
              const fieldMap: Record<string, string> = { fg: "fg_color", outline: "outline_color", shadow: "shadow_color", glow: "glow_color" };
              regions.filter(r => r.id !== region.id && r.role === region.role).forEach(r => {
                Object.entries(style).forEach(([key, val]) => onUpdateRegion(r.idx, fieldMap[key] ?? key, val));
              });
            }}>Apply to Same Role</button>
          </div>
        </div>
      </details>

      <details className="editor-section">
        <summary>Cleanup / Debug</summary>
        <div className="editor-section-body">
        <div className="insp-section-title">Cleanup Override</div>
        <div className="prop-grid">
          <div className="prop-cell">
            <span className="prop-label">Mode</span>
            <select
              className="prop-val"
              value={cleanupValue("cleanup_override_mode", "")}
              onChange={e => updateCleanupOverride({ cleanup_override_mode: e.target.value || null })}
            >
              <option value="">Use global settings</option>
              <option value="skip">Skip cleanup for this region</option>
              <option value="force_solid">Force solid fill</option>
              <option value="force_telea">Force TELEA</option>
              <option value="force_ns">Force OpenCV NS</option>
              <option value="force_iopaint">Force IOPaint</option>
              <option value="review">Review/manual only</option>
              <option value="force_allow">Force allow cleanup</option>
              <option value="force_review">Force review after cleanup</option>
            </select>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Treat as</span>
            <select
              className="prop-val"
              value={cleanupValue("cleanup_region_class", "")}
              onChange={e => updateCleanupOverride({ cleanup_region_class: e.target.value || null })}
            >
              <option value="">Detected type</option>
              <option value="speech_bubble">Speech bubble</option>
              <option value="caption_box">Caption</option>
              <option value="text_on_art">Text-over-art</option>
              <option value="sfx">SFX</option>
            </select>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Edge cleanup px</span>
            <input className="prop-val" type="number" min={0} max={8} value={cleanupValue("cleanup_halo_max_px")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_halo_max_px: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Leftover retry</span>
            <select className="prop-val" value={cleanupValue("cleanup_residual_retry_enabled", "")} onChange={e => updateCleanupOverride({ cleanup_residual_retry_enabled: e.target.value === "" ? null : e.target.value === "true" })}>
              <option value="">Global</option>
              <option value="true">On</option>
              <option value="false">Off</option>
            </select>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Retry growth px</span>
            <input className="prop-val" type="number" min={0} max={8} value={cleanupValue("cleanup_residual_retry_dilate_px")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_residual_retry_dilate_px: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Min conf</span>
            <input className="prop-val" type="number" step="0.05" min={0} max={1} value={cleanupValue("cleanup_min_container_confidence")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_min_container_confidence: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Max mask/container</span>
            <input className="prop-val" type="number" step="0.01" min={0} max={2} value={cleanupValue("cleanup_max_mask_container_ratio")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_max_mask_container_ratio: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Max mask/region</span>
            <input className="prop-val" type="number" step="0.01" min={0} max={2} value={cleanupValue("cleanup_max_mask_region_ratio")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_max_mask_region_ratio: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Border touch</span>
            <input className="prop-val" type="number" step="0.01" min={0} max={1} value={cleanupValue("cleanup_max_border_touch_ratio")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_max_border_touch_ratio: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Rectangularity</span>
            <input className="prop-val" type="number" step="0.01" min={0} max={1} value={cleanupValue("cleanup_max_rectangularity")} placeholder="global" onChange={e => updateCleanupOverride({ cleanup_max_rectangularity: e.target.value === "" ? null : Number(e.target.value) })} />
          </div>
          <div className="prop-cell">
            <span className="prop-label">Danger</span>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="checkbox" checked={cleanupChecked("cleanup_allow_low_confidence")} onChange={e => updateCleanupOverride({ cleanup_allow_low_confidence: e.target.checked || null })} />
              Low conf
            </label>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Texture</span>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="checkbox" checked={cleanupChecked("cleanup_allow_texture_inpaint")} onChange={e => updateCleanupOverride({ cleanup_allow_texture_inpaint: e.target.checked || null })} />
              Allow
            </label>
          </div>
          <div className="prop-cell">
            <span className="prop-label">Translucent</span>
            <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="checkbox" checked={cleanupChecked("cleanup_allow_translucent_caption")} onChange={e => updateCleanupOverride({ cleanup_allow_translucent_caption: e.target.checked || null })} />
              Captions
            </label>
          </div>
        </div>
        <div className="settings-warning" style={{ marginTop: 8 }}>Force allow, SFX, and text-over-art cleanup can damage artwork. Cleanup still writes only cleanup-mask pixels.</div>
        <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
          <div className="prop-grid">
            <div className="prop-cell">
              <span className="prop-label">Last cleanup</span>
              <span className="prop-val" title={region.cleanup_patch?.cleanup_reason || region.cleanup_reason || undefined}>
                {region.cleanup_patch
                  ? `${region.cleanup_patch.cleanup_status || region.cleanup_status || "applied"} · ${region.cleanup_patch.strategy || "cleanup"}`
                  : (region.cleanup_status ? `${region.cleanup_status}${region.cleanup_reason ? ` · ${region.cleanup_reason}` : ""}` : "No patch")}
              </span>
            </div>
            <div className="prop-cell">
              <span className="prop-label">Patch</span>
              <span className="prop-val">
                {region.cleanup_patch?.created_at ? `saved ${region.cleanup_patch.created_at}` : "none"}
              </span>
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <button className="btn-ghost" onClick={() => onPreviewCleanup(region)}>Preview cleanup</button>
            <button className="btn-ghost" onClick={() => onApplyCleanup(region.idx, currentManualMask())}>Apply cleanup</button>
            <button className="btn-ghost" onClick={() => onRerunCleanup(region.idx, currentManualMask())}>Rerun cleanup</button>
            <button className="btn-ghost" disabled={!region.cleanup_patch} onClick={() => onDeleteCleanup(region.idx)}>Undo/delete cleanup</button>
          </div>
          <div style={{ border: "1px solid var(--br-1)", background: "var(--bg-2)", padding: 6, display: "grid", gap: 7 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              <button className="btn-ghost" disabled={cleanupCompareLoading} onClick={compareCleanupCandidates}>
                {cleanupCompareLoading ? "Comparing…" : "Compare cleanup methods"}
              </button>
              <button
                className="btn-ghost"
                disabled={cleanupCompareLoading || !selectedCandidate?.is_available}
                onClick={() => selectedCandidate && onApplyCleanupCandidate(region.idx, selectedCandidate.candidate_id, currentManualMask())}
              >
                Apply selected candidate
              </button>
            </div>
            <div className="cleanup-candidate-meta">Scores are hints; inspect visually.</div>
            {activeCleanupCandidates && (activeCleanupCandidates.candidates ?? []).length > 0 && (
              <>
                <div className="cleanup-candidate-grid">
                  {(activeCleanupCandidates.candidates ?? []).map(candidate => {
                    const isSelected = selectedCandidateId === candidate.candidate_id;
                    const score = candidate.scores?.score;
                    const warnings = candidate.warnings ?? [];
                    return (
                      <div
                        key={candidate.candidate_id}
                        className={`cleanup-candidate ${isSelected ? "sel" : ""} ${candidate.is_available ? "" : "off"}`}
                        onClick={() => candidate.is_available && onSelectCleanupCandidate(candidate.candidate_id)}
                        title={candidate.unavailable_reason || warnings.join("\n") || candidate.label}
                      >
                        {candidate.b64 && <img src={`data:image/png;base64,${candidate.b64}`} alt="" draggable={false} />}
                        <div className="cleanup-candidate-title">
                          {candidate.label}
                          {activeCleanupCandidates.recommended_candidate_id === candidate.candidate_id ? " · suggested" : ""}
                        </div>
                        <div className="cleanup-candidate-meta">
                          score {candidateScoreLabel(score)}
                          <br />
                          {candidate.strategy || "cleanup"} / {candidate.method || candidate.backend || "auto"}
                          <br />
                          mask {candidate.scores?.mask_area ?? 0}px · seam {candidateScoreLabel(candidate.scores?.seam_score)}
                        </div>
                        {!candidate.is_available && (
                          <div className="cleanup-candidate-warn">{candidate.unavailable_reason || "Unavailable"}</div>
                        )}
                        {candidate.is_available && warnings.length > 0 && (
                          <div className="cleanup-candidate-warn">{warnings.slice(0, 2).join(" · ")}</div>
                        )}
                      </div>
                    );
                  })}
                </div>
                <button
                  className="btn-ghost"
                  disabled={cleanupCompareLoading || !selectedCandidate?.is_available}
                  onClick={() => selectedCandidate && onUseCleanupCandidatePreview(region, selectedCandidate)}
                >
                  Use as preview result
                </button>
              </>
            )}
          </div>
          <div className="prop-grid">
            <div className="prop-cell">
              <span className="prop-label">SAM2 mode</span>
              <select className="prop-val" value={sam2Mode} onChange={e => setSam2Mode(e.target.value as Sam2UiMode)}>
                <option value="cleanup">Cleanup mask</option>
                <option value="container">Container/protect mask</option>
                <option value="protect">Protect mask</option>
              </select>
            </div>
            <div className="prop-cell">
              <span className="prop-label">Merge</span>
              <select className="prop-val" value={sam2Merge} onChange={e => setSam2Merge(e.target.value as Sam2MergeMode)}>
                <option value="replace">Replace current mask</option>
                <option value="add">Add to current mask</option>
                <option value="subtract">Subtract/protect from current mask</option>
              </select>
            </div>
            <button className="btn-ghost" disabled={!sam2Enabled || Boolean(sam2DisabledReason)} title={sam2DisabledReason || sam2StatusError || "Suggest a manual cleanup mask with SAM2"} onClick={suggestSam2Mask}>
              Suggest mask with SAM2
            </button>
            <span className="prop-val" style={{ color: sam2Enabled && !sam2StatusError ? "var(--t2)" : "var(--amr)" }}>
              {sam2Status || (sam2Enabled ? `${sam2BackendMode}: ${sam2StatusLabel}${sam2StatusError ? ` - ${sam2StatusError}` : ""}` : sam2DisabledReason)}
            </span>
          </div>
          <label className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={showCleanupMask} onChange={e => { setShowCleanupMask(e.target.checked); if (e.target.checked && !cleanupPreview?.mask_b64) ensureGeneratedMask(); }} />
            Show generated cleanup mask
          </label>
          {showCleanupMask && (
            <div className="prop-grid">
              <div className="prop-cell">
                <span className="prop-label">Brush</span>
                <select className="prop-val" value={brushMode} onChange={e => setBrushMode(e.target.value === "erase" ? "erase" : "add")}>
                  <option value="add">Add to mask</option>
                  <option value="erase">Erase from mask</option>
                </select>
              </div>
              <div className="prop-cell">
                <span className="prop-label">Size</span>
                <input className="prop-val" type="range" min={4} max={72} value={brushSize} onChange={e => setBrushSize(Number(e.target.value) || 18)} />
              </div>
              <button className="btn-ghost" onClick={previewEditedMask}>Preview edited mask</button>
              <button className="btn-ghost" onClick={() => onApplyCleanup(region.idx, currentManualMask())}>Apply edited cleanup</button>
              <button className="btn-ghost" onClick={clearMaskCanvas}>Clear manual edits</button>
              <button className="btn-ghost" onClick={ensureGeneratedMask}>Reset to generated mask</button>
            </div>
          )}
          <div style={{ border: "1px solid var(--br-1)", background: "var(--bg-2)", padding: 6 }}>
            <div className="insp-section-title" style={{ marginBottom: 6 }}>Debug overlays</div>
            <div className="prop-grid">
              {debugItems.map(([key, label, available]) => (
                <label key={key} className="prop-val" style={{ display: "flex", gap: 6, alignItems: "center", opacity: available ? 1 : 0.55 }} title={available ? label : `${label} unavailable`}>
                  <input
                    type="checkbox"
                    checked={debugOverlays[key]}
                    onChange={e => {
                      onDebugOverlayChange(key, e.target.checked);
                      if (e.target.checked) onRefreshCleanupDebug(region, currentManualMask());
                    }}
                  />
                  {label}
                </label>
              ))}
            </div>
            <button className="btn-ghost" style={{ marginTop: 6 }} onClick={() => onRefreshCleanupDebug(region, currentManualMask())}>
              Refresh overlays
            </button>
          </div>
          {cleanupPreview?.regionId === region.id && cleanupPreview.b64 && (
            <div style={{ border: "1px solid var(--br-1)", background: "var(--bg-2)", padding: 6 }}>
              <div style={{ position: "relative" }}>
                <img
                  src={`data:image/png;base64,${cleanupPreview.b64}`}
                  alt="Cleanup preview"
                  style={{ display: "block", width: "100%", maxHeight: 180, objectFit: "contain" }}
                />
                {showCleanupMask && cleanupPreview.mask_b64 && (
                  <canvas
                    ref={maskCanvasRef}
                    style={{
                      position: "absolute",
                      inset: 0,
                      width: "100%",
                      height: "100%",
                      opacity: 0.48,
                      mixBlendMode: "screen",
                      cursor: "crosshair",
                    }}
                    onPointerDown={e => {
                      maskDrawingRef.current = true;
                      e.currentTarget.setPointerCapture(e.pointerId);
                      brushMask(e);
                    }}
                    onPointerMove={e => { if (maskDrawingRef.current) brushMask(e); }}
                    onPointerUp={e => {
                      maskDrawingRef.current = false;
                      try { e.currentTarget.releasePointerCapture(e.pointerId); } catch {}
                      if (debugOverlays.manualMask) onRefreshCleanupDebug(region, currentManualMask());
                    }}
                    onPointerLeave={() => { maskDrawingRef.current = false; }}
                  />
                )}
              </div>
              <div style={{ marginTop: 5, color: "var(--t3)", fontSize: 10, fontFamily: "var(--fnt-mono)" }}>
                {String(cleanupPreview.plan?.cleanup_strategy ?? "cleanup")} / {String(cleanupPreview.plan?.inpaint_method ?? "auto")}
                {cleanupPreview.plan?.skip_reason ? ` · ${String(cleanupPreview.plan.skip_reason)}` : ""}
                {cleanupPreview.manual_mask_used ? " · manual mask" : ""}
              </div>
            </div>
          )}
        </div>
        <button className="btn-ghost" style={{ marginTop: 8 }} onClick={() => onUpdateRegion(region.idx, "reset_cleanup_override", true)}>
          Reset cleanup override
        </button>
        </div>
      </details>

      <details className="editor-section">
        <summary>Cleanup Analysis</summary>
        <div className="editor-section-body">
        <div className="prop-grid">
          {[
            ["Page index", qaValue(qa?.page_index ?? pageIndex), "page_index"],
            ["Region id", qaValue(qa?.region_id ?? region.id), "region_id"],
            ["Detected type", qaValue(qa?.region_type ?? region.role), "region_type"],
            ["Action", qaValue(qa?.effective_cleanup_action), "effective_cleanup_action"],
            ["Mode", qaValue(qa?.effective_cleanup_mode), "effective_cleanup_mode"],
            ["Status", qaValue(qa?.cleanup_status ?? region.cleanup_status), "cleanup_status"],
            ["Reason", qaValue(qa?.cleanup_reason ?? region.cleanup_reason ?? qa?.skip_reason), "cleanup_reason / skip_reason"],
            ["Background", qaValue(qa?.background_model), "background_model"],
            ["Container conf", qaValue(qa?.container_confidence ?? region.cleanup_container_confidence), "container_confidence"],
            ["Text mask conf", qaValue(qa?.text_mask_confidence), "text_mask_confidence"],
            ["Mask/container", qaValue(qa?.mask_container_ratio), "mask_container_ratio"],
            ["Mask/region", qaValue(qa?.mask_region_ratio), "mask_region_ratio"],
            ["Mask pixels", qaValue(qa?.mask_area), "mask_area"],
            ["Border touch", qaValue(qa?.border_touch_ratio), "border_touch_ratio"],
            ["Border basis", qaValue(qa?.border_collision_bbox_source), "border_collision_bbox_source"],
            ["Rectangularity", qaValue(qa?.rectangularity), "rectangularity"],
            ["Mask rejected", qaBool(qa?.cleanup_mask_rejected), "cleanup_mask_rejected"],
            ["Reject reason", qaValue(qa?.cleanup_mask_rejection_reason), "cleanup_mask_rejection_reason"],
            ["Solid fill", qaBool(qa?.solid_fill_eligible), "solid_fill_eligible"],
            ["Edge cleanup", qaBool(qa?.halo_mask_used), "halo_mask_used"],
            ["Leftover text retry", qaBool(qa?.residual_retry_used), "residual_retry_used"],
            ["Grouped fallback", qaBool(qa?.grouped_fallback_used ?? region.cleanup_patch?.grouped_inpaint), "grouped_fallback_used"],
            ["Selected candidate", qaValue(selectedCandidate?.candidate_id ?? qa?.selected_cleanup_candidate ?? region.cleanup_patch?.candidate_id), "selected_cleanup_candidate"],
            ["Candidate source", qaValue(qa?.selected_text_mask_candidate_source), "selected_text_mask_candidate_source"],
            ["Candidate warnings", candidateWarnings.length ? candidateWarnings.join(" · ") : "—", "candidate_warnings"],
            ["YOLO train", qaValue(region.yolo_train_class_id ?? "—"), "yolo_train_class_id"],
            ["Last patch", qaValue(qa?.last_patch_status ?? region.cleanup_patch?.cleanup_status), "last_patch_status"],
            ["Patch reason", qaValue(qa?.last_patch_reason ?? region.cleanup_patch?.cleanup_reason ?? region.cleanup_patch?.fallback_error), "last_patch_reason"],
          ].map(([label, value, title]) => (
            <div className="prop-cell" key={label}>
              <span className="prop-label" title={title}>{label}</span>
              <span className="prop-val" title={title}>{value}</span>
            </div>
          ))}
        </div>
        <button className="btn-ghost" style={{ marginTop: 8 }} onClick={copyCleanupQaSummary}>
          Copy cleanup QA summary
        </button>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
          {[
            ["good", "Mask OK"],
            ["bad_glyph_mask", "Bad glyph"],
            ["bad_container_mask", "Bad container"],
            ["unsafe_cleanup_validation", "Bad safety"],
            ["bad_fill_inpaint", "Bad fill"],
            ["bad_routing_strategy", "Bad routing"],
            ["legacy_candidate_wrong", "Legacy won"],
          ].map(([label, text]) => (
            <button key={label} className="btn-ghost" onClick={() => onRecordMaskQaLabel(region, label)}>
              {text}
            </button>
          ))}
          <button className="btn-ghost" onClick={onTrainMaskQaModel}>Train mask QA</button>
        </div>
        <details style={{ marginTop: 8 }}>
          <summary className="prop-val">Recommended QA cases</summary>
          <div className="settings-warning" style={{ marginTop: 6 }}>
            easy white bubble · off-white bubble · colored bubble · dark caption · gradient bubble · halftone/screentone bubble · translucent caption · text over art · SFX · large bold Korean text
          </div>
        </details>
        </div>
      </details>

      {/* Pass 6: per-stage diagnostic breakdown. Shows whichever stages have
          produced data; hidden rows are omitted to keep the panel compact. */}
      {(
        (typeof region.detector_confidence === "number" && region.detector_confidence > 0) ||
        (typeof (region as any).ocr_confidence === "number" && (region as any).ocr_confidence > 0) ||
        (region as any).cleanup_status ||
        (region as any).typeset_status ||
        (region as any).translation_status
      ) && (
        <details className="editor-section">
          <summary>Diagnostics</summary>
          <div className="editor-section-body">
          <div className="prop-grid">
            {typeof region.detector_confidence === "number" && region.detector_confidence > 0 && (
              <div className="prop-cell">
                <span className="prop-label">Detector</span>
                <span className="prop-val" title={`source: ${(region as any).detector_source ?? "ocr"}${(region as any).yolo_kind ? ` · kind: ${(region as any).yolo_kind}` : ""}`}>
                  {Math.round(region.detector_confidence * 100)}%
                  {(region as any).yolo_kind ? ` · ${(region as any).yolo_kind}` : ""}
                </span>
              </div>
            )}
            {typeof (region as any).ocr_confidence === "number" && (region as any).ocr_confidence > 0 && (
              <div className="prop-cell">
                <span className="prop-label">OCR</span>
                <span className="prop-val">{Math.round((region as any).ocr_confidence * 100)}%</span>
              </div>
            )}
            {(region as any).cleanup_status && (
              <div className="prop-cell">
                <span className="prop-label">Cleanup</span>
                <span
                  className="prop-val"
                  title={(region as any).cleanup_reason || undefined}
                  style={{
                    color: (region as any).cleanup_status === "ok" ? "var(--grn)"
                      : (region as any).cleanup_status === "skipped" ? "var(--amr)"
                      : "var(--t2)",
                  }}
                >
                  {(region as any).cleanup_status}
                  {(region as any).cleanup_reason ? ` · ${(region as any).cleanup_reason}` : ""}
                </span>
              </div>
            )}
            {(region as any).typeset_status && (
              <div className="prop-cell">
                <span className="prop-label">Typeset</span>
                <span
                  className="prop-val"
                  title={(region as any).typeset_reason || undefined}
                  style={{
                    color: (region as any).typeset_status === "ok" ? "var(--grn)"
                      : (region as any).typeset_status === "skipped" ? "var(--amr)"
                      : "var(--t2)",
                  }}
                >
                  {(region as any).typeset_status}
                  {(region as any).typeset_reason ? ` · ${(region as any).typeset_reason}` : ""}
                </span>
              </div>
            )}
            {(region as any).translation_status && (
              <div className="prop-cell">
                <span className="prop-label">Translation</span>
                <span
                  className="prop-val"
                  style={{
                    color: (region as any).translation_status === "ok" ? "var(--grn)"
                      : (region as any).translation_status === "flagged" ? "var(--red)"
                      : (region as any).translation_status === "skipped_sfx" ? "var(--amr)"
                      : "var(--t2)",
                  }}
                >
                  {(region as any).translation_status}
                </span>
              </div>
            )}
          </div>
          </div>
        </details>
      )}
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  LAYERS TAB                                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
const LayersTab = ({
  regions, selectedRegion, onSelectRegion, onPreviewRegion, onUpdateRegion,
}: {
  regions: Region[];
  selectedRegion: Region | null;
  onSelectRegion: (region: Region) => void;
  onPreviewRegion: (regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => void;
  onUpdateRegion: (idx: number, field: string, value: unknown) => void;
}) => {
  if (regions.length === 0) return <div className="empty-state" style={{ flex: 1 }}>No regions</div>;
  return (
    <div style={{ overflowY: "auto", flex: 1 }}>
      {regions.map(r => (
        <div key={r.id} className={`layer-row ${selectedRegion?.id === r.id ? "sel" : ""} ${r.visible ? "" : "off"}`} onClick={() => onSelectRegion(r)}>
          <div className="layer-icon li-region">
            R
          </div>
          <span className="layer-name">{r.label} · {r.tl || r.src || "empty"}</span>
          <div className="layer-actions">
            <button
              className={`layer-action ${r.visible ? "on" : ""}`}
              title={r.visible ? "Hide region" : "Show region"}
              onClick={e => {
                e.stopPropagation();
                onPreviewRegion(r.id, { visible: !r.visible });
                onUpdateRegion(r.idx, "visible", !r.visible);
              }}
            >
              <Svg icon="eye" size={12} />
            </button>
            <button
              className={`layer-action ${r.locked ? "on" : ""}`}
              title={r.locked ? "Unlock region" : "Lock region"}
              onClick={e => {
                e.stopPropagation();
                onPreviewRegion(r.id, { locked: !r.locked });
                onUpdateRegion(r.idx, "locked", !r.locked);
              }}
            >
              <Svg icon="lock" size={12} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  REVIEW TAB                                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
const ReviewTab = ({ issues }: { issues: Issue[] }) => (
  <div style={{ overflowY: "auto", flex: 1 }}>
    {issues.length === 0 ? (
      <div className="empty-state"><div>✓</div><div>No issues</div></div>
    ) : issues.map(iss => (
      <div key={iss.id} className={`issue-item ${iss.sev}`}>
        <div className="issue-msg">{iss.msg}</div>
        <div className="issue-ref">
          {iss.region && <span>{iss.region} · </span>}
          Page {iss.page}
        </div>
      </div>
    ))}
  </div>
);

const splitCsv = (value: string) =>
  value.split(",").map(v => v.trim()).filter(Boolean);

const MemoryTab = ({
  data, region, onMemoryMutation,
}: {
  data: Bootstrap;
  region: Region | null;
  onMemoryMutation: (result: Bootstrap) => void;
}) => {
  const [nameKr, setNameKr] = useState("");
  const [nameEn, setNameEn] = useState("");
  const [nameAliases, setNameAliases] = useState("");
  const [glossKr, setGlossKr] = useState("");
  const [glossEn, setGlossEn] = useState("");
  const [glossAliases, setGlossAliases] = useState("");
  const [glossAlternatives, setGlossAlternatives] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  const submitName = async () => {
    if (!nameKr.trim() || !nameEn.trim()) return;
    setBusyId("add-name");
    const resp = await api.addSeriesName(nameKr, nameEn, splitCsv(nameAliases));
    setBusyId(null);
    if (!resp.ok) return;
    setNameKr(""); setNameEn(""); setNameAliases("");
    onMemoryMutation(resp as unknown as Bootstrap);
  };

  const submitGlossary = async () => {
    if (!glossKr.trim() || !glossEn.trim()) return;
    setBusyId("add-glossary");
    const resp = await api.addSeriesGlossary(
      glossKr,
      glossEn,
      splitCsv(glossAlternatives),
      splitCsv(glossAliases),
    );
    setBusyId(null);
    if (!resp.ok) return;
    setGlossKr(""); setGlossEn(""); setGlossAliases(""); setGlossAlternatives("");
    onMemoryMutation(resp as unknown as Bootstrap);
  };

  const editName = async (entry: NameMemoryEntry) => {
    const kr = window.prompt("Korean name", entry.kr_name);
    if (kr === null) return;
    const en = window.prompt("English name", entry.en_name);
    if (en === null) return;
    const aliases = window.prompt("Korean aliases", entry.aliases_kr.join(", "));
    if (aliases === null) return;
    setBusyId(entry.id);
    const resp = await api.updateSeriesName(entry.id, {
      kr_name: kr,
      en_name: en,
      aliases_kr: splitCsv(aliases),
    });
    setBusyId(null);
    if (resp.ok) onMemoryMutation(resp as unknown as Bootstrap);
  };

  const editGlossary = async (entry: GlossaryEntry) => {
    const kr = window.prompt("Korean term", entry.source_kr);
    if (kr === null) return;
    const en = window.prompt("English term", entry.target_en);
    if (en === null) return;
    const alternatives = window.prompt("English alternatives", entry.alternatives_en.join(", "));
    if (alternatives === null) return;
    const aliases = window.prompt("Korean aliases", entry.aliases_kr.join(", "));
    if (aliases === null) return;
    setBusyId(entry.id);
    const resp = await api.updateSeriesGlossary(entry.id, {
      source_kr: kr,
      target_en: en,
      alternatives_en: splitCsv(alternatives),
      aliases_kr: splitCsv(aliases),
    });
    setBusyId(null);
    if (resp.ok) onMemoryMutation(resp as unknown as Bootstrap);
  };

  const deleteName = async (entry: NameMemoryEntry) => {
    if (!window.confirm(`Delete ${entry.kr_name} -> ${entry.en_name}?`)) return;
    setBusyId(entry.id);
    const resp = await api.deleteSeriesName(entry.id);
    setBusyId(null);
    if (resp.ok) onMemoryMutation(resp as unknown as Bootstrap);
  };

  const deleteGlossary = async (entry: GlossaryEntry) => {
    if (!window.confirm(`Delete ${entry.source_kr} -> ${entry.target_en}?`)) return;
    setBusyId(entry.id);
    const resp = await api.deleteSeriesGlossary(entry.id);
    setBusyId(null);
    if (resp.ok) onMemoryMutation(resp as unknown as Bootstrap);
  };

  if (!data.memory.available) {
    return <div className="empty-state" style={{ flex: 1 }}>{data.memory.error || "Memory unavailable"}</div>;
  }

  return (
    <div className="insp-body">
      <div className="insp-section">
        <div className="insp-section-title">Selected Hits</div>
        {!region || region.memory_hits.length === 0 ? (
          <div className="empty-state" style={{ padding: "10px 0" }}>No memory hits</div>
        ) : (
          <div className="mem-list" style={{ padding: 0 }}>
            {region.memory_hits.map((hit, idx) => (
              <div key={`${hit.type}-${idx}`} className="mem-card">
                <div className="mem-term">{hit.kr} {"->"} {hit.en}</div>
                <div className="mem-meta">{hit.type} · {hit.trust} · {hit.scope}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="insp-section-title" style={{ padding: "10px 10px 0" }}>
        Names · {data.memory.series_title}
      </div>
      <div className="mem-form">
        <div className="mem-two">
          <input className="mem-input" value={nameKr} onChange={e => setNameKr(e.target.value)} placeholder="Korean" />
          <input className="mem-input" value={nameEn} onChange={e => setNameEn(e.target.value)} placeholder="English" />
        </div>
        <input className="mem-input" value={nameAliases} onChange={e => setNameAliases(e.target.value)} placeholder="Aliases" />
        <button className="btn-ghost" disabled={busyId === "add-name"} onClick={submitName}>Add Name</button>
      </div>
      <div className="mem-list">
        {data.memory.names.length === 0 ? (
          <div className="empty-state" style={{ padding: 10 }}>No names</div>
        ) : data.memory.names.map(entry => (
          <div key={entry.id} className="mem-card">
            <div className="mem-row">
              <div className="mem-main">
                <div className="mem-term">{entry.kr_name} {"->"} {entry.en_name}</div>
                <div className="mem-meta">{entry.aliases_kr.join(", ") || "manual"} · {entry.scope}</div>
              </div>
              <div className="mem-actions">
                <button className="mem-mini" disabled={busyId === entry.id} onClick={() => editName(entry)}>Edit</button>
                <button className="mem-mini danger" disabled={busyId === entry.id} onClick={() => deleteName(entry)}>Del</button>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="insp-section-title" style={{ padding: "10px 10px 0" }}>Glossary</div>
      <div className="mem-form">
        <div className="mem-two">
          <input className="mem-input" value={glossKr} onChange={e => setGlossKr(e.target.value)} placeholder="Korean" />
          <input className="mem-input" value={glossEn} onChange={e => setGlossEn(e.target.value)} placeholder="English" />
        </div>
        <input className="mem-input" value={glossAlternatives} onChange={e => setGlossAlternatives(e.target.value)} placeholder="Alternatives" />
        <input className="mem-input" value={glossAliases} onChange={e => setGlossAliases(e.target.value)} placeholder="Aliases" />
        <button className="btn-ghost" disabled={busyId === "add-glossary"} onClick={submitGlossary}>Add Term</button>
      </div>
      <div className="mem-list">
        {data.memory.glossary.length === 0 ? (
          <div className="empty-state" style={{ padding: 10 }}>No glossary</div>
        ) : data.memory.glossary.map(entry => (
          <div key={entry.id} className="mem-card">
            <div className="mem-row">
              <div className="mem-main">
                <div className="mem-term">{entry.source_kr} {"->"} {entry.target_en}</div>
                <div className="mem-meta">{entry.alternatives_en.join(", ") || "manual"} · {entry.scope}</div>
              </div>
              <div className="mem-actions">
                <button className="mem-mini" disabled={busyId === entry.id} onClick={() => editGlossary(entry)}>Edit</button>
                <button className="mem-mini danger" disabled={busyId === entry.id} onClick={() => deleteGlossary(entry)}>Del</button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  RIGHT PANEL                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */
const RightPanel = ({
  data, region, issues, pageIndex, pageSize, onSelectRegion, onUpdateRegion, onPreviewRegion, onCommitBBox, onAddRegion,
  onDeleteRegion, onSetYoloTrainClass, onOcrRegion, onTranslateRegion, onMemoryMutation, fontOptions,
  cleanupPreview, cleanupDebug, debugOverlays, onDebugOverlayChange,
  cleanupCandidates, selectedCandidateId, onSelectCleanupCandidate,
  onPreviewCleanup, onApplyCleanup, onRerunCleanup, onDeleteCleanup, onSuggestSam2Mask, onRefreshCleanupDebug,
  onRecordMaskQaLabel, onTrainMaskQaModel,
  onCompareCleanupCandidates, onApplyCleanupCandidate, onUseCleanupCandidatePreview, cleanupCompareLoading,
}: {
  data:           Bootstrap;
  region:         Region | null;
  issues:         Issue[];
  pageIndex:      number;
  pageSize:       { w: number; h: number } | null;
  fontOptions:    FontOptions;
  onSelectRegion: (region: Region) => void;
  onUpdateRegion: (idx: number, field: string, value: unknown) => void;
  onPreviewRegion: (regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => void;
  onCommitBBox:   (region: Region, bbox: RegionBBox) => Promise<boolean>;
  onAddRegion:    () => void;
  onDeleteRegion: (idx: number, yoloRejectReason?: string) => void;
  onSetYoloTrainClass: (region: Region, classId: number) => void;
  onOcrRegion:    (idx: number) => void;
  onTranslateRegion: (idx: number) => void;
  onMemoryMutation: (result: Bootstrap) => void;
  cleanupPreview: (CleanupPreviewResponse & { regionId: string }) | null;
  cleanupDebug: (CleanupDebugResponse & { regionId: string }) | null;
  debugOverlays: DebugOverlayToggles;
  onDebugOverlayChange: (key: DebugOverlayKey, value: boolean) => void;
  cleanupCandidates: (CleanupCandidateCompareResponse & { regionId: string }) | null;
  selectedCandidateId: string;
  onSelectCleanupCandidate: (candidateId: string) => void;
  onPreviewCleanup: (region: Region, manualMask?: CleanupMaskPayload) => Promise<CleanupPreviewResponse | null>;
  onApplyCleanup: (idx: number, manualMask?: CleanupMaskPayload) => void;
  onRerunCleanup: (idx: number, manualMask?: CleanupMaskPayload) => void;
  onDeleteCleanup: (idx: number) => void;
  onSuggestSam2Mask: (region: Region, prompt: Record<string, unknown>) => Promise<Sam2MaskResponse>;
  onRefreshCleanupDebug: (region: Region, manualMask?: CleanupMaskPayload) => Promise<CleanupDebugResponse | null>;
  onRecordMaskQaLabel: (region: Region, label: string) => void;
  onTrainMaskQaModel: () => void;
  onCompareCleanupCandidates: (region: Region, manualMask?: CleanupMaskPayload) => Promise<CleanupCandidateCompareResponse | null>;
  onApplyCleanupCandidate: (idx: number, candidateId: string, manualMask?: CleanupMaskPayload) => void;
  onUseCleanupCandidatePreview: (region: Region, candidate: CleanupCandidate) => void;
  cleanupCompareLoading: boolean;
}) => {
  const [tab, setTab] = useState<"editor" | "layers" | "issues" | "memory">("editor");
  useEffect(() => { if (region) setTab("editor"); }, [region?.id]);
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div className="tab-bar">
        {(["editor", "layers", "issues", "memory"] as const).map(t => (
          <div key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {t === "issues"
              ? `Issues${issues.length > 0 ? ` (${issues.length})` : ""}`
              : t === "memory"
                ? `Memory${data.memory.names.length + data.memory.glossary.length > 0 ? ` (${data.memory.names.length + data.memory.glossary.length})` : ""}`
              : t.charAt(0).toUpperCase() + t.slice(1)}
          </div>
        ))}
      </div>
      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {tab === "editor" && (
          <InspectorTab
            region={region}
            regions={data.regions}
            pageIndex={pageIndex}
            pageSize={pageSize}
            fontOptions={fontOptions}
            onUpdateRegion={onUpdateRegion}
            onPreviewRegion={onPreviewRegion}
            onCommitBBox={onCommitBBox}
            onAddRegion={onAddRegion}
            onDeleteRegion={onDeleteRegion}
            onSetYoloTrainClass={onSetYoloTrainClass}
            onOcrRegion={onOcrRegion}
            onTranslateRegion={onTranslateRegion}
            cleanupPreview={cleanupPreview}
            cleanupDebug={cleanupDebug}
            debugOverlays={debugOverlays}
            onDebugOverlayChange={onDebugOverlayChange}
            cleanupCandidates={cleanupCandidates}
            selectedCandidateId={selectedCandidateId}
            onSelectCleanupCandidate={onSelectCleanupCandidate}
            sam2Settings={data.meta.settings}
            onPreviewCleanup={onPreviewCleanup}
            onApplyCleanup={onApplyCleanup}
            onRerunCleanup={onRerunCleanup}
            onDeleteCleanup={onDeleteCleanup}
            onSuggestSam2Mask={onSuggestSam2Mask}
            onRefreshCleanupDebug={onRefreshCleanupDebug}
            onRecordMaskQaLabel={onRecordMaskQaLabel}
            onTrainMaskQaModel={onTrainMaskQaModel}
            onCompareCleanupCandidates={onCompareCleanupCandidates}
            onApplyCleanupCandidate={onApplyCleanupCandidate}
            onUseCleanupCandidatePreview={onUseCleanupCandidatePreview}
            cleanupCompareLoading={cleanupCompareLoading}
          />
        )}
        {tab === "layers"    && (
          <LayersTab
            regions={data.regions}
            selectedRegion={region}
            onSelectRegion={onSelectRegion}
            onPreviewRegion={onPreviewRegion}
            onUpdateRegion={onUpdateRegion}
          />
        )}
        {tab === "issues"    && <ReviewTab issues={issues} />}
        {tab === "memory"    && <MemoryTab data={data} region={region} onMemoryMutation={onMemoryMutation} />}
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  STATUS BAR                                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
const StatusBar = ({
  data, activePage, selectedRegion, pipeRunning, liveStatus, pipelineProgress,
}: {
  data:           Bootstrap;
  activePage:     number;
  selectedRegion: Region | null;
  pipeRunning:    boolean;
  liveStatus:     string;
  pipelineProgress: ProgressEvent | null;
}) => {
  const page = data.pages[activePage];
  const progress = data.meta.chapterProgress ?? 0;
  const progressPage = typeof pipelineProgress?.page_idx === "number" ? pipelineProgress.page_idx + 1 : null;
  const progressTotal = pipelineProgress?.page_total ?? pipelineProgress?.total ?? data.pages.length;
  const progressRegion = typeof pipelineProgress?.region_idx === "number" && pipelineProgress?.region_total
    ? ` region ${Math.min(pipelineProgress.region_idx + 1, pipelineProgress.region_total)} / ${pipelineProgress.region_total}`
    : "";
  const progressLabel = pipelineProgress?.job
    ? `${pipelineProgress.job === "run_all" ? "Run All" : pipelineProgress.job === "run_page" ? "Run Page" : pipelineProgress.job}: ${pipelineProgress.stage ?? "working"}${progressPage ? ` page ${progressPage} / ${progressTotal}` : ""}${progressRegion}`
    : "";
  return (
    <div className="ml-statusbar no-select">
      <div className="sb-item">
        <div className={`sb-dot ${pipeRunning ? "busy" : ""}`} />
        <span style={{ color: "var(--t2)", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {liveStatus || (pipeRunning ? "Processing…" : "Ready")}
        </span>
      </div>
      {data.pages.length > 0 && (
        <>
          <div className="divider-v" />
          <div className="sb-item">Pg <span>{activePage + 1}</span> / {data.pages.length}</div>
          <div className="sb-item">Regions <span>{page?.regions ?? 0}</span></div>
          {progressLabel && (
            <div className="sb-item" title={pipelineProgress?.message}>
              Processing <span>{progressLabel}</span>
            </div>
          )}
        </>
      )}
      <div className="sb-item">
        Issues <span style={{ color: data.issues.filter(i => i.sev === "err").length ? "var(--red)" : "var(--t2)" }}>
          {data.issues.filter(i => i.sev !== "info").length}
        </span>
      </div>
      {selectedRegion && <div className="sb-item">Selected <span>{selectedRegion.label}</span></div>}
      <div style={{ flex: 1 }} />
      {data.meta.chapterDir && (
        <>
          <div className="sb-progress">
            <div className="sb-progress-fill" style={{ width: `${pipelineProgress?.running && typeof pipelineProgress.percent === "number" ? pipelineProgress.percent : progress}%` }} />
          </div>
          <div className="sb-item"><span>{Math.round(pipelineProgress?.running && typeof pipelineProgress.percent === "number" ? pipelineProgress.percent : progress)}%</span> {pipelineProgress?.running ? "job" : "chapter"}</div>
          <div className="divider-v" />
        </>
      )}
      <div className="sb-item">
        <kbd className="kbd">R</kbd> <span style={{ color: "var(--t4)" }}>Run All</span>
        <span style={{ margin: "0 4px", color: "var(--t4)" }}>·</span>
        <kbd className="kbd">Esc</kbd> <span style={{ color: "var(--t4)" }}>Deselect</span>
        <span style={{ margin: "0 4px", color: "var(--t4)" }}>·</span>
        <kbd className="kbd">+/−</kbd> <span style={{ color: "var(--t4)" }}>Zoom</span>
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────────────────────────────────────── */
/*  ROOT APP                                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */
export default function App() {
  // Inject global CSS once
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
    return () => { document.head.removeChild(el); };
  }, []);

  // ── State ──────────────────────────────────────────────────────────────────
  const [data,            setData]            = useState<Bootstrap>(EMPTY_BOOTSTRAP);
  const [activeSeries,    setActiveSeries]    = useState<string>("");
  const [activeChapter,   setActiveChapter]   = useState<string>("");
  const [activePage,      setActivePage]      = useState<number>(0);
  const [selectedRegion,  setSelectedRegion]  = useState<Region | null>(null);
  const [cleanupPreview,  setCleanupPreview]  = useState<(CleanupPreviewResponse & { regionId: string }) | null>(null);
  const [cleanupDebug,    setCleanupDebug]    = useState<(CleanupDebugResponse & { regionId: string }) | null>(null);
  const [cleanupCandidates, setCleanupCandidates] = useState<(CleanupCandidateCompareResponse & { regionId: string }) | null>(null);
  const [cleanupCompareLoading, setCleanupCompareLoading] = useState(false);
  const [selectedCandidateId, setSelectedCandidateId] = useState("");
  const [debugOverlays,   setDebugOverlays]   = useState<DebugOverlayToggles>(() => {
    try {
      const raw = localStorage.getItem("ml.cleanupDebugOverlays");
      return raw ? { ...DEBUG_OVERLAY_DEFAULTS, ...JSON.parse(raw) } : DEBUG_OVERLAY_DEFAULTS;
    } catch {
      return DEBUG_OVERLAY_DEFAULTS;
    }
  });
  const [regionDrafts,    setRegionDrafts]    = useState<Record<string, RegionDraft>>({});
  const [showEnglishOverlay, setShowEnglishOverlay] = useState(true);
  const [fontOptions,     setFontOptions]     = useState<FontOptions>({ roles: DEFAULT_FONT_ROLES, fonts: [] });
  const [settingsOpen,    setSettingsOpen]    = useState(false);
  const [modelConfig,     setModelConfig]     = useState<Record<string, string>>({});
  const [toast,           setToast]           = useState("");
  const [zoom,            setZoom]            = useState<number>(85);
  const [canvasImageMode, setCanvasImageMode] = useState<ImageMode>("best");
  const [pageVersions,    setPageVersions]    = useState<Record<number, number>>({});
  const [pageSizes,       setPageSizes]       = useState<Record<number, { w: number; h: number }>>({});
  const [pipelineProgress, setPipelineProgress] = useState<ProgressEvent | null>(null);
  const [yoloTraining, setYoloTraining] = useState<{ status?: string; running?: boolean; log?: string; onnx?: string; error?: string } | null>(null);
  const [readerMode,      setReaderMode]      = useState<"single" | "continuous">("single");
  const [showPageIndicator, setShowPageIndicator] = useState(true);
  const [leftOpen,        setLeftOpen]        = useState(true);
  const [leftWidth,       setLeftWidth]       = useState(() => {
    try {
      const raw = localStorage.getItem("ml.leftPanelWidth");
      return clampNumber(Number(raw) || 220, 220, 520);
    } catch {
      return 220;
    }
  });
  const [resizingLeft,    setResizingLeft]    = useState(false);
  const [rightOpen,       setRightOpen]       = useState(true);
  const [rightWidth,      setRightWidth]      = useState(() => {
    try {
      const raw = localStorage.getItem("ml.rightPanelWidth");
      return clampNumber(Number(raw) || 380, 340, Math.min(680, Math.floor(window.innerWidth * 0.5)));
    } catch {
      return 380;
    }
  });
  const [resizingRight,   setResizingRight]   = useState(false);
  const [pipeRunning,     setPipeRunning]     = useState(false);
  const [pipeState,       setPipeState]       = useState<StepState[]>(["idle","idle","idle","idle","idle"]);
  const [liveStatus,      setLiveStatus]      = useState("");
  const [showBrowse,      setShowBrowse]      = useState(false);
  const [availableSources, setAvailableSources] = useState<string[]>(["naver-comic"]);
  const [selectedSeriesTitle, setSelectedSeriesTitle] = useState<string | null>(null);
  const [selectedSeriesKey, setSelectedSeriesKey] = useState("");
  const [coverPreview, setCoverPreview] = useState<{ src: string; title: string; source: string } | null>(null);
  const [seriesDetailRefreshKey, setSeriesDetailRefreshKey] = useState(0);
  const [continuousScrollTarget, setContinuousScrollTarget] = useState<number | null>(null);
  const activePageRef = useRef(0);
  const dataRef = useRef<Bootstrap>(EMPTY_BOOTSTRAP);
  const applyBootstrapRef = useRef<(b: Bootstrap) => void>(() => {});
  const suppressBackendPageSyncRef = useRef(false);
  const progressRefreshTimerRef = useRef<number | null>(null);
  const pageSyncTimerRef = useRef<number | null>(null);
  const pageSyncSeqRef = useRef(0);

  useEffect(() => {
    activePageRef.current = activePage;
  }, [activePage]);

  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  const clearPageSyncTimer = () => {
    if (pageSyncTimerRef.current !== null) {
      window.clearTimeout(pageSyncTimerRef.current);
      pageSyncTimerRef.current = null;
    }
  };

  // ── Bootstrap on mount ─────────────────────────────────────────────────────
  useEffect(() => {
    api.getBootstrap().then(resp => {
      if (!resp.ok) return;
      applyBootstrap(resp as unknown as Bootstrap);
    });
    api.listFonts().then(resp => {
      if (!resp.ok) return;
      setFontOptions({
        roles: resp.roles?.length ? resp.roles : DEFAULT_FONT_ROLES,
        fonts: resp.fonts ?? [],
      });
    });
    api.getModelConfig().then(resp => {
      if (resp.ok) setModelConfig((resp.config ?? {}) as Record<string, string>);
    });
    api.listSources().then(resp => {
      if (resp.ok && resp.sources?.length) setAvailableSources(resp.sources);
    });
  }, []);

  useEffect(() => {
    if (!resizingLeft) return;
    const onMove = (ev: MouseEvent) => {
      const next = clampNumber(ev.clientX, 220, 520);
      setLeftWidth(next);
      try { localStorage.setItem("ml.leftPanelWidth", String(next)); } catch { /* noop */ }
    };
    const onUp = () => setResizingLeft(false);
    document.body.style.cursor = "col-resize";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      document.body.style.cursor = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizingLeft]);

  useEffect(() => {
    try { localStorage.setItem("ml.cleanupDebugOverlays", JSON.stringify(debugOverlays)); } catch { /* noop */ }
  }, [debugOverlays]);

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      if (detail === "raw" || detail === "cleaned" || detail === "typeset" || detail === "best") {
        setCanvasImageMode(detail);
      }
    };
    window.addEventListener("ml:set-image-mode", handler);
    return () => window.removeEventListener("ml:set-image-mode", handler);
  }, []);

  useEffect(() => {
    if (!resizingRight) return;
    const onMove = (ev: MouseEvent) => {
      const max = Math.min(620, Math.floor(window.innerWidth * 0.45));
      const next = clampNumber(window.innerWidth - ev.clientX, 280, Math.max(280, max));
      setRightWidth(next);
      try { localStorage.setItem("ml.rightPanelWidth", String(next)); } catch { /* noop */ }
    };
    const onUp = () => setResizingRight(false);
    document.body.style.cursor = "ew-resize";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      document.body.style.cursor = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizingRight]);

  useEffect(() => () => {
    clearPageSyncTimer();
    if (progressRefreshTimerRef.current !== null) window.clearTimeout(progressRefreshTimerRef.current);
  }, []);

  // ── Progress / busy events from Python ────────────────────────────────────
  useEffect(() => {
    const bumpPages = (pages: number[]) => {
      const valid = Array.from(new Set(pages.filter(idx => Number.isFinite(idx) && idx >= 0).map(idx => Math.floor(idx))));
      if (!valid.length) return;
      setPageVersions(prev => {
        const next = { ...prev };
        valid.forEach(idx => { next[idx] = (next[idx] ?? 0) + 1; });
        return next;
      });
      const chapterDir = dataRef.current.meta.chapterDir;
      if (chapterDir) valid.forEach(idx => invalidatePageImageCache(chapterDir, idx));
    };

    const scheduleBootstrapRefresh = () => {
      if (progressRefreshTimerRef.current !== null) return;
      progressRefreshTimerRef.current = window.setTimeout(() => {
        progressRefreshTimerRef.current = null;
        const visiblePage = activePageRef.current;
        suppressBackendPageSyncRef.current = true;
        api.getBootstrap().then(resp => {
          if (resp.ok) applyBootstrapRef.current(resp as unknown as Bootstrap);
          setActivePage(visiblePage);
        }).finally(() => {
          suppressBackendPageSyncRef.current = false;
        });
      }, 450);
    };

    const unsub1 = onProgress((ev: ProgressEvent) => {
      setPipelineProgress(ev);
      setLiveStatus(ev.error ? `Error: ${ev.error}` : ev.message);
      const updatedPages = ev.updated_pages ?? [];
      if (updatedPages.length) {
        bumpPages(updatedPages);
        scheduleBootstrapRefresh();
      }
      if (ev.running === false && !ev.error) {
        window.setTimeout(() => {
          setPipelineProgress(current => current === ev ? null : current);
        }, 1800);
      }
    });
    const unsub2 = onBusyChange((busy) => setPipeRunning(busy));
    return () => { unsub1(); unsub2(); };
  }, []);

  // ── Derive pipeline state from page status ─────────────────────────────────
  useEffect(() => {
    const page = data.pages[activePage];
    if (!page) {
      setPipeState(["idle","idle","idle","idle","idle"]);
      return;
    }
    setPipeState(page.status.map((s, i) => {
      if (s === "done") return "done";
      if (i === 0 || page.status[i - 1] === "done") return "active";
      return "idle";
    }) as StepState[]);
  }, [data, activePage]);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const showToast = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(current => current === message ? "" : current), 1100);
  };

  const setSelectedSeries = (series: Series | null) => {
    if (!series) {
      setActiveSeries("");
      setSelectedSeriesTitle(null);
      setSelectedSeriesKey("");
      return;
    }
    setActiveSeries(series.id);
    setSelectedSeriesTitle(series.title);
    setSelectedSeriesKey(seriesStableKey(series));
  };

  const applyBootstrap = (b: Bootstrap) => {
    setData(b);
    setRegionDrafts({});
    const backendActive = b.series.find(s => s.id === b.meta.activeSeriesId) ?? null;
    const preservedActive = selectedSeriesKey
      ? (b.series.find(s => seriesStableKey(s) === selectedSeriesKey) ?? null)
      : null;
    const nextSeries = preservedActive ?? backendActive ?? b.series[0] ?? null;
    setSelectedSeries(nextSeries);
    if (b.meta.activeChapterId) setActiveChapter(b.meta.activeChapterId);
    if (!suppressBackendPageSyncRef.current && typeof b.meta.activePageIdx === "number") setActivePage(b.meta.activePageIdx);
    const selectionPage = suppressBackendPageSyncRef.current ? activePageRef.current : (b.meta.activePageIdx ?? activePageRef.current);
    setSelectedRegion(prev => prev ? (editorRegionsForPage(b, selectionPage).find(r => r.id === prev.id) ?? null) : null);
    setLiveStatus(b.meta.status || "");
    // Pass 4: keep modelConfig in sync with backend settings so the settings
    // panel always reflects current state (e.g. process_sfx_regions) without
    // needing a separate getModelConfig call after every pipeline step.
    const backendSettings = (b.meta as any).settings as Record<string, unknown> | undefined;
    if (backendSettings && typeof backendSettings === "object") {
      setModelConfig(prev => ({
        ...prev,
        ...Object.fromEntries(
          Object.entries(backendSettings).map(([k, v]) => [k, String(v)]),
        ),
      }));
    }
  };
  applyBootstrapRef.current = applyBootstrap;

  const bumpPageImageVersions = (pages: number[]) => {
    const valid = Array.from(new Set(pages.filter(idx => Number.isFinite(idx) && idx >= 0).map(idx => Math.floor(idx))));
    if (!valid.length) return;
    setPageVersions(prev => {
      const next = { ...prev };
      valid.forEach(idx => { next[idx] = (next[idx] ?? 0) + 1; });
      return next;
    });
    if (data.meta.chapterDir) valid.forEach(idx => invalidatePageImageCache(data.meta.chapterDir, idx));
  };

  const withBusy = async (fn: () => Promise<Bootstrap | null>) => {
    if (pipeRunning) return;
    setPipeRunning(true);
    try {
      const result = await fn();
      if (result) applyBootstrap(result);
    } finally {
      setPipeRunning(false);
    }
  };

  // ── Actions ────────────────────────────────────────────────────────────────
  const handleOpenChapter = () => withBusy(async () => {
    const resp = await api.openChapterFolder();
    if (!resp.ok || resp.cancelled) return null;
    return resp as unknown as Bootstrap;
  });

  const handleBrowseSelect = async (title: string, _source: string, _sourceId: string, _card: BrowseCard) => {
    setSelectedSeriesTitle(title);
    setShowBrowse(false);
    const resp = await api.getBootstrap();
    if (resp.ok) {
      const next = resp as unknown as Bootstrap;
      applyBootstrap(next);
      const selected = next.series.find(s => s.title === title);
      if (selected) setSelectedSeries(selected);
      setSeriesDetailRefreshKey(k => k + 1);
    }
    showToast("Series indexed");
  };

  const refreshBootstrap = () => api.getBootstrap().then(resp => {
    if (!resp.ok) return;
    applyBootstrap(resp as unknown as Bootstrap);
    setSeriesDetailRefreshKey(k => k + 1);
  });

  const handleDeleteSeries = (title: string) => {
    const wasSelected = selectedSeriesTitle === title;
    if (wasSelected) {
      setSelectedSeriesTitle(null);
      setSelectedSeriesKey("");
    }
    api.getBootstrap().then(resp => {
      if (!resp.ok) return;
      const next = resp as unknown as Bootstrap;
      if (wasSelected) {
        const fallback = next.series[0] ?? null;
        setData(next);
        setRegionDrafts({});
        setSelectedSeries(fallback);
        if (next.meta.activeChapterId) setActiveChapter(next.meta.activeChapterId);
        if (typeof next.meta.activePageIdx === "number") setActivePage(next.meta.activePageIdx);
      } else {
        applyBootstrap(next);
      }
      setSeriesDetailRefreshKey(k => k + 1);
    });
    showToast("Series removed from library");
  };

  const handleOpenFolderFromDetail = (folder: string) => withBusy(async () => {
    const resp = await api.importChapter(folder);
    if (!resp.ok) return null;
    return resp as unknown as Bootstrap;
  });

  const handleStep = useCallback((idx: number) => withBusy(async () => {
    if (!data.meta.chapterDir || data.pages.length === 0) {
      showToast("Open or import a chapter before running the pipeline.");
      return null;
    }
    const step = PIPELINE_STEPS[idx].toLowerCase() as "detect" | "ocr" | "translate" | "cleanup" | "typeset";
    setPipeState(prev => prev.map((_s, i) => i < idx ? "done" : i === idx ? "running" : "idle") as StepState[]);
    const resp = await api.runStep(step);
    if (!resp.ok) { setLiveStatus(`Error: ${resp.error}`); return null; }
    bumpPageImageVersions([activePageRef.current]);
    return resp as unknown as Bootstrap;
  }), [pipeRunning, data.meta.chapterDir, data.pages.length]);

  const handleRunAll = useCallback(() => withBusy(async () => {
    if (!data.meta.chapterDir || data.pages.length === 0) {
      showToast("Open or import a chapter before running the pipeline.");
      return null;
    }
    const visiblePage = activePageRef.current;
    const allPages = data.pages.map(pg => pg.idx);
    suppressBackendPageSyncRef.current = true;
    const resp = await api.runAll().finally(() => {
      suppressBackendPageSyncRef.current = false;
    });
    if (!resp.ok) { setLiveStatus(`Error: ${resp.error}`); return null; }
    bumpPageImageVersions(((resp as unknown as { updatedPages?: number[] }).updatedPages ?? allPages));
    applyBootstrap(resp as unknown as Bootstrap);
    setActivePage(visiblePage);
    showToast("Run complete");
    return null;
  }), [pipeRunning, data.meta.chapterDir, data.pages.length]);

  const handleContinueRunAll = useCallback(() => withBusy(async () => {
    if (!data.meta.chapterDir || data.pages.length === 0) {
      showToast("Open or import a chapter before continuing.");
      return null;
    }
    const visiblePage = activePageRef.current;
    const allPages = data.pages.map(pg => pg.idx);
    suppressBackendPageSyncRef.current = true;
    const resp = await api.continueRunAll().finally(() => {
      suppressBackendPageSyncRef.current = false;
    });
    if (!resp.ok) { setLiveStatus(`Error: ${resp.error}`); return null; }
    bumpPageImageVersions(((resp as unknown as { updatedPages?: number[] }).updatedPages ?? allPages));
    applyBootstrap(resp as unknown as Bootstrap);
    setActivePage(visiblePage);
    showToast("Run continued");
    return null;
  }), [pipeRunning, data.meta.chapterDir, data.pages.length]);

  const handleRunPage = useCallback(() => withBusy(async () => {
    if (!data.meta.chapterDir || data.pages.length === 0) {
      showToast("Open or import a chapter before running the pipeline.");
      return null;
    }
    clearPageSyncTimer();
    const requestedPage = activePageRef.current;
    const selectResp = await api.goToPage(requestedPage);
    if (!selectResp.ok) { setLiveStatus(`Error: ${selectResp.error}`); return null; }
    const resp = await api.runCurrentPage();
    if (!resp.ok) { setLiveStatus(`Error: ${resp.error}`); return null; }
    const processedPage = (resp as unknown as { processedPageIdx?: number }).processedPageIdx ?? requestedPage;
    const processedSummary = (resp as unknown as Bootstrap).pages?.find(pg => pg.idx === processedPage);
    bumpPageImageVersions([processedPage]);
    console.debug("run_page.refresh", {
      requestedPageIdx: requestedPage,
      processedPageIdx: processedPage,
      imageMode: canvasImageMode,
      cacheKey: `${data.meta.chapterDir}:page:${processedPage}:${canvasImageMode}`,
      hasTypeset: processedSummary?.status?.[4] === "done",
    });
    showToast(canvasImageMode === "raw" ? "Typeset complete. Switch to Best or Typeset to view result." : "Page complete");
    return resp as unknown as Bootstrap;
  }), [pipeRunning, data.meta.chapterDir, data.pages.length, canvasImageMode]);

  const handleDetectAll = useCallback(() => withBusy(async () => {
    if (!data.meta.chapterDir || data.pages.length === 0) {
      showToast("Open or import a chapter before detecting boxes.");
      return null;
    }
    const visiblePage = activePageRef.current;
    const allPages = data.pages.map(pg => pg.idx);
    suppressBackendPageSyncRef.current = true;
    const resp = await api.detectAll().finally(() => {
      suppressBackendPageSyncRef.current = false;
    });
    if (!resp.ok) { setLiveStatus(`Error: ${resp.error}`); return null; }
    bumpPageImageVersions(((resp as unknown as { updatedPages?: number[] }).updatedPages ?? allPages));
    applyBootstrap(resp as unknown as Bootstrap);
    setActivePage(visiblePage);
    showToast("YOLO detection complete");
    return null;
  }), [pipeRunning, data.meta.chapterDir, data.pages]);

  const handleExportYoloDataset = useCallback(() => withBusy(async () => {
    const resp = await api.exportYoloFinetuneDataset();
    if (!resp.ok) { setLiveStatus(`YOLO dataset export failed: ${resp.error}`); return null; }
    showToast(`YOLO dataset exported (${resp.pages ?? 0} pages)`);
    setLiveStatus(`YOLO dataset: ${resp.dataset_dir ?? ""}`);
    return null;
  }), []);

  const handleTrainYolo = useCallback(() => withBusy(async () => {
    const resp = await api.trainYoloDetector();
    if (!resp.ok) { setLiveStatus(`YOLO training failed to start: ${resp.error}`); return null; }
    setYoloTraining({ status: resp.status ?? "starting", running: resp.running, log: resp.log });
    showToast("YOLO training started");
    setLiveStatus(`YOLO training started. Log: ${resp.log ?? ""}`);
    return null;
  }), []);

  useEffect(() => {
    if (!yoloTraining?.running) return;
    const id = window.setInterval(async () => {
      const resp = await api.getYoloTrainingStatus();
      if (!resp.ok) return;
      setYoloTraining({
        status: resp.status,
        running: resp.running,
        log: resp.log,
        onnx: resp.onnx,
        error: resp.error,
      });
      if (!resp.running) {
        setLiveStatus(
          resp.status === "complete"
            ? `YOLO training complete: ${resp.onnx ?? ""}`
            : `YOLO training ${resp.status ?? "stopped"}${resp.error ? `: ${resp.error}` : ""}`
        );
      }
    }, 5000);
    return () => window.clearInterval(id);
  }, [yoloTraining?.running]);

  const handleExport = () => withBusy(async () => {
    const resp = await api.exportProject();
    if (!resp.ok) { setLiveStatus(`Export failed: ${(resp as { error?: string }).error}`); return null; }
    setLiveStatus(`Exported to: ${resp.export_dir}`);
    return null;
  });

  const handleSaveSettings = useCallback(async () => {
    const resp = await api.updateModelConfig(modelConfig);
    if (resp.ok) {
      setModelConfig((resp.config ?? modelConfig) as Record<string, string>);
      setSettingsOpen(false);
      showToast("Settings saved");
    }
  }, [modelConfig]);

  const handlePageSelect = useCallback((idx: number, mode: PageSelectMode = "immediate") => {
    const maxIdx = Math.max(0, data.pages.length - 1);
    const nextIdx = clampNumber(idx, 0, maxIdx);
    setActivePage(nextIdx);
    setSelectedRegion(null);
    setCleanupPreview(null);
    setCleanupDebug(null);
    setCleanupCandidates(null);
    setSelectedCandidateId("");
    if (mode === "immediate") setContinuousScrollTarget(nextIdx);
    if (mode === "local") return;

    const syncPage = async (targetIdx: number, seq: number) => {
      const resp = await api.goToPage(targetIdx);
      if (!resp.ok) {
        setLiveStatus(`Error: ${resp.error}`);
        return;
      }
      if (seq === pageSyncSeqRef.current) applyBootstrap(resp as unknown as Bootstrap);
    };

    pageSyncSeqRef.current += 1;
    const seq = pageSyncSeqRef.current;
    clearPageSyncTimer();
    if (mode === "debounced") {
      pageSyncTimerRef.current = window.setTimeout(() => {
        pageSyncTimerRef.current = null;
        syncPage(nextIdx, seq);
      }, 220);
      return;
    }
    syncPage(nextIdx, seq);
  }, [data.pages.length, selectedSeriesKey]);

  const handleUpdateRegion = useCallback(async (idx: number, field: string, value: unknown) => {
    renderDebug("backend.mutation", { action: "updateRegion", page: activePage, region: idx, field, value });
    const ownerPage = selectedRegion?.idx === idx ? regionOwnerPage(selectedRegion, activePage) : activePage;
    suppressBackendPageSyncRef.current = true;
    let resp: any = { ok: false };
    try {
      resp = await api.updateRegion(idx, field, value, ownerPage);
    } finally {
      suppressBackendPageSyncRef.current = false;
    }
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      showToast(field === "translation" ? "Translation saved" : field === "text" ? "Source saved" : "Region updated");
    }
  }, [activePage, selectedRegion]);

  const handlePreviewRegion = useCallback((regionId: string, patch: RegionDraft, options?: { requestSprite?: boolean; reason?: string }) => {
    renderDebug("local.previewMutation", { action: "previewRegion", page: activePage, regionId, patch });
    const mergedDraft = { ...(regionDrafts[regionId] ?? {}), ...patch };
    setRegionDrafts(prev => ({ ...prev, [regionId]: { ...(prev[regionId] ?? {}), ...patch } }));
    setSelectedRegion(prev => prev?.id === regionId ? { ...prev, ...patch } : prev);
    if (options?.requestSprite || !isBboxOnlyDraft(patch)) {
      window.dispatchEvent(new CustomEvent("ml:preview-sprite-request", {
        detail: { page: activePage, regionId, reason: options?.reason ?? previewReason(patch), draft: mergedDraft },
      }));
    }
  }, [activePage, regionDrafts]);

  const handlePageSizeChange = useCallback((pageIdx: number, size: { w: number; h: number }) => {
    if (!size.w || !size.h) return;
    setPageSizes(prev => {
      const current = prev[pageIdx];
      if (current?.w === size.w && current?.h === size.h) return prev;
      return { ...prev, [pageIdx]: size };
    });
  }, []);

  const handleSelectRegion = useCallback((region: Region | null) => {
    renderDebug("select", {
      action: "select",
      page: activePage,
      region: region?.idx ?? null,
      style: region ? regionStyleDebug(region) : null,
    });
    setSelectedRegion(region);
    setCleanupDebug(current => current && current.regionId === region?.id ? current : null);
    setCleanupCandidates(current => current && current.regionId === region?.id ? current : null);
    setSelectedCandidateId("");
  }, [activePage]);

  const handleCommitBBox = useCallback(async (region: Region, bbox: RegionBBox) => {
    const ownerPage = regionOwnerPage(region, activePage);
    const editPage = typeof region.display_page_idx === "number" ? region.display_page_idx : activePage;
    const baseRegion = editorRegionsForPage(data, editPage).find(r => r.id === region.id);
    const pendingStyle: Array<[string, unknown]> = [];
    if (baseRegion) {
      if (region.font !== baseRegion.font) pendingStyle.push(["font_name", region.font]);
      if (region.size !== baseRegion.size) pendingStyle.push(["font_size", region.size]);
      if (region.align !== baseRegion.align) pendingStyle.push(["align", region.align]);
      if (region.fg !== baseRegion.fg) pendingStyle.push(["fg_color", region.fg]);
      if (region.outline !== baseRegion.outline) pendingStyle.push(["outline_color", region.outline]);
      if (region.outline_width !== baseRegion.outline_width) pendingStyle.push(["outline_width", region.outline_width]);
      if (region.shadow !== baseRegion.shadow) pendingStyle.push(["shadow_color", region.shadow]);
      if (region.shadow_on !== baseRegion.shadow_on) pendingStyle.push(["shadow_on", region.shadow_on]);
      if (region.shadow_offset_x !== baseRegion.shadow_offset_x) pendingStyle.push(["shadow_offset_x", region.shadow_offset_x]);
      if (region.shadow_offset_y !== baseRegion.shadow_offset_y) pendingStyle.push(["shadow_offset_y", region.shadow_offset_y]);
      if (region.shadow_opacity !== baseRegion.shadow_opacity) pendingStyle.push(["shadow_opacity", region.shadow_opacity]);
      if (region.shadow_blur !== baseRegion.shadow_blur) pendingStyle.push(["shadow_blur", region.shadow_blur]);
      if (region.glow !== baseRegion.glow) pendingStyle.push(["glow_color", region.glow]);
      if (region.glow_on !== baseRegion.glow_on) pendingStyle.push(["glow_on", region.glow_on]);
      if (region.glow_radius !== baseRegion.glow_radius) pendingStyle.push(["glow_radius", region.glow_radius]);
      if (region.glow_intensity !== baseRegion.glow_intensity) pendingStyle.push(["glow_intensity", region.glow_intensity]);
      if (region.reflection_on !== baseRegion.reflection_on) pendingStyle.push(["reflection_on", region.reflection_on]);
      if (region.reflection_opacity !== baseRegion.reflection_opacity) pendingStyle.push(["reflection_opacity", region.reflection_opacity]);
      if (region.reflection_offset !== baseRegion.reflection_offset) pendingStyle.push(["reflection_offset", region.reflection_offset]);
      if (region.reflection_blur !== baseRegion.reflection_blur) pendingStyle.push(["reflection_blur", region.reflection_blur]);
      if (region.reflection_fade !== baseRegion.reflection_fade) pendingStyle.push(["reflection_fade", region.reflection_fade]);
      if (region.gradient_on !== baseRegion.gradient_on) pendingStyle.push(["gradient_on", region.gradient_on]);
      if (region.gradient_start !== baseRegion.gradient_start) pendingStyle.push(["gradient_start", region.gradient_start]);
      if (region.gradient_end !== baseRegion.gradient_end) pendingStyle.push(["gradient_end", region.gradient_end]);
      if (region.gradient_angle !== baseRegion.gradient_angle) pendingStyle.push(["gradient_angle", region.gradient_angle]);
      if (region.rotation_angle !== baseRegion.rotation_angle) pendingStyle.push(["rotation_angle", region.rotation_angle]);
    }
    for (const [field, value] of pendingStyle) {
      renderDebug("api.updateRegion.beforeBBox", { page: editPage, ownerPage, region: region.idx, field, value });
      await api.updateRegion(region.idx, field, value, ownerPage);
    }
    const cleanBbox = {
      x: Number(bbox.x),
      y: Number(bbox.y),
      w: Number(bbox.w),
      h: Number(bbox.h),
    };
    if (cleanBbox.x === region.x && cleanBbox.y === region.y && cleanBbox.w === region.w && cleanBbox.h === region.h) {
      renderDebug("api.updateRegionBBox.skip", { page: activePage, region: region.idx, reason: "unchanged", bbox: cleanBbox });
      return false;
    }
    renderDebug("backend.mutation", {
      action: "updateRegionBBox",
      page: editPage,
      ownerPage,
      region: region.idx,
      bbox: cleanBbox,
      beforeStyle: regionStyleDebug(region),
    });
    renderDebug("api.updateRegionBBox", { page: editPage, ownerPage, region: region.idx, bbox: cleanBbox });
    suppressBackendPageSyncRef.current = true;
    let resp: any = { ok: false };
    try {
      resp = await api.updateRegionBBox(region.idx, cleanBbox.x, cleanBbox.y, cleanBbox.w, cleanBbox.h, ownerPage, editPage);
    } finally {
      suppressBackendPageSyncRef.current = false;
    }
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      showToast("Region updated; cleanup/typeset may need refresh");
      return true;
    }
    setRegionDrafts(prev => {
      const next = { ...prev };
      delete next[region.id];
      return next;
    });
    return false;
  }, [activePage, data.regions]);

  useEffect(() => {
    const isTypingTarget = (target: EventTarget | null) => {
      const el = target as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName?.toLowerCase();
      return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
    };
    const onKeyDown = (ev: KeyboardEvent) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(ev.key)) return;
      if (isTypingTarget(ev.target) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (!selectedRegion || selectedRegion.locked) return;
      const displayPage = typeof selectedRegion.display_page_idx === "number" ? selectedRegion.display_page_idx : activePage;
      const pageSize = pageSizes[displayPage];
      if (!pageSize) return;
      ev.preventDefault();
      const step = ev.shiftKey ? 10 : 1;
      const dx = ev.key === "ArrowLeft" ? -step : ev.key === "ArrowRight" ? step : 0;
      const dy = ev.key === "ArrowUp" ? -step : ev.key === "ArrowDown" ? step : 0;
      const draft = regionDrafts[selectedRegion.id] ?? {};
      const current = { ...selectedRegion, ...draft } as Region;
      const next = clampBboxToSize({ x: current.x + dx, y: current.y + dy, w: current.w, h: current.h }, pageSize);
      handlePreviewRegion(selectedRegion.id, next, { requestSprite: false, reason: "keyboard_nudge" });
      void handleCommitBBox(current, next);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activePage, handleCommitBBox, handlePreviewRegion, pageSizes, regionDrafts, selectedRegion]);

  const handleAddRegion = useCallback(async () => {
    const resp = await api.addRegion(24, 24, 160, 80, "", activePage);
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      showToast("Region added");
    }
  }, [activePage]);

  const runForRegionOwner = useCallback(async <T,>(region: Region, fn: (pageIdx: number) => Promise<T>): Promise<T> => {
    const ownerPage = regionOwnerPage(region, activePage);
    suppressBackendPageSyncRef.current = true;
    try {
      return await fn(ownerPage);
    } finally {
      suppressBackendPageSyncRef.current = false;
    }
  }, [activePage]);

  const handleDeleteRegion = useCallback(async (idx: number, yoloRejectReason = "") => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.deleteRegion(idx, yoloRejectReason, pageIdx))
      : await api.deleteRegion(idx, yoloRejectReason, activePage);
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      showToast(yoloRejectReason ? "Region deleted and saved as YOLO correction" : "Region deleted");
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handleSetYoloTrainClass = useCallback(async (region: Region, classId: number) => {
    const resp = await runForRegionOwner(region, pageIdx => api.setYoloTrainClass(region.idx, classId, pageIdx));
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      const label = ["dialogue", "caption", "sfx", "shout"][classId] ?? String(classId);
      showToast(`YOLO positive label saved: ${label}`);
    } else {
      showToast(`YOLO label failed: ${resp.error ?? "unknown error"}`);
    }
  }, [activePage, runForRegionOwner]);

  const handleOcrRegion = useCallback(async (idx: number) => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.ocrRegion(idx, pageIdx))
      : await api.ocrRegion(idx, activePage);
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      showToast("OCR updated");
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handleTranslateRegion = useCallback(async (idx: number) => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.translateRegion(idx, pageIdx))
      : await api.translateRegion(idx, activePage);
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      showToast("Translation saved");
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handlePreviewCleanup = useCallback(async (region: Region, manualMask?: CleanupMaskPayload) => {
    const resp = await runForRegionOwner(region, pageIdx => api.previewRegionCleanup(region.idx, manualMask ?? null, pageIdx));
    if (resp.ok) {
      setCleanupPreview({ ...resp, regionId: region.id });
      showToast("Cleanup preview ready");
      return resp;
    } else {
      showToast(`Preview failed: ${resp.error ?? "unknown error"}`);
    }
    return null;
  }, [runForRegionOwner]);

  const handleSuggestSam2Mask = useCallback(async (region: Region, prompt: Record<string, unknown>) => {
    const resp = await runForRegionOwner(region, pageIdx => api.proposeCleanupMaskSam2(region.idx, prompt, pageIdx));
    if (resp.ok) {
      showToast("SAM2 mask suggested");
      return resp as Sam2MaskResponse;
    }
    showToast(`SAM2 unavailable: ${resp.error ?? resp.status ?? "unknown error"}`);
    return resp as Sam2MaskResponse;
  }, [runForRegionOwner]);

  const handleRefreshCleanupDebug = useCallback(async (region: Region, manualMask?: CleanupMaskPayload) => {
    const resp = await runForRegionOwner(region, pageIdx => api.getRegionCleanupDebug(region.idx, manualMask ?? null, pageIdx));
    if (resp.ok) {
      setCleanupDebug({ ...resp, regionId: region.id });
      return resp;
    }
    showToast(`Debug overlays failed: ${resp.error ?? "unknown error"}`);
    return null;
  }, [runForRegionOwner]);

  const handleRecordMaskQaLabel = useCallback(async (region: Region, label: string) => {
    const resp = await runForRegionOwner(region, pageIdx => api.recordMaskQaLabel(region.idx, label, "", pageIdx));
    if (resp.ok) {
      showToast(`Mask QA label saved: ${resp.label ?? label}`);
    } else {
      showToast(`Mask QA label failed: ${resp.error ?? "unknown error"}`);
    }
  }, [runForRegionOwner]);

  const handleTrainMaskQaModel = useCallback(async () => {
    const resp = await api.trainMaskQaModel();
    if (resp.ok) {
      showToast(`Mask QA trained (${resp.records ?? 0} labels)`);
      setLiveStatus(`Mask QA model: ${resp.model_path ?? ""}`);
    } else {
      setLiveStatus(`Mask QA training failed: ${resp.error ?? "unknown error"}`);
      showToast("Mask QA training failed");
    }
  }, []);

  const handleCompareCleanupCandidates = useCallback(async (region: Region, manualMask?: CleanupMaskPayload) => {
    setCleanupCompareLoading(true);
    try {
      const resp = await runForRegionOwner(region, pageIdx => api.compareRegionCleanupCandidates(region.idx, manualMask ?? null, pageIdx));
      if (resp.ok) {
        setCleanupCandidates({ ...resp, regionId: region.id });
        const first = (resp.candidates ?? []).find(c => c.candidate_id === resp.recommended_candidate_id && c.is_available)
          ?? (resp.candidates ?? []).find(c => c.is_available);
        setSelectedCandidateId(first?.candidate_id ?? "");
        showToast("Cleanup candidates ready");
        return resp;
      }
      showToast(`Compare failed: ${resp.error ?? "unknown error"}`);
      return null;
    } finally {
      setCleanupCompareLoading(false);
    }
  }, [runForRegionOwner]);

  const handleDebugOverlayChange = useCallback((key: DebugOverlayKey, value: boolean) => {
    setDebugOverlays(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleUseCleanupCandidatePreview = useCallback((region: Region, candidate: CleanupCandidate) => {
    if (!candidate.b64 || !candidate.bbox) return;
    setCleanupPreview({
      ok: true,
      regionId: region.id,
      b64: candidate.b64,
      bbox: candidate.bbox,
      mask_b64: null,
      mask_bbox: candidate.bbox,
      plan: {
        cleanup_strategy: candidate.strategy ?? "",
        inpaint_method: candidate.method ?? "",
        candidate_id: candidate.candidate_id,
      },
      debug: { scores: candidate.scores ?? {}, warnings: candidate.warnings ?? [] },
    });
    setSelectedCandidateId(candidate.candidate_id);
    showToast("Candidate loaded as preview");
  }, []);

  const handleApplyCleanupCandidate = useCallback(async (idx: number, candidateId: string, manualMask?: CleanupMaskPayload) => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.applyRegionCleanupCandidate(idx, candidateId, manualMask ?? null, pageIdx))
      : await api.applyRegionCleanupCandidate(idx, candidateId, manualMask ?? null, activePage);
    if (resp.ok) {
      setCleanupPreview(null);
      setCleanupDebug(null);
      setCleanupCandidates(null);
      setSelectedCandidateId("");
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      bumpPageImageVersions([activePageRef.current]);
      showToast("Cleanup candidate applied");
    } else {
      showToast(`Candidate apply failed: ${resp.error ?? "unknown error"}`);
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handleApplyCleanup = useCallback(async (idx: number, manualMask?: CleanupMaskPayload) => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.applyRegionCleanup(idx, manualMask ?? null, pageIdx))
      : await api.applyRegionCleanup(idx, manualMask ?? null, activePage);
    if (resp.ok) {
      setCleanupPreview(null);
      setCleanupDebug(null);
      setCleanupCandidates(null);
      setSelectedCandidateId("");
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      bumpPageImageVersions([activePageRef.current]);
      showToast("Cleanup applied");
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handleRerunCleanup = useCallback(async (idx: number, manualMask?: CleanupMaskPayload) => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.rerunRegionCleanup(idx, manualMask ?? null, pageIdx))
      : await api.rerunRegionCleanup(idx, manualMask ?? null, activePage);
    if (resp.ok) {
      setCleanupPreview(null);
      setCleanupDebug(null);
      setCleanupCandidates(null);
      setSelectedCandidateId("");
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      bumpPageImageVersions([activePageRef.current]);
      showToast("Cleanup rerun");
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handleDeleteCleanup = useCallback(async (idx: number) => {
    const region = selectedRegion?.idx === idx ? selectedRegion : null;
    const resp = region
      ? await runForRegionOwner(region, pageIdx => api.deleteRegionCleanup(idx, pageIdx))
      : await api.deleteRegionCleanup(idx, activePage);
    if (resp.ok) {
      setCleanupPreview(null);
      setCleanupDebug(null);
      setCleanupCandidates(null);
      setSelectedCandidateId("");
      applyBootstrap(resp as unknown as Bootstrap);
      setActivePage(activePage);
      bumpPageImageVersions([activePageRef.current]);
      showToast("Cleanup patch removed");
    }
  }, [activePage, runForRegionOwner, selectedRegion]);

  const handleUndo = useCallback(async () => {
    const resp = await api.undo();
    if (resp.ok) {
      applyBootstrap(resp as unknown as Bootstrap);
      showToast("Undo applied");
    }
  }, []);

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = document.activeElement?.tagName;
      const editingText = tag === "TEXTAREA" || tag === "INPUT";
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        if (editingText) return;
        e.preventDefault();
        handleUndo();
        return;
      }
      if (editingText) return; // don't capture when editing
      if (e.key === "Escape") setSelectedRegion(null);
      if (e.key === "ArrowLeft" && activePage > 0) {
        e.preventDefault();
        handlePageSelect(activePage - 1);
      }
      if (e.key === "ArrowRight" && activePage < data.pages.length - 1) {
        e.preventDefault();
        handlePageSelect(activePage + 1);
      }
      if ((e.key === "r" || e.key === "R") && !e.ctrlKey) handleRunAll();
      if ((e.key === "+" || e.key === "=") && !e.ctrlKey) setZoom(z => Math.min(z + 15, 300));
      if (e.key === "-" && !e.ctrlKey) setZoom(z => Math.max(z - 15, 20));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activePage, data.pages.length, handleRunAll, handleUndo]);

  // ── Render ─────────────────────────────────────────────────────────────────
  const canRunPipeline = Boolean(data.meta.chapterDir && data.pages.length > 0);
  const runAllCheckpoint = data.meta.runAllCheckpoint;
  const canContinueRunAll = Boolean(runAllCheckpoint?.active);
  const continueTitle = runAllCheckpoint?.message
    || `Continue Run All from ${runAllCheckpoint?.phase ?? "saved"} page ${((runAllCheckpoint?.page_idx ?? 0) + 1)} / ${runAllCheckpoint?.page_total ?? data.pages.length}`;
  const visiblePageData: Bootstrap = { ...data, regions: editorRegionsForPage(data, activePage) };

  return (
    <>
      <TopBar
        pipeState={pipeState}
        onStep={handleStep}
        onRunAll={handleRunAll}
        onContinueRunAll={handleContinueRunAll}
        onRunPage={handleRunPage}
        onDetectAll={handleDetectAll}
        onExportYoloDataset={handleExportYoloDataset}
        onTrainYolo={handleTrainYolo}
        onUndo={handleUndo}
        onExport={handleExport}
        onOpenChapter={handleOpenChapter}
        onBrowse={() => setShowBrowse(true)}
        onSettings={() => setSettingsOpen(true)}
        leftOpen={leftOpen}
        rightOpen={rightOpen}
        onToggleLeft={() => setLeftOpen(o => !o)}
        onToggleRight={() => setRightOpen(o => !o)}
        busy={pipeRunning}
        canRun={canRunPipeline}
        canContinueRunAll={canContinueRunAll}
        continueTitle={continueTitle}
      />

      <div className="ml-body">
        <div
          className={`ml-left ${leftOpen ? "" : "collapsed"}`}
          style={{ width: leftOpen ? leftWidth : 0 }}
        >
          <LeftPanel
            data={data}
            activeSeries={activeSeries}
            setActiveSeries={setActiveSeries}
            activeChapter={activeChapter}
            setActiveChapter={setActiveChapter}
            activePage={activePage}
            onPageSelect={handlePageSelect}
            onSeriesSelect={setSelectedSeries}
            selectedSeriesTitle={selectedSeriesTitle}
            onBrowseSource={() => setShowBrowse(true)}
            onOpenFolder={handleOpenFolderFromDetail}
            onBootstrap={applyBootstrap}
            onSourceChange={refreshBootstrap}
            onDeleteSeries={handleDeleteSeries}
            onCoverPreview={(src, title, source) => setCoverPreview({ src, title, source })}
            detailRefreshKey={seriesDetailRefreshKey}
            pageVersions={pageVersions}
            pipelineProgress={pipelineProgress}
          />
          {leftOpen && (
            <div
              className="left-resize-handle"
              onMouseDown={(e) => {
                e.preventDefault();
                setResizingLeft(true);
              }}
              title="Resize sidebar"
            />
          )}
        </div>

        <CanvasArea
          data={data}
          activePage={activePage}
          selectedRegion={selectedRegion}
          setSelectedRegion={handleSelectRegion}
          zoom={zoom}
          setZoom={setZoom}
          regionDrafts={regionDrafts}
          onPreviewRegion={handlePreviewRegion}
          onCommitBBox={handleCommitBBox}
          showEnglishOverlay={showEnglishOverlay}
          setShowEnglishOverlay={setShowEnglishOverlay}
          readerMode={readerMode}
          setReaderMode={setReaderMode}
          showPageIndicator={showPageIndicator}
          setShowPageIndicator={setShowPageIndicator}
          scrollTarget={continuousScrollTarget}
          onPageSelect={handlePageSelect}
          imageMode={canvasImageMode}
          setImageMode={setCanvasImageMode}
          pageVersions={pageVersions}
          cleanupDebug={cleanupDebug?.regionId === selectedRegion?.id ? cleanupDebug : null}
          debugOverlays={debugOverlays}
          onPageSizeChange={handlePageSizeChange}
        />

        <div
          className={`ml-right ${rightOpen ? "" : "collapsed"}`}
          style={{ width: rightOpen ? rightWidth : 0 }}
        >
          <RightPanel
            data={visiblePageData}
            region={selectedRegion}
            issues={data.issues}
            pageIndex={activePage}
            pageSize={pageSizes[activePage] ?? null}
            fontOptions={fontOptions}
            onSelectRegion={handleSelectRegion}
            onUpdateRegion={handleUpdateRegion}
            onPreviewRegion={handlePreviewRegion}
            onCommitBBox={handleCommitBBox}
            onAddRegion={handleAddRegion}
            onDeleteRegion={handleDeleteRegion}
            onSetYoloTrainClass={handleSetYoloTrainClass}
            onOcrRegion={handleOcrRegion}
            onTranslateRegion={handleTranslateRegion}
            onMemoryMutation={applyBootstrap}
            cleanupPreview={cleanupPreview}
            cleanupDebug={cleanupDebug?.regionId === selectedRegion?.id ? cleanupDebug : null}
            debugOverlays={debugOverlays}
            onDebugOverlayChange={handleDebugOverlayChange}
            cleanupCandidates={cleanupCandidates?.regionId === selectedRegion?.id ? cleanupCandidates : null}
            selectedCandidateId={selectedCandidateId}
            onSelectCleanupCandidate={setSelectedCandidateId}
            onPreviewCleanup={handlePreviewCleanup}
            onApplyCleanup={handleApplyCleanup}
            onRerunCleanup={handleRerunCleanup}
            onDeleteCleanup={handleDeleteCleanup}
            onSuggestSam2Mask={handleSuggestSam2Mask}
            onRefreshCleanupDebug={handleRefreshCleanupDebug}
            onRecordMaskQaLabel={handleRecordMaskQaLabel}
            onTrainMaskQaModel={handleTrainMaskQaModel}
            onCompareCleanupCandidates={handleCompareCleanupCandidates}
            onApplyCleanupCandidate={handleApplyCleanupCandidate}
            onUseCleanupCandidatePreview={handleUseCleanupCandidatePreview}
            cleanupCompareLoading={cleanupCompareLoading}
          />
          {rightOpen && (
            <div
              className="right-resize-handle"
              onMouseDown={(e) => {
                e.preventDefault();
                setResizingRight(true);
              }}
              title="Resize inspector"
            />
          )}
        </div>
      </div>

      <StatusBar
        data={visiblePageData}
        activePage={activePage}
        selectedRegion={selectedRegion}
        pipeRunning={pipeRunning}
        liveStatus={liveStatus}
        pipelineProgress={pipelineProgress}
      />
      {toast && <div className="toast">{toast}</div>}
      {settingsOpen && (
        <SettingsModal
          config={modelConfig}
          onChange={setModelConfig}
          onSave={handleSaveSettings}
          onClose={() => setSettingsOpen(false)}
        />
      )}
      {showBrowse && (
        <BrowseModal
          sources={availableSources}
          onSelect={handleBrowseSelect}
          onClose={() => setShowBrowse(false)}
        />
      )}
      {coverPreview && (
        <CoverLightbox
          src={coverPreview.src}
          title={coverPreview.title}
          source={coverPreview.source}
          onClose={() => setCoverPreview(null)}
        />
      )}
    </>
  );
}
