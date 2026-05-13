/**
 * frontend/vite.config.ts
 * ────────────────────────
 * Vite config for the Manhwa Localizer frontend.
 *
 * Key decisions:
 *  - base: "./"   → relative asset paths so the built index.html works when
 *                    pywebview loads it via http_server (file:// equivalent).
 *  - outDir: "dist" → launcher.py looks for frontend/dist/index.html.
 *  - server.port: 5173  → matches the default in launcher.py --dev.
 *  - No proxy needed: pywebview JS-bridge calls go through window.pywebview,
 *    not HTTP, so there's nothing to proxy.
 */

import { defineConfig } from "vite";
import react            from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  // Relative base so assets resolve correctly when served by pywebview's
  // built-in http_server from the dist/ directory.
  base: "./",

  build: {
    outDir:        "dist",
    emptyOutDir:   true,
    sourcemap:     false,
    // Inline small assets so pywebview's http_server doesn't need extra
    // MIME-type handling for fonts / tiny SVGs.
    assetsInlineLimit: 8192,
  },

  server: {
    port:        5173,
    strictPort:  true,   // fail fast if the port is taken
    // No cors / proxy needed — pywebview injects window.pywebview directly.
  },

  // Tell Vite that .ts / .tsx files should use the React JSX transform.
  esbuild: {
    jsxInject: undefined,   // we rely on the automatic JSX transform
  },
});
