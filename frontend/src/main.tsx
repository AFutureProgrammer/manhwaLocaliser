/**
 * frontend/src/main.tsx
 * ─────────────────────
 * React 18 entry point — mounts <App /> into the #root div.
 *
 * StrictMode is intentionally left ON so that double-invocation of effects
 * surfaces any non-idempotent side-effects early.  Remove it only if
 * specific third-party libs break under strict mode.
 */

import { StrictMode } from "react";
import { createRoot }  from "react-dom/client";
import App             from "./App";

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("[main] #root element not found — check index.html");
}

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>
);
