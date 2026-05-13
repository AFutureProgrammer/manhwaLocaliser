"""
launcher.py
───────────
Entry point for the Manhwa Localizer desktop app.

Usage:
    python launcher.py              # production — serves pre-built frontend dist/
    python launcher.py --dev        # dev mode — proxies to Vite on localhost:5173
    python launcher.py --dev --port 5174

What this file does:
  1. Parses CLI args.
  2. In production: locates the compiled frontend/dist/index.html.
  3. Instantiates LocalizerEngine (model loading starts in a background thread).
  4. Instantiates PywebviewAPI (the JS bridge object).
  5. Creates the pywebview window.
  6. Wires the window back into the API (needed for push events).
  7. Starts the pywebview event loop (blocks until the window is closed).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# ── Make sure the project root is on sys.path so that
#    `from backend.engine import ...` and `from translator_v14 import ...` work
#    regardless of the working directory the user launches from.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import webview  # pywebview — must come after sys.path fix

from backend.api    import PywebviewAPI
from backend.engine import LocalizerEngine


# ── Constants ────────────────────────────────────────────────────────────────

APP_TITLE   = "Manhwa Localizer"
WINDOW_W    = 1440
WINDOW_H    = 900
MIN_W       = 1024
MIN_H       = 600
VITE_PORT   = 5173                          # default Vite dev-server port
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist" / "index.html"


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_TITLE)
    p.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: proxy UI to the Vite dev server instead of dist/",
    )
    p.add_argument(
        "--port",
        type=int,
        default=VITE_PORT,
        metavar="PORT",
        help=f"Vite dev-server port (default: {VITE_PORT})",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable pywebview debug mode (opens DevTools on start)",
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve UI source ────────────────────────────────────────────────────
    if args.dev:
        url = f"http://localhost:{args.port}"
        print(f"[launcher] Dev mode — pointing at {url}")
        print(f"[launcher] Make sure to run:  cd frontend && npm run dev")
    else:
        if not FRONTEND_DIST.exists():
            print(
                f"[launcher] ERROR: built frontend not found at {FRONTEND_DIST}\n"
                f"           Run:  cd frontend && npm install && npm run build",
                file=sys.stderr,
            )
            sys.exit(1)
        url = FRONTEND_DIST.as_uri()
        print(f"[launcher] Production mode — loading {FRONTEND_DIST}")

    # ── Create engine + API ──────────────────────────────────────────────────
    engine = LocalizerEngine()
    js_api = PywebviewAPI(engine)

    # ── Create window ────────────────────────────────────────────────────────
    window = webview.create_window(
        title         = APP_TITLE,
        url           = url,
        js_api        = js_api,
        width         = WINDOW_W,
        height        = WINDOW_H,
        min_size      = (MIN_W, MIN_H),
        resizable     = True,
        text_select   = False,
        # Allow file:// → api calls in production builds.
        # In dev mode the Vite dev-server is on http:// so this is harmless.
    )

    # Give the API a reference to the window so it can push events.
    js_api.set_window(window)

    # ── Register close callback ──────────────────────────────────────────────
    window.events.closed += js_api.on_window_closed

    # ── Start event loop (blocks until window is closed) ────────────────────
    webview.start(
        debug    = args.debug,
        # http_server=True is required when url is a file:// URI so that
        # pywebview serves assets correctly on all platforms.
        http_server = not args.dev,
    )


if __name__ == "__main__":
    main()
