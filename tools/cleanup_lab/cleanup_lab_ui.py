from __future__ import annotations

import json
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.cleanup_plan import (
    CleanupPolicy,
    build_cleanup_plan,
    execute_cleanup_plan,
    normalize_mask_to_image,
    validate_cleanup_proposal,
)
from backend.core.config import ModelConfig
from backend.core.regions import OCRBlock, RegionKind, RegionOverride


def _bbox(value: Any, default: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return default
    try:
        x, y, w, h = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return default
    if w <= 0 or h <= 0:
        return default
    return (x, y, w, h)


def _kind(raw: str) -> Optional[RegionKind]:
    name = str(raw).strip().upper()
    aliases = {
        "PLAIN": "PLAIN_BUBBLE", "PLAIN_BUBBLE": "PLAIN_BUBBLE",
        "TEXTURED": "TEXTURED_BUBBLE", "TEXTURED_BUBBLE": "TEXTURED_BUBBLE",
        "GRADIENT": "GRADIENT_BUBBLE", "GRADIENT_BUBBLE": "GRADIENT_BUBBLE",
        "CAPTION": "CAPTION_BOX", "CAPTION_BOX": "CAPTION_BOX",
        "SFX": "SFX_OVER_ART", "SFX_OVER_ART": "SFX_OVER_ART",
        "TEXT_ON_ART": "DIALOGUE_OVER_ART", "DIALOGUE_OVER_ART": "DIALOGUE_OVER_ART",
        "UNKNOWN": "UNKNOWN",
    }
    try:
        return RegionKind[aliases.get(name, name)]
    except KeyError:
        return None


def _coerce_boxes(region: Dict[str, Any]) -> List[Any]:
    for key in ("boxes", "ocr_boxes", "text_boxes"):
        boxes = region.get(key)
        if isinstance(boxes, list) and boxes:
            return boxes
    return []


def _make_block(region: Dict[str, Any]) -> OCRBlock:
    bbox = _bbox(region.get("bbox") or region.get("region_bbox"), (0, 0, 1, 1))
    assert bbox is not None
    block = OCRBlock(
        text=str(region.get("text") or ""),
        boxes=_coerce_boxes(region),
        confidence=float(region.get("detector_confidence", region.get("confidence", 0.0)) or 0.0),
    )
    block.bbox_override = bbox
    block.detector_source = str(region.get("detector") or region.get("detector_source") or "yolo")
    block.bubble_role = str(region.get("role") or region.get("bubble_role") or "dialog")
    rk = _kind(str(region.get("kind") or region.get("region_kind") or ""))
    if rk is not None:
        block.region_kind = rk
    yk = str(region.get("yolo_kind") or region.get("yolo_class") or "")
    if yk:
        setattr(block, "yolo_kind", yk)
    yci = region.get("yolo_class_id")
    if yci is not None:
        try:
            setattr(block, "yolo_class_id", int(yci))
        except Exception:
            pass
    for attr in ("bubble_bbox", "safe_rect", "cleanup_safe_rect", "cleanup_container_bbox", "detector_text_bbox"):
        b = _bbox(region.get(attr))
        if b is not None:
            setattr(block, attr, b)
    if region.get("cleanup_container_confidence") is not None:
        block.cleanup_container_confidence = float(region.get("cleanup_container_confidence") or 0.0)
    if region.get("cleanup_safe_rect_confidence") is not None:
        block.cleanup_safe_rect_confidence = float(region.get("cleanup_safe_rect_confidence") or 0.0)
    if block.bubble_bbox is not None and bool(region.get("assume_rect_bubble_mask", True)):
        bx, by, bw, bh = block.bubble_bbox
        block.bubble_mask = np.full((max(1, bh), max(1, bw)), 255, dtype=np.uint8)
    override_data = region.get("override")
    if isinstance(override_data, dict):
        block.override = RegionOverride.from_dict(override_data)
    rc = region.get("cleanup_region_class")
    if rc:
        if block.override is None:
            block.override = RegionOverride()
        block.override.cleanup_region_class = str(rc)
    return block


class CleanupLabUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Cleanup Lab")
        self.root.geometry("1300x860")

        self.image_path: Optional[Path] = None
        self.fixture_path: Optional[Path] = None
        self.raw_image: Optional[np.ndarray] = None
        self.regions: List[Tuple[str, OCRBlock, Dict[str, Any]]] = []
        self.plan: Any = None
        self.cleaned: Optional[np.ndarray] = None
        self._container_full: Optional[np.ndarray] = None
        self._show_overlay = tk.BooleanVar(value=True)
        self._running = False

        self._build_ui()
        self.root.mainloop()

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Image:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self._img_entry = ttk.Entry(top)
        self._img_entry.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(top, text="Browse", command=self._browse_image).grid(row=0, column=2, padx=(4, 0))

        ttk.Label(top, text="Fixture:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4))
        self._fix_entry = ttk.Entry(top)
        self._fix_entry.grid(row=1, column=1, sticky=tk.EW, padx=4)
        ttk.Button(top, text="Browse", command=self._browse_fixture).grid(row=1, column=2, padx=(4, 0))

        top.columnconfigure(1, weight=1)

        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))

        left = ttk.Frame(panes, width=260)
        panes.add(left, weight=0)

        ttk.Label(left, text="Regions", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 2))
        self._region_list = tk.Listbox(left, height=10, exportselection=False)
        self._region_list.pack(fill=tk.BOTH, expand=True)
        self._region_list.bind("<<ListboxSelect>>", self._on_region_select)

        self._run_btn = ttk.Button(left, text="Run Cleanup", command=self._run_cleanup, state=tk.DISABLED)
        self._run_btn.pack(fill=tk.X, pady=(6, 2))

        ovf = ttk.Frame(left)
        ovf.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(ovf, text="Show overlay", variable=self._show_overlay, command=self._refresh_display).pack(side=tk.LEFT)

        self._info_text = tk.Text(left, height=18, width=34, font=("Consolas", 9), state=tk.DISABLED, wrap=tk.NONE)
        self._info_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        right = ttk.Frame(panes)
        panes.add(right, weight=1)

        self._before_canvas = tk.Canvas(right, bg="#222", highlightthickness=0)
        self._after_canvas = tk.Canvas(right, bg="#222", highlightthickness=0)

        self._before_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        self._after_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        self._status_var = tk.StringVar(value="Load an image and fixture to begin.")
        status = ttk.Label(self.root, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.bind("<Control-o>", lambda _: self._browse_image())
        self.root.bind("<Control-f>", lambda _: self._browse_fixture())
        self.root.bind("<F5>", lambda _: self._run_cleanup())

    # ── File Browsing ──────────────────────────────────────────────────────

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select page image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff")],
        )
        if not path:
            return
        self.image_path = Path(path)
        self._img_entry.delete(0, tk.END)
        self._img_entry.insert(0, str(self.image_path))
        self._load_image()

    def _browse_fixture(self) -> None:
        path = filedialog.askopenfilename(
            title="Select regions fixture",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        self.fixture_path = Path(path)
        self._fix_entry.delete(0, tk.END)
        self._fix_entry.insert(0, str(self.fixture_path))
        self._load_fixture()

    def _load_image(self) -> None:
        if not self.image_path or not self.image_path.exists():
            return
        data = np.fromfile(str(self.image_path), dtype=np.uint8)
        if data.size == 0:
            return
        self.raw_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self._refresh_display()
        self._update_run_state()

    def _load_fixture(self) -> None:
        if not self.fixture_path or not self.fixture_path.exists():
            return
        with self.fixture_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            regions_data = data.get("regions") or []
            img_ref = data.get("page_image") or data.get("image")
            if img_ref and not self.image_path:
                p = Path(img_ref).expanduser()
                if not p.is_absolute():
                    p = (self.fixture_path.parent / p).resolve()
                if p.exists():
                    self.image_path = p
                    self._img_entry.delete(0, tk.END)
                    self._img_entry.insert(0, str(self.image_path))
                    self._load_image()
        elif isinstance(data, list):
            regions_data = data
        else:
            regions_data = []
        self.regions = []
        self._region_list.delete(0, tk.END)
        for idx, r in enumerate(regions_data):
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id") or r.get("region_id") or f"R-{idx + 1:02d}")
            block = _make_block(r)
            self.regions.append((rid, block, r))
            label = f"{rid}  {r.get('role',''):>8}  {r.get('kind','') or r.get('region_kind','') or '?'}"
            self._region_list.insert(tk.END, label)
        self._update_run_state()

    # ── Region Selection ───────────────────────────────────────────────────

    def _on_region_select(self, _event: Any = None) -> None:
        sel = self._region_list.curselection()
        if not sel or not self.regions:
            return
        idx = sel[0]
        _rid, _block, raw = self.regions[idx]
        lines = []
        for k in ("bbox", "region_bbox", "bubble_bbox", "safe_rect", "cleanup_safe_rect",
                  "cleanup_container_bbox", "detector_text_bbox", "role", "kind", "region_kind",
                  "detector", "detector_source", "text"):
            v = raw.get(k)
            if v is not None:
                lines.append(f"{k}: {v}")
        lines.append(f"conf: {raw.get('confidence', raw.get('detector_confidence', '?'))}")
        lines.append(f"container_conf: {raw.get('cleanup_container_confidence', '?')}")
        self._info_text.configure(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        for line in lines:
            self._info_text.insert(tk.END, line + "\n")
        self._info_text.configure(state=tk.DISABLED)

    # ── Run ────────────────────────────────────────────────────────────────

    def _update_run_state(self) -> None:
        ready = self.raw_image is not None and bool(self.regions)
        self._run_btn.configure(state=tk.NORMAL if ready else tk.DISABLED)

    def _run_cleanup(self) -> None:
        if self._running:
            return
        sel = self._region_list.curselection()
        if not sel:
            self._status_var.set("Select a region first.")
            return
        idx = sel[0]
        region_id, block, _raw = self.regions[idx]
        self._status_var.set(f"Running cleanup on {region_id}...")
        self._running = True
        self._run_btn.configure(state=tk.DISABLED)
        Thread(target=self._do_cleanup, args=(region_id, block), daemon=True).start()

    def _do_cleanup(self, region_id: str, block: OCRBlock) -> None:
        try:
            page_index = 0
            cfg = ModelConfig()
            policy = CleanupPolicy.from_config(cfg)
            self.plan = build_cleanup_plan(
                self.raw_image,
                block,
                page_index=page_index,
                region_id=region_id,
                cleanup_debug_artifacts=False,
                cleanup_policy=policy,
                model_config=cfg,
            )
            self.plan.cleanup_backend = "opencv"
            self.cleaned = self.raw_image.copy()
            execute_cleanup_plan(self.raw_image, self.cleaned, self.plan)
            validate_cleanup_proposal(
                self.raw_image,
                self.cleaned,
                self.plan,
                destructive_allowed=True,
                production_patch_accepted=False,
                validation_source="cleanup_lab_ui",
            )
            if self.plan.container_mask is not None and self.plan.container_bbox is not None:
                self._container_full = normalize_mask_to_image(
                    self.plan.container_mask, self.plan.container_bbox, self.raw_image.shape
                )
            else:
                self._container_full = None
            self.root.after(0, self._on_cleanup_done)
        except Exception as exc:
            self.root.after(0, lambda: self._on_cleanup_error(str(exc)))

    def _on_cleanup_done(self) -> None:
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._refresh_display()
        self._show_plan_metrics()
        self._status_var.set(
            f"Done — {self.plan.cleanup_strategy}/{self.plan.inpaint_method} "
            f"bg={self.plan.background_model}"
        )

    def _on_cleanup_error(self, msg: str) -> None:
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_var.set(f"Error: {msg}")

    # ── Display ────────────────────────────────────────────────────────────

    def _refresh_display(self) -> None:
        for canvas, arr in ((self._before_canvas, self.raw_image), (self._after_canvas, self.cleaned)):
            self._show_on_canvas(canvas, arr, overlay=(canvas is self._after_canvas))

    def _show_on_canvas(self, canvas: tk.Canvas, arr: Optional[np.ndarray], overlay: bool = False) -> None:
        canvas.delete("all")
        if arr is None:
            canvas.create_text(canvas.winfo_width() // 2 or 200, canvas.winfo_height() // 2 or 200,
                               text="No image", fill="#666", font=("", 14))
            return
        if overlay and self._show_overlay.get() and self.plan is not None:
            disp = self._build_overlay(arr)
        else:
            disp = arr
        h, w = disp.shape[:2]
        cw = max(canvas.winfo_width(), 50)
        ch = max(canvas.winfo_height(), 50)
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.image = tk_img
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)

    def _build_overlay(self, cleaned: np.ndarray) -> np.ndarray:
        out = cleaned.copy()
        plan = self.plan
        layers: List[Tuple[Optional[np.ndarray], Tuple[int, int, int], float]] = [
            (plan.text_mask, (0, 255, 0), 0.45),
            (plan.cleanup_mask, (0, 0, 255), 0.3),
            (self._container_full, (80, 80, 255), 0.15),
        ]
        for mask, color, alpha in layers:
            if mask is None or not np.any(mask):
                continue
            active = mask > 0
            c = np.array(color, dtype=np.float32)
            out[active] = (out[active].astype(np.float32) * (1.0 - alpha) + c * alpha).clip(0, 255).astype(np.uint8)
        return out

    def _show_plan_metrics(self) -> None:
        plan = self.plan
        q = (plan.debug_metrics.get("quality") or {}) if hasattr(plan, "debug_metrics") else {}
        lines = [
            f"Strategy:    {plan.cleanup_strategy}",
            f"Method:      {plan.inpaint_method}",
            f"Backend:     {plan.cleanup_backend}",
            f"Bg model:    {plan.background_model}",
            f"Skip reason: {plan.skip_reason or '-'}",
            f"Text conf:   {plan.text_mask_confidence:.3f}",
            f"Cont conf:   {plan.container_confidence:.3f}",
            f"Mask conf:   {plan.cleanup_mask_confidence:.3f}",
            f"Region bbox: {list(plan.region_bbox) if plan.region_bbox else '-'}",
            f"Text bbox:   {list(plan.text_bbox) if plan.text_bbox else '-'}",
            f"Container:   {list(plan.container_bbox) if plan.container_bbox else '-'}",
            f"Region cls:  {plan.region_class}",
            f"Text src:    {plan.text_mask_reason or '-'}",
            f"",
            f"Mask reg ratio:  {q.get('mask_region_ratio', 0):.4f}",
            f"Mask cont ratio: {q.get('mask_container_ratio', 0):.4f}",
            f"Rectangularity:  {q.get('rectangularity', 0):.4f}",
            f"Border touch:    {q.get('border_touch_ratio', 0):.4f}",
            f"Components:      {q.get('component_count', 0)}",
            f"Effective:       {getattr(plan, 'cleanup_effective', '?')}",
        ]
        self._info_text.configure(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        for line in lines:
            self._info_text.insert(tk.END, line + "\n")
        self._info_text.configure(state=tk.DISABLED)


if __name__ == "__main__":
    CleanupLabUI()
