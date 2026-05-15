from __future__ import annotations

import base64
import os
import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import torch as _torch  # type: ignore[import]
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if hasattr(os, "add_dll_directory") and os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
except Exception:  # pragma: no cover - optional runtime
    pass

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional runtime
    ort = None

from backend.core.constants import SFX_MAP, debug_print
from backend.core.regions import OCRBlock

def image_to_base64(path: str) -> str:
    debug_print(f"Encoding image to base64: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def group_ocr_blocks(
    blocks: List["OCRBlock"],
    # FIX-1: tightened from (40, 60) to (18, 25).
    # Old values were loose enough to merge text from *adjacent* stacked
    # balloons into one logical block, which then got a single translation
    # rendered into a mis-sized bubble.  The new values keep lines within
    # the same balloon together (typical inter-line gap ≤ 15 px) while
    # leaving the inter-balloon gap (usually ≥ 30 px) as a clean break.
    y_threshold: int = 18,
    x_threshold: int = 25,
) -> List["OCRBlock"]:
    if not blocks:
        debug_print("group_ocr_blocks: no blocks to merge")
        return []

    debug_print(f"group_ocr_blocks: received {len(blocks)} block(s)")
    blocks = sorted(blocks, key=lambda b: b.bbox()[1])
    used   = [False] * len(blocks)
    result: List[OCRBlock] = []

    for i, anchor in enumerate(blocks):
        if used[i]:
            continue

        current = OCRBlock(
            text=anchor.text,
            boxes=[list(row) for row in anchor.boxes],
            confidence=anchor.confidence,
            bg_color=anchor.bg_color,
            fg_color=anchor.fg_color,
        )
        used[i] = True

        for j in range(i + 1, len(blocks)):
            if used[j]:
                continue

            ax, ay, aw, ah = current.bbox()
            bx, by, bw, bh = blocks[j].bbox()

            # Vertical: gap between bottom of current and top of candidate
            vert_gap = by - (ay + ah)
            vert_close = vert_gap < y_threshold or abs(ay - by) < y_threshold

            # Horizontal: require genuine X overlap, not just proximity.
            # Two blocks "overlap" when neither lies entirely outside the
            # other with the threshold as a tolerance zone on each side.
            horiz_close = not (
                ax + aw + x_threshold < bx or bx + bw + x_threshold < ax
            )

            if vert_close and horiz_close:
                current.merge(blocks[j])
                used[j] = True

        result.append(current)

    debug_print(f"group_ocr_blocks: merged into {len(result)} block(s)")
    return result

def build_mask(
    shape: Tuple[int, int, int],
    bubbles: List["OCRBlock"],
    pad: int = 5,
) -> np.ndarray:
    h_max, w_max = shape[:2]
    mask = np.zeros((h_max, w_max), dtype=np.uint8)
    debug_print(f"build_mask: image_shape={shape}, bubbles={len(bubbles)}, pad={pad}")

    for idx, block in enumerate(bubbles):
        x, y, w, bh = block.bbox()
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_max, x + w + pad)
        y2 = min(h_max, y + bh + pad)
        debug_print(f"build_mask[{idx}]: bbox={(x, y, w, bh)}, masked_area={(x2 - x1) * (y2 - y1)}")
        mask[y1:y2, x1:x2] = 255

    return mask

class OCRProcessor:
    def __init__(self) -> None:
        debug_print("OCRProcessor: initialising EasyOCR (Korean, CPU)…")
        import easyocr
        self._reader = easyocr.Reader(["ko"], gpu=False, verbose=False)
        debug_print("OCRProcessor: EasyOCR ready")

    _MIN_CONFIDENCE        = 0.10   # Detect above this → always erase
    _MIN_TRANSLATE_CONFIDENCE = 0.25  # Translate only above this

    def detect(self, image_path: str) -> "List[OCRBlock]":
        debug_print(f"OCRProcessor.detect: running on {image_path!r}")
        results = self._reader.readtext(
            image_path,
            text_threshold=0.25,
            low_text=0.35,
            canvas_size=3200,
        )
        if not results:
            debug_print("OCRProcessor.detect: no text found")
            return []

        blocks: "List[OCRBlock]" = []
        skipped = 0
        for bbox, text, confidence in results:
            text = str(text).strip()
            if not text:
                skipped += 1
                continue
            if confidence < self._MIN_CONFIDENCE:
                debug_print(f"OCRProcessor.detect: skipped low-conf {text!r} ({confidence:.3f})")
                skipped += 1
                continue
            erase_only = confidence < self._MIN_TRANSLATE_CONFIDENCE
            block = OCRBlock(
                text=text,
                boxes=[bbox],
                confidence=float(confidence),
                erase_only=erase_only,
            )
            # ── Auto-flag: reviewable conditions set at detect time ────────
            if erase_only:
                block.flag("low_confidence", {
                    "conf": round(float(confidence), 3),
                    "threshold": self._MIN_TRANSLATE_CONFIDENCE,
                })
            blocks.append(block)
            tag = " [erase-only]" if erase_only else ""
            debug_print(f"OCRProcessor.detect: {text!r} conf={confidence:.3f} bbox={block.bbox()}{tag}")

        debug_print(f"OCRProcessor.detect: returning {len(blocks)} block(s) ({skipped} skipped)")
        return blocks

    def shutdown(self) -> None:
        pass


class RegionDetector:
    """Small detector interface so OCR and YOLO backends return OCRBlock objects."""

    source = "ocr"

    def detect(self, image_path: str) -> "List[OCRBlock]":
        raise NotImplementedError


class OCRRegionDetector(RegionDetector):
    source = "ocr"

    def __init__(self, ocr_processor: OCRProcessor) -> None:
        self._ocr = ocr_processor

    def detect(self, image_path: str) -> "List[OCRBlock]":
        blocks = self._ocr.detect(image_path)
        for block in blocks:
            block.detector_source = "ocr"
        return blocks


class YoloV8RegionDetector(RegionDetector):
    """
    Optional lightweight detector wrapper.

    Currently supports YOLOv8 ONNX outputs through onnxruntime, with OpenCV
    DNN retained as a fallback for older compatible detector exports.
    Missing models, unsupported formats, or inference errors are allowed to
    raise; the engine catches them and falls back to OCR detection.
    """

    source = "yolo"

    def __init__(self, model_path: str, confidence: float = 0.25, nms_iou: float = 0.45) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path!r}")
        if not model_path.lower().endswith(".onnx"):
            raise ValueError("Only ONNX YOLO detector models are supported by this optional wrapper.")
        self.model_path = model_path
        self.confidence = confidence
        self.nms_iou = float(nms_iou) if nms_iou else 0.45
        self._input_size = 1024
        self._session = None
        self._input_name = ""
        self._net = None
        if ort is not None:
            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            debug_print(f"YoloV8RegionDetector: loaded ONNX with onnxruntime: {model_path}")
        else:
            self._net = cv2.dnn.readNetFromONNX(model_path)
            debug_print(f"YoloV8RegionDetector: loaded ONNX with OpenCV DNN: {model_path}")

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        scale = min(self._input_size / max(1, w), self._input_size / max(1, h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_x = (self._input_size - nw) // 2
        pad_y = (self._input_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _run(self, lb: np.ndarray) -> np.ndarray:
        if self._session is not None:
            inp = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            pred = self._session.run(None, {self._input_name: inp})[0]
        else:
            blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self._input_size, self._input_size), swapRB=True, crop=False)
            self._net.setInput(blob)
            pred = self._net.forward()
        arr = np.squeeze(pred)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
            arr = arr.T
        return arr

    @staticmethod
    def _yolo_box_area(b: List[int]) -> float:
        return float(max(0, b[2]) * max(0, b[3]))

    @staticmethod
    def _yolo_intersection_area(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        return float(inter_w * inter_h)

    @staticmethod
    def _yolo_union_box(a: List[int], b: List[int]) -> List[int]:
        x1 = min(a[0], b[0])
        y1 = min(a[1], b[1])
        x2 = max(a[0] + a[2], b[0] + b[2])
        y2 = max(a[1] + a[3], b[1] + b[3])
        return [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]

    @classmethod
    def _merge_overlapping_yolo_detections(
        cls,
        detections: List[Dict[str, Any]],
        overlap_threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        if len(detections) < 2:
            return detections

        class_group = {0: 0, 1: 0, 2: 1, 3: 1}
        merged = [dict(det) for det in detections]
        changed = True
        while changed:
            changed = False
            next_items: List[Dict[str, Any]] = []
            used = [False] * len(merged)
            for i, det in enumerate(merged):
                if used[i]:
                    continue
                current = dict(det)
                used[i] = True
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    other = merged[j]
                    if class_group.get(int(current["class_id"]), 0) != class_group.get(int(other["class_id"]), 0):
                        continue
                    inter = cls._yolo_intersection_area(current["box"], other["box"])
                    smaller = min(cls._yolo_box_area(current["box"]), cls._yolo_box_area(other["box"]))
                    if smaller <= 0 or inter / smaller < overlap_threshold:
                        continue
                    if float(other["score"]) > float(current["score"]):
                        current["class_id"] = int(other["class_id"])
                    current["score"] = max(float(current["score"]), float(other["score"]))
                    current["box"] = cls._yolo_union_box(current["box"], other["box"])
                    current["merged_count"] = int(current.get("merged_count", 1)) + int(other.get("merged_count", 1))
                    used[j] = True
                    changed = True
                next_items.append(current)
            merged = next_items
        return merged

    def detect(self, image_path: str) -> "List[OCRBlock]":
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {image_path!r}")
        h, w = img.shape[:2]
        lb, scale, pad_x, pad_y = self._letterbox(img)
        arr = self._run(lb)

        boxes: List[List[int]] = []
        scores: List[float] = []
        classes: List[int] = []
        blocks: List[OCRBlock] = []
        raw_total = 0
        after_conf = 0
        for row in arr:
            raw_total += 1
            if len(row) < 5:
                continue
            class_scores = np.asarray(row[4:], dtype=np.float32)
            score = float(np.max(class_scores)) if class_scores.size else float(row[4])
            class_id = int(np.argmax(class_scores)) if class_scores.size else 0
            if score < self.confidence:
                continue
            after_conf += 1
            cx, cy, bw, bh = [float(v) for v in row[:4]]
            if cx <= 1.5 and cy <= 1.5 and bw <= 1.5 and bh <= 1.5:
                cx *= self._input_size
                bw *= self._input_size
                cy *= self._input_size
                bh *= self._input_size
            cx = (cx - pad_x) / max(scale, 1e-6)
            cy = (cy - pad_y) / max(scale, 1e-6)
            bw = bw / max(scale, 1e-6)
            bh = bh / max(scale, 1e-6)
            x = int(round(cx - bw / 2))
            y = int(round(cy - bh / 2))
            ww = int(round(bw))
            hh = int(round(bh))
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            ww = max(1, min(ww, w - x))
            hh = max(1, min(hh, h - y))
            boxes.append([x, y, ww, hh])
            scores.append(score)
            classes.append(class_id)

        keep_ids: List[int] = []
        for class_id in sorted(set(classes)):
            ids = [i for i, value in enumerate(classes) if value == class_id]
            if not ids:
                continue
            keep = cv2.dnn.NMSBoxes([boxes[i] for i in ids], [scores[i] for i in ids], self.confidence, self.nms_iou)
            if len(keep):
                keep_ids.extend(ids[int(i)] for i in np.array(keep).reshape(-1).tolist())

        after_nms = len(keep_ids)

        # ── Containment suppression ───────────────────────────────────────────
        # Pass 7: threshold raised from 0.80 → 0.85 (same-class-group).
        # Cross-class rule: dialogue/narration (group 0) that almost entirely
        # contains an sfx/shout (group 1) box → suppress the sfx box.
        # A dialogue box is NEVER suppressed because an SFX box encloses it.
        # Class ids: 0=dialogue, 1=narration, 2=sfx, 3=shout
        _CLASS_GROUP   = {0: 0, 1: 0, 2: 1, 3: 1}
        _TEXT_CLASS_IDS = {0, 1}   # dialogue / narration
        _SFX_CLASS_IDS  = {2, 3}   # sfx / shout

        suppressed: set = set()
        sorted_by_conf = sorted(keep_ids, key=lambda i: scores[i], reverse=True)
        for ai, a in enumerate(sorted_by_conf):
            if a in suppressed:
                continue
            cls_a = classes[a]
            grp_a = _CLASS_GROUP.get(cls_a, 0)
            for b in sorted_by_conf[ai + 1:]:
                if b in suppressed:
                    continue
                cls_b = classes[b]
                grp_b = _CLASS_GROUP.get(cls_b, 0)

                inter   = self._yolo_intersection_area(boxes[a], boxes[b])
                area_b  = self._yolo_box_area(boxes[b])
                if area_b <= 0:
                    continue
                contained_ratio = inter / area_b

                if grp_a == grp_b:
                    # Same class group — tighter threshold 0.85 (was 0.80).
                    if contained_ratio >= 0.85:
                        suppressed.add(b)
                elif cls_a in _TEXT_CLASS_IDS and cls_b in _SFX_CLASS_IDS:
                    # Cross-class: dialogue/narration nearly containing sfx →
                    # sfx is a sub-region of the dialogue bubble, suppress it.
                    if contained_ratio >= 0.85:
                        debug_print(
                            f"[YOLO_SUPPRESS] cross-class dialogue⊃sfx "
                            f"cls_a={cls_a} cls_b={cls_b} "
                            f"ratio={contained_ratio:.2f}"
                        )
                        suppressed.add(b)
                # sfx cannot suppress dialogue/narration — no action.

        final_ids = [i for i in keep_ids if i not in suppressed]
        after_containment = len(final_ids)
        merged_detections = self._merge_overlapping_yolo_detections([
            {
                "box": boxes[i],
                "score": float(scores[i]),
                "class_id": int(classes[i]),
                "merged_count": 1,
            }
            for i in final_ids
        ])
        after_merge = len(merged_detections)

        debug_print(
            f"[YOLO_FILTER] model={self.model_path!r} "
            f"raw={raw_total} after_conf={after_conf} "
            f"after_nms={after_nms} after_containment={after_containment} "
            f"after_overlap_merge={after_merge}"
        )

        for det in sorted(merged_detections, key=lambda item: (item["box"][1], item["box"][0])):
            x, y, ww, hh = [int(v) for v in det["box"]]
            score = float(det["score"])
            class_id = int(det["class_id"])
            tight_bbox = (x, y, ww, hh)
            block = OCRBlock(
                text="",
                boxes=[],
                confidence=score,
                detector_source="yolo",
            )
            # Pass 7: store tight YOLO output as detector_text_bbox.
            # bbox_override starts equal to it; _enrich_region must NOT inflate it.
            # bubble_bbox is set to same value here; detect_bubble_region inside
            # _enrich_region will expand it to the container — separate from overlay.
            block.detector_text_bbox = tight_bbox
            block.bbox_override = tight_bbox
            block.bubble_bbox = tight_bbox
            block.yolo_class_id = class_id  # type: ignore[attr-defined]
            block.yolo_kind = ("dialogue", "narration", "sfx", "shout")[class_id] if 0 <= class_id < 4 else "dialogue"  # type: ignore[attr-defined]
            debug_print(
                f"[YOLO_BLOCK] kind={block.yolo_kind!r} "
                f"conf={score:.3f} detector_text_bbox={tight_bbox} "
                f"merged_count={int(det.get('merged_count', 1))}"
            )
            blocks.append(block)
        return blocks


YoloV6RegionDetector = YoloV8RegionDetector
