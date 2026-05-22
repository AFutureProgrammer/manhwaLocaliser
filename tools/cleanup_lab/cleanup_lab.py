from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from backend.core.cleanup_failure_taxonomy import (  # noqa: E402
    classify_cleanup_failure,
    primary_cleanup_failure_class,
)
from backend.core.cleanup_plan import (  # noqa: E402
    CleanupPolicy,
    build_cleanup_plan,
    execute_cleanup_plan,
    normalize_mask_to_image,
    validate_cleanup_proposal,
)
from backend.core.config import ModelConfig  # noqa: E402
from backend.core.regions import OCRBlock, RegionKind, RegionOverride, _block_from_dict  # noqa: E402

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = LAB_DIR / "outputs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def imread_unicode(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, arr: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    ok, encoded = cv2.imencode(ext, arr)
    if not ok:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


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


def _rect_poly(bbox: Tuple[int, int, int, int]) -> List[List[float]]:
    x, y, w, h = bbox
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
    ]


def _coerce_boxes(region: Dict[str, Any]) -> List[Any]:
    for key in ("boxes", "ocr_boxes", "text_boxes"):
        boxes = region.get(key)
        if isinstance(boxes, list) and boxes:
            return boxes
    return []


def _kind_from_fixture(region: Dict[str, Any]) -> Optional[RegionKind]:
    raw = str(region.get("kind") or region.get("region_kind") or "").strip().upper()
    if not raw:
        return None
    aliases = {
        "PLAIN": "PLAIN_BUBBLE",
        "PLAIN_BUBBLE": "PLAIN_BUBBLE",
        "TEXTURED": "TEXTURED_BUBBLE",
        "TEXTURED_BUBBLE": "TEXTURED_BUBBLE",
        "GRADIENT": "GRADIENT_BUBBLE",
        "GRADIENT_BUBBLE": "GRADIENT_BUBBLE",
        "CAPTION": "CAPTION_BOX",
        "CAPTION_BOX": "CAPTION_BOX",
        "SFX": "SFX_OVER_ART",
        "SFX_OVER_ART": "SFX_OVER_ART",
        "TEXT_ON_ART": "DIALOGUE_OVER_ART",
        "DIALOGUE_OVER_ART": "DIALOGUE_OVER_ART",
        "UNKNOWN": "UNKNOWN",
    }
    name = aliases.get(raw, raw)
    try:
        return RegionKind[name]
    except KeyError:
        return None


def _make_fixture_block(region: Dict[str, Any]) -> OCRBlock:
    bbox = _bbox(region.get("bbox") or region.get("region_bbox"), (0, 0, 1, 1))
    assert bbox is not None
    boxes = _coerce_boxes(region)
    block = OCRBlock(
        text=str(region.get("text") or ""),
        boxes=boxes,
        confidence=float(region.get("detector_confidence", region.get("confidence", 0.0)) or 0.0),
    )
    block.bbox_override = bbox
    block.detector_source = str(region.get("detector") or region.get("detector_source") or "yolo")
    block.bubble_role = str(region.get("role") or region.get("bubble_role") or "dialog")
    region_kind = _kind_from_fixture(region)
    if region_kind is not None:
        block.region_kind = region_kind
    yolo_kind = str(region.get("yolo_kind") or region.get("yolo_class") or "")
    if yolo_kind:
        setattr(block, "yolo_kind", yolo_kind)
    yolo_class_id = region.get("yolo_class_id")
    if yolo_class_id is not None:
        try:
            setattr(block, "yolo_class_id", int(yolo_class_id))
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
        _x, _y, bw, bh = block.bubble_bbox
        block.bubble_mask = np.full((max(1, bh), max(1, bw)), 255, dtype=np.uint8)
    override_data = region.get("override")
    if isinstance(override_data, dict):
        block.override = RegionOverride.from_dict(override_data)
    cleanup_region_class = region.get("cleanup_region_class")
    if cleanup_region_class:
        if block.override is None:
            block.override = RegionOverride()
        block.override.cleanup_region_class = str(cleanup_region_class)
    return block


def _load_fixture(path: Path) -> Tuple[Optional[Path], List[Tuple[str, OCRBlock, Dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        fixture = {"regions": data}
    elif isinstance(data, dict):
        fixture = data
    else:
        raise ValueError("regions fixture must be a JSON object or array")
    image_value = fixture.get("page_image") or fixture.get("image")
    image_path = Path(image_value).expanduser() if image_value else None
    if image_path is not None and not image_path.is_absolute():
        image_path = (path.parent / image_path).resolve()
    regions = fixture.get("regions") or []
    if not isinstance(regions, list):
        raise ValueError("fixture 'regions' must be a list")
    out: List[Tuple[str, OCRBlock, Dict[str, Any]]] = []
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        region_id = str(region.get("id") or region.get("region_id") or f"R-{idx + 1:02d}")
        out.append((region_id, _make_fixture_block(region), dict(region)))
    return image_path, out


def _page_index_from_arg(page: int, zero_based: bool) -> int:
    return int(page if zero_based else page - 1)


def _iter_chapter_images(chapter: Path) -> List[Path]:
    return sorted(p for p in chapter.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_chapter_page(chapter: Path, page: int, zero_based: bool) -> Tuple[Path, List[Tuple[str, OCRBlock, Dict[str, Any]]], int]:
    page_idx = _page_index_from_arg(page, zero_based)
    images = _iter_chapter_images(chapter)
    if page_idx < 0 or page_idx >= len(images):
        raise ValueError(f"page index out of range: requested={page} resolved_zero_based={page_idx} total={len(images)}")
    state_path = chapter / ".ml_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing .ml_state.json: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    pages = state.get("pages") or []
    page_entry: Optional[Dict[str, Any]] = None
    image_path = images[page_idx].resolve()
    for idx, entry in enumerate(pages):
        if idx == page_idx:
            page_entry = entry
            break
        raw_path = entry.get("image_path") if isinstance(entry, dict) else None
        if raw_path and Path(raw_path).name == image_path.name:
            page_entry = entry
            break
    if not page_entry:
        raise ValueError(f"no page entry found in .ml_state.json for page {page}")
    regions = page_entry.get("regions") or []
    out: List[Tuple[str, OCRBlock, Dict[str, Any]]] = []
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        block = _block_from_dict(region)
        if region.get("yolo_kind") and not getattr(block, "yolo_kind", ""):
            setattr(block, "yolo_kind", str(region.get("yolo_kind") or ""))
        if region.get("yolo_class") and not getattr(block, "yolo_kind", ""):
            setattr(block, "yolo_kind", str(region.get("yolo_class") or ""))
        if region.get("yolo_class_id") is not None and getattr(block, "yolo_class_id", None) is None:
            try:
                setattr(block, "yolo_class_id", int(region.get("yolo_class_id")))
            except Exception:
                pass
        region_id = str(region.get("id") or region.get("region_id") or f"R-{idx + 1:02d}")
        out.append((region_id, block, dict(region)))
    return image_path, out, page_idx


def _fixture_region_from_block(region_id: str, block: OCRBlock, raw_region: Dict[str, Any]) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": region_id,
        "bbox": list(block.bbox()),
        "text": str(getattr(block, "text", "") or ""),
        "role": str(getattr(block, "bubble_role", "") or ""),
        "kind": str(getattr(getattr(block, "region_kind", None), "name", "") or ""),
        "yolo_kind": str(getattr(block, "yolo_kind", "") or raw_region.get("yolo_kind", "") or ""),
        "detector": str(getattr(block, "detector_source", "") or ""),
        "detector_confidence": float(getattr(block, "confidence", 0.0) or 0.0),
        "boxes": list(getattr(block, "boxes", []) or []),
    }
    for key in (
        "bubble_bbox",
        "safe_rect",
        "cleanup_safe_rect",
        "cleanup_container_bbox",
        "detector_text_bbox",
    ):
        value = getattr(block, key, None)
        if value is not None:
            item[key] = [int(v) for v in value]
    if getattr(block, "cleanup_container_confidence", 0.0):
        item["cleanup_container_confidence"] = float(block.cleanup_container_confidence)
    if getattr(block, "cleanup_safe_rect_confidence", 0.0):
        item["cleanup_safe_rect_confidence"] = float(block.cleanup_safe_rect_confidence)
    yolo_class_id = getattr(block, "yolo_class_id", raw_region.get("yolo_class_id", None))
    if yolo_class_id is not None:
        item["yolo_class_id"] = yolo_class_id
    override = raw_region.get("override")
    if isinstance(override, dict) and override:
        item["override"] = override
    return item


def export_fixture(args: argparse.Namespace) -> int:
    if not args.chapter or not args.page:
        raise ValueError("--export-fixture requires --chapter and --page")
    image_path, regions, page_index = _load_chapter_page(
        Path(args.chapter).resolve(),
        args.page,
        args.zero_based_page,
    )
    region_id, block, raw_region = _select_region(regions, args.region_id)
    out_path = Path(args.export_fixture).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "page_image": str(image_path),
        "source": {
            "chapter": str(Path(args.chapter).resolve()),
            "page_arg": args.page,
            "page_index_zero_based": page_index,
            "region_id": region_id,
            "read_only_export": True,
        },
        "regions": [_fixture_region_from_block(region_id, block, raw_region)],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(fixture), f, indent=2, ensure_ascii=False)
    print(f"exported fixture: {out_path}")
    print(f"  image: {image_path}")
    print(f"  page_index_zero_based: {page_index}")
    print(f"  region: {region_id} bbox={block.bbox()}")
    return 0


def _select_region(regions: Iterable[Tuple[str, OCRBlock, Dict[str, Any]]], region_id: str) -> Tuple[str, OCRBlock, Dict[str, Any]]:
    wanted = str(region_id).strip()
    for rid, block, raw in regions:
        if str(rid) == wanted:
            return rid, block, raw
    if wanted.upper().startswith("R-"):
        try:
            wanted_idx = int(wanted.split("-", 1)[1]) - 1
            for idx, item in enumerate(regions):
                if idx == wanted_idx:
                    return item
        except Exception:
            pass
    raise ValueError(f"region-id not found: {region_id}")


def _crop(arr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    h, w = arr.shape[:2]
    x, y, bw, bh = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return arr[y1:y2, x1:x2]


def _mask_px(mask: Optional[np.ndarray]) -> int:
    return int(np.count_nonzero(mask)) if mask is not None else 0


def _mask_bbox(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _union_bboxes(image_shape: Tuple[int, ...], bboxes: Iterable[Optional[Tuple[int, int, int, int]]]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    xs1: List[int] = []
    ys1: List[int] = []
    xs2: List[int] = []
    ys2: List[int] = []
    for b in bboxes:
        if b is None:
            continue
        x, y, bw, bh = b
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 > x1 and y2 > y1:
            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)
    if not xs1:
        return (0, 0, w, h)
    x1, y1 = min(xs1), min(ys1)
    x2, y2 = max(xs2), max(ys2)
    pad = 8
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _container_full(plan: Any, image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    if plan.container_mask is None or plan.container_bbox is None:
        return None
    return normalize_mask_to_image(plan.container_mask, plan.container_bbox, image_shape)


def _overlay(raw_crop: np.ndarray, crop_bbox: Tuple[int, int, int, int], layers: List[Tuple[Optional[np.ndarray], Tuple[int, int, int], float]]) -> np.ndarray:
    out = raw_crop.copy()
    x, y, _w, _h = crop_bbox
    for mask, color, alpha in layers:
        if mask is None:
            continue
        local = _crop(mask, crop_bbox)
        if local is None or not np.any(local):
            continue
        active = local > 0
        c = np.array(color, dtype=np.float32)
        out[active] = (out[active].astype(np.float32) * (1.0 - alpha) + c * alpha).clip(0, 255).astype(np.uint8)
    return out


def _write_artifacts(out_dir: Path, image: np.ndarray, cleaned: np.ndarray, plan: Any, source: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    container = _container_full(plan, image.shape)
    crop_bbox = _union_bboxes(
        image.shape,
        [
            plan.region_bbox,
            plan.text_bbox,
            plan.container_bbox,
            _mask_bbox(plan.text_mask),
            _mask_bbox(container),
            _mask_bbox(plan.cleanup_mask),
        ],
    )
    raw_crop = _crop(image, crop_bbox)
    cleaned_crop = _crop(cleaned, crop_bbox)
    files: Dict[str, str] = {}

    def write(name: str, arr: Optional[np.ndarray]) -> None:
        if arr is None:
            return
        path = out_dir / name
        if imwrite_unicode(str(path), arr):
            files[name] = str(path)

    write("original_crop.png", raw_crop)
    write("text_mask.png", _crop(plan.text_mask, crop_bbox) if plan.text_mask is not None else None)
    write("container_mask.png", _crop(container, crop_bbox) if container is not None else None)
    write("cleanup_mask.png", _crop(plan.cleanup_mask, crop_bbox) if plan.cleanup_mask is not None else None)
    write("cleaned_crop.png", cleaned_crop)
    write("cleaned_page_preview.png", cleaned)
    if raw_crop is not None:
        write(
            "overlay.png",
            _overlay(
                raw_crop,
                crop_bbox,
                [
                    (container, (80, 80, 255), 0.25),
                    (plan.text_mask, (0, 255, 0), 0.55),
                    (plan.cleanup_mask, (0, 0, 255), 0.35),
                ],
            ),
        )
    report = {
        "source": source,
        "region_id": plan.region_id,
        "page_index_zero_based": int(plan.page_index),
        "region_bbox": plan.region_bbox,
        "text_bbox": plan.text_bbox,
        "container_bbox": plan.container_bbox,
        "bubble_bbox": source.get("raw_region", {}).get("bubble_bbox"),
        "safe_rect": source.get("raw_region", {}).get("safe_rect"),
        "crop_bbox": crop_bbox,
        "role": plan.debug_metrics.get("region_role", ""),
        "detector_source": plan.detector_source,
        "yolo_class": plan.yolo_class,
        "yolo_kind": plan.yolo_class,
        "region_kind": plan.debug_metrics.get("region_kind", ""),
        "region_class": plan.region_class,
        "background_model": plan.background_model,
        "strategy": plan.cleanup_strategy,
        "inpaint_method": plan.inpaint_method,
        "skip_reason": plan.skip_reason,
        "text_mask_confidence": plan.text_mask_confidence,
        "container_confidence": plan.container_confidence,
        "cleanup_mask_confidence": plan.cleanup_mask_confidence,
        "text_mask_px": _mask_px(plan.text_mask),
        "container_mask_px": _mask_px(container),
        "cleanup_mask_px": _mask_px(plan.cleanup_mask),
        "text_mask_reason": plan.text_mask_reason,
        "container_reason": plan.container_reason,
        "effectiveness": {
            key: value
            for key, value in plan.debug_metrics.items()
            if key in {
                "raw_cleaned_diff_px",
                "raw_cleaned_diff_ratio",
                "diff_inside_cleanup_mask_px",
                "diff_outside_cleanup_mask_px",
                "cleanup_mask_px",
                "text_mask_px",
                "cleaned_same_as_raw",
                "near_identical_raw_cleaned",
                "near_identical_tolerance_px",
                "cleanup_effective",
                "cleanup_failure_reason",
                "cleanup_validation_source",
                "manual_visual_success",
                "manual_visual_partial",
                "diagnostic_only",
                "diagnostic_cleanup_ran",
                "destructive_cleanup_executed",
                "production_patch_accepted",
                "proposal_valid",
                "proposal_failure_reason",
                "gate_violation",
                "text_removed",
                "residual_text_visible",
                "visual_quality_ok",
                "fill_patch_visible",
                "cleanup_partial",
                "residual_component_count",
                "residual_component_px",
                "residual_component_bboxes",
                "residual_verifier_reason",
                "residual_retry_safe",
                "residual_retry_rejection_reason",
                "fill_patch_component_count",
                "fill_patch_component_bboxes",
                "fill_patch_reason",
            }
        },
        "debug_metrics": plan.debug_metrics,
        "artifacts": files,
    }
    report.update(report["effectiveness"])
    failure_classes = classify_cleanup_failure(report)
    report["failure_classes"] = failure_classes
    report["failure_class"] = primary_cleanup_failure_class(failure_classes)
    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, ensure_ascii=False)
    files["report.json"] = str(report_path)
    return files


def _source_for_inline_fixture(case: Dict[str, Any], manifest_path: Path) -> Tuple[Path, List[Tuple[str, OCRBlock, Dict[str, Any]]], int, Dict[str, Any]]:
    image_value = case.get("image") or case.get("page_image")
    if not image_value:
        raise ValueError("manifest case requires 'image' or 'page_image'")
    image_path = Path(str(image_value)).expanduser()
    if not image_path.is_absolute():
        image_path = (manifest_path.parent / image_path).resolve()
    region = case.get("region")
    if not isinstance(region, dict):
        raise ValueError("manifest case requires inline 'region' object when 'regions' is not provided")
    region_id = str(case.get("region_id") or region.get("id") or region.get("region_id") or "R-01")
    raw_region = dict(region)
    raw_region.setdefault("id", region_id)
    page_index = int(case.get("page_index", case.get("page_index_zero_based", 0)) or 0)
    return image_path, [(region_id, _make_fixture_block(raw_region), raw_region)], page_index, {
        "mode": "manifest_inline",
        "manifest": str(manifest_path),
        "case_id": str(case.get("case_id") or region_id),
    }


def _load_manifest_case(case: Dict[str, Any], manifest_path: Path) -> Tuple[Path, List[Tuple[str, OCRBlock, Dict[str, Any]]], int, Dict[str, Any]]:
    if case.get("chapter"):
        page = int(case.get("page") or 0)
        if page <= 0 and not bool(case.get("zero_based_page")):
            raise ValueError("chapter manifest case requires one-based 'page' unless zero_based_page=true")
        image_path, regions, page_index = _load_chapter_page(
            (manifest_path.parent / str(case["chapter"])).resolve()
            if not Path(str(case["chapter"])).is_absolute()
            else Path(str(case["chapter"])).resolve(),
            page,
            bool(case.get("zero_based_page", False)),
        )
        return image_path, regions, page_index, {
            "mode": "manifest_chapter",
            "manifest": str(manifest_path),
            "chapter": str(case["chapter"]),
            "page_arg": page,
        }

    if case.get("regions"):
        regions_path = Path(str(case["regions"])).expanduser()
        if not regions_path.is_absolute():
            regions_path = (manifest_path.parent / regions_path).resolve()
        fixture_image, regions = _load_fixture(regions_path)
        image_value = case.get("image") or case.get("page_image")
        image_path = Path(str(image_value)).expanduser() if image_value else fixture_image
        if image_path is None:
            raise ValueError("manifest case requires image/page_image or fixture page_image")
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        page_index = int(case.get("page_index", case.get("page_index_zero_based", 0)) or 0)
        return image_path, regions, page_index, {
            "mode": "manifest_fixture",
            "manifest": str(manifest_path),
            "regions": str(regions_path),
        }

    return _source_for_inline_fixture(case, manifest_path)


def _cleanup_verdict(report: Dict[str, Any]) -> str:
    if report.get("cleanup_effective") and not report.get("failure_classes"):
        return "pass"
    if report.get("cleanup_effective"):
        return "review"
    if report.get("strategy") in {"review", "skip"} or report.get("skip_reason"):
        return "review"
    return "fail"


def _case_output_dir(batch_out: Path, case_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id).strip("_")
    return batch_out / (safe or "case")


def _string_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def run_manifest(manifest_path: Path, out_dir: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list):
        raise ValueError("cleanup QA manifest must contain a 'cases' list")

    out_dir.mkdir(parents=True, exist_ok=True)
    review_cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, case_obj in enumerate(cases, start=1):
        if not isinstance(case_obj, dict):
            continue
        case_id = str(case_obj.get("case_id") or f"case_{idx:03d}")
        try:
            image_path, regions, page_index, source = _load_manifest_case(case_obj, manifest_path)
            inline_region = case_obj.get("region")
            region_id = str(
                case_obj.get("region_id")
                or (inline_region.get("id") if isinstance(inline_region, dict) else "")
                or "R-01"
            )
            region_id, block, raw_region = _select_region(regions, region_id)
            image = imread_unicode(str(image_path))
            if image is None:
                raise FileNotFoundError(f"could not read image: {image_path}")
            cfg = ModelConfig()
            policy = CleanupPolicy.from_config(cfg)
            plan = build_cleanup_plan(
                image,
                block,
                page_index=page_index,
                region_id=region_id,
                cleanup_debug_artifacts=False,
                cleanup_policy=policy,
                model_config=cfg,
            )
            plan.cleanup_backend = "opencv"
            cleaned = image.copy()
            execute_cleanup_plan(image, cleaned, plan)
            validate_cleanup_proposal(
                image,
                cleaned,
                plan,
                destructive_allowed=True,
                production_patch_accepted=False,
                validation_source="cleanup_lab_manifest",
            )
            source.update({"image": str(image_path), "raw_region": raw_region})
            case_dir = _case_output_dir(out_dir, case_id)
            files = _write_artifacts(case_dir, image, cleaned, plan, source)
            report = _read_report(files["report.json"])
            expected_outcome = str(case_obj.get("expected_outcome") or "").strip()
            actual_outcome = _cleanup_verdict(report)
            expected_classes = _string_list(
                case_obj.get("expected_failure_classes")
                if "expected_failure_classes" in case_obj
                else case_obj.get("expected_failure_class")
            )
            failure_classes = list(report.get("failure_classes") or [])
            expected_met = not expected_outcome or expected_outcome == actual_outcome
            if expected_classes:
                expected_met = expected_met and all(item in failure_classes for item in expected_classes)
            quality = report.get("debug_metrics", {}).get("quality", {}) or {}
            review_cases.append({
                "case_id": case_id,
                "region_id": region_id,
                "page_index_zero_based": int(page_index),
                "expected_outcome": expected_outcome,
                "actual_outcome": actual_outcome,
                "expected_failure_classes": expected_classes,
                "actual_failure_class": report.get("failure_class", ""),
                "failure_classes": failure_classes,
                "backend_used": report.get("debug_metrics", {}).get("cleanup_backend_used", report.get("cleanup_backend", "opencv")),
                "strategy": report.get("strategy", ""),
                "method": report.get("inpaint_method", ""),
                "mask_quality": {
                    "mask_region_ratio": quality.get("mask_region_ratio", report.get("mask_region_ratio", 0.0)),
                    "mask_container_ratio": quality.get("mask_container_ratio", report.get("mask_container_ratio", 0.0)),
                    "border_touch_ratio": quality.get("border_touch_ratio", report.get("border_touch_ratio", 0.0)),
                    "rectangularity": quality.get("rectangularity", report.get("rectangularity", 0.0)),
                },
                "inpaint_quality": {
                    "cleanup_effective": bool(report.get("cleanup_effective", False)),
                    "residual_text_visible": bool(report.get("residual_text_visible", False)),
                    "visual_quality_ok": bool(report.get("visual_quality_ok", False)),
                    "cleanup_failure_reason": str(report.get("cleanup_failure_reason", "") or ""),
                },
                "final_result": "pass" if expected_met else "fail",
                "artifacts": files,
                "output_dir": str(case_dir),
                "notes": str(case_obj.get("notes") or ""),
            })
        except Exception as exc:
            errors.append({"case_id": case_id, "error": str(exc)})
            review_cases.append({
                "case_id": case_id,
                "actual_outcome": "error",
                "actual_failure_class": "needs_manual_mask",
                "failure_classes": ["needs_manual_mask"],
                "final_result": "fail",
                "error": str(exc),
                "output_dir": str(_case_output_dir(out_dir, case_id)),
            })

    review_manifest = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "summary": {
            "case_count": len(review_cases),
            "pass_count": sum(1 for item in review_cases if item.get("final_result") == "pass"),
            "fail_count": sum(1 for item in review_cases if item.get("final_result") == "fail"),
            "error_count": len(errors),
        },
        "cases": review_cases,
        "errors": errors,
    }
    review_path = out_dir / "review_manifest.json"
    with review_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(review_manifest), fh, ensure_ascii=False, indent=2)
    return {"review_manifest": review_manifest, "review_path": str(review_path), "out_dir": str(out_dir)}


def _read_report(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _print_manifest_summary(result: Dict[str, Any]) -> None:
    manifest = result["review_manifest"]
    print(f"cleanup_lab manifest cases={manifest['summary']['case_count']} out={result['out_dir']}")
    print("case_id\tstrategy\tmethod\tfailure_class\tmask_region\tmask_container\tresult\toutput")
    for case in manifest.get("cases", []):
        quality = case.get("mask_quality") or {}
        print(
            "\t".join([
                str(case.get("case_id", "")),
                str(case.get("strategy", "")),
                str(case.get("method", "")),
                str(case.get("actual_failure_class", "")),
                f"{float(quality.get('mask_region_ratio', 0.0) or 0.0):.4f}",
                f"{float(quality.get('mask_container_ratio', 0.0) or 0.0):.4f}",
                str(case.get("final_result", "")),
                str(case.get("output_dir", "")),
            ])
        )
    print(f"review_manifest: {result['review_path']}")


def _default_out(image_path: Path, region_id: str) -> Path:
    safe_region = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in region_id)
    return DEFAULT_OUTPUTS_DIR / f"{image_path.stem}_{safe_region}"


def run(args: argparse.Namespace) -> int:
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        out_dir = Path(args.out).resolve() if args.out else (DEFAULT_OUTPUTS_DIR / manifest_path.stem).resolve()
        result = run_manifest(manifest_path, out_dir)
        _print_manifest_summary(result)
        return 0
    if args.export_fixture:
        return export_fixture(args)
    if not args.region_id:
        raise ValueError("--region-id is required unless --manifest is used")
    page_index = 0
    source: Dict[str, Any] = {"mode": "fixture"}
    if args.chapter:
        if not args.page:
            raise ValueError("--page is required with --chapter")
        image_path, regions, page_index = _load_chapter_page(Path(args.chapter).resolve(), args.page, args.zero_based_page)
        source = {"mode": "ml_state", "chapter": str(Path(args.chapter).resolve()), "page_arg": args.page}
    else:
        if not args.regions:
            raise ValueError("--regions is required unless --chapter is used")
        fixture_image, regions = _load_fixture(Path(args.regions).resolve())
        image_path = Path(args.image).resolve() if args.image else fixture_image
        if image_path is None:
            raise ValueError("provide --image or page_image in the fixture")
        source = {"mode": "fixture", "regions": str(Path(args.regions).resolve())}
        page_index = int(args.page_index or 0)
    if not regions:
        raise ValueError("no regions available")
    region_id, block, raw_region = _select_region(regions, args.region_id)
    image = imread_unicode(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    policy = CleanupPolicy.from_config(ModelConfig())
    plan = build_cleanup_plan(
        image,
        block,
        page_index=page_index,
        region_id=region_id,
        cleanup_debug_artifacts=False,
        cleanup_policy=policy,
        model_config=ModelConfig(),
    )
    plan.cleanup_backend = "opencv"
    cleaned = image.copy()
    execute_cleanup_plan(image, cleaned, plan)
    validate_cleanup_proposal(
        image,
        cleaned,
        plan,
        destructive_allowed=True,
        production_patch_accepted=False,
        validation_source="cleanup_lab",
    )
    out_dir = Path(args.out).resolve() if args.out else _default_out(image_path, region_id).resolve()
    source.update({"image": str(image_path), "raw_region": raw_region})
    files = _write_artifacts(out_dir, image, cleaned, plan, source)
    quality = plan.debug_metrics.get("quality", {}) or {}
    print(f"cleanup_lab region={region_id} image={image_path}")
    print(f"  bbox={plan.region_bbox} text_bbox={plan.text_bbox} container_bbox={plan.container_bbox}")
    print(f"  text_mask_px={_mask_px(plan.text_mask)} container_mask_px={_mask_px(_container_full(plan, image.shape))} cleanup_mask_px={_mask_px(plan.cleanup_mask)}")
    print(f"  text_conf={plan.text_mask_confidence:.3f} container_conf={plan.container_confidence:.3f} cleanup_conf={plan.cleanup_mask_confidence:.3f}")
    print(f"  strategy={plan.cleanup_strategy} method={plan.inpaint_method} background={plan.background_model} skip_reason={plan.skip_reason or '-'}")
    print(f"  selected_candidate={plan.text_mask_reason or '-'} mask_region_ratio={float(quality.get('mask_region_ratio', 0.0) or 0.0):.4f}")
    print("  artifacts:")
    for name, path in sorted(files.items()):
        print(f"    {name}: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cleanup/mask planning experiments without starting the backend server.")
    parser.add_argument("--image", help="raw page image path for standalone fixture mode")
    parser.add_argument("--regions", help="standalone region fixture JSON path")
    parser.add_argument("--chapter", help="chapter folder containing images and .ml_state.json")
    parser.add_argument("--page", type=int, help="page number for --chapter mode; one-based unless --zero-based-page is set")
    parser.add_argument("--zero-based-page", action="store_true", help="treat --page as zero-based")
    parser.add_argument("--page-index", type=int, default=0, help="zero-based page index to store in report for fixture mode")
    parser.add_argument("--region-id", help="region id such as R-01")
    parser.add_argument("--out", help="output directory; defaults to tools/cleanup_lab/outputs/<image>_<region>")
    parser.add_argument("--manifest", help="run a cleanup QA manifest containing multiple cases")
    parser.add_argument("--export-fixture", help="write a standalone fixture JSON from --chapter/--page/--region-id and exit")
    try:
        return run(parser.parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
