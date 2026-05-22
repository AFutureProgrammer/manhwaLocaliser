"""
Offline cleanup finetuner.

This tool reads real cleanup debug artifacts, scores cleanup/mask quality, and
groups failures into general rule suggestions. It does not mutate app state or
write per-region overrides.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEBUG_DIR = ROOT / "debug_cleanup"
DEFAULT_RUNS_DIR = ROOT / "tools" / "cleanup_lab" / "runs"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _image_exists(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _read_gray(path: Path) -> Optional[np.ndarray]:
    if not _image_exists(path):
        return None
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return arr


def _read_rgb(path: Path) -> Optional[np.ndarray]:
    if not _image_exists(path):
        return None
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _mask_binary(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 12, 255, 0).astype(np.uint8)


def _connected_components(mask: np.ndarray) -> Tuple[int, int, int]:
    if mask is None or mask.size == 0:
        return 0, 0, 0
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    if not areas:
        return 0, 0, 0
    return len(areas), max(areas), sum(1 for area in areas if area <= 18)


@dataclass
class CleanupCase:
    case_id: str
    meta_path: Path
    page_dir: Path
    region_id: str
    page_index: int
    region_index: int
    region_class: str
    role: str
    background_model: str
    strategy: str
    method: str
    route: str
    backend: str
    mask_source: str
    selected_candidate: str
    text_candidates: List[Dict[str, Any]]
    mask_region_ratio: float
    mask_container_ratio: float
    rectangularity: float
    border_touch_ratio: float
    component_count: int
    largest_component_ratio: float
    residual_speck_px: int
    cleanup_effective: bool
    visual_quality_ok: bool
    residual_text_visible: bool
    artifacts: Dict[str, str] = field(default_factory=dict)
    terminal_events: List[str] = field(default_factory=list)

    @property
    def signature(self) -> str:
        parts = [
            self.region_class or "unknown",
            self.role or "unknown",
            self.background_model or "unknown",
            self.mask_source or "none",
            self.strategy or "none",
            self.route or "none",
        ]
        return "|".join(parts)


def parse_terminal_log(path: Optional[Path]) -> Dict[Tuple[int, str], List[str]]:
    if path is None or not path.exists():
        return {}
    events: Dict[Tuple[int, str], List[str]] = {}
    pattern = re.compile(r"page=(?P<page>\d+).*?region=(?P<region>R-\d+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "cleanup_" not in line and "CleanupPlan" not in line and "lama" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        key = (int(match.group("page")), match.group("region"))
        events.setdefault(key, []).append(line.strip())
    return events


def load_debug_cases(debug_dir: Path, terminal_log: Optional[Path] = None) -> List[CleanupCase]:
    debug_dir = debug_dir.resolve()
    terminal_events = parse_terminal_log(terminal_log.resolve() if terminal_log else None)
    cases: List[CleanupCase] = []
    for meta_path in sorted(debug_dir.glob("page_*/*_meta.json")):
        meta = _read_json(meta_path)
        debug_metrics = meta.get("debug_metrics") or {}
        quality = debug_metrics.get("quality") or {}
        region_id = str(meta.get("region_id") or meta_path.stem.replace("_meta", ""))
        page_index = _as_int(meta.get("page_index", meta.get("page_idx", -1)), -1)
        region_index = _as_int(meta.get("region_index", meta.get("region_idx", -1)), -1)
        artifacts = {}
        for name in ("raw", "cleaned", "text_mask", "cleanup_mask", "container_mask", "overlay"):
            artifact_path = meta_path.with_name(f"{region_id}_{name}.png")
            if _image_exists(artifact_path):
                artifacts[name] = str(artifact_path)
        case = CleanupCase(
            case_id=f"p{page_index:03d}_{region_id}",
            meta_path=meta_path,
            page_dir=meta_path.parent,
            region_id=region_id,
            page_index=page_index,
            region_index=region_index,
            region_class=str(meta.get("region_class") or ""),
            role=str(meta.get("role") or debug_metrics.get("region_role") or ""),
            background_model=str(meta.get("background_model") or debug_metrics.get("bg_model") or ""),
            strategy=str(meta.get("chosen_cleanup_strategy") or debug_metrics.get("strategy_before_mask_assembly") or ""),
            method=str(meta.get("chosen_inpaint_method") or ""),
            route=str(debug_metrics.get("cleanup_route") or ""),
            backend=str(meta.get("cleanup_backend") or debug_metrics.get("cleanup_backend_used") or ""),
            mask_source=str(meta.get("selected_text_mask_candidate_source") or debug_metrics.get("selected_text_mask_candidate_source") or "none"),
            selected_candidate=str(meta.get("selected_text_mask_candidate") or debug_metrics.get("selected_text_mask_candidate") or "none"),
            text_candidates=list(meta.get("text_mask_candidates") or debug_metrics.get("text_mask_candidate_scores") or []),
            mask_region_ratio=_as_float(meta.get("mask_region_ratio", quality.get("mask_region_ratio"))),
            mask_container_ratio=_as_float(meta.get("mask_container_ratio", quality.get("mask_container_ratio"))),
            rectangularity=_as_float(meta.get("rectangularity", quality.get("rectangularity"))),
            border_touch_ratio=_as_float(quality.get("border_touch_ratio")),
            component_count=_as_int(quality.get("component_count")),
            largest_component_ratio=_as_float(quality.get("largest_component_ratio")),
            residual_speck_px=_as_int(debug_metrics.get("sam2_residual_speck_pass_px")),
            cleanup_effective=bool(meta.get("cleanup_effective", False)),
            visual_quality_ok=bool(meta.get("visual_quality_ok", False)),
            residual_text_visible=bool(meta.get("residual_text_visible", False)),
            artifacts=artifacts,
            terminal_events=terminal_events.get((page_index, region_id), []),
        )
        cases.append(case)
    return cases


def score_case(case: CleanupCase) -> Dict[str, Any]:
    cleaned = _read_rgb(Path(case.artifacts["cleaned"])) if "cleaned" in case.artifacts else None
    cleanup_mask = _read_gray(Path(case.artifacts["cleanup_mask"])) if "cleanup_mask" in case.artifacts else None
    text_mask = _read_gray(Path(case.artifacts["text_mask"])) if "text_mask" in case.artifacts else None
    residual_dark_px = 0
    residual_component_count = 0
    residual_small_components = 0
    largest_residual_component = 0
    if cleaned is not None:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
        shape = gray.shape[:2]
        scope = _mask_binary(cleanup_mask, shape)
        if not np.any(scope):
            scope = _mask_binary(text_mask, shape)
        if np.any(scope):
            dark = np.where((gray < 92) & (scope > 0), 255, 0).astype(np.uint8)
            residual_dark_px = int(np.count_nonzero(dark))
            residual_component_count, largest_residual_component, residual_small_components = _connected_components(dark)

    overbroad = (
        case.mask_region_ratio >= 0.30
        or case.border_touch_ratio >= 0.50
        or (case.rectangularity > 0.0 and case.rectangularity < 0.33 and case.mask_region_ratio >= 0.24)
    )
    fragmented = case.component_count >= 8 and case.largest_component_ratio <= 0.45
    residual_risk = residual_dark_px >= 24 or residual_component_count >= 2 or case.residual_speck_px >= 24
    route_mismatch = (
        case.backend in {"lama_pt", "lama_onnx", "iopaint"}
        and case.route == "solid_bubble_cv"
        and case.strategy == "flat_fill"
        and (case.mask_region_ratio >= 0.24 or fragmented)
    )
    art_damage_risk = case.mask_region_ratio >= 0.45 or (case.border_touch_ratio >= 0.50 and case.mask_region_ratio >= 0.25)
    skipped_review = case.strategy == "review" or case.method == "skip"

    risk_score = 0
    risk_score += 3 if overbroad else 0
    risk_score += 2 if fragmented else 0
    risk_score += 2 if residual_risk else 0
    risk_score += 2 if route_mismatch else 0
    risk_score += 3 if art_damage_risk else 0
    risk_score += 1 if skipped_review else 0

    if risk_score >= 5 or art_damage_risk or route_mismatch:
        verdict = "fail"
    elif risk_score >= 2 or not case.cleanup_effective:
        verdict = "review"
    else:
        verdict = "pass"

    reasons: List[str] = []
    if overbroad:
        reasons.append("overbroad_mask")
    if fragmented:
        reasons.append("fragmented_mask")
    if residual_risk:
        reasons.append("residual_dark_components")
    if route_mismatch:
        reasons.append("lama_backend_routed_through_flat_fill")
    if art_damage_risk:
        reasons.append("art_damage_risk")
    if skipped_review:
        reasons.append("cleanup_skipped_review")
    if not case.cleanup_effective:
        reasons.append("cleanup_effective_false")

    return {
        "case_id": case.case_id,
        "signature": case.signature,
        "verdict": verdict,
        "risk_score": risk_score,
        "reasons": reasons,
        "metrics": {
            "mask_region_ratio": round(case.mask_region_ratio, 4),
            "mask_container_ratio": round(case.mask_container_ratio, 4),
            "rectangularity": round(case.rectangularity, 4),
            "border_touch_ratio": round(case.border_touch_ratio, 4),
            "component_count": case.component_count,
            "largest_component_ratio": round(case.largest_component_ratio, 4),
            "residual_dark_px": residual_dark_px,
            "residual_component_count": residual_component_count,
            "largest_residual_component": largest_residual_component,
            "residual_small_components": residual_small_components,
            "sam2_residual_speck_pass_px": case.residual_speck_px,
        },
    }


def propose_candidate_variants(case: CleanupCase, score: Dict[str, Any]) -> List[Dict[str, Any]]:
    reasons = set(score.get("reasons") or [])
    variants: List[Dict[str, Any]] = []

    if "lama_backend_routed_through_flat_fill" in reasons:
        variants.append({
            "id": "lama_first_mask_inpaint",
            "type": "route",
            "description": "Use mask_inpaint/texture_clone for LaMa backends instead of solid_bubble_cv flat_fill when masks are nontrivial.",
            "expected_effect": "Avoids CV flat-fill behavior on broad/fragmented masks while still using the selected cleanup mask.",
        })
    if case.mask_source == "fallback_cv_no_bbox" and ("overbroad_mask" in reasons or "fragmented_mask" in reasons):
        variants.append({
            "id": "reject_blocky_fallback_cv",
            "type": "mask_ranking",
            "description": "Reject fallback CV no-bbox masks when region ratio is high and components are fragmented/block-shaped.",
            "expected_effect": "Forces selection of a tighter candidate or review instead of erasing a whole text block.",
        })
    if case.mask_source == "sam2" and ("residual_dark_components" in reasons or case.residual_speck_px > 0):
        variants.append({
            "id": "sam2_residual_speck_expansion",
            "type": "mask_refinement",
            "description": "Add conservative dark connected components near the text bbox/container before halo/growth.",
            "expected_effect": "Catches missed thin debris without expanding across the bubble.",
        })
    if case.strategy == "flat_fill" and "overbroad_mask" in reasons:
        variants.append({
            "id": "reduced_flat_fill_growth",
            "type": "mask_growth",
            "description": "Disable or reduce flat-fill ladder growth for broad masks; prefer growth 0 unless ring metrics are clearly safe.",
            "expected_effect": "Prevents broad masks from expanding into art, borders, or bubble outlines.",
        })
    if case.border_touch_ratio >= 0.50:
        variants.append({
            "id": "border_touch_review_or_split",
            "type": "safety_gate",
            "description": "Route border-touching masks to review or cross-page/split cleanup unless the mask is small and tight.",
            "expected_effect": "Avoids page-edge wipes and seam artifacts.",
        })
    if not variants:
        variants.append({
            "id": "manual_review_required",
            "type": "review",
            "description": "Metrics are inconclusive; include this case in the review sheet.",
            "expected_effect": "Avoids overfitting a general rule to an ambiguous case.",
        })
    return variants


def synthesize_rules(cases: List[CleanupCase], scores: List[Dict[str, Any]], candidates: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    by_rule: Dict[str, Dict[str, Any]] = {}
    by_case = {case.case_id: case for case in cases}
    for score in scores:
        case = by_case[score["case_id"]]
        if score["verdict"] == "pass":
            continue
        for variant in candidates.get(case.case_id, []):
            rule_id = str(variant["id"])
            entry = by_rule.setdefault(rule_id, {
                "rule_id": rule_id,
                "type": variant.get("type", ""),
                "description": variant.get("description", ""),
                "affected_cases": [],
                "signatures": {},
                "priority": 0,
            })
            entry["affected_cases"].append(case.case_id)
            entry["signatures"][case.signature] = entry["signatures"].get(case.signature, 0) + 1
            entry["priority"] += int(score.get("risk_score", 0) or 0)
    rules = list(by_rule.values())
    for rule in rules:
        rule["affected_count"] = len(rule["affected_cases"])
        rule["signatures"] = [
            {"signature": sig, "count": count}
            for sig, count in sorted(rule["signatures"].items(), key=lambda item: (-item[1], item[0]))
        ]
    rules.sort(key=lambda item: (-int(item.get("priority", 0)), -int(item.get("affected_count", 0)), str(item.get("rule_id", ""))))
    return rules


def replay_chapter_cases(chapter: Path, cases: List[CleanupCase]) -> Dict[str, Any]:
    """Replay baseline cleanup planning for each case against a chapter folder."""
    lab_dir = Path(__file__).resolve().parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))
    from cleanup_lab import _load_chapter_page, _select_region  # type: ignore
    from backend.core.cleanup_plan import CleanupPolicy, build_cleanup_plan, execute_cleanup_plan, validate_cleanup_proposal
    from backend.core.config import ModelConfig
    from backend.core.ocr import imread_unicode

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    page_cache: Dict[int, Tuple[Path, List[Dict[str, Any]], int]] = {}
    for case in cases:
        try:
            if case.page_index not in page_cache:
                page_cache[case.page_index] = _load_chapter_page(chapter.resolve(), case.page_index + 1, False)
            image_path, regions, page_index = page_cache[case.page_index]
            region_id, block, _raw_region = _select_region(regions, case.region_id)
            image = imread_unicode(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
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
                validation_source="cleanup_tuner_replay",
            )
            quality = plan.debug_metrics.get("quality", {}) or {}
            records.append({
                "case_id": case.case_id,
                "page_index": int(page_index),
                "region_id": region_id,
                "strategy": plan.cleanup_strategy,
                "method": plan.inpaint_method,
                "background_model": plan.background_model,
                "selected_candidate": plan.text_mask_reason,
                "selected_source": plan.debug_metrics.get("selected_text_mask_candidate_source", ""),
                "mask_region_ratio": round(_as_float(quality.get("mask_region_ratio")), 4),
                "border_touch_ratio": round(_as_float(quality.get("border_touch_ratio")), 4),
                "component_count": _as_int(quality.get("component_count")),
                "cleanup_effective": bool(getattr(plan, "cleanup_effective", False)),
            })
        except Exception as exc:
            errors.append({"case_id": case.case_id, "error": str(exc)})
    return {
        "chapter": str(chapter.resolve()),
        "records": records,
        "errors": errors,
        "record_count": len(records),
        "error_count": len(errors),
    }


def _case_manifest_entry(case: CleanupCase) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "page_index": case.page_index,
        "region_index": case.region_index,
        "region_id": case.region_id,
        "signature": case.signature,
        "region_class": case.region_class,
        "role": case.role,
        "background_model": case.background_model,
        "strategy": case.strategy,
        "method": case.method,
        "route": case.route,
        "backend": case.backend,
        "mask_source": case.mask_source,
        "selected_candidate": case.selected_candidate,
        "artifacts": case.artifacts,
        "meta_path": str(case.meta_path),
        "terminal_event_count": len(case.terminal_events),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(data), fh, ensure_ascii=False, indent=2)


def _make_review_sheet(cases: List[CleanupCase], scores: List[Dict[str, Any]], candidates: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    score_by_id = {score["case_id"]: score for score in scores}
    review_cases = [case for case in cases if score_by_id[case.case_id]["verdict"] in {"fail", "review"}]
    if not review_cases:
        review_cases = cases[:]
    cell_w, cell_h = 190, 145
    label_h = 58
    columns = ["raw", "text_mask", "cleanup_mask", "cleaned", "overlay"]
    width = cell_w * len(columns)
    height = max(1, len(review_cases)) * (cell_h + label_h)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, case in enumerate(review_cases):
        y = row_idx * (cell_h + label_h)
        score = score_by_id[case.case_id]
        variant_ids = ",".join(v["id"] for v in candidates.get(case.case_id, [])[:2])
        label = (
            f"{case.case_id} {score['verdict']} risk={score['risk_score']} "
            f"{case.region_class} {case.strategy}/{case.method} src={case.mask_source} "
            f"ratio={case.mask_region_ratio:.3f} {variant_ids}"
        )
        draw.text((4, y + 4), label[:160], fill=(0, 0, 0))
        for col_idx, name in enumerate(columns):
            x = col_idx * cell_w
            image_path = case.artifacts.get(name)
            if image_path and Path(image_path).exists():
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((cell_w, cell_h), Image.LANCZOS)
                canvas = Image.new("RGB", (cell_w, cell_h), "white")
                canvas.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
            else:
                canvas = Image.new("RGB", (cell_w, cell_h), (238, 238, 238))
                ImageDraw.Draw(canvas).text((8, 8), f"missing {name}", fill=(80, 80, 80))
            sheet.paste(canvas, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def _write_rules_markdown(path: Path, rules: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines = [
        "# Generalized Cleanup Rules",
        "",
        f"Cases analyzed: {summary['case_count']}",
        f"Verdicts: pass={summary['verdicts'].get('pass', 0)} review={summary['verdicts'].get('review', 0)} fail={summary['verdicts'].get('fail', 0)}",
        "",
    ]
    if not rules:
        lines.append("No generalized rule suggestions were produced.")
    for idx, rule in enumerate(rules, 1):
        lines.extend([
            f"## {idx}. {rule['rule_id']}",
            "",
            f"- Type: {rule.get('type', '')}",
            f"- Priority: {rule.get('priority', 0)}",
            f"- Affected cases: {rule.get('affected_count', 0)}",
            f"- Rule: {rule.get('description', '')}",
        ])
        signatures = rule.get("signatures") or []
        if signatures:
            lines.append("- Top signatures:")
            for sig in signatures[:5]:
                lines.append(f"  - {sig['count']}x `{sig['signature']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(debug_dir: Path, terminal_log: Optional[Path], out_dir: Path, chapter: Optional[Path] = None) -> Dict[str, Any]:
    cases = load_debug_cases(debug_dir, terminal_log)
    scores = [score_case(case) for case in cases]
    candidate_results = {case.case_id: propose_candidate_variants(case, score) for case, score in zip(cases, scores)}
    rules = synthesize_rules(cases, scores, candidate_results)
    chapter_replay = replay_chapter_cases(chapter, cases) if chapter else None
    verdicts: Dict[str, int] = {}
    strategies: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for case, score in zip(cases, scores):
        verdicts[score["verdict"]] = verdicts.get(score["verdict"], 0) + 1
        strategies[case.strategy] = strategies.get(case.strategy, 0) + 1
        sources[case.mask_source] = sources.get(case.mask_source, 0) + 1
    summary = {
        "case_count": len(cases),
        "debug_dir": str(debug_dir.resolve()),
        "terminal_log": str(terminal_log.resolve()) if terminal_log else "",
        "chapter": str(chapter.resolve()) if chapter else "",
        "created_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdicts": verdicts,
        "strategies": strategies,
        "mask_sources": sources,
        "rules_count": len(rules),
        "chapter_replay_records": int((chapter_replay or {}).get("record_count", 0) or 0),
        "chapter_replay_errors": int((chapter_replay or {}).get("error_count", 0) or 0),
    }
    manifest = {
        "summary": summary,
        "cases": [_case_manifest_entry(case) for case in cases],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "case_manifest.json", manifest)
    _write_json(out_dir / "case_scores.json", {"summary": summary, "scores": scores})
    _write_json(out_dir / "candidate_results.json", {"summary": summary, "candidates": candidate_results, "rules": rules})
    if chapter_replay is not None:
        _write_json(out_dir / "chapter_replay.json", chapter_replay)
    _write_rules_markdown(out_dir / "generalized_rules.md", rules, summary)
    _make_review_sheet(cases, scores, candidate_results, out_dir / "review_sheet.jpg")
    return {
        "summary": summary,
        "out_dir": str(out_dir),
        "files": {
            "case_manifest": str(out_dir / "case_manifest.json"),
            "case_scores": str(out_dir / "case_scores.json"),
            "candidate_results": str(out_dir / "candidate_results.json"),
            "chapter_replay": str(out_dir / "chapter_replay.json") if chapter_replay is not None else "",
            "review_sheet": str(out_dir / "review_sheet.jpg"),
            "generalized_rules": str(out_dir / "generalized_rules.md"),
        },
        "top_rules": rules[:5],
    }


def _default_out_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"cleanup_tuner_{stamp}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze cleanup debug artifacts and synthesize general cleanup tuning rules.")
    parser.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR), help="directory containing page_*/R-*_meta.json debug artifacts")
    parser.add_argument("--terminal-log", default="", help="optional backend terminal log to attach cleanup events")
    parser.add_argument("--chapter", default="", help="optional chapter directory for future full-page replay context")
    parser.add_argument("--out", default="", help="output directory; defaults to tools/cleanup_lab/runs/cleanup_tuner_<timestamp>")
    args = parser.parse_args(argv)
    debug_dir = Path(args.debug_dir).resolve()
    terminal_log = Path(args.terminal_log).resolve() if args.terminal_log else None
    chapter = Path(args.chapter).resolve() if args.chapter else None
    out_dir = Path(args.out).resolve() if args.out else _default_out_dir().resolve()
    if not debug_dir.exists():
        print(f"ERROR: debug dir not found: {debug_dir}", file=sys.stderr)
        return 2
    result = analyze(debug_dir, terminal_log, out_dir, chapter)
    summary = result["summary"]
    print(f"cleanup_tuner cases={summary['case_count']} out={result['out_dir']}")
    print(f"  verdicts={summary['verdicts']}")
    print(f"  strategies={summary['strategies']}")
    print("  files:")
    for name, path in result["files"].items():
        if not path:
            continue
        print(f"    {name}: {path}")
    if result["top_rules"]:
        print("  top rules:")
        for rule in result["top_rules"]:
            print(f"    {rule['rule_id']}: affected={rule['affected_count']} priority={rule['priority']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
