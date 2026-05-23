import unittest
import builtins
import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageFont

from backend.core import sam2_mask
from backend.core.cleanup_plan import (
    CleanupPlan,
    build_cleanup_plan,
    clamp_cleanup_outcome_fields,
    cleanup_production_patch_allowed,
    execute_cleanup_plan,
    select_strategy,
    _execute_telea,
    _constrain_flat_fill_boundary_mask,
    validate_cleanup_proposal,
)
from backend.core.config import ModelConfig
from backend.core.ocr import YoloV8RegionDetector
from backend.core.project import ChapterManager, ChapterPage
from backend.core.regions import OCRBlock, RegionKind, RegionOverride, _block_from_dict, _block_to_dict
from backend.engine import LocalizerEngine


def _dialogue_block(bbox, text="테스트", kind=RegionKind.PLAIN_BUBBLE):
    block = OCRBlock(
        text=text,
        boxes=[],
        confidence=0.9,
        detector_source="yolo",
        bubble_bbox=bbox,
        bubble_mask=np.ones((bbox[3], bbox[2]), dtype=np.uint8) * 255,
        bubble_role="dialog",
        region_kind=kind,
    )
    block.bbox_override = bbox
    return block


def _white_bubble_page():
    img = np.full((180, 260, 3), 255, np.uint8)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (252, 252, 252), -1)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (30, 30, 30), 2)
    cv2.putText(img, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _halftone_bubble_page():
    img = np.full((180, 260, 3), 255, np.uint8)
    bubble = np.zeros((180, 260), np.uint8)
    cv2.ellipse(bubble, (130, 90), (90, 45), 0, 0, 360, 255, -1)
    img[bubble > 0] = (246, 246, 246)
    for yy in range(50, 132, 4):
        for xx in range(50, 212, 4):
            if bubble[yy, xx] > 0:
                cv2.circle(img, (xx, yy), 1, (206, 206, 206), -1)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (30, 30, 30), 2)
    cv2.putText(img, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _solid_colored_bubble_page(fill_bgr=(210, 185, 235)):
    img = np.full((180, 260, 3), 245, np.uint8)
    bubble = np.zeros((180, 260), np.uint8)
    cv2.ellipse(bubble, (130, 90), (90, 45), 0, 0, 360, 255, -1)
    img[bubble > 0] = np.array(fill_bgr, dtype=np.uint8)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (60, 70, 80), 2)
    cv2.putText(img, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 45, 60), 2, cv2.LINE_AA)
    return img


def _yellowed_scan_bubble_page(fill_bgr=(238, 232, 214)):
    img = np.full((180, 260, 3), (224, 222, 218), np.uint8)
    bubble = np.zeros((180, 260), np.uint8)
    cv2.ellipse(bubble, (130, 90), (90, 45), 0, 0, 360, 255, -1)
    img[bubble > 0] = np.array(fill_bgr, dtype=np.uint8)
    yy, xx = np.indices(img.shape[:2])
    noise = (((xx * 7 + yy * 11) % 5) - 2).astype(np.int16)
    for ch in range(3):
        channel = img[:, :, ch].astype(np.int16)
        channel[bubble > 0] = np.clip(channel[bubble > 0] + noise[bubble > 0], 0, 255)
        img[:, :, ch] = channel.astype(np.uint8)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (55, 55, 55), 2)
    cv2.putText(img, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def _dark_caption_page(fill_bgr=(24, 24, 24)):
    img = np.full((140, 260, 3), 235, np.uint8)
    cv2.rectangle(img, (40, 45), (220, 95), fill_bgr, -1)
    cv2.putText(img, "TEXT", (78, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    return img


def _dark_caption_glow_page(fill_bgr=(8, 8, 8)):
    img = np.full((140, 260, 3), 235, np.uint8)
    cv2.rectangle(img, (40, 45), (220, 95), fill_bgr, -1)
    cv2.putText(img, "TEXT", (76, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 210), 7, cv2.LINE_AA)
    cv2.putText(img, "TEXT", (76, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2, cv2.LINE_AA)
    return img


def _flat_black_text_page():
    img = np.zeros((120, 240, 3), np.uint8)
    cv2.putText(img, "TEXT", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 245, 245), 2, cv2.LINE_AA)
    return img


def _translucent_caption_page(detailed=False):
    img = np.full((150, 260, 3), 210, np.uint8)
    yy, xx = np.indices(img.shape[:2])
    img[:, :, 0] = np.clip(70 + xx * 0.25, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(120 + yy * 0.30, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(95 + xx * 0.10 + yy * 0.10, 0, 255).astype(np.uint8)
    if detailed:
        for offset in range(-120, 260, 8):
            cv2.line(img, (offset, 0), (offset + 160, 150), (25, 55, 75), 2, cv2.LINE_AA)
            cv2.line(img, (offset + 4, 150), (offset + 120, 0), (210, 225, 240), 1, cv2.LINE_AA)
    x1, y1, x2, y2 = 30, 28, 230, 118
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 54, 42), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 230, 226), 2)
    cv2.putText(img, "TEXT", (75, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    return img


def _bold_korean_like_bubble_page():
    img = np.full((180, 260, 3), 255, np.uint8)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (252, 252, 252), -1)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (30, 30, 30), 2)
    cv2.rectangle(img, (85, 66), (98, 112), (0, 0, 0), -1)
    cv2.rectangle(img, (85, 66), (126, 78), (0, 0, 0), -1)
    cv2.rectangle(img, (118, 82), (132, 112), (0, 0, 0), -1)
    cv2.rectangle(img, (104, 98), (114, 112), (0, 0, 0), -1)
    existing = np.zeros(img.shape[:2], dtype=np.uint8)
    existing[66:113, 85:99] = 255
    existing[66:79, 85:127] = 255
    return img, existing


def _thin_aa_bubble_page():
    img = np.full((180, 260, 3), 255, np.uint8)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (252, 252, 252), -1)
    cv2.ellipse(img, (130, 90), (90, 45), 0, 0, 360, (30, 30, 30), 2)
    core = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(core, (95, 76), (105, 108), 255, -1)
    cv2.rectangle(core, (95, 76), (135, 84), 255, -1)
    halo = cv2.bitwise_and(
        cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1),
        cv2.bitwise_not(core),
    )
    img[halo > 0] = (238, 238, 238)
    img[core > 0] = (0, 0, 0)
    return img, core


def _punctuation_bubble_page():
    img = _white_bubble_page()
    cv2.circle(img, (163, 101), 2, (0, 0, 0), -1, cv2.LINE_AA)
    return img


def _dialogue_block_with_mask(bbox, mask, text="테스트"):
    block = _dialogue_block(bbox, text=text, kind=RegionKind.PLAIN_BUBBLE)
    local = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    cv2.ellipse(local, (bbox[2] // 2, bbox[3] // 2), (bbox[2] // 2 - 2, bbox[3] // 2 - 5), 0, 0, 360, 255, -1)
    block.bubble_mask = local
    block.text_mask = mask
    return block


def _caption_block(bbox, text="캡션"):
    block = OCRBlock(
        text=text,
        boxes=[],
        confidence=0.9,
        detector_source="yolo",
        bubble_bbox=bbox,
        bubble_mask=np.ones((bbox[3], bbox[2]), dtype=np.uint8) * 255,
        bubble_role="caption",
        region_kind=RegionKind.CAPTION_BOX,
    )
    block.bbox_override = bbox
    return block


def _protected_art_block(bbox, kind, role, text="효과"):
    block = OCRBlock(
        text=text,
        boxes=[],
        confidence=0.9,
        detector_source="yolo",
        bubble_bbox=bbox,
        bubble_mask=np.ones((bbox[3], bbox[2]), dtype=np.uint8) * 255,
        bubble_role=role,
        region_kind=kind,
    )
    block.bbox_override = bbox
    mask = np.zeros((180, 260), dtype=np.uint8)
    x, y, w, h = bbox
    mask[y + 20:y + min(h, 38), x + 20:x + min(w, 70)] = 255
    block.text_mask = mask
    if kind == RegionKind.SFX_OVER_ART:
        block.yolo_kind = "sfx"
        block.yolo_class_id = 2
    return block


def _cleanup_engine_for_blocks(blocks, cfg=None):
    img = _white_bubble_page()
    page = ChapterPage(image_path="synthetic.png")
    page.regions = blocks
    page.translations = ["" for _ in blocks]
    engine = LocalizerEngine.__new__(LocalizerEngine)
    engine.chapter_mgr = SimpleNamespace(
        current_page=page,
        current_idx=0,
        pages=[page],
        save_state=lambda: None,
        total_pages=lambda: 1,
    )
    engine._raw_cv = img
    engine._regions = blocks
    engine._translations = page.translations
    engine.model_config = cfg or ModelConfig()
    engine._push_undo_snapshot = lambda: None
    engine._flush_working_state_to_page = lambda: None
    engine._notify = lambda *args, **kwargs: None
    engine.get_bootstrap = lambda: {"ok": True}
    return engine, page, img


def _assert_changed_pixels_within_mask(testcase, before, after, mask):
    testcase.assertIsNotNone(mask)
    changed = np.any(before != after, axis=2) if before.ndim == 3 else before != after
    outside = changed & ~(mask > 0)
    testcase.assertEqual(int(np.count_nonzero(outside)), 0)


class CleanupPipelineTests(unittest.TestCase):
    def test_sam2_cleanup_mask_can_drive_planner(self):
        img = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100))
        sam_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.putText(sam_mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 4, cv2.LINE_AA)
        ok, buf = cv2.imencode(".png", sam_mask)
        self.assertTrue(ok)
        cfg = ModelConfig()
        cfg.sam2_enabled = True
        cfg.sam2_mask_mode = "cleanup_assist"
        cfg.cleanup_mask_backend = "sam2"

        with patch("backend.core.sam2_mask.propose_mask", return_value={
            "ok": True,
            "mask_b64": __import__("base64").b64encode(buf.tobytes()).decode("utf-8"),
            "bbox": [0, 0, img.shape[1], img.shape[0]],
            "confidence": 0.91,
        }):
            plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.debug_metrics.get("selected_text_mask_candidate_source"), "sam2")
        self.assertTrue(plan.debug_metrics.get("sam2_mask_used"))
        self.assertIsNotNone(plan.cleanup_mask)

    def test_force_cleanup_falls_back_to_bbox_without_text_signal(self):
        img = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")
        cfg = ModelConfig()
        cfg.sam2_enabled = False
        cfg.cleanup_force_enabled = True

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertIsNotNone(plan.cleanup_mask)
        self.assertNotIn(plan.cleanup_strategy, {"skip", "review"})
        self.assertEqual(plan.skip_reason, "")
        self.assertTrue(plan.debug_metrics.get("force_cleanup_no_text_signal"))

    def test_textured_dark_caption_routes_to_model_backend(self):
        img = _dark_caption_page(fill_bgr=(7, 3, 105))
        block = OCRBlock(
            text="테스트",
            boxes=[],
            confidence=0.9,
            detector_source="yolo",
            bubble_bbox=(40, 45, 180, 50),
            bubble_mask=np.ones((50, 180), dtype=np.uint8) * 255,
            bubble_role="thought",
            region_kind=RegionKind.TEXTURED_BUBBLE,
        )
        block.bbox_override = (40, 45, 180, 50)
        block.yolo_kind = "narration"
        cfg = ModelConfig()
        cfg.sam2_enabled = False
        cfg.cleanup_backend = "lama_pt"

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.debug_metrics.get("cleanup_route"), "model_inpaint_solid_bubble")
        self.assertEqual(plan.cleanup_backend, "lama_pt")
        self.assertTrue(plan.debug_metrics.get("model_inpaint_required"))

    def test_yolo_dialogue_without_qwen_text_skips_cleanup(self):
        img = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertIn(plan.cleanup_strategy, {"skip", "review"})
        self.assertIsNone(plan.cleanup_mask)
        self.assertEqual(plan.skip_reason, "no_ocr_text_for_cleanup")

    def test_white_bubble_cleanup_mask_is_stroke_shaped_not_region(self):
        img = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        metrics = plan.debug_metrics["mask"]

        self.assertEqual(plan.background_model, "flat_light")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertEqual(plan.debug_metrics.get("cleanup_route"), "solid_bubble_cv")
        self.assertTrue(plan.debug_metrics.get("flat_fill_ladder_enabled"))
        self.assertGreater(len(plan.debug_metrics.get("flat_fill_ladder_candidates", [])), 0)
        self.assertIn("flat_fill_ladder_selected_growth_px", plan.debug_metrics)
        self.assertLess(metrics["mask_region_ratio"], 0.25)
        self.assertNotEqual(metrics["mask_bbox"], plan.region_bbox)
        self.assertIsNotNone(plan.cleanup_mask)

        result = img.copy()
        execute_cleanup_plan(img, result, plan)
        self.assertFalse(np.array_equal(img, result))

    def test_large_bold_glyph_recovery_adds_nearby_components(self):
        img, existing = _bold_korean_like_bubble_page()
        block = _dialogue_block_with_mask((40, 40, 180, 100), existing)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.background_model, "flat_light")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertLessEqual(plan.debug_metrics.get("flat_fill_ladder_selected_growth_px", 0), 10)
        self.assertGreater(plan.debug_metrics["large_component_kept_count"], 0)
        self.assertGreater(plan.cleanup_mask[84, 124], 0)
        self.assertLess(plan.debug_metrics["mask"]["mask_region_ratio"], 0.25)
        self.assertNotEqual(plan.debug_metrics["mask"]["mask_bbox"], plan.region_bbox)

    def test_anti_aliased_gray_halo_is_included_on_flat_bubble(self):
        img, core = _thin_aa_bubble_page()
        block = _dialogue_block_with_mask((40, 40, 180, 100), core)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.background_model, "flat_light")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertGreater(plan.debug_metrics["halo_added_px"], 0)
        self.assertGreater(plan.cleanup_mask[74, 100], 0)
        self.assertLess(plan.debug_metrics["mask"]["mask_region_ratio"], 0.25)

    def test_tiny_punctuation_dot_is_kept_in_cleanup_mask(self):
        img = _punctuation_bubble_page()
        block = _dialogue_block((40, 40, 180, 100))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.background_model, "flat_light")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertGreater(plan.cleanup_mask[101, 163], 0)
        self.assertLess(plan.debug_metrics["mask"]["mask_region_ratio"], 0.25)

    def test_large_easy_white_bubble_mask_is_not_skipped_for_tight_yolo_bbox(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for x in range(58, 180, 18):
            mask[58:120, x:x + 12] = 255
        img[mask > 0] = (0, 0, 0)
        block = _dialogue_block_with_mask((40, 40, 180, 100), mask)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.background_model, "flat_light")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertTrue(plan.debug_metrics.get("easy_cleanup_eligible"))
        self.assertGreaterEqual(plan.debug_metrics["quality"]["mask_region_ratio"], 0.28)
        self.assertNotEqual(plan.skip_reason, "cleanup_mask_too_large_region_ratio")

    def test_halftone_bubble_still_classifies_as_texture(self):
        img = _halftone_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.TEXTURED_BUBBLE)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertIn(plan.cleanup_strategy, {"skip", "review"})
        self.assertEqual(plan.skip_reason, "skipped_texture_inpaint_disabled")

    def test_halftone_texture_opencv_policy_is_review_only(self):
        cfg = ModelConfig()
        cfg.cleanup_backend = "opencv"
        cfg.cleanup_allow_texture_inpaint = True
        cfg.cleanup_fallback_backend = "telea"
        img = _halftone_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.TEXTURED_BUBBLE)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertEqual(plan.cleanup_strategy, "review")
        self.assertEqual(plan.inpaint_method, "skip")

    def test_lama_backend_routes_halftone_bubble_to_model_inpaint(self):
        cfg = ModelConfig()
        cfg.cleanup_backend = "lama_pt"
        img = _halftone_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.TEXTURED_BUBBLE)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertEqual(plan.cleanup_backend, "lama_pt")
        self.assertEqual(plan.cleanup_strategy, "texture_clone")
        self.assertEqual(plan.inpaint_method, "telea")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertEqual(plan.debug_metrics.get("cleanup_route"), "model_inpaint")

    def test_aggressive_halftone_caption_attempts_cleanup_with_review_flag(self):
        cfg = ModelConfig()
        cfg.cleanup_mode = "aggressive"
        cfg.cleanup_backend = "opencv"
        cfg.cleanup_allow_texture_inpaint = True
        cfg.cleanup_fallback_backend = "telea"
        cfg.cleanup_risky_action = "attempt"
        img = _halftone_bubble_page()
        block = _caption_block((40, 40, 180, 100))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertEqual(plan.cleanup_strategy, "texture_clone")
        self.assertEqual(plan.inpaint_method, "telea")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertTrue(plan.debug_metrics.get("review_required_after_cleanup"))
        self.assertIn("aggressive_review_attempt", plan.debug_metrics)

    def test_halftone_texture_prefers_iopaint_when_configured(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_texture_inpaint = True
        cfg.cleanup_prefer_iopaint_for_texture = True
        cfg.iopaint_url = "http://127.0.0.1:9/inpaint"
        img = _halftone_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.TEXTURED_BUBBLE)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertEqual(plan.cleanup_strategy, "texture_clone")
        self.assertEqual(plan.cleanup_backend, "iopaint")
        self.assertEqual(plan.debug_metrics.get("cleanup_route"), "model_inpaint")
        self.assertEqual(plan.debug_metrics["cleanup_fallback_backend"], "iopaint")

    def test_halftone_texture_missing_iopaint_marks_review(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_texture_inpaint = True
        cfg.cleanup_prefer_iopaint_for_texture = True
        cfg.iopaint_url = ""
        img = _halftone_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.TEXTURED_BUBBLE)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertEqual(plan.debug_metrics.get("cleanup_route"), "model_inpaint")
        self.assertEqual(plan.cleanup_strategy, "review")
        self.assertIn("iopaint_required_unavailable", plan.skip_reason)

    def test_halftone_candidate_compare_does_not_offer_solid_fill(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        img = _halftone_bubble_page()
        page = ChapterPage(image_path="synthetic.png")
        engine.chapter_mgr = SimpleNamespace(current_page=page, current_idx=0)
        engine._raw_cv = img
        engine._regions = [_dialogue_block((40, 40, 180, 100), kind=RegionKind.TEXTURED_BUBBLE)]
        engine.model_config = ModelConfig()

        resp = engine.compare_region_cleanup_candidates(0)

        solid = next(c for c in resp["candidates"] if c["candidate_id"] == "solid_fill")
        self.assertFalse(solid["is_available"])
        self.assertIn("background", solid["unavailable_reason"])
        self.assertEqual(resp["recommended_candidate_id"], "")

        telea = next(c for c in resp["candidates"] if c["candidate_id"] == "telea")
        ns = next(c for c in resp["candidates"] if c["candidate_id"] == "opencv_ns")
        self.assertTrue(telea["is_available"])
        self.assertTrue(ns["is_available"])
        self.assertTrue(telea["review_required"])
        self.assertTrue(ns["review_required"])
        self.assertIn("Review: texture blur risk", telea["warnings"])
        self.assertIn("Review: texture blur risk", ns["warnings"])

    def test_cleanup_patch_rebuild_composites_only_mask_pixels(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        raw = np.full((20, 22, 3), 10, np.uint8)
        crop = np.full((8, 9, 3), 200, np.uint8)
        mask = np.zeros((8, 9), np.uint8)
        mask[2:6, 3:8] = 255
        page = ChapterPage(image_path="synthetic.png")
        page.cleanup_patches = [{
            "region_id": "r1",
            "region_idx": 0,
            "bbox": [5, 6, 9, 8],
            "patch_png_b64": engine._encode_cv_png_b64(crop),
            "mask_png_b64": engine._encode_cv_png_b64(mask),
        }]

        rebuilt = engine._rebuild_cleaned_from_cleanup_patches(page, raw)

        changed = np.any(rebuilt != raw, axis=2)
        expected = np.zeros(raw.shape[:2], dtype=bool)
        expected[8:12, 8:13] = True
        self.assertTrue(np.array_equal(changed, expected))
        self.assertTrue(np.all(rebuilt[expected] == 200))
        self.assertTrue(np.all(rebuilt[~expected] == 10))

    def test_cleanup_patch_rebuild_skips_legacy_unmasked_crop(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        raw = np.full((12, 12, 3), 25, np.uint8)
        crop = np.full((6, 6, 3), 240, np.uint8)
        page = ChapterPage(image_path="synthetic.png")
        page.cleanup_patches = [{
            "region_id": "r1",
            "region_idx": 0,
            "bbox": [3, 3, 6, 6],
            "patch_png_b64": engine._encode_cv_png_b64(crop),
        }]

        rebuilt = engine._rebuild_cleaned_from_cleanup_patches(page, raw)

        self.assertTrue(np.array_equal(rebuilt, raw))

    def test_sfx_region_apply_is_blocked_by_default(self):
        cfg = ModelConfig()
        cfg.auto_clean_sfx = True
        cfg.cleanup_allow_sfx_cleanup = False
        block = _protected_art_block((35, 35, 185, 85), RegionKind.SFX_OVER_ART, "sfx")
        engine, page, raw = _cleanup_engine_for_blocks([block], cfg)

        engine.apply_region_cleanup(0)

        self.assertEqual(page.cleanup_patches, [])
        self.assertTrue(np.array_equal(page.cleaned_cv, raw))
        self.assertFalse(block.cleanup_meta["proposal_valid"])
        self.assertTrue(block.cleanup_meta["diagnostic_only"])
        self.assertFalse(block.cleanup_meta["destructive_cleanup_executed"])
        self.assertFalse(block.cleanup_meta["production_patch_accepted"])
        self.assertFalse(block.cleanup_meta["cleanup_effective"])
        self.assertIn("protected_sfx", block.cleanup_meta["proposal_failure_reason"])

    def test_text_over_art_region_apply_is_blocked_by_default(self):
        cfg = ModelConfig()
        cfg.auto_clean_text_over_art = True
        cfg.cleanup_allow_text_over_art = False
        block = _protected_art_block((35, 35, 185, 85), RegionKind.DIALOGUE_OVER_ART, "dialog")
        engine, page, raw = _cleanup_engine_for_blocks([block], cfg)

        engine.apply_region_cleanup(0)

        self.assertEqual(page.cleanup_patches, [])
        self.assertTrue(np.array_equal(page.cleaned_cv, raw))
        self.assertFalse(block.cleanup_meta["proposal_valid"])
        self.assertTrue(block.cleanup_meta["diagnostic_only"])
        self.assertFalse(block.cleanup_meta["production_patch_accepted"])
        self.assertFalse(block.cleanup_meta["cleanup_effective"])

    def test_cleanup_risky_attempt_does_not_override_sfx_gate(self):
        cfg = ModelConfig()
        cfg.auto_clean_sfx = True
        cfg.cleanup_risky_action = "attempt"
        cfg.cleanup_allow_sfx_cleanup = False
        block = _protected_art_block((35, 35, 185, 85), RegionKind.SFX_OVER_ART, "sfx")
        engine, _page, _raw = _cleanup_engine_for_blocks([block], cfg)

        resp = engine.compare_region_cleanup_candidates(0)

        self.assertTrue(resp["candidates"])
        self.assertTrue(all(not c["is_available"] for c in resp["candidates"]))
        self.assertTrue(all("protected_sfx" in c["unavailable_reason"] for c in resp["candidates"]))

    def test_sfx_region_class_override_does_not_enable_pipeline_cleanup(self):
        cfg = ModelConfig()
        cfg.process_sfx_regions = False
        cfg.cleanup_allow_sfx_cleanup = False
        block = _protected_art_block((35, 35, 185, 85), RegionKind.SFX_OVER_ART, "sfx")
        block.override = RegionOverride(cleanup_region_class="sfx")
        engine, page, raw = _cleanup_engine_for_blocks([block], cfg)
        engine._progress_ctx = {}
        engine._active_op_depth = 0
        engine._active_ops = set()
        engine._active_op_names = []

        engine.cleanup_current_page()

        self.assertTrue(page.cleaned_cv is None or np.array_equal(page.cleaned_cv, raw))
        self.assertEqual(getattr(page, "cleanup_patches", []) or [], [])

    def test_protected_region_candidates_unavailable_by_default(self):
        cfg = ModelConfig()
        block = _protected_art_block((35, 35, 185, 85), RegionKind.SFX_OVER_ART, "sfx")
        engine, _page, _raw = _cleanup_engine_for_blocks([block], cfg)

        resp = engine.compare_region_cleanup_candidates(0)

        solid = next(c for c in resp["candidates"] if c["candidate_id"] == "solid_fill")
        telea = next(c for c in resp["candidates"] if c["candidate_id"] == "telea")
        self.assertFalse(solid["is_available"])
        self.assertFalse(telea["is_available"])
        self.assertIn("protected_sfx", solid["unavailable_reason"])
        self.assertIn("protected_sfx", telea["unavailable_reason"])

    def test_grouped_fallback_excludes_sfx_by_default(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_grouped_inpaint = True
        normal = _dialogue_block((35, 35, 185, 85), kind=RegionKind.PLAIN_BUBBLE)
        sfx = _protected_art_block((45, 45, 150, 70), RegionKind.SFX_OVER_ART, "sfx")
        engine, _page, raw = _cleanup_engine_for_blocks([normal, sfx], cfg)
        selected = build_cleanup_plan(raw, normal, page_index=0, region_id="R-01", model_config=cfg)
        result = raw.copy()

        group = engine._try_grouped_fallback(0, selected, raw, result, False)

        self.assertNotIn(1, group.get("indices", []))

    def test_explicit_force_allow_still_requires_valid_proposal_for_patch(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_sfx_cleanup = True
        block = _protected_art_block((35, 35, 185, 85), RegionKind.SFX_OVER_ART, "sfx")
        block.override = RegionOverride(cleanup_override_mode="force_allow")
        engine, page, raw = _cleanup_engine_for_blocks([block], cfg)

        engine.apply_region_cleanup(0)

        self.assertEqual(page.cleanup_patches, [])
        self.assertTrue(np.array_equal(page.cleaned_cv, raw))
        self.assertFalse(block.cleanup_meta["proposal_valid"])
        self.assertTrue(block.cleanup_meta["diagnostic_only"])
        self.assertFalse(block.cleanup_meta["production_patch_accepted"])

    def test_valid_flat_unknown_region_is_not_metadata_blocked(self):
        img = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.UNKNOWN)
        block.bubble_role = "manual"
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        block.text_mask = mask
        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        result = img.copy()

        execute_cleanup_plan(img, result, plan)
        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertEqual(plan.background_model, "flat_light")
        self.assertEqual(plan.region_class, "unknown")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertTrue(plan.debug_metrics["proposal_valid"])
        self.assertFalse(plan.debug_metrics["diagnostic_only"])
        self.assertTrue(cleanup_production_patch_allowed(plan))

    def test_manual_review_proposal_does_not_accept_production_patch(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="review",
            inpaint_method="skip",
            skip_reason="manual_review_required",
            text_mask=mask,
            cleanup_mask=mask,
            text_mask_confidence=0.9,
        )
        result = img.copy()

        execute_cleanup_plan(img, result, plan)
        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertFalse(plan.debug_metrics["proposal_valid"])
        self.assertEqual(plan.debug_metrics["proposal_failure_reason"], "manual_review_required")
        self.assertTrue(plan.debug_metrics["diagnostic_only"])
        self.assertFalse(plan.debug_metrics["destructive_cleanup_executed"])
        self.assertFalse(plan.debug_metrics["production_patch_accepted"])
        self.assertFalse(plan.debug_metrics["cleanup_effective"])
        self.assertFalse(cleanup_production_patch_allowed(plan))

    def test_visible_rectangular_fill_patch_marks_cleanup_non_effective(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[50:120, 45:215] = 255
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            text_mask=mask,
            cleanup_mask=mask,
            text_mask_confidence=0.9,
            container_mask=np.ones((100, 180), dtype=np.uint8) * 255,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
            text_bbox=(45, 50, 170, 70),
        )
        result = img.copy()

        execute_cleanup_plan(img, result, plan)
        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertFalse(plan.debug_metrics["proposal_valid"])
        self.assertTrue(plan.debug_metrics["fill_patch_visible"])
        self.assertFalse(plan.debug_metrics["visual_quality_ok"])
        self.assertFalse(plan.debug_metrics["cleanup_effective"])
        self.assertFalse(cleanup_production_patch_allowed(plan))

    def test_flat_fill_boundary_constraint_shrinks_unsupported_side_band(self):
        img = _white_bubble_page()
        text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(text_mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        cleanup_mask = text_mask.copy()
        cleanup_mask[60:118, 45:56] = 255
        container = np.zeros((100, 180), dtype=np.uint8)
        cv2.ellipse(container, (90, 50), (88, 45), 0, 0, 360, 255, -1)
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            text_mask=text_mask,
            cleanup_mask=cleanup_mask,
            text_mask_confidence=0.9,
            container_mask=container,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
            text_bbox=(40, 40, 180, 100),
        )

        constrained = _constrain_flat_fill_boundary_mask(img, plan, cleanup_mask)

        self.assertTrue(plan.debug_metrics["preclean_boundary_damage_risk"])
        self.assertTrue(plan.debug_metrics["boundary_mask_constrained"])
        self.assertLess(int(np.count_nonzero(constrained[60:118, 45:56])), 64)
        self.assertGreater(int(np.count_nonzero(constrained & text_mask)), int(np.count_nonzero(text_mask)) * 0.82)

    def test_boundary_damage_marks_visual_quality_failure_without_residual_text(self):
        img = _white_bubble_page()
        text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(text_mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        cleanup_mask = text_mask.copy()
        cleanup_mask[60:118, 45:56] = 255
        container = np.zeros((100, 180), dtype=np.uint8)
        cv2.ellipse(container, (90, 50), (88, 45), 0, 0, 360, 255, -1)
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            text_mask=text_mask,
            cleanup_mask=cleanup_mask,
            text_mask_confidence=0.9,
            container_mask=container,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
            text_bbox=(40, 40, 180, 100),
        )
        result = img.copy()
        result[text_mask > 0] = (252, 252, 252)
        result[60:118, 45:56] = (232, 232, 232)

        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertTrue(plan.debug_metrics["text_removed"])
        self.assertFalse(plan.debug_metrics["residual_text_visible"])
        self.assertTrue(plan.debug_metrics["fill_patch_visible"])
        self.assertFalse(plan.debug_metrics["visual_quality_ok"])
        self.assertFalse(plan.debug_metrics["cleanup_effective"])
        self.assertEqual(plan.debug_metrics["cleanup_failure_reason"], "cleanup_fill_patch_visible")
        self.assertTrue(plan.debug_metrics["visual_boundary_damage_risk"])
        self.assertFalse(cleanup_production_patch_allowed(plan))

    def test_boundary_fill_patch_takes_precedence_over_non_authoritative_residual(self):
        img = _white_bubble_page()
        text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(text_mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        cleanup_mask = text_mask.copy()
        cleanup_mask[58:118, 44:54] = 255
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            skip_reason="cleanup_residual_text_remains",
            text_mask=text_mask,
            cleanup_mask=cleanup_mask,
            text_mask_confidence=0.9,
            container_mask=np.ones((100, 180), dtype=np.uint8) * 255,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
            text_bbox=(82, 75, 80, 30),
        )
        plan.debug_metrics["residual_score"] = {"bad": True}
        result = img.copy()
        result[text_mask > 0] = (252, 252, 252)
        result[58:118, 44:54] = (232, 232, 232)

        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertTrue(plan.debug_metrics["fill_patch_visible"])
        self.assertFalse(plan.debug_metrics["visual_quality_ok"])
        self.assertFalse(plan.debug_metrics["residual_text_visible"])
        self.assertEqual(plan.debug_metrics["cleanup_failure_reason"], "cleanup_fill_patch_visible")
        self.assertEqual(plan.skip_reason, "cleanup_fill_patch_visible")
        self.assertTrue(plan.debug_metrics["residual_suppressed_by_fill_patch"])
        self.assertFalse(cleanup_production_patch_allowed(plan))

    def test_residual_component_verifier_retries_tiny_leftover_dot(self):
        img = _white_bubble_page()
        cv2.circle(img, (164, 100), 2, (0, 0, 0), -1, cv2.LINE_AA)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            text_mask=mask,
            cleanup_mask=mask.copy(),
            text_mask_confidence=0.9,
            container_mask=np.ones((100, 180), dtype=np.uint8) * 255,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
            text_bbox=(82, 75, 80, 30),
        )
        result = img.copy()

        execute_cleanup_plan(img, result, plan)
        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertTrue(plan.debug_metrics.get("residual_component_retry_used", False))
        self.assertEqual(plan.debug_metrics.get("residual_component_count"), 0)
        self.assertFalse(plan.debug_metrics["residual_text_visible"])
        self.assertTrue(plan.debug_metrics["proposal_valid"])
        self.assertTrue(cleanup_production_patch_allowed(plan))

    def test_glyph_shaped_changed_components_do_not_trigger_patch_failure(self):
        img = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100))
        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        result = img.copy()

        execute_cleanup_plan(img, result, plan)
        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertFalse(plan.debug_metrics["fill_patch_visible"])
        self.assertTrue(plan.debug_metrics["visual_quality_ok"])
        self.assertTrue(plan.debug_metrics["cleanup_effective"])

    def test_diagnostic_cleanup_ran_does_not_set_production_gate_violation(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            skip_reason="manual_review_required",
            text_mask=mask,
            cleanup_mask=mask,
            text_mask_confidence=0.9,
            container_mask=np.ones((100, 180), dtype=np.uint8) * 255,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
        )
        plan.debug_metrics["proposal_failure_reason"] = "manual_review_required"
        result = img.copy()
        result[mask > 0] = (252, 252, 252)

        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertFalse(plan.debug_metrics["proposal_valid"])
        self.assertTrue(plan.debug_metrics["diagnostic_only"])
        self.assertTrue(plan.debug_metrics["diagnostic_cleanup_ran"])
        self.assertTrue(plan.debug_metrics["destructive_cleanup_executed"])
        self.assertFalse(plan.debug_metrics["production_patch_accepted"])
        self.assertFalse(plan.debug_metrics["gate_violation"])
        self.assertEqual(plan.debug_metrics["proposal_failure_reason"], "manual_review_required")

    def test_cleanup_reason_does_not_overwrite_manual_proposal_reason(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            text_mask=mask,
            cleanup_mask=mask,
            text_mask_confidence=0.9,
            container_mask=np.ones((100, 180), dtype=np.uint8) * 255,
            container_bbox=(40, 40, 180, 100),
            container_confidence=0.9,
        )
        plan.debug_metrics["proposal_failure_reason"] = "manual_review_required"
        plan.debug_metrics["residual_score"] = {"bad": True}
        result = img.copy()
        result[mask > 0] = (252, 252, 252)

        validate_cleanup_proposal(img, result, plan, destructive_allowed=True, validation_source="test")

        self.assertEqual(plan.debug_metrics["proposal_failure_reason"], "manual_review_required")
        self.assertEqual(plan.debug_metrics["cleanup_failure_reason"], "cleanup_residual_text_remains")
        self.assertTrue(plan.debug_metrics["residual_text_visible"])
        self.assertFalse(plan.debug_metrics["cleanup_effective"])

    def test_cleanup_outcome_clamp_blocks_invalid_proposal(self):
        outcome = {
            "proposal_valid": False,
            "production_patch_accepted": True,
            "cleanup_effective": True,
            "residual_text_visible": False,
            "visual_quality_ok": True,
            "fill_patch_visible": False,
            "gate_violation": False,
            "manual_visual_success": True,
        }

        clamp_cleanup_outcome_fields(outcome)

        self.assertFalse(outcome["cleanup_effective"])
        self.assertFalse(outcome["production_patch_accepted"])
        self.assertEqual(outcome["proposal_failure_reason"], "cleanup_proposal_invalid")

    def test_cleanup_outcome_clamp_blocks_residual_text(self):
        outcome = {
            "proposal_valid": True,
            "cleanup_effective": True,
            "residual_text_visible": True,
            "visual_quality_ok": True,
            "fill_patch_visible": False,
            "gate_violation": False,
            "manual_visual_success": True,
        }

        clamp_cleanup_outcome_fields(outcome)

        self.assertFalse(outcome["cleanup_effective"])
        self.assertTrue(outcome["cleanup_partial"])
        self.assertEqual(outcome["cleanup_failure_reason"], "cleanup_residual_text_remains")

    def test_cleanup_outcome_clamp_blocks_fill_patch(self):
        outcome = {
            "proposal_valid": True,
            "cleanup_effective": True,
            "residual_text_visible": False,
            "visual_quality_ok": True,
            "fill_patch_visible": True,
            "gate_violation": False,
            "manual_visual_success": True,
        }

        clamp_cleanup_outcome_fields(outcome)

        self.assertFalse(outcome["cleanup_effective"])
        self.assertFalse(outcome["visual_quality_ok"])
        self.assertEqual(outcome["cleanup_failure_reason"], "cleanup_fill_patch_visible")

    def test_cleanup_outcome_clamp_blocks_visual_quality_failure(self):
        outcome = {
            "proposal_valid": True,
            "cleanup_effective": True,
            "residual_text_visible": False,
            "visual_quality_ok": False,
            "fill_patch_visible": False,
            "gate_violation": False,
            "manual_visual_success": True,
        }

        clamp_cleanup_outcome_fields(outcome)

        self.assertFalse(outcome["cleanup_effective"])
        self.assertEqual(outcome["cleanup_failure_reason"], "visual_quality_failed")

    def test_cleanup_outcome_clamp_blocks_gate_violation(self):
        outcome = {
            "proposal_valid": True,
            "cleanup_effective": True,
            "residual_text_visible": False,
            "visual_quality_ok": True,
            "fill_patch_visible": False,
            "gate_violation": True,
            "manual_visual_success": True,
        }

        clamp_cleanup_outcome_fields(outcome)

        self.assertFalse(outcome["cleanup_effective"])
        self.assertEqual(outcome["cleanup_failure_reason"], "cleanup_gate_violation")

    def test_solid_colored_bubble_uses_sampled_color_not_white(self):
        fill_bgr = np.array([210, 185, 235], dtype=np.uint8)
        img = _solid_colored_bubble_page(tuple(int(v) for v in fill_bgr))
        block = _dialogue_block((40, 40, 180, 100), kind=RegionKind.PLAIN_BUBBLE)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        result = img.copy()
        execute_cleanup_plan(img, result, plan)

        self.assertEqual(plan.background_model, "flat_colored")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertEqual(plan.debug_metrics.get("flat_fill_color_source"), "ladder_border_median")
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)
        cleaned = result[plan.cleanup_mask > 0].reshape(-1, 3)
        self.assertLess(float(np.linalg.norm(np.median(cleaned, axis=0) - fill_bgr)), 8.0)

    def test_yellowed_scan_bubble_uses_local_median_fill(self):
        fill_bgr = np.array([238, 232, 214], dtype=np.uint8)
        img = _yellowed_scan_bubble_page(tuple(int(v) for v in fill_bgr))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (82, 99), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        block = _dialogue_block_with_mask((40, 40, 180, 100), mask)

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        result = img.copy()
        execute_cleanup_plan(img, result, plan)

        self.assertIn(plan.background_model, {"flat_light", "flat_colored"})
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertEqual(plan.debug_metrics.get("flat_fill_color_source"), "ladder_border_median")
        cleaned = result[plan.cleanup_mask > 0].reshape(-1, 3)
        self.assertLess(float(np.linalg.norm(np.median(cleaned, axis=0) - fill_bgr)), 14.0)

    def test_dark_caption_fill_samples_dark_background(self):
        fill_bgr = np.array([24, 24, 24], dtype=np.uint8)
        img = _dark_caption_page(tuple(int(v) for v in fill_bgr))
        block = _caption_block((40, 45, 181, 51))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        result = img.copy()
        execute_cleanup_plan(img, result, plan)

        self.assertEqual(plan.background_model, "dark_bubble")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertEqual(plan.debug_metrics.get("flat_fill_color_source"), "ladder_border_median")
        self.assertLess(max(plan.debug_metrics.get("flat_fill_ladder_fill_bgr", [255])), 80)
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)
        cleaned = result[plan.cleanup_mask > 0].reshape(-1, 3)
        self.assertLess(float(np.linalg.norm(np.median(cleaned, axis=0) - fill_bgr)), 8.0)

    def test_dark_caption_light_text_uses_tight_mask_candidate(self):
        fill_bgr = np.array([18, 54, 42], dtype=np.uint8)
        img = _dark_caption_page(tuple(int(v) for v in fill_bgr))
        block = _caption_block((40, 45, 181, 51))
        block.text_mask = None

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.background_model, "dark_bubble")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIn("dark_caption_light_text", plan.text_mask_reason)
        self.assertGreaterEqual(plan.text_mask_confidence, 0.25)
        self.assertLess(plan.debug_metrics["quality"]["mask_container_ratio"], 0.35)

    def test_low_confidence_dark_caption_still_uses_flat_fill(self):
        strategy, method = select_strategy("caption_box", "dark_bubble", 0.22, 0.20)

        self.assertEqual(strategy, "flat_fill")
        self.assertEqual(method, "local_sample")

    def test_forced_telea_accepts_bool_mask(self):
        img = np.full((32, 32, 3), 220, np.uint8)
        img[10:20, 10:20] = 20
        result = img.copy()
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True

        _execute_telea(result, mask)

        self.assertGreater(int(np.count_nonzero(result != img)), 0)

    def test_partial_caption_container_does_not_clip_cleanup_mask(self):
        img = _dark_caption_page((24, 24, 24))
        block = _caption_block((40, 45, 181, 51))
        partial_container = np.zeros((51, 181), dtype=np.uint8)
        partial_container[:, 90:] = 255
        block.bubble_mask = partial_container

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertIsNotNone(plan.cleanup_mask)
        self.assertEqual(plan.debug_metrics.get("container_confine_ignored"), "partial_container_mask")
        self.assertGreaterEqual(
            int(np.count_nonzero(plan.cleanup_mask)),
            int(np.count_nonzero(plan.text_mask)) - 64,
        )

    def test_dark_caption_red_glow_is_included_in_cleanup_mask(self):
        img = _dark_caption_glow_page()
        block = _caption_block((40, 45, 181, 51))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")
        result = img.copy()
        execute_cleanup_plan(img, result, plan)

        self.assertEqual(plan.background_model, "dark_bubble")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)
        self.assertGreater(plan.debug_metrics["halo_added_px"], 0)
        mask_pixels = img[plan.cleanup_mask > 0].reshape(-1, 3)
        self.assertGreater(int(np.count_nonzero(mask_pixels[:, 2] > 160)), 0)
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)

    def test_flat_black_text_over_art_uses_solid_fill_when_allowed(self):
        cfg = ModelConfig()
        cfg.auto_clean_text_over_art = True
        cfg.cleanup_allow_text_over_art = True
        img = _flat_black_text_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "TEXT", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 3, cv2.LINE_AA)
        block = _protected_art_block((35, 35, 175, 55), RegionKind.DIALOGUE_OVER_ART, "text_on_art")
        block.text_mask = mask
        block.bubble_mask = None
        block.bubble_bbox = None

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.region_class, "text_on_art")
        self.assertEqual(plan.background_model, "dark_bubble")
        self.assertEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIsNotNone(plan.cleanup_mask)

    def test_mild_translucent_caption_uses_deterministic_gradient_cleanup(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_translucent_caption = True
        cfg.allow_gradient_fill = True
        img = _translucent_caption_page(detailed=False)
        block = _caption_block((30, 28, 201, 91))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertIn(plan.background_model, {"smooth_gradient", "translucent_gradient", "flat_colored", "dark_bubble"})
        self.assertIn(plan.cleanup_strategy, {"gradient_fill", "flat_fill"})
        self.assertNotEqual(plan.cleanup_backend, "iopaint")

    def test_detailed_translucent_caption_routes_to_iopaint_or_review(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_translucent_caption = True
        cfg.allow_gradient_fill = True
        cfg.allow_texture_inpaint = True
        cfg.cleanup_prefer_iopaint_for_translucent = True
        cfg.iopaint_url = "http://127.0.0.1:9/inpaint"
        img = _translucent_caption_page(detailed=True)
        block = _caption_block((30, 28, 201, 91))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertNotEqual(plan.cleanup_strategy, "flat_fill")
        self.assertIn(plan.cleanup_strategy, {"texture_clone", "review", "skip"})
        if plan.background_model == "translucent_gradient" and plan.cleanup_strategy == "texture_clone":
            self.assertEqual(plan.cleanup_backend, "iopaint")

    def test_halftone_caption_does_not_use_flat_fill(self):
        cfg = ModelConfig()
        cfg.cleanup_allow_texture_inpaint = True
        img = _halftone_bubble_page()
        block = _caption_block((40, 40, 180, 100))

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01", model_config=cfg)

        self.assertEqual(plan.background_model, "halftone_texture")
        self.assertNotEqual(plan.cleanup_strategy, "flat_fill")
        self.assertNotEqual(plan.inpaint_method, "local_sample")

    def test_gradient_color_plane_writes_only_cleanup_mask_pixels(self):
        img = np.zeros((100, 140, 3), np.uint8)
        yy, xx = np.indices(img.shape[:2])
        img[:, :, 0] = np.clip(40 + xx * 0.7 + yy * 0.2, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(80 + xx * 0.3 + yy * 0.5, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(120 + xx * 0.2 + yy * 0.4, 0, 255).astype(np.uint8)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[42:54, 62:76] = 255
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(20, 20, 100, 60),
            region_class="speech_bubble",
            background_model="smooth_gradient",
            cleanup_strategy="gradient_fill",
            inpaint_method="idw_lab",
            cleanup_mask=mask,
            container_mask=np.ones((60, 100), dtype=np.uint8) * 255,
            container_bbox=(20, 20, 100, 60),
            container_confidence=0.9,
            text_mask_confidence=0.9,
            text_bbox=(62, 42, 14, 12),
        )
        result = img.copy()

        execute_cleanup_plan(img, result, plan)

        self.assertEqual(plan.debug_metrics.get("gradient_color_plane"), "ok")
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)

    def test_opencv_inpaint_writes_only_cleanup_mask_pixels(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[80:95, 95:120] = 255
        for method in ("telea", "ns"):
            with self.subTest(method=method):
                plan = CleanupPlan(
                    region_id="R-01",
                    region_bbox=(40, 40, 180, 100),
                    region_class="speech_bubble",
                    background_model="busy_art",
                    cleanup_strategy="mask_inpaint",
                    inpaint_method=method,
                    cleanup_mask=mask,
                )
                result = img.copy()

                execute_cleanup_plan(img, result, plan)

                _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)

    def test_rectangular_yolo_container_does_not_authorize_full_fill(self):
        img = _white_bubble_page()
        block = _dialogue_block((342, 82, 310, 182), text="")

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertIn("no_ocr_text_for_cleanup", plan.skip_reason)
        self.assertIsNone(plan.cleanup_mask)
        self.assertIn(plan.cleanup_strategy, {"skip", "review"})

    def test_colored_gradient_bubble_never_forces_white_rectangle(self):
        img = np.full((200, 300, 3), 245, np.uint8)
        for x in range(60, 241):
            t = (x - 60) / 180
            color = (int(210 - 50 * t), int(160 + 30 * t), int(230 - 20 * t))
            cv2.line(img, (x, 55), (x, 145), color, 1)
        bubble = np.zeros((200, 300), np.uint8)
        cv2.ellipse(bubble, (150, 100), (90, 45), 0, 0, 360, 255, -1)
        page = np.full_like(img, 245)
        page[bubble > 0] = img[bubble > 0]
        cv2.putText(page, "COLOR", (86, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 40, 170), 2, cv2.LINE_AA)
        block = _dialogue_block((50, 45, 200, 110), text="컬러", kind=RegionKind.GRADIENT_BUBBLE)

        plan = build_cleanup_plan(page, block, page_index=0, region_id="R-01")

        self.assertNotEqual(plan.cleanup_strategy, "texture_clone")
        if plan.cleanup_mask is None:
            self.assertIn(plan.cleanup_strategy, {"skip", "review"})
        else:
            self.assertLess(plan.debug_metrics["mask"]["mask_region_ratio"], 0.25)
            self.assertNotEqual(plan.debug_metrics["mask"]["mask_bbox"], plan.region_bbox)

    def test_sfx_uses_tight_mask_inpaint_or_skips(self):
        rng = np.random.default_rng(1)
        img = rng.integers(80, 180, (160, 260, 3), dtype=np.uint8)
        cv2.putText(img, "BAM", (55, 95), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, "BAM", (55, 95), cv2.FONT_HERSHEY_DUPLEX, 1.8, (20, 30, 220), 2, cv2.LINE_AA)
        block = OCRBlock(
            text="쾅",
            boxes=[],
            confidence=0.9,
            detector_source="yolo",
            bubble_bbox=(35, 35, 185, 85),
            bubble_role="sfx",
            region_kind=RegionKind.SFX_OVER_ART,
        )
        block.bbox_override = (35, 35, 185, 85)
        block.yolo_kind = "sfx"

        plan = build_cleanup_plan(img, block, page_index=0, region_id="R-01")

        self.assertEqual(plan.region_class, "sfx")
        self.assertNotEqual(plan.cleanup_strategy, "flat_fill")
        if plan.cleanup_mask is not None:
            self.assertEqual(plan.cleanup_strategy, "mask_inpaint")
            self.assertLess(plan.debug_metrics["mask"]["mask_region_ratio"], 0.25)
            self.assertNotEqual(plan.debug_metrics["mask"]["mask_bbox"], plan.region_bbox)
        else:
            self.assertIn(plan.cleanup_strategy, {"skip", "review"})

    def test_qwen_coordinate_like_fields_are_ignored(self):
        class FakeClient:
            def chat_json(self, **kwargs):
                return {
                    "text_blocks": [
                        {
                            "source_text": "안녕",
                            "role": "dialogue",
                            "confidence": 0.91,
                            "reading_order": 1,
                            "spatial_hint": "center",
                            "notes": "plain",
                            "bbox": [1, 2, 3, 4],
                            "polygon": [[1, 2], [3, 4]],
                        }
                    ],
                }

        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.client = FakeClient()
        engine.model_config = SimpleNamespace(
            qwen_ocr_model="qwen3-vl:8b",
            ocr_model="qwen3-vl:8b",
            vision_model="qwen3-vl:8b",
            keep_alive="5m",
            easyocr_fallback_enabled=False,
        )
        engine._ocr_proc = None
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")
        engine._regions = [block]

        text = engine._ocr_one_region(0)

        self.assertEqual(text, "안녕")
        self.assertEqual(block.boxes, [])
        self.assertEqual(block.text_mask, None)
        self.assertNotIn("bbox", block.qwen_text_blocks[0])
        self.assertNotIn("polygon", block.qwen_text_blocks[0])

    def test_multiple_qwen_blocks_are_review_flagged(self):
        class FakeClient:
            def chat_json(self, **kwargs):
                return {
                    "text_blocks": [
                        {"source_text": "하나", "role": "dialogue", "confidence": 0.8, "reading_order": 2, "spatial_hint": "bottom", "notes": ""},
                        {"source_text": "둘", "role": "dialogue", "confidence": 0.8, "reading_order": 1, "spatial_hint": "top", "notes": ""},
                    ],
                }

        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.client = FakeClient()
        engine.model_config = SimpleNamespace(qwen_ocr_model="qwen3-vl:8b", keep_alive="5m", easyocr_fallback_enabled=False)
        engine._ocr_proc = None
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")
        engine._regions = [block]

        text = engine._ocr_one_region(0)

        self.assertEqual(text, "둘 하나")
        self.assertTrue(block.is_flagged)
        self.assertEqual(block.review.flag_reason, "multiple_qwen_text_blocks")

    def test_manual_move_preserves_cleaned_layer_and_invalidates_typeset_only(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        cleaned = np.full((10, 10, 3), 127, np.uint8)
        page = ChapterPage(image_path="synthetic.png", cleaned_cv=cleaned.copy(), typeset_pil=Image.new("RGB", (10, 10)))
        engine.chapter_mgr = SimpleNamespace(current_page=page)
        engine._regions = [_dialogue_block((1, 1, 5, 5))]

        engine._invalidate_page_outputs(preserve_cleanup=True)

        self.assertIsNotNone(page.cleaned_cv)
        self.assertTrue(np.array_equal(page.cleaned_cv, cleaned))
        self.assertIsNone(page.typeset_pil)
        self.assertTrue(page.render_dirty)

    def test_easyocr_is_disabled_by_default(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.ocr_backend, "cascade")
        self.assertFalse(cfg.easyocr_fallback_enabled)

    def test_paddleocr_cascade_uses_fast_result_without_qwen(self):
        class FakeClient:
            def chat_json(self, **kwargs):
                raise AssertionError("Qwen should not be called for confident PaddleOCR")

        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.client = FakeClient()
        engine.model_config = SimpleNamespace(
            ocr_backend="cascade",
            paddleocr_service_url="",
            paddleocr_lang="korean",
            ocr_vlm_fallback_confidence=0.70,
            ocr_cache_enabled=True,
        )
        engine._ocr_cache = {}
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")
        engine._regions = [block]
        calls = {"paddle": 0}

        def fake_paddle(_crop):
            calls["paddle"] += 1
            return {"ok": True, "text": "안녕", "confidence": 0.93, "boxes": []}

        engine._run_paddleocr_on_crop = fake_paddle

        self.assertEqual(engine._ocr_one_region(0), "안녕")
        self.assertEqual(engine._ocr_one_region(0), "안녕")
        self.assertEqual(calls["paddle"], 1)
        self.assertEqual(block.ocr_backend, "paddleocr")
        self.assertGreaterEqual(block.ocr_confidence, 0.9)

    def test_paddleocr_cascade_falls_back_to_qwen_on_low_confidence(self):
        class FakeClient:
            def chat_json(self, **kwargs):
                return {
                    "text_blocks": [
                        {"source_text": "정답", "role": "dialogue", "confidence": 0.91, "reading_order": 1, "spatial_hint": "center", "notes": ""}
                    ],
                }

        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.client = FakeClient()
        engine.model_config = SimpleNamespace(
            ocr_backend="cascade",
            qwen_ocr_model="qwen3-vl:8b",
            ocr_model="qwen3-vl:8b",
            vision_model="qwen3-vl:8b",
            keep_alive="5m",
            paddleocr_service_url="",
            paddleocr_lang="korean",
            ocr_vlm_fallback_confidence=0.70,
            ocr_cache_enabled=True,
            easyocr_fallback_enabled=False,
        )
        engine._ocr_cache = {}
        engine._ocr_proc = None
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")
        engine._regions = [block]
        engine._run_paddleocr_on_crop = lambda _crop: {"ok": True, "text": "저", "confidence": 0.20, "boxes": []}

        text = engine._ocr_one_region(0)

        self.assertEqual(text, "정답")
        self.assertEqual(block.ocr_backend, "qwen_vl")

    def test_cross_page_ocr_uses_paddleocr_before_qwen(self):
        class FakeClient:
            def chat_json(self, **kwargs):
                raise AssertionError("Qwen should not be called for PaddleOCR cross-page success")

        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.client = FakeClient()
        engine.model_config = SimpleNamespace(
            ocr_backend="paddleocr",
            paddleocr_service_url="",
            paddleocr_lang="korean",
            ocr_vlm_fallback_confidence=0.70,
            ocr_cache_enabled=True,
        )
        engine._ocr_cache = {}
        calls = {"paddle": 0}

        def fake_paddle(_crop):
            calls["paddle"] += 1
            return {"ok": True, "text": "이어진 말", "confidence": 0.88, "boxes": []}

        engine._run_paddleocr_on_crop = fake_paddle
        crop = _white_bubble_page()

        self.assertEqual(engine._ocr_composite_crop(crop, "cp-test"), "이어진 말")
        self.assertEqual(engine._ocr_composite_crop(crop, "cp-test"), "이어진 말")
        self.assertEqual(calls["paddle"], 1)

    def test_cross_page_pair_only_mutates_after_stitched_ocr_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            page0_path = os.path.join(tmp, "p0.png")
            page1_path = os.path.join(tmp, "p1.png")
            cv2.imwrite(page0_path, _white_bubble_page())
            cv2.imwrite(page1_path, _white_bubble_page())

            engine = LocalizerEngine.__new__(LocalizerEngine)
            engine.model_config = SimpleNamespace(cross_page_merge_sfx=False)
            engine._raw_cv = cv2.imread(page0_path)
            primary = _dialogue_block((80, 140, 90, 38), text="앞")
            secondary = _dialogue_block((80, 0, 90, 38), text="뒤")
            engine._regions = [primary]
            engine._translations = ["old primary"]
            engine.chapter_mgr = ChapterManager()
            engine.chapter_mgr.pages = [
                ChapterPage(image_path=page0_path),
                ChapterPage(image_path=page1_path),
            ]
            engine.chapter_mgr.pages[0].regions = engine._regions
            engine.chapter_mgr.pages[0].translations = engine._translations
            engine.chapter_mgr.pages[1].regions = [secondary]
            engine.chapter_mgr.pages[1].translations = ["stale secondary"]
            engine.chapter_mgr.current_idx = 0
            engine._ocr_composite_crop = lambda *_args, **_kwargs: ""

            engine._try_cross_page_ocr_groups(0)

            self.assertFalse(primary.cross_page)
            self.assertFalse(secondary.cross_page)
            self.assertEqual(primary.text, "앞")
            self.assertEqual(secondary.text, "뒤")
            self.assertEqual(engine.chapter_mgr.pages[1].translations[0], "stale secondary")

            engine._ocr_composite_crop = lambda *_args, **_kwargs: "앞 뒤"
            engine._try_cross_page_ocr_groups(0)

            self.assertTrue(primary.cross_page)
            self.assertEqual(primary.text, "앞 뒤")
            self.assertTrue(engine._is_cross_page_secondary(secondary))
            self.assertEqual(secondary.text, "")
            self.assertEqual(engine.chapter_mgr.pages[1].translations[0], "")
            self.assertEqual(primary.cross_page_pages, [0, 1])
            self.assertIn(0, primary.page_local_bboxes)
            self.assertIn(1, primary.page_local_bboxes)

    def test_cross_page_candidates_ignore_mid_page_masks(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.chapter_mgr = ChapterManager()
        engine.chapter_mgr.current_idx = 0
        engine._raw_cv = _white_bubble_page()
        mid_page = _dialogue_block((80, 70, 90, 38), text="middle")

        candidates = engine._log_cross_page_candidates(0, [mid_page], engine._raw_cv)

        self.assertEqual(candidates["top"], [])
        self.assertEqual(candidates["bottom"], [])

    def test_cross_page_secondary_cannot_be_reused_as_next_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                path = os.path.join(tmp, f"p{i}.png")
                cv2.imwrite(path, _white_bubble_page())
                paths.append(path)

            engine = LocalizerEngine.__new__(LocalizerEngine)
            engine.model_config = SimpleNamespace(cross_page_merge_sfx=False)
            engine.chapter_mgr = ChapterManager()
            engine.chapter_mgr.pages = [ChapterPage(image_path=p) for p in paths]
            p0_bottom = _dialogue_block((80, 140, 90, 38), text="p0")
            p1_top = _dialogue_block((80, 0, 90, 38), text="p1 top")
            p1_bottom = _dialogue_block((80, 140, 90, 38), text="p1 bottom")
            p2_top = _dialogue_block((80, 0, 90, 38), text="p2")
            engine.chapter_mgr.pages[0].regions = [p0_bottom]
            engine.chapter_mgr.pages[0].translations = [""]
            engine.chapter_mgr.pages[1].regions = [p1_top, p1_bottom]
            engine.chapter_mgr.pages[1].translations = ["stale", ""]
            engine.chapter_mgr.pages[2].regions = [p2_top]
            engine.chapter_mgr.pages[2].translations = [""]
            engine._ocr_composite_crop = lambda *_args, **_kwargs: "stitched"

            engine.chapter_mgr.current_idx = 0
            engine._raw_cv = cv2.imread(paths[0])
            engine._regions = engine.chapter_mgr.pages[0].regions
            engine._translations = engine.chapter_mgr.pages[0].translations
            engine._try_cross_page_ocr_groups(0)

            self.assertTrue(engine._is_cross_page_secondary(p1_top))

            engine.chapter_mgr.current_idx = 1
            engine._raw_cv = cv2.imread(paths[1])
            engine._regions = engine.chapter_mgr.pages[1].regions
            engine._translations = engine.chapter_mgr.pages[1].translations
            engine._try_cross_page_ocr_groups(1)

            self.assertTrue(engine._is_cross_page_secondary(p1_top))
            self.assertFalse(p1_top.cross_page_group_id == p1_bottom.cross_page_group_id)
            self.assertTrue(p1_bottom.cross_page)
            self.assertTrue(engine._is_cross_page_secondary(p2_top))

    def test_cross_page_primary_bbox_edit_updates_page_local_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            page0_path = os.path.join(tmp, "p0.png")
            page1_path = os.path.join(tmp, "p1.png")
            cv2.imwrite(page0_path, _white_bubble_page())
            cv2.imwrite(page1_path, _white_bubble_page())

            engine = LocalizerEngine.__new__(LocalizerEngine)
            engine.model_config = ModelConfig()
            engine._raw_cv = cv2.imread(page0_path)
            engine._progress_ctx = {}
            engine._active_op_depth = 0
            engine._active_ops = set()
            engine._active_op_names = []
            engine._undo_stack = []
            engine._memory_hits = {}
            engine._consistency_warnings = []
            engine._notify = lambda *args, **kwargs: None
            block = _dialogue_block((80, 140, 90, 38), text="stitched")
            block.cross_page = True
            block.cross_page_group_id = "cp-0-0-1-0"
            block.cross_page_pages = [0, 1]
            block.page_local_bboxes = {0: (80, 140, 90, 38), 1: (80, 0, 90, 38)}
            block.composite_bbox = (80, 140, 90, 78)
            engine._regions = [block]
            engine._translations = ["combined"]
            engine.chapter_mgr = ChapterManager()
            engine.chapter_mgr.pages = [
                ChapterPage(image_path=page0_path),
                ChapterPage(image_path=page1_path),
            ]
            engine.chapter_mgr.pages[0].regions = engine._regions
            engine.chapter_mgr.pages[0].translations = engine._translations
            engine.chapter_mgr.pages[1].regions = []
            engine.chapter_mgr.pages[1].translations = []
            engine.chapter_mgr.current_idx = 0
            engine.chapter_mgr.request_save_state = lambda: None
            engine.get_bootstrap = lambda: {"ok": True}

            engine.update_region_bbox(0, 90, 10, 90, 38, page_idx=1)

            self.assertEqual(block.page_local_bboxes[1], (90, 10, 90, 38))
            self.assertEqual(block.page_local_bboxes[0], (90, 142, 90, 38))
            self.assertEqual(block.bbox(), (90, 142, 90, 38))
            self.assertEqual(block.composite_bbox, (90, 142, 90, 86))

    def test_cross_page_cleanup_without_context_does_not_recurse(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = ModelConfig()
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100))
        block.cross_page = True
        engine._regions = [block]
        engine.chapter_mgr = ChapterManager()
        engine.chapter_mgr.pages = [ChapterPage(image_path="")]
        engine.chapter_mgr.current_idx = 0
        engine._cross_page_context_for_bbox = lambda *args, **kwargs: (None, {}, [], (0, 0, 0, 0), {})
        engine._update_cross_page_metadata = lambda *args, **kwargs: None

        run = engine._run_selected_region_cleanup(0, mutate_block=False)

        self.assertIn("plan", run)
        self.assertNotEqual(run.get("cross_page"), True)

    def test_typeset_uses_large_safe_rect_for_yolo_bubble(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        block = _dialogue_block((80, 80, 100, 50))
        block.detector_source = "yolo"
        block.bubble_mask = np.ones((50, 100), dtype=np.uint8) * 255
        block.safe_rect = (50, 55, 120, 80)

        self.assertEqual(engine._get_typeset_box(block), (50, 55, 120, 80))

    def test_typeset_rejects_inflated_yolo_safe_rect(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        block = _dialogue_block((317, 747, 282, 199))
        block.detector_source = "yolo"
        block.bbox_override = (317, 747, 282, 199)
        block.bubble_bbox = (64, 701, 583, 424)
        block.bubble_mask = np.ones((199, 282), dtype=np.uint8) * 255
        block.safe_rect = (64, 701, 583, 424)

        self.assertEqual(engine._get_typeset_box(block), (317, 747, 282, 199))

    def test_typeset_rejects_inflated_cleanup_safe_rect(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        block = _dialogue_block((104, 1332, 463, 91))
        block.detector_source = "yolo"
        block.bbox_override = (104, 1332, 463, 91)
        block.bubble_bbox = (0, 1287, 690, 182)
        block.cleanup_safe_rect = (105, 1359, 460, 922)
        block.cleanup_safe_rect_confidence = 0.88

        self.assertEqual(engine._get_typeset_box(block), (104, 1332, 463, 91))

    def test_cross_page_secondary_survives_serialization(self):
        block = _dialogue_block((40, 40, 180, 100))
        block.cross_page = True
        block.cross_page_group_id = "cp-0-0-1-0"
        block.cross_page_pages = [0, 1]
        block.composite_bbox = (40, 160, 180, 120)
        block.page_local_bboxes = {0: (40, 160, 180, 20), 1: (40, 0, 180, 100)}
        block.cleanup_meta["cross_page_secondary"] = True
        block.typeset_meta["cross_page_secondary"] = True

        restored = _block_from_dict(_block_to_dict(block))
        engine = LocalizerEngine.__new__(LocalizerEngine)

        self.assertTrue(engine._is_cross_page_secondary(restored))
        self.assertEqual(restored.cross_page_group_id, "cp-0-0-1-0")
        self.assertEqual(restored.cross_page_pages, [0, 1])
        self.assertEqual(restored.composite_bbox, (40, 160, 180, 120))
        self.assertEqual(restored.page_local_bboxes[0], (40, 160, 180, 20))
        self.assertEqual(restored.page_local_bboxes[1], (40, 0, 180, 100))

    def test_typeset_noops_when_only_sfx_is_skipped(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = SimpleNamespace(process_sfx_regions=False)
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100))
        block.bubble_role = "sfx"
        block.yolo_kind = "sfx"
        block.yolo_class_id = 2
        engine._regions = [block]
        engine._translations = [""]
        engine.chapter_mgr = ChapterManager()
        engine.chapter_mgr.pages = [ChapterPage(image_path="")]
        engine.chapter_mgr.current_idx = 0
        engine.chapter_mgr.save_state = lambda: None
        engine._progress_ctx = {}
        engine._notify = lambda *args, **kwargs: None
        engine._flush_working_state_to_page = lambda: None
        engine.get_bootstrap = lambda: {"ok": True}

        resp = engine.typeset_current_page()

        self.assertTrue(resp["ok"])
        self.assertIsNotNone(engine.chapter_mgr.current_page.typeset_pil)
        self.assertFalse(engine.chapter_mgr.current_page.render_dirty)

    def test_typeset_noops_when_regions_have_no_ocr_text(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = SimpleNamespace(process_sfx_regions=False)
        engine._raw_cv = _white_bubble_page()
        block = _dialogue_block((40, 40, 180, 100), text="")
        engine._regions = [block]
        engine._translations = [""]
        engine.chapter_mgr = ChapterManager()
        engine.chapter_mgr.pages = [ChapterPage(image_path="")]
        engine.chapter_mgr.current_idx = 0
        engine.chapter_mgr.save_state = lambda: None
        engine._progress_ctx = {}
        engine._notify = lambda *args, **kwargs: None
        engine._flush_working_state_to_page = lambda: None
        engine.get_bootstrap = lambda: {"ok": True}

        resp = engine.typeset_current_page()

        self.assertTrue(resp["ok"])
        self.assertIsNotNone(engine.chapter_mgr.current_page.typeset_pil)
        self.assertFalse(engine.chapter_mgr.current_page.render_dirty)

    def test_ocr_noops_after_detecting_zero_regions(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = SimpleNamespace()
        engine._regions = []
        engine._translations = []
        engine.chapter_mgr = ChapterManager()
        engine.chapter_mgr.pages = [ChapterPage(image_path="", detected=True)]
        engine.chapter_mgr.current_idx = 0
        engine.chapter_mgr.save_state = lambda: None
        engine._progress_ctx = {}
        engine._notify = lambda *args, **kwargs: None
        engine.get_bootstrap = lambda: {"ok": True}

        resp = engine.ocr_current_page()

        self.assertTrue(resp["ok"])

    def test_translation_slots_trim_to_region_count(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine._regions = [_dialogue_block((40, 40, 180, 100))]
        engine._translations = ["current", "stale"]

        engine._sync_translation_slots()

        self.assertEqual(engine._translations, ["current"])

    def test_run_all_processes_all_pages_phase_first(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        calls = []
        chapter_mgr = SimpleNamespace(
            pages=[ChapterPage(image_path="p0.png"), ChapterPage(image_path="p1.png")],
            current_idx=0,
            total_pages=lambda: 2,
            save_state=lambda: None,
            save_run_all_checkpoint=lambda checkpoint: None,
            clear_run_all_checkpoint=lambda: None,
        )
        engine.chapter_mgr = chapter_mgr
        engine._progress_ctx = {}
        engine._active_op_depth = 0
        engine._active_ops = set()
        engine._notify = lambda *args, **kwargs: None
        engine._set_progress = lambda **updates: engine._progress_ctx.update(updates)
        engine._begin_active_operation = lambda name: None
        engine._end_active_operation = lambda name: None
        engine.get_bootstrap = lambda: {"ok": True}

        def go_to_page(idx):
            chapter_mgr.current_idx = idx
            calls.append(("go", idx))

        engine.go_to_page = go_to_page
        for step in ("detect", "ocr", "translate", "cleanup", "typeset"):
            setattr(
                engine,
                f"{step}_current_page" if step != "detect" else "detect_current_page",
                lambda step=step: calls.append((step, chapter_mgr.current_idx)) or {"ok": True},
            )

        engine._run_all_steps_from(None, 0)

        step_calls = [call for call in calls if call[0] != "go"]
        self.assertEqual(
            step_calls,
            [
                ("detect", 0), ("detect", 1),
                ("ocr", 0), ("ocr", 1),
                ("translate", 0), ("translate", 1),
                ("cleanup", 0), ("cleanup", 1),
                ("typeset", 0), ("typeset", 1),
            ],
        )

    def test_missing_required_translation_keeps_cross_page_primary(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = SimpleNamespace(process_sfx_regions=False)
        block = _dialogue_block((40, 40, 180, 100), text="테스트")
        block.cross_page = True
        engine._regions = [block]
        engine._translations = [""]

        self.assertEqual(engine._missing_required_translation_indices(), [0])

    def test_missing_required_translation_skips_cross_page_secondary(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = SimpleNamespace(process_sfx_regions=False)
        block = _dialogue_block((40, 40, 180, 100), text="테스트")
        block.cross_page = True
        block.cleanup_meta["cross_page_secondary"] = True
        engine._regions = [block]
        engine._translations = [""]

        self.assertEqual(engine._missing_required_translation_indices(), [])

    def test_missing_required_translation_skips_sfx_when_disabled(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.model_config = SimpleNamespace(process_sfx_regions=False)
        block = _dialogue_block((40, 40, 180, 100), text="쾅", kind=RegionKind.SFX_OVER_ART)
        block.bubble_role = "sfx"
        block.yolo_kind = "sfx"
        engine._regions = [block]
        engine._translations = [""]

        self.assertEqual(engine._missing_required_translation_indices(), [])

    def test_api_forwards_page_indices_to_engine(self):
        sys.modules.setdefault("webview", SimpleNamespace(windows=[]))
        from backend.api import PywebviewAPI

        calls = []

        class FakeEngine:
            busy = False

            def update_region_field(self, *args):
                calls.append(("field", args))
                return {"ok": True}

            def update_region_bbox(self, *args):
                calls.append(("bbox", args))
                return {"ok": True}

            def ocr_region(self, *args):
                calls.append(("ocr", args))
                return {"ok": True}

        api = PywebviewAPI(FakeEngine())

        self.assertTrue(api.update_region(2, "text", "x", 5)["ok"])
        self.assertTrue(api.update_region_bbox(2, 1, 2, 3, 4, 5, 6)["ok"])
        self.assertTrue(api.ocr_region(2, 5)["ok"])

        self.assertEqual(calls[0], ("field", (2, "text", "x", 5)))
        self.assertEqual(calls[1], ("bbox", (2, 1, 2, 3, 4, 5, 6)))
        self.assertEqual(calls[2], ("ocr", (2, 5)))

    def test_run_all_flags_missing_translation_and_continues(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        calls = []
        block = _dialogue_block((40, 40, 180, 100), text="테스트")
        page = ChapterPage(image_path="p0.png")
        page.regions = [block]
        page.translations = [""]
        chapter_mgr = SimpleNamespace(
            pages=[page],
            current_page=page,
            current_idx=0,
            total_pages=lambda: 1,
            save_state=lambda: None,
            save_run_all_checkpoint=lambda checkpoint: None,
            clear_run_all_checkpoint=lambda: None,
        )
        engine.chapter_mgr = chapter_mgr
        engine.model_config = ModelConfig()
        engine._raw_cv = _white_bubble_page()
        engine._regions = page.regions
        engine._translations = page.translations
        engine._progress_ctx = {}
        engine._active_op_depth = 0
        engine._active_ops = set()
        engine._notify = lambda *args, **kwargs: None
        engine._set_progress = lambda **updates: engine._progress_ctx.update(updates)
        engine._begin_active_operation = lambda name: None
        engine._end_active_operation = lambda name: None
        engine._flush_working_state_to_page = lambda: setattr(page, "translations", list(engine._translations))
        engine.get_bootstrap = lambda: {"ok": True}
        engine.go_to_page = lambda idx: calls.append(("go", idx))
        engine.detect_current_page = lambda: calls.append(("detect", 0)) or {"ok": True}
        engine.ocr_current_page = lambda: calls.append(("ocr", 0)) or {"ok": True}
        engine.translate_current_page = lambda: calls.append(("translate", 0)) or {"ok": True}
        engine.cleanup_current_page = lambda: calls.append(("cleanup", 0)) or {"ok": True}
        engine.typeset_current_page = lambda: calls.append(("typeset", 0)) or {"ok": True}

        resp = engine._run_all_steps_from(None, 0)

        self.assertTrue(resp["ok"])
        self.assertIn(("cleanup", 0), calls)
        self.assertIsNotNone(page.typeset_pil)
        self.assertTrue(block.is_flagged)
        self.assertEqual(block.review.flag_reason, "empty_translation")

    def test_run_all_catches_translate_empty_failure_and_continues(self):
        engine = LocalizerEngine.__new__(LocalizerEngine)
        calls = []
        block = _dialogue_block((40, 40, 180, 100), text="테스트")
        page = ChapterPage(image_path="p0.png")
        page.regions = [block]
        page.translations = [""]
        chapter_mgr = SimpleNamespace(
            pages=[page],
            current_page=page,
            current_idx=0,
            total_pages=lambda: 1,
            save_state=lambda: None,
            save_run_all_checkpoint=lambda checkpoint: None,
            clear_run_all_checkpoint=lambda: None,
        )
        engine.chapter_mgr = chapter_mgr
        engine.model_config = ModelConfig()
        engine._raw_cv = _white_bubble_page()
        engine._regions = page.regions
        engine._translations = page.translations
        engine._progress_ctx = {}
        engine._active_op_depth = 0
        engine._active_ops = set()
        engine._notify = lambda *args, **kwargs: None
        engine._set_progress = lambda **updates: engine._progress_ctx.update(updates)
        engine._begin_active_operation = lambda name: None
        engine._end_active_operation = lambda name: None
        engine._flush_working_state_to_page = lambda: setattr(page, "translations", list(engine._translations))
        engine.get_bootstrap = lambda: {"ok": True}
        engine.go_to_page = lambda idx: calls.append(("go", idx))
        engine.detect_current_page = lambda: calls.append(("detect", 0)) or {"ok": True}
        engine.ocr_current_page = lambda: calls.append(("ocr", 0)) or {"ok": True}
        def fail_translate():
            calls.append(("translate", 0))
            raise RuntimeError("Translation failed: OCR text exists but translation is empty for region(s) 1.")
        engine.translate_current_page = fail_translate
        engine.cleanup_current_page = lambda: calls.append(("cleanup", 0)) or {"ok": True}
        engine.typeset_current_page = lambda: calls.append(("typeset", 0)) or {"ok": True}

        resp = engine._run_all_steps_from(None, 0)

        self.assertTrue(resp["ok"])
        self.assertIn(("cleanup", 0), calls)
        self.assertIsNotNone(page.typeset_pil)
        self.assertTrue(block.is_flagged)
        self.assertEqual(block.review.flag_reason, "empty_translation")

    def test_sfx_font_fallback_uses_bold_when_primary_fails_sanity(self):
        calls = []
        default_font = ImageFont.load_default()

        class FakeFontLib:
            def get(self, role, size):
                calls.append((role, size))
                return default_font

            def get_by_name(self, name, size):
                calls.append((name, size))
                return default_font

        engine = LocalizerEngine.__new__(LocalizerEngine)
        engine.font_lib = FakeFontLib()
        engine._font_visible_sanity = lambda font: (False, "forced_failure")

        font = engine._load_fit_font("sfx", 18)

        self.assertIs(font, default_font)
        self.assertEqual(calls, [("sfx", 18), ("bold", 18)])

    def test_iopaint_client_posts_json_base64(self):
        from backend.core.iopaint_client import call_iopaint_inpaint

        img = np.zeros((20, 30, 3), dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        out_bgr = np.full_like(img, 200)

        def _fake_post(url, json=None, timeout=None):
            self.assertEqual(url, "http://127.0.0.1:8080/api/v1/inpaint")
            self.assertIn("image", json)
            self.assertIn("mask", json)
            self.assertNotIn("files", json or {})
            ok, buf = cv2.imencode(".png", out_bgr)
            resp = types.SimpleNamespace()
            resp.content = buf.tobytes()
            resp.raise_for_status = lambda: None
            return resp

        with patch("backend.core.iopaint_client.requests.post", side_effect=_fake_post):
            decoded = call_iopaint_inpaint(
                "http://127.0.0.1:8080/api/v1/inpaint", img, mask, timeout=3
            )
        self.assertTrue(np.array_equal(decoded, out_bgr))

    def test_iopaint_failure_falls_back_to_opencv(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[80:95, 95:120] = 255
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="busy_art",
            cleanup_strategy="mask_inpaint",
            inpaint_method="telea",
            cleanup_mask=mask,
            cleanup_backend="iopaint",
            iopaint_url="http://127.0.0.1:9/inpaint",
        )
        result = img.copy()

        with patch("backend.core.iopaint_client.requests.post", side_effect=RuntimeError("unreachable")):
            execute_cleanup_plan(img, result, plan)

        self.assertIn("iopaint_fallback", plan.debug_metrics["cleanup_backend_fallback"])
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)

    def test_force_telea_bypasses_external_backend(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[80:95, 95:120] = 255
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="busy_art",
            cleanup_strategy="mask_inpaint",
            inpaint_method="telea",
            cleanup_mask=mask,
            cleanup_backend="lama_pt",
        )
        plan.debug_metrics["cleanup_override_mode"] = "force_telea"
        result = img.copy()
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "simple_lama_inpainting":
                raise AssertionError("force_telea should not import LaMa")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            execute_cleanup_plan(img, result, plan)

        self.assertNotIn("cleanup_backend_fallback", plan.debug_metrics)
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)

    def test_lama_backend_runs_before_flat_fill_for_solid_bubbles(self):
        img = _white_bubble_page()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[80:95, 95:120] = 255
        plan = CleanupPlan(
            region_id="R-01",
            region_bbox=(40, 40, 180, 100),
            region_class="speech_bubble",
            background_model="flat_light",
            cleanup_strategy="flat_fill",
            inpaint_method="local_sample",
            cleanup_mask=mask,
            cleanup_backend="lama_pt",
        )
        result = img.copy()

        def fake_lama(img_cv, out, cleanup_mask, cleanup_plan):
            out[cleanup_mask > 0] = (12, 34, 56)
            cleanup_plan.debug_metrics["fake_lama_used"] = True
            return True

        with patch("backend.core.cleanup_plan._try_external_inpaint_backend", side_effect=fake_lama):
            execute_cleanup_plan(img, result, plan)

        self.assertTrue(plan.debug_metrics.get("fake_lama_used"))
        self.assertFalse(np.array_equal(img[mask > 0], result[mask > 0]))
        _assert_changed_pixels_within_mask(self, img, result, plan.cleanup_mask)

    def test_sam2_purges_conflicting_torchvision_namespace(self):
        names = [name for name in list(sys.modules) if name in {"torch", "torchvision", "sam2"} or name.startswith(("torch.", "torchvision.", "sam2."))]
        saved = {name: sys.modules[name] for name in names}
        for name in names:
            sys.modules.pop(name, None)
        try:
            torch_mod = types.ModuleType("torch")
            torch_mod.__file__ = "C:/other/site-packages/torch/__init__.py"
            vision_mod = types.ModuleType("torchvision")
            vision_mod.__file__ = "C:/other/site-packages/torchvision/__init__.py"
            sam2_mod = types.ModuleType("sam2")
            sam2_mod.__file__ = str(sam2_mask._project_root() / "external" / "sam2" / "sam2" / "__init__.py")
            sys.modules["torch"] = torch_mod
            sys.modules["torchvision"] = vision_mod
            sys.modules["sam2"] = sam2_mod

            removed = sam2_mask._purge_conflicting_imports(sam2_mask._project_root() / "external" / "sam2")

            self.assertIn("torch", removed)
            self.assertIn("torchvision", removed)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("torchvision", sys.modules)
            self.assertIn("sam2", sys.modules)
        finally:
            for name in [name for name in list(sys.modules) if name in {"torch", "torchvision", "sam2"} or name.startswith(("torch.", "torchvision.", "sam2."))]:
                sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_sam2_import_error_reports_module_origins(self):
        err = sam2_mask._sam2_import_error(RuntimeError("boom"), retry_removed={"torchvision": "C:/bad/torchvision/__init__.py"})

        self.assertIn("Could not import torch/SAM2", err)
        self.assertIn("Retried after clearing conflicting modules", err)
        self.assertIn("Restart the app", err)

    def test_sam2_auto_device_does_not_probe_cuda(self):
        class BrokenCuda:
            @staticmethod
            def is_available():
                raise RuntimeError("cudnnGetLibConfig")

        fake_torch = SimpleNamespace(cuda=BrokenCuda())

        self.assertEqual(sam2_mask._select_device(fake_torch, "auto"), "cpu")

    def test_yolo_overlap_merge_unions_duplicate_regions(self):
        detections = [
            {"box": [100, 100, 120, 80], "score": 0.80, "class_id": 0, "merged_count": 1},
            {"box": [115, 110, 118, 78], "score": 0.92, "class_id": 0, "merged_count": 1},
            {"box": [400, 100, 90, 70], "score": 0.70, "class_id": 0, "merged_count": 1},
        ]

        merged = YoloV8RegionDetector._merge_overlapping_yolo_detections(detections)

        self.assertEqual(len(merged), 2)
        union = next(item for item in merged if item["merged_count"] == 2)
        self.assertEqual(union["box"], [100, 100, 133, 88])
        self.assertEqual(union["class_id"], 0)
        self.assertAlmostEqual(union["score"], 0.92)


if __name__ == "__main__":
    unittest.main()
