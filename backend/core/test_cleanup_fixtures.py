import json
import os
import unittest

import cv2
import numpy as np

from backend.core import constants

constants.DEBUG = False

from backend.core.cleanup_plan import (  # noqa: E402
    build_cleanup_plan,
    cleanup_production_patch_allowed,
    execute_cleanup_plan,
    validate_cleanup_proposal,
)
from backend.core.config import ModelConfig  # noqa: E402
from backend.core.regions import OCRBlock, RegionKind  # noqa: E402


FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "cleanup_fixtures")
MANIFEST_PATH = os.path.join(FIXTURE_ROOT, "manifest.json")


def _load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _model_config(meta):
    cfg = ModelConfig(**(meta.get("default_model_config") or {}))
    cfg.cleanup_debug_artifacts = False
    return cfg


def _block_from_case(case, image_shape):
    region = case["region"]
    x, y, w, h = [int(v) for v in region["bbox"]]
    block = OCRBlock(
        text=str(region.get("text") or "fixture"),
        boxes=[],
        confidence=float(region.get("confidence", 0.9)),
        detector_source=str(region.get("detector") or "fixture"),
        bubble_bbox=(x, y, w, h),
        bubble_mask=np.ones((h, w), dtype=np.uint8) * 255,
        bubble_role=str(region.get("role") or "dialog"),
        region_kind=getattr(RegionKind, str(region.get("kind") or "PLAIN_BUBBLE")),
    )
    block.bbox_override = (x, y, w, h)
    if region.get("yolo_kind"):
        block.yolo_kind = str(region.get("yolo_kind"))
    if region.get("yolo_class_id") is not None:
        block.yolo_class_id = int(region.get("yolo_class_id"))
    text_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for rx, ry, rw, rh in region.get("text_mask_rects") or []:
        x1, y1 = max(0, int(rx)), max(0, int(ry))
        x2, y2 = min(image_shape[1], x1 + int(rw)), min(image_shape[0], y1 + int(rh))
        if x2 > x1 and y2 > y1:
            text_mask[y1:y2, x1:x2] = 255
    if np.any(text_mask):
        block.text_mask = text_mask
    return block


class CleanupFixtureCorpusTests(unittest.TestCase):
    def test_manifest_documents_representative_cases(self):
        manifest = _load_manifest()
        cases = manifest.get("cases") or []
        self.assertGreaterEqual(len(cases), 10)
        seen = set()
        required = {
            "plain_white_bubble",
            "off_white_bubble",
            "colored_bubble",
            "textured_bubble",
            "dark_caption",
            "gradient_bubble",
            "halftone_bubble",
            "translucent_caption",
            "text_over_art",
            "sfx_protected",
        }
        for case in cases:
            with self.subTest(case=case.get("id")):
                case_id = str(case.get("id") or "")
                self.assertTrue(case_id)
                seen.add(case_id)
                self.assertTrue(case.get("description"))
                self.assertTrue(case.get("image"))
                self.assertIn("region", case)
                expected = case.get("expected") or {}
                self.assertTrue(expected.get("strategies"))
                self.assertTrue(expected.get("methods"))
                image_path = os.path.join(FIXTURE_ROOT, str(case["image"]))
                self.assertTrue(os.path.exists(image_path), image_path)
                self.assertLess(os.path.getsize(image_path), 150_000)
        self.assertTrue(required.issubset(seen))

    def test_cleanup_fixture_manifest_planner_bands(self):
        manifest = _load_manifest()
        cfg = _model_config(manifest)
        for case in manifest.get("cases") or []:
            with self.subTest(case=case.get("id")):
                image_path = os.path.join(FIXTURE_ROOT, str(case["image"]))
                img = cv2.imread(image_path)
                self.assertIsNotNone(img, image_path)
                block = _block_from_case(case, img.shape)
                plan = build_cleanup_plan(
                    img,
                    block,
                    page_index=0,
                    region_id=str(case.get("region_id") or "R-01"),
                    model_config=cfg,
                )
                expected = case.get("expected") or {}
                self.assertIn(plan.cleanup_strategy, set(expected.get("strategies") or []))
                self.assertIn(plan.inpaint_method, set(expected.get("methods") or []))
                skip_contains = str(expected.get("skip_contains") or "")
                if skip_contains:
                    self.assertIn(skip_contains, str(plan.skip_reason or ""))

                quality = plan.debug_metrics.get("quality", {}) or {}
                for key in ("mask_region_ratio", "mask_container_ratio"):
                    low, high = [float(v) for v in expected[key]]
                    actual = float(quality.get(key, 0.0) or 0.0)
                    self.assertGreaterEqual(actual, low, key)
                    self.assertLessEqual(actual, high, key)

                result = img.copy()
                execute_cleanup_plan(img, result, plan)
                validate_cleanup_proposal(
                    img,
                    result,
                    plan,
                    destructive_allowed=True,
                    validation_source="fixture",
                )
                self.assertEqual(
                    cleanup_production_patch_allowed(plan),
                    bool(expected.get("production_allowed", False)),
                )


if __name__ == "__main__":
    unittest.main()
