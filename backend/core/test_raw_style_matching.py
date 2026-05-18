import json
import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from backend.core.project import ChapterManager
from backend.core.regions import OCRBlock, RegionKind, TextStyle, _block_from_dict, _block_to_dict
from backend.engine import LocalizerEngine


def _region_block(
    bbox=(30, 25, 170, 70),
    *,
    role="dialog",
    kind=RegionKind.PLAIN_BUBBLE,
    text="테스트",
) -> OCRBlock:
    x, y, w, h = bbox
    block = OCRBlock(
        text=text,
        boxes=[[
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ]],
        confidence=0.95,
        detector_source="yolo",
        bubble_role=role,
        region_kind=kind,
    )
    block.bbox_override = bbox
    block.bubble_bbox = bbox
    return block


def _raw_dialogue() -> np.ndarray:
    img = np.full((130, 240, 3), 255, np.uint8)
    cv2.putText(img, "TEXT", (48, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def _mixed_style_raw() -> np.ndarray:
    img = np.full((130, 260, 3), 245, np.uint8)
    cv2.putText(img, "TE", (42, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "XT", (108, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (20, 20, 220), 3, cv2.LINE_AA)
    return img


def _engine_with_raw(img: np.ndarray) -> LocalizerEngine:
    engine = LocalizerEngine.__new__(LocalizerEngine)
    engine._raw_cv = img
    return engine


def _fixture_block(meta) -> OCRBlock:
    x, y, w, h = [int(v) for v in meta["bbox"]]
    block = OCRBlock(
        text="테스트",
        boxes=[[
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ]],
        confidence=0.95,
        detector_source="fixture",
        bubble_role=str(meta.get("role") or "dialog"),
        region_kind=getattr(RegionKind, str(meta.get("kind") or "PLAIN_BUBBLE")),
    )
    block.bbox_override = (x, y, w, h)
    block.bubble_bbox = (x, y, w, h)
    return block


def _raw_style_fixtures():
    root = os.path.join(os.path.dirname(__file__), "raw_style_fixtures")
    with open(os.path.join(root, "manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return root, manifest["fixtures"]


class RawStyleMatchingTests(unittest.TestCase):
    def test_high_confidence_dialogue_auto_applies(self) -> None:
        engine = _engine_with_raw(_raw_dialogue())
        block = _region_block()

        self.assertTrue(engine._auto_apply_raw_style_if_safe(block, force_analysis=True))
        self.assertIsNotNone(block.style)
        self.assertEqual(block.style.source, "raw:auto")
        self.assertEqual(block.raw_style_match.get("status"), "high")
        self.assertTrue(block.raw_style_match.get("auto_applied"))

    def test_low_confidence_falls_back(self) -> None:
        engine = _engine_with_raw(np.full((120, 220, 3), 255, np.uint8))
        block = _region_block()

        match = engine._analyze_raw_style_for_block(block, force=True)

        self.assertEqual(match.get("status"), "fallback")
        self.assertIn("insufficient_glyph_pixels", match.get("downgrade_reasons", []))
        self.assertFalse(engine._auto_apply_raw_style_if_safe(block, force_analysis=False))

    def test_manual_style_override_is_not_overwritten(self) -> None:
        engine = _engine_with_raw(_raw_dialogue())
        block = _region_block()
        block.style = TextStyle(fg_color=(220, 0, 0), source="manual")

        self.assertFalse(engine._auto_apply_raw_style_if_safe(block, force_analysis=True))
        self.assertEqual(block.style.source, "manual")
        self.assertEqual(block.style.fg_color, (220, 0, 0))

    def test_sfx_is_proposal_only_and_not_auto_applied(self) -> None:
        engine = _engine_with_raw(_raw_dialogue())
        block = _region_block(role="sfx", kind=RegionKind.SFX_OVER_ART)

        self.assertFalse(engine._auto_apply_raw_style_if_safe(block, force_analysis=True))
        self.assertIn(block.raw_style_match.get("status"), {"medium", "fallback"})
        self.assertIn("unsafe_role", block.raw_style_match.get("downgrade_reasons", []))
        self.assertIsNone(block.style)

    def test_mixed_styles_downgrade_confidence(self) -> None:
        engine = _engine_with_raw(_mixed_style_raw())
        block = _region_block(bbox=(30, 25, 205, 70))

        match = engine._analyze_raw_style_for_block(block, force=True)

        self.assertIn("mixed_styles", match.get("downgrade_reasons", []))
        self.assertNotEqual(match.get("status"), "high")

    def test_raw_style_fixture_manifest_regression(self) -> None:
        root, fixtures = _raw_style_fixtures()

        for meta in fixtures:
            with self.subTest(meta["id"]):
                img = cv2.imread(os.path.join(root, meta["file"]))
                self.assertIsNotNone(img)
                engine = _engine_with_raw(img)
                block = _fixture_block(meta)

                match = engine._analyze_raw_style_for_block(block, force=True)
                self.assertIn(match.get("status"), set(meta.get("expect_status") or []))
                for effect in meta.get("expect_matched") or []:
                    self.assertIn(effect, match.get("matched", []))
                expected_reasons = set(meta.get("expect_reasons_any") or [])
                if expected_reasons:
                    self.assertTrue(expected_reasons.intersection(set(match.get("downgrade_reasons", []))))
                self.assertTrue(match.get("analysis_crop_signature"))
                self.assertEqual(match.get("review_status"), "unreviewed")

                qa = engine._raw_match_quality_summary(0, 0, block, img)
                self.assertEqual(qa.get("status"), match.get("status"))
                self.assertIn(qa.get("auto_state"), {"auto_applied", "proposed", "fallback"})

                auto_applied = engine._auto_apply_raw_style_if_safe(block, force_analysis=True)
                if meta.get("expect_auto_apply"):
                    self.assertTrue(auto_applied)
                    self.assertEqual(block.raw_style_match.get("auto_state"), "auto_applied")
                else:
                    self.assertFalse(auto_applied)
                if meta.get("expect_proposal_only"):
                    self.assertIn("unsafe_role", block.raw_style_match.get("downgrade_reasons", []))
                    self.assertNotEqual(block.raw_style_match.get("auto_state"), "auto_applied")

    def test_raw_review_metadata_and_stale_rematch_flow(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        img = _raw_dialogue()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(os.path.join(tmp.name, "000.png"))
        mgr = ChapterManager()
        self.assertEqual(mgr.load_from_folder(tmp.name), 1)
        block = _region_block()
        mgr.pages[0].regions = [block]
        mgr.pages[0].translations = [""]
        engine = LocalizerEngine()
        engine.chapter_mgr = mgr
        engine._load_page_into_working_state()

        match = engine._analyze_raw_style_for_block(block, force=True)
        self.assertTrue(match.get("analysis_crop_signature"))
        bootstrap = engine.get_bootstrap()
        self.assertIsNotNone(bootstrap["regions"][0].get("raw_match_qa"))

        engine.update_region_field(0, "raw_review_status", "accepted")
        self.assertEqual(block.raw_style_match.get("review_status"), "accepted")
        self.assertFalse(block.raw_style_match.get("needs_review"))

        engine.update_region_bbox(0, 35, 30, 170, 70)
        self.assertTrue(block.raw_style_match.get("analysis_crop_signature_stale"))
        self.assertEqual(block.raw_style_match.get("review_status"), "unreviewed")
        stale_bootstrap = engine.get_bootstrap()
        self.assertTrue(any(issue.get("kind") == "raw_match_qa" for issue in stale_bootstrap["issues"]))

        engine.update_region_field(0, "rematch_stale_raw_styles", True)
        self.assertFalse(block.raw_style_match.get("analysis_crop_signature_stale"))
        self.assertTrue(block.raw_style_match.get("analysis_crop_signature"))

    def test_geometry_and_rotation_do_not_create_typeset_override(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        Image.new("RGB", (240, 130), "white").save(os.path.join(tmp.name, "000.png"))
        mgr = ChapterManager()
        self.assertEqual(mgr.load_from_folder(tmp.name), 1)
        block = _region_block()
        mgr.pages[0].regions = [block]
        mgr.pages[0].translations = [""]
        engine = LocalizerEngine()
        engine.chapter_mgr = mgr
        engine._load_page_into_working_state()

        engine.update_region_field(0, "rotation_angle", 12)
        engine.update_region_bbox(0, 35, 30, 170, 70)

        self.assertFalse(mgr.pages[0].regions[0].typeset_override)
        self.assertAlmostEqual(mgr.pages[0].regions[0].rotation_angle, 12.0)

    def test_transform_and_reflection_serialization_defaults_and_roundtrip(self) -> None:
        old_block = _block_from_dict({
            "text": "old",
            "boxes": [[[10, 20], [80, 20], [80, 60], [10, 60]]],
            "confidence": 1.0,
        })
        self.assertEqual(old_block.rotation_angle, 0.0)
        self.assertIsNone(old_block.style)

        block = _region_block()
        block.rotation_angle = 17.5
        block.style = TextStyle(
            reflection_on=True,
            reflection_opacity=0.4,
            reflection_offset=8,
            reflection_blur=2.0,
            reflection_fade=0.9,
            shadow_on=True,
            shadow_blur=3.0,
            source="manual",
        )
        restored = _block_from_dict(_block_to_dict(block))

        self.assertAlmostEqual(restored.rotation_angle, 17.5)
        self.assertIsNotNone(restored.style)
        self.assertTrue(restored.style.reflection_on)
        self.assertAlmostEqual(restored.style.reflection_opacity, 0.4)
        self.assertEqual(restored.style.reflection_offset, 8)
        self.assertAlmostEqual(restored.style.shadow_blur, 3.0)


if __name__ == "__main__":
    unittest.main()
