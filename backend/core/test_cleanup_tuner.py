import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.cleanup_lab.cleanup_tuner import (
    load_debug_cases,
    propose_candidate_variants,
    score_case,
    synthesize_rules,
)
from tools.cleanup_lab.cleanup_lab import run_manifest


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("failed to encode png")
    encoded.tofile(str(path))


def _write_case(root: Path, page: int, region: str, *, source: str = "fallback_cv_no_bbox") -> Path:
    page_dir = root / f"page_{page:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "page_index": page,
        "region_id": region,
        "region_index": 0,
        "role": "dialog",
        "region_class": "speech_bubble",
        "chosen_cleanup_strategy": "flat_fill",
        "chosen_inpaint_method": "local_sample",
        "selected_text_mask_candidate_source": source,
        "selected_text_mask_candidate": "region_cv_no_bbox(area=0.220)",
        "mask_region_ratio": 0.39,
        "mask_container_ratio": 0.20,
        "rectangularity": 0.42,
        "cleanup_backend": "lama_pt",
        "cleanup_effective": False,
        "visual_quality_ok": True,
        "text_mask_candidates": [
            {
                "reason": "region_cv_no_bbox(area=0.220)",
                "source": source,
                "confidence": 0.48,
                "mask_px": 90,
                "accepted": True,
                "selected": True,
            }
        ],
        "debug_metrics": {
            "cleanup_route": "solid_bubble_cv",
            "quality": {
                "mask_region_ratio": 0.39,
                "mask_container_ratio": 0.20,
                "rectangularity": 0.42,
                "border_touch_ratio": 0.0,
                "component_count": 12,
                "largest_component_ratio": 0.25,
            },
        },
    }
    meta_path = page_dir / f"{region}_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    cleaned = np.full((32, 32, 3), 255, np.uint8)
    cleaned[12:15, 12:15] = 0
    cleaned[20:22, 18:21] = 0
    mask = np.zeros((32, 32), np.uint8)
    mask[8:26, 8:26] = 255
    raw = np.full((32, 32, 3), 255, np.uint8)
    _write_png(page_dir / f"{region}_raw.png", raw)
    _write_png(page_dir / f"{region}_cleaned.png", cleaned)
    _write_png(page_dir / f"{region}_text_mask.png", mask)
    _write_png(page_dir / f"{region}_cleanup_mask.png", mask)
    _write_png(page_dir / f"{region}_overlay.png", raw)
    return meta_path


class CleanupTunerTests(unittest.TestCase):
    def test_load_debug_cases_reads_meta_artifacts_and_terminal_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, 4, "R-01")
            log = root / "terminal.txt"
            log.write_text("[DEBUG] cleanup_debug_artifacts: wrote page=4 region=R-01 mask_px=123\n", encoding="utf-8")

            cases = load_debug_cases(root, log)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "p004_R-01")
            self.assertEqual(cases[0].mask_source, "fallback_cv_no_bbox")
            self.assertIn("cleaned", cases[0].artifacts)
            self.assertEqual(len(cases[0].terminal_events), 1)

    def test_score_case_flags_overbroad_fragmented_lama_flat_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, 4, "R-01")
            case = load_debug_cases(root)[0]

            score = score_case(case)

            self.assertEqual(score["verdict"], "fail")
            self.assertIn("overbroad_mask", score["reasons"])
            self.assertIn("fragmented_mask", score["reasons"])
            self.assertIn("lama_backend_routed_through_flat_fill", score["reasons"])
            self.assertGreater(score["metrics"]["residual_dark_px"], 0)

    def test_candidate_variants_are_general_rules_not_region_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, 4, "R-01")
            case = load_debug_cases(root)[0]
            score = score_case(case)

            variants = propose_candidate_variants(case, score)
            variant_ids = {item["id"] for item in variants}

            self.assertIn("lama_first_mask_inpaint", variant_ids)
            self.assertIn("reject_blocky_fallback_cv", variant_ids)
            self.assertTrue(all("R-01" not in item["description"] for item in variants))

    def test_synthesize_rules_groups_multiple_cases_by_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, 4, "R-01")
            _write_case(root, 5, "R-02")
            cases = load_debug_cases(root)
            scores = [score_case(case) for case in cases]
            candidates = {case.case_id: propose_candidate_variants(case, score) for case, score in zip(cases, scores)}

            rules = synthesize_rules(cases, scores, candidates)

            first_ids = {rule["rule_id"] for rule in rules[:3]}
            self.assertIn("lama_first_mask_inpaint", first_ids)
            lama_rule = next(rule for rule in rules if rule["rule_id"] == "lama_first_mask_inpaint")
            self.assertEqual(lama_rule["affected_count"], 2)

    def test_cleanup_lab_manifest_runs_inline_cases_and_writes_review_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.png"
            img = np.full((96, 140, 3), 245, np.uint8)
            cv2.ellipse(img, (70, 48), (46, 24), 0, 0, 360, (252, 252, 252), -1)
            cv2.putText(img, "TEXT", (38, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            _write_png(image_path, img)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({
                    "cases": [
                        {
                            "case_id": "plain_bubble",
                            "image": "page.png",
                            "region_id": "R-01",
                            "expected_outcome": "pass",
                            "region": {
                                "id": "R-01",
                                "bbox": [20, 20, 100, 56],
                                "text": "테스트",
                                "role": "dialog",
                                "kind": "PLAIN_BUBBLE",
                                "boxes": [[[38, 40], [100, 40], [100, 60], [38, 60]]],
                                "bubble_bbox": [20, 20, 100, 56],
                                "safe_rect": [24, 24, 92, 48],
                            },
                        }
                    ]
                }),
                encoding="utf-8",
            )

            result = run_manifest(manifest_path, root / "out")

            review_path = Path(result["review_path"])
            self.assertTrue(review_path.exists())
            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["summary"]["case_count"], 1)
            case = review["cases"][0]
            self.assertIn(case["final_result"], {"pass", "fail"})
            self.assertTrue((Path(case["output_dir"]) / "report.json").exists())
            self.assertTrue((Path(case["output_dir"]) / "cleaned_crop.png").exists())


if __name__ == "__main__":
    unittest.main()
