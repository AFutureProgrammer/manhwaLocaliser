import unittest

from backend.core.cleanup_failure_taxonomy import (
    classify_cleanup_failure,
    primary_cleanup_failure_class,
)


class CleanupFailureTaxonomyTests(unittest.TestCase):
    def test_rejected_broad_mask_maps_to_safety_and_overcleanup(self):
        classes = classify_cleanup_failure({
            "cleanup_mask_rejected": True,
            "skip_reason": "cleanup_mask_too_large_region_ratio(0.52)",
            "debug_metrics": {
                "quality": {
                    "mask_region_ratio": 0.52,
                    "mask_container_ratio": 0.74,
                    "border_touch_ratio": 0.1,
                }
            },
        })

        self.assertIn("unsafe_mask_rejection", classes)
        self.assertIn("art_damage", classes)
        self.assertIn("overcleanup", classes)

    def test_fragmented_fallback_maps_to_bad_text_mask(self):
        classes = classify_cleanup_failure({
            "debug_metrics": {
                "selected_text_mask_candidate_source": "fallback_cv_no_bbox",
                "text_mask_candidate_scores": [
                    {"rejection_reason": "fragmented_broad_fallback_cv_no_bbox"}
                ],
            }
        })

        self.assertEqual(primary_cleanup_failure_class(classes), "bad_text_mask")

    def test_residual_and_fill_patch_map_to_specific_classes(self):
        classes = classify_cleanup_failure({
            "cleanup_failure_reason": "cleanup_residual_text_remains",
            "residual_component_count": 3,
            "debug_metrics": {"fill_patch_reason": "fill_patch_component_visible"},
        })

        self.assertIn("leftover_glyphs", classes)
        self.assertIn("bad_flat_fill", classes)

    def test_backend_failure_maps_to_inpaint_backend(self):
        classes = classify_cleanup_failure({
            "cleanup_backend": "iopaint",
            "cleanup_failure_reason": "backend timeout unavailable",
        })

        self.assertIn("bad_inpaint_backend", classes)


if __name__ == "__main__":
    unittest.main()
