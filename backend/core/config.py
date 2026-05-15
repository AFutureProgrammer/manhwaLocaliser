from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from backend.core.constants import MODEL_CONFIG_FILE, debug_print

DEFAULT_YOLO_MODEL_PATH = (
    "external/manhwa-text-detection/models/manhwa-yolo-v8.onnx"
)
DEFAULT_SAM2_MODEL_PATH = "external/sam2"
DEFAULT_SAM2_CHECKPOINT_PATH = "external/sam2_checkpoints/sam2.1_hiera_tiny.pt"

@dataclass
class ModelConfig:
    """Holds Ollama model names.  Persisted to model_config.json."""
    ocr_model:       str = "qwen3-vl:8b"
    translate_model: str = "translategemma:12b"
    vision_model:    str = "qwen3-vl:8b"
    polisher_model:  str = "qwen2.5:7b"

    # keep_alive string sent to Ollama for each request
    keep_alive: str = "30m"

    # Region detector backend. "ocr" preserves the existing EasyOCR detector.
    # "yolo" is strict by default; fallback must be explicitly enabled.
    detector_backend: str = "yolo"
    yolo_model_path:  str = DEFAULT_YOLO_MODEL_PATH
    detector_allow_fallback: bool = False

    # OCR path. Cascade uses fast OCR first, then falls back to Qwen-VL only
    # when confidence/output quality is not good enough.
    ocr_backend: str = "cascade"
    qwen_ocr_model: str = "qwen3-vl:8b"
    paddleocr_service_url: str = ""
    paddleocr_lang: str = "korean"
    ocr_vlm_fallback_confidence: float = 0.70
    ocr_cache_enabled: bool = True
    easyocr_fallback_enabled: bool = False

    # Cleanup backend abstraction. LaMa can be wired later without making it a
    # required dependency.
    cleanup_backend: str = "lama_pt"
    iopaint_url: str = ""
    lama_model_path: str = ""
    max_tile_size: int = 1024
    cleanup_debug_artifacts: bool = False
    cleanup_debug_dir: str = ""
    auto_clean_sfx: bool = False
    auto_typeset_sfx: bool = False
    auto_clean_text_over_art: bool = False
    auto_clean_busy_background: bool = False
    require_review_for_tier2: bool = True
    allow_gradient_fill: bool = True
    allow_texture_inpaint: bool = True
    sfx_experimental_cleanup_mode: str = "off"
    busy_background_cleanup_mode: str = "off"
    cleanup_mode: str = "balanced"
    cleanup_solid_bubble_fill_enabled: bool = True
    cleanup_solid_bubble_min_container_confidence: float = 0.60
    cleanup_solid_bubble_max_mask_container_ratio: float = 0.15
    cleanup_solid_bubble_max_rectangularity: float = 0.45
    cleanup_flat_fill_ladder_enabled: bool = True
    cleanup_flat_fill_max_growth_px: int = 10
    cleanup_flat_fill_retry_extra_growth_px: int = 2
    cleanup_flat_fill_ring_px: int = 3
    cleanup_flat_fill_max_ring_gray_std: float = 14.0
    cleanup_flat_fill_max_ring_chroma_std: float = 12.0
    cleanup_flat_fill_max_ring_edge_density: float = 0.08
    cleanup_halo_mask_enabled: bool = True
    cleanup_halo_max_px: int = 2
    cleanup_residual_retry_enabled: bool = True
    cleanup_residual_retry_dilate_px: int = 1
    cleanup_allow_grouped_inpaint: bool = False
    cleanup_manual_review_only: bool = False
    cleanup_min_container_confidence: float = 0.0
    cleanup_max_mask_container_ratio: float = 0.50
    cleanup_max_mask_region_ratio: float = 0.28
    cleanup_max_border_touch_ratio: float = 0.35
    cleanup_max_rectangularity: float = 0.88
    cleanup_allow_translucent_caption: bool = False
    cleanup_allow_texture_inpaint: bool = False
    cleanup_easy_fallback_enabled: bool = False
    cleanup_easy_fallback_backend: str = "telea"
    cleanup_easy_fallback_scope: str = "bubbles"
    cleanup_allow_text_over_art: bool = False
    cleanup_allow_sfx_cleanup: bool = False
    cleanup_allow_texture_telea: bool = False
    cleanup_prefer_iopaint_for_texture: bool = False
    cleanup_prefer_iopaint_for_translucent: bool = False
    cleanup_iopaint_allow_opencv_fallback: bool = False
    cleanup_risky_action: str = "skip"
    cleanup_fallback_backend: str = "telea"
    cleanup_verbose_logs: bool = False
    cleanup_show_diagnostics: bool = False
    cleanup_candidate_timeout_sec: int = 8
    cleanup_iopaint_candidate_timeout_sec: int = 5
    cleanup_skip_unavailable_iopaint_candidate: bool = True

    # Optional SAM2 mask assist. SAM2 is never used as an inpainting backend;
    # it only proposes masks for the existing manual cleanup flow.
    sam2_enabled: bool = True
    sam2_load_mode: str = "lazy"
    sam2_required: bool = False
    sam2_backend_url: str = ""
    sam2_timeout_sec: int = 30
    sam2_model_path: str = DEFAULT_SAM2_MODEL_PATH
    sam2_checkpoint_path: str = DEFAULT_SAM2_CHECKPOINT_PATH
    sam2_device: str = "auto"
    sam2_mask_mode: str = "cleanup_assist"

    # Optional bubble/container segmentation. Disabled unless configured.
    bubble_seg_backend: str = "none"
    bubble_seg_model_path: str = ""

    minimum_dialog_font_size: int = 14
    minimum_sfx_font_size: int = 16

    # ── YOLO detector thresholds ─────────────────────────────────
    # Confidence below this is dropped before NMS. Raise to suppress
    # low-quality boxes; lower to recover missed text. 0.25 is the YOLOv8 default.
    yolo_confidence:  float = 0.25
    # IoU threshold used in class-local NMS.
    yolo_nms_iou:     float = 0.45
    yolo_training_base_model: str = "yolov8n.pt"
    yolo_training_epochs: int = 30
    yolo_training_imgsz: int = 640
    yolo_training_batch: int = 8
    yolo_training_device: str = ""

    # ── SFX handling ─────────────────────────────────────────────
    # Master toggle. When False (default), SFX/shout/sound-effect-like
    # regions are filtered out of the pipeline right after YOLO detection:
    # they are NOT OCRed, translated, cleaned, typeset, or rendered as
    # preview sprites, and the overlay hides them. Existing saved SFX
    # regions remain persisted in .ml_state.json and reappear when this
    # toggle is turned back on.
    process_sfx_regions: bool = False

    # ── Translation provider ─────────────────────────────────────
    # "ollama" uses the local Ollama stack (default).
    # "deepseek" routes to the DeepSeek HTTP API.
    translation_provider: str = "ollama"

    # DeepSeek API settings (only used when translation_provider == "deepseek")
    # API key is read from the environment variable named by deepseek_api_key_env.
    # Never put the actual key here.
    deepseek_api_key_env:  str   = "DEEPSEEK_API_KEY"
    deepseek_base_url:     str   = "https://api.deepseek.com"
    deepseek_model:        str   = "deepseek-chat"
    deepseek_timeout_sec:  int   = 60
    deepseek_temperature:  float = 0.1
    deepseek_fallback_to_ollama: bool = True

    # ── Optional FLUX.2 Klein secondary cleanup ──────────────────
    # Empty string = disabled; set to e.g. "http://localhost:7860/inpaint"
    klein_backend_url:          str   = ""
    # Cleanup strategies that qualify a region for Klein treatment
    # (erase_only and is_flagged are always eligible when Klein is enabled)
    klein_eligible_strategies:  List[str] = field(
        default_factory=lambda: ["texture_clone", "inpaint"])

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def save(self, path: str = MODEL_CONFIG_FILE) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            debug_print(f"ModelConfig saved to {path!r}")
        except Exception as e:
            debug_print(f"ModelConfig.save failed: {e}")

    @classmethod
    def load(cls, path: str = MODEL_CONFIG_FILE) -> "ModelConfig":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            obj = cls()
            for k, v in data.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            if not str(getattr(obj, "yolo_model_path", "") or "").strip():
                obj.yolo_model_path = DEFAULT_YOLO_MODEL_PATH
            if not str(getattr(obj, "sam2_model_path", "") or "").strip():
                obj.sam2_model_path = DEFAULT_SAM2_MODEL_PATH
            if not str(getattr(obj, "sam2_checkpoint_path", "") or "").strip():
                obj.sam2_checkpoint_path = DEFAULT_SAM2_CHECKPOINT_PATH
            if str(getattr(obj, "sam2_load_mode", "") or "").strip().lower() not in {"startup", "lazy"}:
                obj.sam2_load_mode = "lazy"
            if str(getattr(obj, "sam2_device", "") or "").strip().lower() not in {"auto", "cpu", "cuda", "mps"}:
                obj.sam2_device = "auto"
            ocr_backend = str(getattr(obj, "ocr_backend", "") or "cascade").strip().lower()
            if ocr_backend not in {"cascade", "qwen_vl", "paddleocr", "easyocr"}:
                ocr_backend = "cascade"
            obj.ocr_backend = ocr_backend
            if not str(getattr(obj, "paddleocr_lang", "") or "").strip():
                obj.paddleocr_lang = "korean"
            try:
                obj.ocr_vlm_fallback_confidence = float(
                    getattr(obj, "ocr_vlm_fallback_confidence", 0.70) or 0.70
                )
            except Exception:
                obj.ocr_vlm_fallback_confidence = 0.70
            debug_print(f"ModelConfig loaded: {obj.to_dict()}")
            return obj
        except FileNotFoundError:
            debug_print("model_config.json not found, using defaults")
            return cls()
        except Exception as e:
            debug_print(f"ModelConfig.load failed: {e}")
            return cls()
