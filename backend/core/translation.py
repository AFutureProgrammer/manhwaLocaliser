from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from backend.core.constants import OLLAMA_TIMEOUT, OLLAMA_URL, debug_print

try:
    import ctranslate2
    import transformers
    _HAS_CTRANSLATE = True
except ImportError:
    ctranslate2 = None
    transformers = None
    _HAS_CTRANSLATE = False

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":        {"type": "string"},
                    "description": {"type": "string"},
                    "emotion":     {"type": "string"},
                },
                "required": ["name", "description", "emotion"],
                "additionalProperties": False,
            },
        },
        "scene_description": {"type": "string"},
        "tone":              {"type": "string"},
    },
    "required": ["characters", "scene_description", "tone"],
    "additionalProperties": False,
}

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "issues":          {"type": "array", "items": {"type": "string"}},
        "corrected_lines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["issues", "corrected_lines"],
    "additionalProperties": False,
}

POLISHER_SCHEMA = {
    "type": "object",
    "properties": {
        "polished_lines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["polished_lines"],
    "additionalProperties": False,
}

TRANSLATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "translated_lines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["translated_lines"],
    "additionalProperties": False,
}

QWEN_OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "text_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_text": {"type": "string"},
                    "role": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reading_order": {"type": "integer"},
                    "spatial_hint": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": [
                    "source_text",
                    "role",
                    "confidence",
                    "reading_order",
                    "spatial_hint",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["text_blocks"],
    "additionalProperties": False,
}

class OllamaClient:
    def __init__(self, url: str = OLLAMA_URL, timeout: int = OLLAMA_TIMEOUT):
        self.url     = url
        self.timeout = timeout

    # ── Plain-text response (no JSON schema) ────────────────────────────────
    def chat_raw(
        self,
        model:      str,
        prompt:     str,
        image_b64:  Optional[str] = None,
        system:     str = "",
        keep_alive: str = "5m",
    ) -> str:
        """Send a chat request and return the plain-text response."""
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if image_b64 is not None:
            user_msg["images"] = [image_b64]
        messages.append(user_msg)
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "keep_alive": keep_alive,
        }
        debug_print(f"OllamaClient.chat_raw: model={model!r}")
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    # ── JSON-schema response ─────────────────────────────────────────────────
    def chat_json(
        self,
        model:      str,
        prompt:     str,
        schema:     Dict[str, Any],
        image_b64:  Optional[str] = None,
        system:     str = "",
        keep_alive: str = "5m",
    ) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if image_b64 is not None:
            user_msg["images"] = [image_b64]
        messages.append(user_msg)

        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "format":     schema,
            "keep_alive": keep_alive,
        }

        debug_print(f"OllamaClient.chat_json: model={model!r}, keep_alive={keep_alive!r}, url={self.url}")
        debug_print(f"OllamaClient.chat_json: prompt_length={len(prompt)}, system_length={len(system)}")
        if image_b64 is not None:
            debug_print(f"OllamaClient.chat_json: image attached, base64_length={len(image_b64)}")
        started = time.perf_counter()
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        debug_print(f"OllamaClient.chat_json: HTTP {r.status_code}")
        r.raise_for_status()
        body = r.json()
        raw = body.get("message", {}).get("content", "{}")
        debug_print(f"OllamaClient.chat_json: raw_response_length={len(raw)}")
        load_ms = int(float(body.get("load_duration", 0) or 0) / 1_000_000)
        prompt_ms = int(float(body.get("prompt_eval_duration", 0) or 0) / 1_000_000)
        eval_ms = int(float(body.get("eval_duration", 0) or 0) / 1_000_000)
        total_ms = int(float(body.get("total_duration", 0) or 0) / 1_000_000)
        debug_print(
            "OllamaClient.chat_json: timing "
            f"http_ms={elapsed_ms} total_ms={total_ms} load_ms={load_ms} "
            f"prompt_eval_ms={prompt_ms} eval_ms={eval_ms}"
        )
        try:
            data = json.loads(raw)
            debug_print("OllamaClient.chat_json: JSON parsed successfully")
            return data
        except json.JSONDecodeError:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            debug_print("OllamaClient.chat_json: raw JSON parse failed, trying cleaned response")
            data = json.loads(cleaned)
            debug_print("OllamaClient.chat_json: cleaned JSON parsed successfully")
            return data

class NLLBTranslator:
    _REQUIRED_FILES = ("model.bin", "config.json")

    def __init__(self, model_dir: str):
        debug_print(f"Initializing NLLBTranslator with model_dir={model_dir}")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"NLLB model directory not found: '{model_dir}'\n"
            )
        missing = [f for f in self._REQUIRED_FILES
                   if not os.path.isfile(os.path.join(model_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"NLLB model directory '{model_dir}' is incomplete. "
            )
        debug_print("Loading CTranslate2 translator...")
        self._model_dir = model_dir
        try:
            self.translator = ctranslate2.Translator(
                model_dir,
                device="cuda",
                compute_type="int8_float16",
                inter_threads=1,
            )
            debug_print("NLLBTranslator: running on CUDA")
        except Exception as cuda_err:
            debug_print(f"NLLBTranslator: CUDA unavailable ({cuda_err}), falling back to CPU")
            self.translator = ctranslate2.Translator(
                model_dir,
                device="cpu",
                compute_type="int8",
                inter_threads=2,
            )
            debug_print("NLLBTranslator: running on CPU")
        debug_print("Loading tokenizer...")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir,
                src_lang="kor_Hang",
                local_files_only=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir,
                src_lang="kor_Hang",
                local_files_only=True,
            )
        self.tgt_lang = "eng_Latn"

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            debug_print("translate_batch: no texts provided")
            return []

        debug_print(f"translate_batch: translating {len(texts)} text(s)")
        for idx, text in enumerate(texts):
            debug_print(f"translate_batch input[{idx}]={text!r}")
        source = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t))
            for t in texts
        ]

        debug_print("translate_batch: starting CTranslate2 inference")
        try:
            results = self.translator.translate_batch(
                source,
                target_prefix=[[self.tgt_lang]] * len(source),
                beam_size=4,               # more beams = better for short comic lines
                max_decoding_length=256,
                repetition_penalty=1.3,    # penalise repeated n-grams
                no_repeat_ngram_size=3,    # forbid any 3-gram from appearing twice
            )
        except RuntimeError as e:
            if any(k in str(e) for k in ("cublas", "cudnn", "cuda", "CUDA")):
                debug_print(f"translate_batch: CUDA runtime error ({e}), switching to CPU")
                self.translator = ctranslate2.Translator(
                    self._model_dir,
                    device="cpu",
                    compute_type="int8",
                    inter_threads=2,
                )
                debug_print("translate_batch: retrying on CPU")
                results = self.translator.translate_batch(
                    source,
                    target_prefix=[[self.tgt_lang]] * len(source),
                    beam_size=4,
                    max_decoding_length=256,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
            else:
                raise
        debug_print("translate_batch: inference complete")

        translations = []
        for idx, res in enumerate(results):
            token_ids = self.tokenizer.convert_tokens_to_ids(res.hypotheses[0][1:])
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            translations.append(text)
            debug_print(f"translate_batch[{idx}] -> {text!r}")
        return translations

