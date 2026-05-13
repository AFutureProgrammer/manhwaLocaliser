"""
backend/core/deepseek_translate.py
────────────────────────────────────
Thin wrapper around the DeepSeek HTTP chat-completion API for batch
manhwa translation.

Security rules:
  • The API key is NEVER stored in model_config.json or source code.
  • It is read at call-time from the environment variable whose *name* is
    stored in ModelConfig.deepseek_api_key_env (default: DEEPSEEK_API_KEY).
  • Raise DeepSeekConfigError if the key is missing so the caller can
    decide whether to fall back to Ollama.

Usage (called from engine._translate_texts_deepseek):
    from backend.core.deepseek_translate import translate_batch, DeepSeekConfigError
    results = translate_batch(texts, config=self.model_config, prompt_prefix=prefix)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from backend.core.constants import debug_print


class DeepSeekConfigError(RuntimeError):
    """Raised when the DeepSeek API key is absent or the config is invalid."""


class DeepSeekAPIError(RuntimeError):
    """Raised when the DeepSeek API returns a non-200 response."""


def _get_api_key(key_env: str) -> str:
    """Read API key from environment; raise DeepSeekConfigError if absent."""
    key = os.environ.get(str(key_env or "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise DeepSeekConfigError(
            f"DeepSeek API key not found. "
            f"Set the environment variable {key_env!r} before starting the app."
        )
    return key


def translate_batch(
    texts: List[str],
    config: Any,
    prompt_prefix: str = "",
) -> List[str]:
    """Translate a list of Korean strings via DeepSeek chat-completion.

    Args:
        texts:         Korean source strings (one per region).
        config:        ModelConfig instance with deepseek_* fields populated.
        prompt_prefix: Memory/glossary context block (same as Ollama path).

    Returns:
        List of English translations, same length as ``texts``.
        Empty strings are returned for positions the model skipped.

    Raises:
        DeepSeekConfigError: API key missing.
        DeepSeekAPIError:    Non-200 HTTP response after retries.
    """
    key_env    = str(getattr(config, "deepseek_api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY")
    base_url   = str(getattr(config, "deepseek_base_url", "https://api.deepseek.com") or "https://api.deepseek.com").rstrip("/")
    model      = str(getattr(config, "deepseek_model", "deepseek-chat") or "deepseek-chat")
    timeout    = int(getattr(config, "deepseek_timeout_sec", 60) or 60)
    temperature = float(getattr(config, "deepseek_temperature", 0.1) or 0.1)

    api_key = _get_api_key(key_env)

    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    system_msg = (
        "You are a professional manhwa/webtoon translator from Korean to English. "
        "Translate naturally, preserving tone, speech patterns, and character voice. "
        "Output ONLY valid JSON with a single key 'translated_lines' containing an array "
        "of translated strings in the same order as the input. "
        "Do not add commentary, explanations, or extra keys."
    )
    user_msg = (
        f"{prompt_prefix}"
        f"Translate each numbered Korean line to natural English manga dialogue.\n"
        f"Output JSON with 'translated_lines' array in the same order.\n\n"
        f"{numbered}"
    )

    # Pass 8: clamp output length so a broken / chatty response cannot
    # produce runaway output. Roughly "a sentence per source line" with a
    # floor of 1024 tokens (room for JSON scaffolding on small batches).
    max_tokens = max(1024, min(4096, 256 * max(1, len(texts))))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    debug_print(
        f"[DEEPSEEK] model={model!r} texts={len(texts)} "
        f"base_url={base_url!r} timeout={timeout}s max_tokens={max_tokens}"
    )

    def _post(p: Dict[str, Any]) -> List[Any]:
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=p,
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise DeepSeekAPIError(f"DeepSeek request timed out after {timeout}s") from exc
        except requests.exceptions.RequestException as exc:
            raise DeepSeekAPIError(f"DeepSeek request failed: {exc}") from exc
        if not resp.ok:
            raise DeepSeekAPIError(
                f"DeepSeek API error {resp.status_code}: {resp.text[:400]}"
            )
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            parsed  = json.loads(content)
            lines = parsed.get("translated_lines", [])
            if not isinstance(lines, list):
                raise ValueError("translated_lines is not an array")
            return list(lines)
        except Exception as exc:
            raise DeepSeekAPIError(f"Failed to parse DeepSeek response: {exc}") from exc

    raw_lines = _post(payload)

    # Pass 8: if the model dropped or added lines, retry once with a stricter
    # prompt that explicitly repeats the expected count. After that, we accept
    # whatever came back and pad / truncate silently.
    if len(raw_lines) != len(texts):
        debug_print(
            f"[DEEPSEEK] final count mismatch got={len(raw_lines)} want={len(texts)}; padding/truncating"
        )
        retry_user = (
            f"{prompt_prefix}"
            f"You MUST output exactly {len(texts)} items in 'translated_lines'. "
            f"Do NOT merge, split, or omit any input line. "
            f"For blank or untranslatable lines, output an empty string in that slot.\n\n"
            f"{numbered}"
        )
        retry_payload = dict(payload)
        retry_payload["messages"] = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": retry_user},
        ]
        try:
            raw_lines = _post(retry_payload)
        except DeepSeekAPIError as exc:
            debug_print(f"[DEEPSEEK] retry failed: {exc}; padding best-effort")

    results: List[str] = []
    for i, kr in enumerate(texts):
        en = raw_lines[i] if i < len(raw_lines) else ""
        en_str = str(en or "").strip()
        # Pass 8: per-line length validator — reject absurdly long outputs
        # that typically come from hallucinated explanations. Threshold:
        # english-text length <= 6x the source length, with a floor of 240
        # chars for very short KR inputs (single char -> "Huh?" is fine).
        kr_len = max(1, len(kr or ""))
        if en_str and len(en_str) > max(240, 6 * kr_len):
            debug_print(
                f"[DEEPSEEK] line {i} too long ({len(en_str)} chars vs kr={kr_len}); flagging as empty"
            )
            en_str = ""
        results.append(en_str)
    while len(results) < len(texts):
        results.append("")

    debug_print(
        f"[DEEPSEEK] done texts={len(texts)} translated={sum(1 for r in results if r)}"
    )
    return results
