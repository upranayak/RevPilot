"""Minimal Groq client (OpenAI-compatible chat completions with tool calling)."""
import os
import time
from pathlib import Path

import httpx

from config import GROQ_BASE_URL


class LLMError(RuntimeError):
    pass


def _secrets_available() -> bool:
    """True on Streamlit Cloud (apps run from /mount/src) or when a secrets.toml exists.
    Checked first so st.secrets never shows its 'No secrets found' box locally."""
    candidates = [Path.home() / ".streamlit" / "secrets.toml", Path.cwd() / ".streamlit" / "secrets.toml"]
    return Path("/mount/src").exists() or any(p.exists() for p in candidates)


def api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()  # locally from .env (loaded in config)
    if not key and _secrets_available():
        try:
            import streamlit as st
            key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            key = ""
    return key


def chat(messages: list[dict], model: str, tools: list[dict] | None = None, temperature: float = 0.0,
         max_tokens: int = 4096, retries: int = 3) -> dict:
    """Return the assistant message dict. Retries on rate limits and server errors."""
    key = api_key()
    if not key:
        raise LLMError("GROQ_API_KEY is not set.")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = httpx.post(f"{GROQ_BASE_URL}/chat/completions", json=body, timeout=60,
                           headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as e:
            if attempt == retries:
                raise LLMError(f"Network error: {e}") from e
            time.sleep(delay); delay *= 2
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == retries:
                raise LLMError(f"Groq error {r.status_code}: {r.text[:200]}")
            wait = float(r.headers.get("retry-after", delay))
            time.sleep(min(wait, 30)); delay *= 2
            continue
        if r.status_code >= 400:
            raise LLMError(f"Groq error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]
    raise LLMError("Unreachable")
