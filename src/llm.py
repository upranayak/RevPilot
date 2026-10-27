"""Minimal Groq client (OpenAI-compatible chat completions with tool calling)."""
import os
import time
from pathlib import Path

import httpx

from config import GROQ_BASE_URL


class LLMError(RuntimeError):
    pass


key_error = ""  # why the key could not be read (error type only, never the value)


def api_key() -> str:
    """Locally the key comes from .env (loaded in config), so st.secrets is never touched there.
    On Streamlit Cloud it comes from the app's Secrets."""
    global key_error
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        key_error = "" if key else "GROQ_API_KEY not found in secrets"
    except Exception as e:
        key, key_error = "", type(e).__name__
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
