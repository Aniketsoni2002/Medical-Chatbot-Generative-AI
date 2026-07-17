"""A thin wrapper around the Google GenAI client.

Centralizes auth so the rest of the codebase never touches the SDK directly.
Uses the current, supported ``google-genai`` package.
"""

from __future__ import annotations

from google import genai

from . import config

_client: genai.Client | None = None


def configure(api_key: str) -> None:
    """Initialize the shared GenAI client with an API key."""
    global _client
    _client = genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    """Return the configured client, creating one from the env if needed."""
    global _client
    if _client is None:
        key = config.get_api_key()
        if not key:
            raise RuntimeError(
                "Gemini API key not configured. Call client.configure(key) or set "
                "the GEMINI_API_KEY environment variable."
            )
        _client = genai.Client(api_key=key)
    return _client
