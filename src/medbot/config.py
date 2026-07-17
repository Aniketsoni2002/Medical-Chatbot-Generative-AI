"""Central configuration for MedBot.

Values are read from environment variables (or Streamlit secrets, wired up in
``app.py``) so the app runs the same locally and on Streamlit Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

# --- Models ----------------------------------------------------------------
# Gemini is used for both generation and embeddings, so no heavy local model
# (torch / sentence-transformers) is required. This keeps the deploy small.
GENERATION_MODEL = os.getenv("MEDBOT_GENERATION_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("MEDBOT_EMBEDDING_MODEL", "gemini-embedding-001")

# --- Retrieval -------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("MEDBOT_CHUNK_SIZE", "1000"))      # characters
CHUNK_OVERLAP = int(os.getenv("MEDBOT_CHUNK_OVERLAP", "150"))  # characters
TOP_K = int(os.getenv("MEDBOT_TOP_K", "4"))
# Below this cosine similarity we treat the corpus as having no answer.
MIN_SIMILARITY = float(os.getenv("MEDBOT_MIN_SIMILARITY", "0.55"))

# --- Generation ------------------------------------------------------------
TEMPERATURE = float(os.getenv("MEDBOT_TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MEDBOT_MAX_OUTPUT_TOKENS", "1024"))


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment, if set."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
