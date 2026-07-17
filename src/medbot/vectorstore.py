"""A tiny, dependency-light vector store backed by NumPy + Gemini embeddings.

We avoid FAISS/torch on purpose: for the document sizes a personal medical
assistant handles, an in-memory cosine search over a normalized matrix is fast,
trivially serializable, and deploys cleanly on Streamlit Cloud.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
from google.genai import types

from . import config
from .client import get_client
from .ingest import Chunk

# Gemini embeds in batches; this keeps requests comfortably under limits.
_EMBED_BATCH = 64


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    """Embed a list of texts and return an L2-normalized float32 matrix."""
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        resp = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in resp.embeddings)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    """Holds document chunks and their normalized embeddings."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings

    # -- Construction -------------------------------------------------------
    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "VectorStore":
        if not chunks:
            return cls([], np.zeros((0, 1), dtype=np.float32))
        embeddings = _embed([c.text for c in chunks], "retrieval_document")
        return cls(chunks, embeddings)

    # -- Search -------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` most similar chunks with cosine scores."""
        top_k = top_k or config.TOP_K
        if not self.chunks:
            return []
        q = _embed([query], "retrieval_query")[0]
        scores = self.embeddings @ q  # cosine similarity (both normalized)
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    # -- Persistence --------------------------------------------------------
    def save(self, index_dir: Path | None = None) -> None:
        index_dir = index_dir or config.INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "VectorStore | None":
        index_dir = index_dir or config.INDEX_DIR
        emb_path = index_dir / "embeddings.npy"
        chunks_path = index_dir / "chunks.json"
        if not emb_path.exists() or not chunks_path.exists():
            return None
        embeddings = np.load(emb_path)
        with open(chunks_path, encoding="utf-8") as f:
            chunks = [Chunk(**c) for c in json.load(f)]
        return cls(chunks, embeddings)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def __len__(self) -> int:
        return len(self.chunks)
