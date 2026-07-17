"""Load documents (PDF / TXT / MD) and split them into overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config


@dataclass
class Chunk:
    """A slice of a source document, ready to embed."""

    text: str
    source: str  # file name
    page: int | None = None


def _clean(text: str) -> str:
    """Collapse whitespace so chunking is stable across PDF extractors."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` tuples for a PDF (1-indexed pages)."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def read_text(path: Path) -> list[tuple[int, str]]:
    """Return a single ``(None, text)`` entry for a plain-text/markdown file."""
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    return [(None, text)] if text else []


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, preferring natural boundaries.

    Each window is up to ``size`` characters. We try to cut on a paragraph or
    sentence boundary near the end of the window, then advance so consecutive
    windows share about ``overlap`` characters. ``end`` (the true cut point in
    the source) drives advancement, so shrinking a window for readability never
    stalls forward progress.
    """
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    overlap = min(overlap, size // 2)  # keep overlap sane vs. size
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)

        # Try to end on a natural boundary in the back half of the window.
        if end < n:
            best = -1
            for sep in ("\n\n", ". ", ".\n", "\n", " "):
                idx = text.rfind(sep, start + size // 2, end)
                if idx != -1:
                    best = max(best, idx + len(sep))
            if best != -1:
                end = best

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)  # guarantee progress
    return chunks


def load_documents(documents_dir: Path | None = None) -> list[Chunk]:
    """Load and chunk every supported file in ``documents_dir``."""
    documents_dir = documents_dir or config.DOCUMENTS_DIR
    chunks: list[Chunk] = []

    if not documents_dir.exists():
        return chunks

    for path in sorted(documents_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = read_pdf(path)
        elif suffix in {".txt", ".md"}:
            pages = read_text(path)
        else:
            continue

        for page, text in pages:
            for piece in split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append(Chunk(text=piece, source=path.name, page=page))

    return chunks


def chunk_bytes(data: bytes, filename: str) -> list[Chunk]:
    """Chunk an uploaded file provided as raw bytes (used by the UI uploader)."""
    import io

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [
            (i, _clean(p.extract_text() or ""))
            for i, p in enumerate(reader.pages, start=1)
        ]
        pages = [(i, t) for i, t in pages if t]
    elif suffix in {".txt", ".md"}:
        pages = [(None, _clean(data.decode("utf-8", errors="ignore")))]
        pages = [(p, t) for p, t in pages if t]
    else:
        return []

    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append(Chunk(text=piece, source=filename, page=page))
    return chunks
