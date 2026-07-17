"""Tests for the parts of MedBot that don't require a live Gemini key.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import medbot.vectorstore as vs  # noqa: E402
from medbot import safety  # noqa: E402
from medbot.chatbot import MedBot  # noqa: E402
from medbot.ingest import load_documents, split_text  # noqa: E402


# --- Safety ---------------------------------------------------------------- #
def test_emergency_detected():
    assert safety.check_emergency("I have severe chest pain")
    assert safety.check_emergency("she is having trouble breathing")
    assert safety.screen("severe bleeding that won't stop") is not None


def test_crisis_detected_and_prioritized():
    msg = safety.screen("I want to kill myself")
    assert msg is not None and "988" in msg


def test_normal_question_not_flagged():
    assert safety.screen("what is a healthy BMI range?") is None


# --- Chunking -------------------------------------------------------------- #
def test_short_text_single_chunk():
    assert split_text("short text", 1000, 150) == ["short text"]


def test_empty_text_no_chunks():
    assert split_text("   ", 1000, 150) == []


def test_long_text_chunks_bounded_and_progress():
    text = "This is a test sentence. " * 500  # ~12,500 chars
    parts = split_text(text, 1000, 150)
    assert all(len(p) <= 1000 for p in parts)
    assert 10 <= len(parts) <= 20  # no runaway tiny-chunk explosion


def test_sample_documents_load():
    chunks = load_documents()
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


# --- Retrieval & engine (mocked embeddings/generation) --------------------- #
def _mock_embed(texts, task_type):
    vecs = []
    for t in texts:
        v = np.zeros(64, dtype=np.float32)
        for w in t.lower().split():
            v[hash(w) % 64] += 1.0
        vecs.append(v)
    m = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def _make_bot(monkeypatch, min_similarity=0.05):
    monkeypatch.setattr(vs, "_embed", _mock_embed)
    # The bag-of-words mock scores lower than real Gemini embeddings, so relax
    # the retrieval threshold for tests that exercise the grounded path.
    import medbot.config as cfg

    monkeypatch.setattr(cfg, "MIN_SIMILARITY", min_similarity)
    store = vs.VectorStore.from_chunks(load_documents())

    class FakeModels:
        def generate_content(self, **kw):
            class R:
                text = "General information from the documents. Consult a professional."

            return R()

    class FakeClient:
        models = FakeModels()

    bot = MedBot.__new__(MedBot)
    bot.store = store
    bot.client = FakeClient()
    return bot


def test_grounded_answer_has_sources(monkeypatch):
    bot = _make_bot(monkeypatch)
    ans = bot.answer("how is high blood pressure managed?")
    assert ans.grounded
    assert ans.sources


def test_emergency_short_circuits_llm(monkeypatch):
    bot = _make_bot(monkeypatch)
    ans = bot.answer("I have chest pain radiating to my arm")
    assert ans.is_escalation
    assert not ans.grounded


def test_out_of_scope_refuses(monkeypatch):
    # Keep the real (strict) threshold: an unrelated query must be rejected.
    bot = _make_bot(monkeypatch, min_similarity=0.55)
    ans = bot.answer("zxqwv plooble frobnicate wibblewobble snarf")
    assert not ans.grounded
    assert not ans.sources
