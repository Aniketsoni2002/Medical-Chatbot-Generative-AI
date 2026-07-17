"""The MedBot engine: retrieve from documents, then generate a grounded answer.

Answers are constrained to the retrieved context. If the documents don't cover
the question, the bot says so rather than falling back to model memory — this is
the core requirement: responses are limited to the provided documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google.genai import types

from . import config, safety
from .client import get_client
from .ingest import Chunk
from .vectorstore import VectorStore

SYSTEM_PROMPT = """\
You are MedBot, a careful medical information assistant. You help users understand \
health topics using ONLY the reference excerpts provided to you.

Strict rules:
1. Answer USING ONLY the information in the "Context" section. Do NOT use outside \
   knowledge. If the context does not contain the answer, say clearly that the \
   provided documents don't cover it and suggest consulting a healthcare professional.
2. Never give a definitive diagnosis. Describe possibilities and general information only.
3. Never prescribe medication or tell the user to change a prescribed dose.
4. Be clear, structured, and easy to read. Use short paragraphs and bullet points.
5. Explain medical terms in plain language.
6. Always remind the user, briefly, that this is general information and not a \
   substitute for a professional consultation when giving health guidance.
7. If the user describes red-flag symptoms (e.g. chest pain, trouble breathing, \
   stroke signs, severe bleeding, thoughts of self-harm), urge them to seek \
   emergency care immediately.

Be warm, concise, and responsible."""


@dataclass
class Answer:
    """A generated answer plus the sources it was grounded in."""

    text: str
    sources: list[Chunk] = field(default_factory=list)
    is_escalation: bool = False
    grounded: bool = True


def _format_context(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(results, start=1):
        loc = f"{chunk.source}" + (f", p.{chunk.page}" if chunk.page else "")
        blocks.append(f"[Source {i} — {loc}]\n{chunk.text}")
    return "\n\n".join(blocks)


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:  # keep the last few turns for continuity
        role = "User" if turn["role"] == "user" else "MedBot"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


class MedBot:
    """Retrieval-augmented medical assistant."""

    def __init__(self, store: VectorStore):
        self.store = store
        self.client = get_client()

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        history = history or []

        # 1) Deterministic safety screen — runs before anything else.
        escalation = safety.screen(question)

        # 2) Retrieve grounding context from the documents.
        results = self.store.search(question)
        strong = [r for r in results if r[1] >= config.MIN_SIMILARITY]

        # If the escalation fired, prepend it; still try to add helpful info.
        if escalation:
            return Answer(text=escalation, sources=[], is_escalation=True, grounded=False)

        if not strong:
            msg = (
                "I couldn't find anything about that in the loaded documents, so I "
                "can't answer reliably. This assistant only responds using the "
                "medical documents provided to it.\n\n"
                "Try rephrasing, or add a document that covers this topic. For any "
                "health concern, please consult a qualified healthcare professional."
            )
            return Answer(text=msg, sources=[], grounded=False)

        # 3) Generate a grounded answer.
        context = _format_context(strong)
        convo = _format_history(history)
        prompt = (
            (f"Conversation so far:\n{convo}\n\n" if convo else "")
            + f"Context (the ONLY source you may use):\n{context}\n\n"
            + f"User question: {question}\n\n"
            + "Answer using only the context above."
        )

        response = self.client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=config.TEMPERATURE,
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
            ),
        )
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            text = (
                "I wasn't able to generate a response for that. Please try "
                "rephrasing your question."
            )

        return Answer(text=text, sources=[c for c, _ in strong], grounded=True)
