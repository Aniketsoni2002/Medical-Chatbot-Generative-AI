"""MedBot — Streamlit app.

A RAG-based medical assistant that answers strictly from your documents,
with emergency detection and a clean, modern chat interface.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (set GEMINI_API_KEY in Secrets)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the ``src`` package importable both locally and on Streamlit Cloud.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from medbot import client, config, safety  # noqa: E402
from medbot.chatbot import MedBot  # noqa: E402
from medbot.ingest import chunk_bytes, load_documents  # noqa: E402
from medbot.vectorstore import VectorStore  # noqa: E402

# --------------------------------------------------------------------------- #
# Page config & API key
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MedBot — AI Medical Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)


def resolve_api_key() -> str | None:
    """Prefer Streamlit secrets, fall back to environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return config.get_api_key()


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --brand: #0ea5a4;
            --brand-2: #2563eb;
            --bg-soft: #f0f9f9;
        }
        .stApp { background: radial-gradient(1200px 600px at 10% -10%, #e6fffb 0%, transparent 45%),
                              radial-gradient(1000px 500px at 110% 10%, #eef2ff 0%, transparent 40%); }
        .hero {
            text-align: center;
            padding: 1.6rem 1rem 0.4rem;
        }
        .hero h1 {
            font-size: 2.3rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, var(--brand), var(--brand-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .hero p { color: #475569; margin-top: .35rem; font-size: 1.02rem; }
        .pill {
            display:inline-block; padding: .28rem .8rem; border-radius: 999px;
            background: var(--bg-soft); color: #0f766e; font-size: .8rem;
            font-weight: 600; border: 1px solid #99f6e4; margin: .15rem;
        }
        .src-card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: .7rem .9rem; margin: .35rem 0; font-size: .86rem; color:#334155;
            box-shadow: 0 1px 2px rgba(0,0,0,.04);
        }
        .src-tag {
            font-weight: 700; color: var(--brand); font-size: .78rem;
        }
        .disclaimer {
            font-size: .8rem; color:#64748b; text-align:center;
            border-top: 1px dashed #cbd5e1; margin-top: 1.2rem; padding-top: .8rem;
        }
        section[data-testid="stSidebar"] { background: #ffffffcc; backdrop-filter: blur(6px); }
        .stChatInput textarea { border-radius: 14px !important; }
        .suggest-btn button {
            border-radius: 999px !important; border: 1px solid #cbd5e1 !important;
            background: #ffffff !important; color:#334155 !important;
            font-size: .84rem !important; font-weight:500 !important;
        }
        .suggest-btn button:hover { border-color: var(--brand) !important; color: var(--brand) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Knowledge base loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def build_store_from_disk(_key_fingerprint: str) -> VectorStore:
    """Build (or load) the vector store from documents on disk.

    ``_key_fingerprint`` busts the cache when the API key changes.
    """
    store = VectorStore.load()
    if store is not None and not store.is_empty:
        return store
    chunks = load_documents()
    store = VectorStore.from_chunks(chunks)
    if not store.is_empty:
        try:
            store.save()
        except Exception:
            pass  # read-only FS on some hosts — in-memory is fine
    return store


def add_uploaded_files(store: VectorStore, files) -> int:
    """Embed uploaded files and merge them into the active store. Returns count."""
    import numpy as np

    from medbot.vectorstore import _embed

    added_chunks = []
    for f in files:
        added_chunks.extend(chunk_bytes(f.getvalue(), f.name))
    if not added_chunks:
        return 0

    new_emb = _embed([c.text for c in added_chunks], "retrieval_document")
    if store.is_empty:
        store.chunks = added_chunks
        store.embeddings = new_emb
    else:
        store.chunks = store.chunks + added_chunks
        store.embeddings = np.vstack([store.embeddings, new_emb])
    return len(added_chunks)


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>🩺 MedBot</h1>
            <p>Your AI medical assistant — grounded strictly in trusted documents.</p>
            <div>
                <span class="pill">📚 Document-grounded</span>
                <span class="pill">🚨 Emergency-aware</span>
                <span class="pill">🔒 No diagnosis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = resolve_api_key()

    # ----- Sidebar --------------------------------------------------------- #
    with st.sidebar:
        st.header("⚙️ Setup")

        if not api_key:
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey",
                placeholder="AIza...",
            )
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
        else:
            st.success("Gemini API key loaded ✓")

        st.divider()
        st.subheader("📂 Knowledge base")
        uploads = st.file_uploader(
            "Add documents (PDF / TXT / MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="The bot answers only from these documents.",
        )
        index_uploads = st.button("➕ Add to knowledge base", use_container_width=True)

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "MedBot provides general health information and is not a substitute "
            "for professional medical advice."
        )

    # ----- Guard: need a key ---------------------------------------------- #
    if not api_key:
        st.info(
            "👋 **Welcome!** Enter your free **Gemini API key** in the sidebar to begin. "
            "Get one at [Google AI Studio](https://aistudio.google.com/apikey)."
        )
        st.stop()

    client.configure(api_key)

    # ----- Build / retrieve the knowledge base ---------------------------- #
    if "store" not in st.session_state:
        with st.spinner("Indexing medical documents…"):
            st.session_state.store = build_store_from_disk(api_key[:8])
    store = st.session_state.store

    if index_uploads and uploads:
        with st.spinner("Embedding your documents…"):
            n = add_uploaded_files(store, uploads)
        if n:
            st.sidebar.success(f"Added {n} chunks from {len(uploads)} file(s).")
        else:
            st.sidebar.warning("No readable text found in those files.")

    with st.sidebar:
        st.metric("Indexed chunks", len(store))

    if store.is_empty:
        st.warning(
            "No documents are loaded yet. Upload a medical PDF or text file in the "
            "sidebar and click **Add to knowledge base** to start chatting."
        )
        st.stop()

    bot = MedBot(store)

    # ----- Chat state ------------------------------------------------------ #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested prompts on a fresh chat.
    if not st.session_state.messages:
        st.markdown("###### 💡 Try asking")
        cols = st.columns(2)
        suggestions = [
            "What are common symptoms of the flu?",
            "How is high blood pressure managed?",
            "What is a healthy BMI range?",
            "How should a minor burn be treated?",
        ]
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending = s
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # Render history.
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="src-card"><span class="src-tag">'
                            f'{src["label"]}</span><br>{src["snippet"]}</div>',
                            unsafe_allow_html=True,
                        )

    # ----- Handle input ---------------------------------------------------- #
    user_input = st.chat_input("Ask a health question…")
    if "pending" in st.session_state:
        user_input = st.session_state.pop("pending")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Consulting your documents…"):
                history = st.session_state.messages[:-1]
                try:
                    answer = bot.answer(user_input, history=history)
                except Exception as e:  # surface API/errors gracefully
                    answer = None
                    st.error(f"Something went wrong: {e}")

            if answer is not None:
                st.markdown(answer.text)
                if not answer.is_escalation:
                    st.markdown(
                        f'<div class="disclaimer">{safety.DISCLAIMER}</div>',
                        unsafe_allow_html=True,
                    )

                sources = []
                if answer.sources:
                    with st.expander(f"📎 Sources ({len(answer.sources)})"):
                        for src in answer.sources:
                            label = src.source + (f", p.{src.page}" if src.page else "")
                            snippet = src.text[:280].strip() + "…"
                            sources.append({"label": label, "snippet": snippet})
                            st.markdown(
                                f'<div class="src-card"><span class="src-tag">'
                                f"{label}</span><br>{snippet}</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.text,
                        "sources": sources,
                    }
                )


if __name__ == "__main__":
    main()
