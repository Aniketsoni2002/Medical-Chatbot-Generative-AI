<div align="center">

# 🩺 MedBot — Medical Chatbot (Generative AI + RAG)

**An AI medical assistant that answers *strictly* from trusted documents — with built-in emergency detection, source citations, and a clean, modern chat UI.**

Built with **Streamlit** + **Google Gemini** + **Retrieval-Augmented Generation (RAG)**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ What it does

MedBot is a **document-grounded** medical assistant. It does **not** answer from the
model's memory — every reply is generated only from the medical documents you give it,
and it says so honestly when the documents don't cover a question. This keeps answers
traceable and reduces hallucination, which matters for health information.

### Highlights

- 📚 **RAG grounded in your documents** — answers come only from your PDFs / text files, with **source citations** shown for every response.
- 🚨 **Emergency & crisis detection** — red-flag symptoms (chest pain, stroke signs, severe bleeding) and self-harm language trigger an immediate, deterministic escalation message *before* the LLM is ever called.
- 🔒 **Responsible by design** — never diagnoses, never prescribes, always reminds users to consult a professional.
- 🎨 **Awesome, modern UI** — gradient hero, suggestion chips, chat bubbles, expandable source cards, document uploader, and a clean sidebar.
- 📎 **Bring your own documents** — upload PDFs / TXT / MD at runtime, or drop them in `data/documents/`.
- ⚡ **Lightweight & deploy-friendly** — no torch/FAISS. Uses Gemini embeddings + a NumPy cosine index, so it deploys on **Streamlit Community Cloud for free**.

---

## 🖼️ Architecture

```
                ┌──────────────────────────────────────────────┐
  User ──────►  │                 Streamlit UI (app.py)         │
                └───────────────┬──────────────────────────────┘
                                │  question
                                ▼
                  ┌─────────────────────────────┐
                  │   Safety screen (safety.py)  │  ← emergency / self-harm
                  └─────────────┬───────────────┘     → deterministic escalation
                                │ (if clear)
                                ▼
        ┌───────────────────────────────────────────────┐
        │   RAG retrieval (vectorstore.py)               │
        │   Gemini embeddings → NumPy cosine top-k       │
        └───────────────┬───────────────────────────────┘
                        │ relevant chunks (or "not covered")
                        ▼
        ┌───────────────────────────────────────────────┐
        │   Grounded generation (chatbot.py)             │
        │   Gemini + strict "use only this context" prompt│
        └───────────────┬───────────────────────────────┘
                        ▼
                 Answer + cited sources
```

Documents are chunked (`ingest.py`), embedded once, and cached to `data/index/`.

---

## 🚀 Quickstart (local)

**1. Clone & install**

```bash
git clone https://github.com/Aniketsoni2002/Medical-Chatbot-Generative-AI.git
cd Medical-Chatbot-Generative-AI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Gemini API key** (free at [Google AI Studio](https://aistudio.google.com/apikey))

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

> You can also just paste the key into the sidebar when the app opens.

**3. Run**

```bash
streamlit run app.py
```

Open <http://localhost:8501>. A sample medical reference is bundled, so you can start
chatting immediately — or upload your own documents in the sidebar.

---

## ☁️ Deploy to Streamlit Community Cloud (free)

1. Push this repo to your GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**.
3. Pick this repo, set the main file to **`app.py`**.
4. Under **Advanced settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```

5. **Deploy.** That's it — the app installs `requirements.txt` and launches.

> The bundled sample document lets the deployed app work out of the box. Uploaded
> documents live in the session; to make documents permanent, commit them to
> `data/documents/`.

---

## 🐳 Run with Docker

```bash
docker build -t medbot .
docker run -p 8501:8501 -e GEMINI_API_KEY=your-key medbot
```

---

## 🧠 Using your own medical documents

- **Permanent:** drop `.pdf`, `.txt`, or `.md` files into `data/documents/` and restart. They're indexed automatically.
- **At runtime:** use the sidebar uploader and click **Add to knowledge base**.

The bundled `health_reference.md` covers common conditions, first aid, vitals, and
preventive care as a starting point — replace it with your own trusted sources
(e.g. clinical guideline PDFs) for production use.

---

## ⚙️ Configuration

All settings have sensible defaults and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Your Google Gemini API key |
| `MEDBOT_GENERATION_MODEL` | `gemini-2.0-flash` | Chat model |
| `MEDBOT_EMBEDDING_MODEL` | `text-embedding-004` | Embedding model |
| `MEDBOT_TOP_K` | `4` | Chunks retrieved per query |
| `MEDBOT_MIN_SIMILARITY` | `0.55` | Below this, the bot says the docs don't cover it |
| `MEDBOT_CHUNK_SIZE` | `1000` | Chunk size (characters) |
| `MEDBOT_CHUNK_OVERLAP` | `150` | Overlap between chunks |

---

## 📁 Project structure

```
Medical-Chatbot-Generative-AI/
├── app.py                     # Streamlit UI
├── src/medbot/
│   ├── config.py              # Settings
│   ├── client.py              # Google GenAI client wrapper
│   ├── ingest.py              # Load & chunk documents
│   ├── vectorstore.py         # Embeddings + cosine retrieval
│   ├── safety.py              # Emergency / crisis detection
│   └── chatbot.py             # RAG engine (retrieve → generate)
├── data/documents/            # Your knowledge base (sample included)
├── .streamlit/config.toml     # Theme
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚠️ Medical disclaimer

MedBot provides **general health information only** and is **not** a substitute for
professional medical advice, diagnosis, or treatment. It does not diagnose conditions
or prescribe medication. In an emergency, contact your local emergency services
immediately. Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

Released under the [MIT License](LICENSE).
