# retrieval-agent

A production-grade RAG (Retrieval-Augmented Generation) chatbot built with Python and the Claude API. Answers questions grounded in your own documents — no hallucinations, sources always cited.

**Status:** config + document model ✅ | chunker ✅ | embedder ✅ | vector store 🔜 | query pipeline 🔜 | deployment 🔜

---

## What it does

1. Ingests documents (PDF, web pages, plain text, Markdown)
2. Chunks and embeds them into a vector store
3. On query: retrieves the most relevant chunks
4. Passes them as context to Claude and streams the answer back
5. Cites the source for every claim

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Claude API (Anthropic) |
| Embeddings | `sentence-transformers` (local, free) |
| Vector DB | ChromaDB (local dev) → Qdrant (production) |
| API server | FastAPI |
| UI | Streamlit (dev) → Telegram bot (prod) |
| Observability | Langfuse |
| Deploy | Railway / Fly.io |

---

## Project structure

```
retrieval-agent/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py        # PDF, web, text loaders
│   │   ├── chunker.py        # text splitting strategies
│   │   ├── embedder.py       # embedding model wrapper
│   │   └── pipeline.py       # orchestrates full ingest flow
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── store.py          # ChromaDB / Qdrant abstraction
│   │   └── retriever.py      # top-k semantic search
│   └── api/
│       ├── __init__.py
│       └── chat.py           # Claude API + prompt builder
├── data/
│   ├── raw/                  # drop your source files here
│   └── processed/            # ChromaDB persists here
├── tests/
│   └── test_ingestion.py
├── scripts/
│   └── ingest.py             # CLI: python scripts/ingest.py --source data/raw/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Clone and set up environment
git clone https://github.com/dimipash/retrieval-agent.git
cd retrieval-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-ant-...

# 3. Drop documents into data/raw/
cp your_docs.pdf data/raw/

# 4. Run ingestion
python scripts/ingest.py --source data/raw/ --collection my_docs

# 5. Ask a question (Week 2 — coming soon)
# python scripts/query.py --question "What is the refund policy?"
```

---

## Roadmap

- [x] **Week 1** — Ingestion pipeline (loaders, chunker, embedder, ChromaDB)
- [ ] **Week 2** — Query pipeline (retrieval + Claude API + streaming)
- [ ] **Week 3** — Eval + observability (Langfuse, golden test set)
- [ ] **Week 4** — Deploy to Railway, collect real user feedback
- [ ] **Phase 2** — Agent tools (web search, file creation, calculator)
- [ ] **Phase 2** — MCP server integration
- [ ] **Phase 2** — Long-term memory store
