# Briefly — Stage 1: LLM + Legal AI Prototype

## What this is

A modular LLM/RAG/agent foundation for a legal case workspace. Stage 1 scope only: document ingestion, retrieval, grounded Q&A, summarization, and one controlled tool-using agent. No UI. No irreversible or external actions — the agent only reads authorized documents and answers questions.

## Architecture

Each layer is independent and only talks to the layer below it through a defined interface, so any one layer can be swapped without rewriting the others.

- **Ingestion** (`src/ingestion`) — loads PDF/TXT/DOCX, extracts text with PyMuPDF, falls back to PaddleOCR-VL per-page for scanned content, cleans and chunks text while preserving document/page metadata.
- **Retrieval** (`src/retrieval`) — embeds chunks (OpenAI embeddings by default), stores them in a local Chroma vector store, retrieves top-k relevant chunks per query.
- **Context / permissions** (`src/context`) — enforces the shared/client/firm/private visibility boundary from the product's case-record model. Every retrieval call is scoped through `AuthorizedScope`; nothing bypasses it.
- **LLM** (`src/llm`) — single factory for the chat model, isolated so provider/model can change without touching agent or retrieval code.
- **Tools** (`src/tools`) — controlled functions the agent can call: `document_search` and `document_metadata_lookup`. Each tool is bound to an `AuthorizedScope` at construction time, so the agent can never retrieve outside its authorized visibility layers regardless of what it decides to call.
- **Agent** (`src/agent`) — a LangGraph graph with a real tool-calling loop: `agent` node (LLM bound to tools via `bind_tools`) → conditional edge → `tools` (`ToolNode`) → back to `agent`, looping until the model stops requesting tools or hits the `agent.max_tool_calls` budget, then `finalize`. The model decides for itself whether/when to call `document_search` or `document_metadata_lookup` — this isn't a fixed retrieval step. Grounding is still enforced structurally, not just by prompt: `finalize` only treats an answer as grounded if a `document_search` call actually returned evidence during that run; otherwise it returns the fixed fallback message no matter what the model said. Summarization is a separate, simpler function (`src/agent/summarize.py`) since it doesn't need tools at all.
- **Evaluation** (`src/evaluation`) — runs `eval/eval_dataset.json` through the agent and scores answer correctness, citation presence, and fallback correctness.

Prompts live in `prompts/` as plain text files, not embedded in code, so they can change without a redeploy.

## Model

The reasoning/agent LLM is **Kimi K2.5** (Moonshot AI) — the finalized, HIGH PRIORITY candidate from `Model_Verification_and_Evaluation_Final_Report`. Moonshot's API is OpenAI-compatible, so `src/llm/model.py` just points `ChatOpenAI` at `https://api.moonshot.ai/v1` with `MOONSHOT_API_KEY` rather than needing a separate client. Provider/model are configurable in `configs/settings.yaml` (or via `LLM_PROVIDER`/`LLM_MODEL` env vars) — check Moonshot's current model list before deploying, since some sources indicate `kimi-k2.5` has a scheduled sunset in favor of newer K2/K3-family ids.

The report also finalizes Qwen3-Embedding-8B (embedding), Qwen3-VL-Reranker-8B (reranker), PaddleOCR-VL vs. NaviDC-OCR (OCR, undecided), and Whisper Large-v3 (speech-to-text) — only the LLM and OCR layers are wired in so far.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in MOONSHOT_API_KEY (and OPENAI_API_KEY for embeddings)
```

Drop legal documents (PDF/TXT/DOCX) into `data/raw/`.

## Usage

```bash
# 1. Ingest documents into the vector store
python scripts/ingest.py --visibility shared

# 2. Ask a question
python scripts/ask.py "What is the termination clause in the NDA?"

# 3. Run evaluation
python -m src.evaluation.runner
```

## Design notes / limitations

- Vector store is local Chroma, persisted to `data/vectorstore/`. Fine for Stage 1's small document set; will need a hosted vector DB before this scales to many cases.
- OCR fallback triggers per-page only when direct text extraction returns nothing — text-native PDFs never touch PaddleOCR-VL, keeping ingestion fast.
- The agent can call tools multiple times per question (bounded by `agent.max_tool_calls`), so it can reformulate a query and retry within one run if the first search comes back thin — it isn't limited to one retrieval pass.
- `AuthorizedScope` is currently constructed manually in scripts (`AuthorizedScope.firm_member(...)`) since there's no auth/session layer yet — this will need to come from the real session/user context once this connects to the actual product.
- The eval dataset's `expected_answer_contains` fields are placeholders — fill them in with real keywords once real case documents are ingested.
- No hard-coded answers anywhere in the agent path; every answer comes from the LLM grounded in retrieved chunks, or the fixed fallback message when evidence is insufficient.
