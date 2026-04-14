# 🧠 Smart RAG System

A **modular, multimodal Retrieval-Augmented Generation (RAG) backend** built with FastAPI, FAISS, and Groq-hosted LLaMA3.  
Supports document understanding across **PDFs, DOCX, images, audio, web URLs, and YouTube** — with session memory, reranking, TTS, web fallback, file conversion, and a validated retrieval pipeline.

---

## 🚀 Features

- 📥 Upload and parse multiple file types: PDF, DOCX, images, audio, URLs, YouTube transcripts
- ✂️ Chunk and embed content using `intfloat/e5-large-v2` (1024-dim dense vectors)
- 🗄️ Store and search vectors in FAISS with `doc_id|chunk` tagging for session scoping
- 🔀 Hybrid retrieval: dense FAISS + sparse BM25 fused via **Reciprocal Rank Fusion (RRF)**
- 🏆 Rerank candidates using `BAAI/bge-reranker-base` cross-encoder for precision
- 🤖 Answer queries using **LLaMA 3 (via Groq API)** with injected document context
- 🧠 Session-scoped 30-turn conversation memory via **ChromaDB** (DuckDB backend)
- 🌐 Automatic fallback to **Tavily web search** when retrieval confidence is low
- 🔊 Optional **Text-to-Speech** audio response via Microsoft Edge-TTS
- 🔄 File conversion: PDF ⇄ DOCX, Image ⇄ PDF
- 📊 Compare multiple documents with side-by-side LLM summaries
- ✅ Fully validated pipeline: Recall@5 = 1.0, MRR = 0.854, ROC AUC = 0.986

---

## 🧱 Tech Stack

| Layer | Tools |
|---|---|
| 🔍 Retrieval | FAISS (IndexFlatIP), E5-Large-v2 embeddings, BM25Okapi, RRF fusion |
| 🏆 Reranking | BAAI/bge-reranker-base (cross-encoder, sigmoid scoring) |
| 🤖 LLM | LLaMA 3 via Groq API (`llama-3.1-8b-instant` / `llama3-70b-8192`) |
| 📄 Parsing | PyMuPDF, pdfplumber, python-docx, Tesseract OCR, faster-whisper, trafilatura |
| 🧠 Memory | ChromaDB (session-based, DuckDB + Parquet backend) |
| 🌐 Web Search | Tavily API (relevance-gated fallback) |
| 🔊 TTS | Microsoft Edge-TTS (local synthesis, async streaming) |
| 🧰 API | FastAPI + Uvicorn + Pydantic |
| 💾 Storage | Local filesystem (uploads, converted files, audio responses, FAISS index) |

---

## 🗂️ Project Structure

```
RAG_PROJECT/
├── app/
│   ├── main.py                    # FastAPI app entry point, router registration
│   ├── Routes/
│   │   ├── upload.py              # /upload/ — parse, chunk, embed, index
│   │   ├── query.py               # /query/ — retrieve, rerank, LLM, TTS
│   │   ├── compare.py             # /compare/ — multi-doc comparison
│   │   ├── convert.py             # /convert/ — PDF/DOCX/image conversion
│   │   ├── audio.py               # /audio/ — serve TTS audio files
│   │   ├── docs.py                # /docs-list/ — list uploaded documents
│   │   └── frontend.py            # / — serve static frontend
│   ├── Services/
│   │   ├── embedder.py            # Sentence-transformer wrapper (e5-large-v2)
│   │   ├── retriever.py           # Hybrid BM25+FAISS search with RRF
│   │   ├── reranker.py            # BGE cross-encoder reranker
│   │   ├── vector_db.py           # FAISS index management (add, search, persist)
│   │   ├── bm25_index.py          # BM25Okapi index over raw chunks
│   │   ├── llm.py                 # Groq LLM client with memory injection
│   │   ├── web_search.py          # Tavily API fallback handler
│   │   ├── tts.py                 # Edge-TTS async synthesis
│   │   ├── extractor.py           # Key-phrase extraction utilities
│   │   ├── model_loader.py        # Lazy model loader (singleton pattern)
│   │   ├── parsers/               # File-type-specific parsers
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── image_parser.py    # Tesseract OCR
│   │   │   ├── audio_parser.py    # faster-whisper transcription
│   │   │   ├── url_parser.py      # trafilatura web scraping
│   │   │   └── youtube_parser.py  # youtube-transcript-api
│   │   ├── chunkings/             # Chunking strategies
│   │   └── convertors/            # PDF/DOCX/image conversion logic
│   ├── Memory/
│   │   ├── memory_db.py           # ChromaDB session memory (30-turn window)
│   │   └── session_docs.py        # Session → doc_id mapping
│   └── static/                    # Frontend HTML/CSS/JS
├── data/
│   ├── index.faiss                # FAISS vector index (persisted)
│   └── chunks.pkl                 # Raw chunk list (doc_id|text format)
├── validate.py                    # Standalone validation audit (Tasks 1–6)
├── validation_results/            # Validation plots and report
│   ├── validation_report.md
│   └── *.png                      # 10 validation plots
├── requirements.txt
└── .env
```

---

## 🌐 Query Flow

```
User Query
    │
    ▼
[1] Embed query (e5-large-v2)
    │
    ▼
[2] Hybrid search: FAISS (dense) + BM25 (sparse) → Reciprocal Rank Fusion → Top-10 chunks
    │
    ▼
[3] Session scoping: filter chunks to session-specific doc_ids (if session docs exist)
    │
    ▼
[4] Rerank with BGE cross-encoder → relevance score per chunk
    │
    ├─── max score ≥ 0.40 → pass top chunks as context to Groq LLM
    │
    └─── max score < 0.40 → Tavily web search fallback
    │
    ▼
[5] Groq LLM generates answer (with 30-turn memory injected)
    │
    ▼
[6] Optional: Edge-TTS synthesizes audio response
```

---

## 📡 API Endpoints

### `POST /upload/`
Upload any supported document. Parses, chunks, embeds, and indexes it into FAISS.  
**Supported types:** PDF, DOCX, PNG/JPG/WEBP (OCR), MP3/WAV/M4A (transcription), URL, YouTube link.  
Returns: `doc_id`, `summary`, `suggested_questions[]`

### `POST /query/`
Ask a question over uploaded documents.  
**Body:** `{ "query": "...", "session_id": "...", "doc_ids": [...], "tts": true/false }`  
Returns: `answer`, `sources[]`, `audio_url` (if TTS enabled)

### `POST /compare/`
Compare two or more uploaded documents.  
**Body:** `{ "doc_ids": [...], "aspect": "..." }`  
Returns: per-document summaries + key differences analysis

### `POST /convert/`
Convert a file between formats.  
**Supported conversions:** PDF → DOCX, DOCX → PDF, Image → PDF  
Returns: download URL of converted file

### `GET /docs-list/`
List all currently indexed documents with their `doc_id` and metadata.

### `GET /audio/{filename}`
Stream a TTS audio file by filename.

---

## 📦 Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/smart-rag.git
cd smart-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Tesseract OCR must be installed separately.  
> Windows: https://github.com/UB-Mannheim/tesseract/wiki  
> Linux: `sudo apt install tesseract-ocr`

### 3. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama3-70b-8192
TAVILY_API_KEY=your_tavily_api_key   # optional, required for web fallback
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive Swagger docs: `http://localhost:8000/docs`

---

## 🧪 Validation

The `validate.py` script performs a standalone, read-only audit of the full pipeline across 6 tasks (retrieval quality, reranker quality, threshold calibration, LLM faithfulness, latency, and fallback accuracy):

```bash
python validate.py
```

Results are saved to `validation_results/` as a markdown report + 10 diagnostic plots.

### Latest Validation Results (2026-04-14)

| Metric | Value | Status |
|---|---|---|
| Recall@1 | 0.750 | ✅ PASS |
| Recall@3 | 0.917 | ✅ PASS |
| Recall@5 | 1.000 | ✅ PASS |
| MRR | 0.854 | ✅ PASS |
| Reranker Gap | 0.890 | ✅ PASS |
| Reranker Accuracy | 0.917 | ✅ PASS |
| Faithfulness (Semantic) | 0.895 | ✅ PASS |
| Faithfulness (Lexical) | 0.200 | ❌ FAIL |
| ROC AUC | 0.986 | ✅ PASS |
| Fallback Accuracy | 0.700 | — |

**7/8 checks passed.** The only failing metric is key-phrase coverage — the LLM correctly paraphrases answers semantically but doesn't echo verbatim terms. No hallucinations detected.

---

## ⚙️ Configuration Reference

| Setting | Default | Description |
|---|---|---|
| Relevance threshold | `0.40` | Min reranker score to use docs vs. web fallback. Optimal is `0.50` per validation. |
| Memory window | 30 turns | ChromaDB session history kept per `session_id` |
| Top-K retrieval | 10 | Number of FAISS + BM25 candidates before reranking |
| Embedding model | `e5-large-v2` | Requires `"query: "` / `"passage: "` E5 prefix convention |
| Reranker model | `bge-reranker-base` | Switch to `bge-reranker-large` for better accuracy at cost of latency |

---

