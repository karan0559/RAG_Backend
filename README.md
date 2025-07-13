# 🧠 Smart RAG System

A powerful Retrieval-Augmented Generation (RAG) system for multimodal document understanding, comparison, summarization, and conversational querying with memory.

## 🚀 Features

- 📄 Upload PDFs, DOCX, images, audio, URLs, and YouTube transcripts
- 🔍 Ask questions about uploaded content using Groq's LLMs (e.g., LLaMA3-70B)
- 🧠 Memory-enabled chat using ChromaDB (retrieves last 30 exchanges per session)
- 📚 Document Comparison and Summarization
- 📦 FastAPI backend with modular architecture
- 🎯 Chunk → Retrieve → Augment → Generate (CRAG pipeline)
- 🧠 Reranking using `bge-reranker-large`
- ✨ Built-in support for OCR, transcription, and file parsing

## 🛠️ Tech Stack

- **FastAPI** – Backend API
- **Groq API** – LLM inference (LLaMA3, Mixtral)
- **SentenceTransformers** – Embeddings (`e5-large-v2`)
- **FAISS** – Vector store for chunk retrieval
- **ChromaDB** – Chat memory store
- **FasterWhisper** – Audio transcription
- **Pytesseract / pdfplumber / PyMuPDF** – Parsing and OCR
- **Cohere Reranker** – Semantic reranking of chunks

## 📂 Folder Structure

app/
├── Routes/ # API endpoints
│ ├── upload.py
│ ├── query.py
│ ├── compare.py
│ └── convert.py
├── Services/ # Core logic (retriever, llm, embedder, etc.)
│ ├── extractor.py
│ ├── retriever.py
│ ├── embedder.py
│ └── vector_db.py
├── Memory/
│ └── memory_db.py # Chat memory using ChromaDB
├── parser/ # File parsing logic (PDF, DOCX, image, audio, URL)
└── main.py # FastAPI app entry point

Create a Conda environment
    conda create -n rag-env python=3.10
    conda activate rag-env

Install dependencies
    pip install -r requirements.txt

Set up your .env file
    GROQ_API_KEY=your_api_key
    GROQ_MODEL=llama3-70b-8192

Run the FastAPI app
    uvicorn app.main:app --reload


