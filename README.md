# 🧠 Smart RAG System

A **modular, multimodal Retrieval-Augmented Generation (RAG) backend** built with FastAPI, FAISS, and Groq-hosted LLaMA3.  
Supports document understanding across **PDFs, DOCX, images, audio, web URLs, and YouTube**—with memory, TTS, web fallback, and more.

---

## 🚀 Features

✅ Upload and parse multiple file types (PDF, DOCX, images, audio, URLs, YouTube)  
✅ Chunk and embed content using `sentence-transformers/e5-large-v2`  
✅ Store vectors in FAISS with `doc_id|chunk` tagging  
✅ Rerank retrieved results using `bge-reranker-large`  
✅ Answer user queries using **LLaMA3-70B (via Groq)**  
✅ Add recent 30-turn memory using **ChromaDB (DuckDB backend)**  
✅ Automatic fallback to web search if retrieval is weak  
✅ Generate text-to-speech output using **Edge-TTS**  
✅ File conversion route: PDF ⇄ DOCX, Image ⇄ PDF  
✅ Fully modular FastAPI structure

---

## 🧱 Tech Stack

| Layer           | Tools Used                                                                 |
|----------------|------------------------------------------------------------------------------|
| 🔍 Retrieval    | FAISS, E5 Embeddings, BGE-Reranker                                           |
| 🧠 LLM          | LLaMA3-70B via Groq API                                                      |
| 📄 Parsing      | PyMuPDF, pdfplumber, python-docx, Tesseract OCR, faster-whisper, trafilatura|
| 🔗 Memory       | ChromaDB (session-based memory, DuckDB + Parquet)                           |
| 🌐 Web Search   | Tavily API fallback (optional)                                              |
| 🔊 TTS          | Microsoft Edge-TTS (local synthesis)                                        |
| 🧰 API Backend  | FastAPI + Uvicorn + Pydantic                                                |
| 💾 Storage      | Local file system for uploads, converted files, and audio responses         |

---

## 🗂️ Folder Structure

<img width="237" height="672" alt="image" src="https://github.com/user-attachments/assets/98df448f-c0df-4410-804d-52e9bf97b298" />


Smart-RAG/
├── app/
│ ├── main.py
│ ├── Routes/
│ │ ├── upload.py
│ │ ├── query.py
│ │ ├── compare.py
│ │ ├── convert.py
│ │ ├── docs.py
│ │ └── audio.py
│ ├── Services/
│ │ ├── extractor.py
│ │ ├── embedder.py
│ │ ├── llm.py
│ │ ├── retriever.py
│ │ ├── reranker.py
│ │ ├── tts.py
│ │ ├── vector_db.py
│ │ └── parsers/
│ │ ├── pdf_parser.py
│ │ ├── docx_parser.py
│ │ ├── image_ocr.py
│ │ ├── audio_transcriber.py
│ │ ├── url_scraper.py
│ │ └── youtube_transcriber.py
│ ├── Memory/
│ │ └── memory_db.py
│ ├── utils/
│ │ └── helpers.py
├── data/
│ ├── uploads/
│ ├── chroma_memory/
│ ├── converted_files/
│ └── audio_responses/
├── .env
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧪 API Endpoints

### `/upload/`
Upload any supported document (PDF, DOCX, image, audio, URL, YouTube). Auto-parsed and embedded.

### `/query/`
Ask questions over uploaded content.  
Supports reranking, memory, fallback search, and TTS response.

### `/compare/`
Compare multiple uploaded documents. Summarize and generate key differences.

### `/convert/`
Convert files between PDF ⇄ DOCX ⇄ Image formats.

---

## 🌐 Sample Query Flow

1. User uploads a PDF and a YouTube link
2. System parses, chunks, and embeds both using `E5`
3. On query, retrieves top-10 relevant chunks from FAISS
4. Reranks using `BGE Reranker`
5. Prepends chat memory and passes context to Groq-hosted LLaMA3
6. If no relevant chunks are found → performs a web search
7. Optionally returns TTS audio link for the response

---

## 📦 Setup Instructions

### 1. Clone and install requirements
```bash
cd Smart-RAG
pip install -r requirements.txt

2. Add your .env
GROQ_API_KEY=your_key
GROQ_MODEL=llama3-70b-8192

3. Run locally
uvicorn app.main:app --reload

