FROM python:3.11-slim

# System dependencies:
#   ffmpeg        - audio transcription (faster-whisper input conversion)
#   tesseract-ocr - image OCR (pytesseract)
#   poppler-utils - PDF rendering (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch build to avoid pulling multi-GB CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image so the Space's ephemeral filesystem
# doesn't have to re-download ~2.5GB from the HF Hub on every restart.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/e5-large-v2'); \
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('BAAI/bge-reranker-base'); \
AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base'); \
from faster_whisper import WhisperModel; \
WhisperModel('base', device='cpu')"

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
