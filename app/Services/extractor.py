from app.Services.parsers import (
    pdf_parser,
    docx_parser,
    image_ocr,
    audio_transcriber,
    url_scraper,
    youtube_transcriber
)
from app.Services.chunkings import semantic_chunk_text
import re

def _looks_like_error_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    error_markers = (
        "ocr failed:",
        "extraction failed:",
        "unsupported file type",
        "unsupported parser output format",
        "no readable text found",
    )
    if any(marker in normalized for marker in error_markers):
        return True

    clean_text = text.strip()
    if clean_text:
        alnum_count = sum(c.isalnum() for c in clean_text)
        if alnum_count / len(clean_text) < 0.2:
            print(f"[Garbage Filter] Text rejected (Low alnum): {clean_text[:30]}...")
            return True
        # Allow numbers and letters to count as words
        words = re.findall(r'[a-zA-Z0-9]{3,}', clean_text)
        if not words and len(clean_text) > 5:
            print(f"[Garbage Filter] Text rejected (No 3+ char words): {clean_text[:30]}...")
            return True

    return False


def extract_content(path_or_input: str, file_type: str) -> list[str]:
    try:
        # Step 1: Parse raw text based on file_type
        if file_type == "pdf":
            text = pdf_parser.parse_pdf(path_or_input)
        elif file_type == "docx":
            text = docx_parser.parse_docx(path_or_input)
        elif file_type == "image":
            text = image_ocr.ocr_image(path_or_input)
        elif file_type == "audio":
            text = audio_transcriber.transcribe_audio(path_or_input)
        elif file_type == "url":
            text = url_scraper.extract_from_url(path_or_input)
        elif file_type == "youtube":
            text = youtube_transcriber.extract_youtube_transcript(path_or_input)
        else:
            print(f"[Extractor] Unsupported file type: {file_type}")
            return []

        # Step 2: Normalize to string and chunk
        # Use larger windows for long-form web transcripts/pages to avoid
        # generating an excessive number of tiny chunks.
        chunk_size = 6 if file_type in {"url", "youtube"} else 3
        overlap = 2 if file_type in {"url", "youtube"} else 1

        if isinstance(text, str):
            if _looks_like_error_text(text):
                print(f"[Extractor] Parser produced non-usable text for {file_type}")
                return []
            return semantic_chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        elif isinstance(text, list):
            clean_items = [t for t in text if isinstance(t, str) and not _looks_like_error_text(t)]
            joined_text = " ".join(clean_items)
            if _looks_like_error_text(joined_text):
                print(f"[Extractor] Parser list output was empty/non-usable for {file_type}")
                return []
            return semantic_chunk_text(joined_text, chunk_size=chunk_size, overlap=overlap)
        else:
            print(f"[Extractor] Unsupported parser output format: {type(text)}")
            return []

    except Exception as e:
        print(f"[Extractor] Extraction failed ({file_type}): {str(e)}")
        return []
