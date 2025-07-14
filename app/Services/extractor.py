from app.Services.parsers import (
    pdf_parser,
    docx_parser,
    image_ocr,
    audio_transcriber,
    url_scraper,
    youtube_transcriber
)
from app.Services.chunkings import semantic_chunk_text  

def extract_content(path_or_input: str, file_type: str):
    # Parse raw content
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
        raise ValueError(f"Unsupported file type: {file_type}")

    #Return chunked output
    if isinstance(text, str):
        return semantic_chunk_text(text, chunk_size=3, overlap=1)
    elif isinstance(text, list):
        joined = " ".join(text)
        return semantic_chunk_text(joined, chunk_size=3, overlap=1)
    else:
        return ["❌ Unsupported output format from parser"]
