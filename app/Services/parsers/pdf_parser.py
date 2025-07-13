import fitz  # PyMuPDF

def parse_pdf(path: str):
    try:
        doc = fitz.open(path)
        chunks = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                chunks.append(text)
        doc.close()
        return chunks or ["No extractable text found."]
    except Exception as e:
        return [f"PDF parsing failed: {e}"]
