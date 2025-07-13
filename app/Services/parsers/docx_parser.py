import docx

def parse_docx(path: str):
    try:
        doc = docx.Document(path)
        return [para.text for para in doc.paragraphs if para.text.strip()]
    except Exception as e:
        return [f"DOCX parsing failed: {e}"]
