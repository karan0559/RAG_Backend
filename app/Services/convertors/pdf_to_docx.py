import pdfplumber
from docx import Document

def convert_pdf_to_docx(input_path, output_path):
    doc = Document()
    with pdfplumber.open(input_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                doc.add_paragraph(text)
    doc.save(output_path)
