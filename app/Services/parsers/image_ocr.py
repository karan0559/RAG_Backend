import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd =r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_image(image_path: str) -> str:
    try:
        print(f" Using Tesseract at: {pytesseract.pytesseract.tesseract_cmd}")
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text if text.strip() else "No readable text found."
    except Exception as e:
        return f"OCR failed: {str(e)}"
