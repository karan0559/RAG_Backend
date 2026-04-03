import os
import pytesseract
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Read Tesseract path from .env, fall back to system PATH
_tesseract_path = os.getenv("TESSERACT_PATH", "")
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path
# If TESSERACT_PATH is not set, pytesseract will look for 'tesseract' on
# the system PATH automatically (works on Linux/Mac and Windows if Tesseract
# is in PATH).


def ocr_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"OCR failed: {str(e)}")
