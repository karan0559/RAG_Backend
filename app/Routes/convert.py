from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import os

router = APIRouter()
CONVERTED_DIR = "data/converted_files"
os.makedirs(CONVERTED_DIR, exist_ok=True)

@router.post("/convert/", summary="Convert a file to another format")
async def convert_file(file: UploadFile = File(...), output_format: str = Form(...)):
    try:
        file_ext = Path(file.filename).suffix.lower().strip(".")
        input_path = os.path.join(CONVERTED_DIR, f"{uuid.uuid4()}.{file_ext}")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        output_filename = f"{Path(file.filename).stem}_converted.{output_format}"
        output_path = os.path.join(CONVERTED_DIR, output_filename)

        if file_ext == "pdf" and output_format == "docx":
            from app.Services.convertors.pdf_to_docx import convert_pdf_to_docx
            convert_pdf_to_docx(input_path, output_path)

        elif file_ext in ["jpg", "jpeg", "png"] and output_format == "pdf":
            from app.Services.convertors.image_to_pdf import convert_image_to_pdf
            convert_image_to_pdf(input_path, output_path)

        elif file_ext == "pdf" and output_format == "jpg":
            from app.Services.convertors.pdf_to_jpg import convert_pdf_to_jpg
            result_path = convert_pdf_to_jpg(input_path, CONVERTED_DIR)
            output_path = result_path
            output_filename = os.path.basename(result_path)

        elif file_ext == "docx" and output_format == "pdf":
            from app.Services.convertors.docx_to_pdf import convert_docx_to_pdf
            convert_docx_to_pdf(input_path, output_path)

        elif file_ext == "txt" and output_format == "xlsx":
            from app.Services.convertors.txt_to_excel import convert_txt_to_excel
            convert_txt_to_excel(input_path, output_path)

        else:
            raise HTTPException(status_code=400, detail=f" Unsupported conversion: {file_ext} → {output_format}")

        return FileResponse(output_path, filename=output_filename, media_type="application/octet-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Conversion failed: {str(e)}")