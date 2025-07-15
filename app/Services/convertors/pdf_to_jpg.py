import os
import zipfile
from pdf2image import convert_from_path

def convert_pdf_to_jpg(input_pdf: str, output_folder: str, dpi: int = 200):
    """
    Converts each page of a PDF file to JPG images.
    If single page, returns JPG path.
    If multiple pages, returns ZIP path containing JPGs.
    Args:
        input_pdf (str): Path to the input PDF file.
        output_folder (str): Folder to save JPG images.
        dpi (int): Resolution for output images.
    Returns:
        str: Path to JPG or ZIP file.
    """
    input_abs = os.path.abspath(input_pdf)
    output_abs = os.path.abspath(output_folder)
    os.makedirs(output_abs, exist_ok=True)

    print(f"Converting PDF → JPG\nInput: {input_abs}\nOutput Folder: {output_abs}")

    try:
        images = convert_from_path(input_abs, dpi=dpi)
        output_files = []
        for i, image in enumerate(images):
            output_path = os.path.join(output_abs, f"page_{i+1}.jpg")
            image.save(output_path, "JPEG")
            output_files.append(output_path)
        print(f"Conversion successful: {len(output_files)} pages converted.")

        if len(output_files) == 1:
            return output_files[0]
        else:
            zip_path = os.path.join(output_abs, "pdf_pages.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for jpg in output_files:
                    zipf.write(jpg, arcname=os.path.basename(jpg))
            return zip_path

    except Exception as e:
        raise RuntimeError(f"PDF to JPG conversion failed:{e}")