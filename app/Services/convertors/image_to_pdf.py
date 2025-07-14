from PIL import Image

def convert_image_to_pdf(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    img.save(output_path, "PDF")
