import os
import comtypes.client

def convert_docx_to_pdf(input_path: str, output_path: str):
    try:
        input_abs = os.path.abspath(input_path)
        output_abs = os.path.abspath(output_path)

        print(f"📄 Converting DOCX → PDF\nInput: {input_abs}\nOutput: {output_abs}")
        word = comtypes.client.CreateObject("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(input_abs)
        doc.SaveAs(output_abs, FileFormat=17)
        doc.Close(False)
        word.Quit()

    except Exception as e:
        raise RuntimeError(f"❌ DOCX to PDF conversion failed: {e}")
