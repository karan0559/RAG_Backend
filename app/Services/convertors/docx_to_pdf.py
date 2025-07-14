import comtypes.client

def convert_docx_to_pdf(input_path, output_path):
    word = comtypes.client.CreateObject('Word.Application')
    doc = word.Documents.Open(input_path)
    doc.SaveAs(output_path, FileFormat=17)  # PDF format
    doc.Close()
    word.Quit()
