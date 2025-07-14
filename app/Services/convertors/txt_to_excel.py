import openpyxl

def convert_txt_to_excel(input_path, output_path):
    wb = openpyxl.Workbook()
    ws = wb.active

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            ws.cell(row=i, column=1, value=line.strip())

    wb.save(output_path)
