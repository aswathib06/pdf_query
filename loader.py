import io
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def extract_text_from_ipynb(file_bytes: bytes) -> str:
    data = json.load(io.BytesIO(file_bytes))
    text = ""
    for cell in data.get("cells", []):
        if cell["cell_type"] == "markdown":
            text += "".join(cell["source"]) + "\n\n"
        elif cell["cell_type"] == "code":
            text += "# code:\n" + "".join(cell["source"]) + "\n\n"
    return text

def extract_text_from_file(uploaded_file) -> str:
    bytes_data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(bytes_data)
    if name.endswith(".ipynb"):
        return extract_text_from_ipynb(bytes_data)

    try:
        return bytes_data.decode("utf-8")
    except:
        return ""
