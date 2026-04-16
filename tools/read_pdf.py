import pdfplumber
from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file. Input must be an absolute file path."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
                if len(text) >= MAX_CHARS:
                    break
        return text[:MAX_CHARS] if text.strip() else "No text could be extracted from this PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"
