from docx import Document
from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def read_docx(file_path: str) -> str:
    """Extract text from a Word document (.docx). Input must be an absolute file path."""
    try:
        doc = Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text[:MAX_CHARS] if text.strip() else "No text could be extracted from this document."
    except Exception as e:
        return f"Error reading DOCX: {e}"
