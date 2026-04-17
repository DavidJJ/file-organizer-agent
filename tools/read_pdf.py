import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MAX_CHARS = 3000

# Module-level converter — instantiated once to avoid reloading models on
# every call.  Docling downloads its layout/OCR models from HuggingFace on
# first use and caches them; subsequent calls are fast.
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
        logger.info("Docling DocumentConverter initialised")
    return _converter


@tool
def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file using Docling. Input must be an absolute
    file path.  Handles both text-based and scanned (image) PDFs automatically
    — OCR is applied when no embedded text layer is detected."""
    try:
        converter = _get_converter()
        result = converter.convert(file_path)
        text = result.document.export_to_text()
        return text[:MAX_CHARS] if text.strip() else "No text could be extracted from this PDF."
    except Exception as e:
        logger.error(f"Docling failed on {file_path}: {e}")
        return f"Error reading PDF: {e}"
