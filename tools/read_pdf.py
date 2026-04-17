import logging

from langchain_core.tools import tool
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

MAX_CHARS = 3000

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
    from telemetry import tracer
    with tracer.start_as_current_span("tool.read_pdf") as span:
        span.set_attribute("file.path", file_path)
        try:
            converter = _get_converter()
            result = converter.convert(file_path, page_range=(1,3))
            text = result.document.export_to_text()

            chars_extracted = len(text.strip())
            span.set_attribute("tool.chars_extracted", chars_extracted)
            span.set_attribute("tool.truncated", len(text) > MAX_CHARS)

            if not text.strip():
                span.set_attribute("tool.outcome", "empty")
                return "No text could be extracted from this PDF."

            span.set_attribute("tool.outcome", "success")
            return text[:MAX_CHARS]
        except Exception as e:
            span.set_attribute("tool.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(f"Docling failed on {file_path}: {e}")
            return f"Error reading PDF: {e}"
