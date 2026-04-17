import logging

from docx import Document
from langchain_core.tools import tool
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

MAX_CHARS = 3000


@tool
def read_docx(file_path: str) -> str:
    """Extract text from a Word document (.docx). Input must be an absolute file path."""
    from telemetry import tracer
    with tracer.start_as_current_span("tool.read_docx") as span:
        span.set_attribute("file.path", file_path)
        try:
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)

            chars_extracted = len(text.strip())
            span.set_attribute("tool.chars_extracted", chars_extracted)
            span.set_attribute("tool.truncated", len(text) > MAX_CHARS)

            if not text.strip():
                span.set_attribute("tool.outcome", "empty")
                return "No text could be extracted from this document."

            span.set_attribute("tool.outcome", "success")
            return text[:MAX_CHARS]
        except Exception as e:
            span.set_attribute("tool.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return f"Error reading DOCX: {e}"
