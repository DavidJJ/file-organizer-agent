import logging

from langchain_core.tools import tool
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

MAX_CHARS = 3000


@tool
def read_text(file_path: str) -> str:
    """Read a plain text file. Input must be an absolute file path."""
    from telemetry import tracer
    with tracer.start_as_current_span("tool.read_text") as span:
        span.set_attribute("file.path", file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(MAX_CHARS)

            chars_extracted = len(text.strip())
            span.set_attribute("tool.chars_extracted", chars_extracted)
            span.set_attribute("tool.truncated", chars_extracted >= MAX_CHARS)

            if not text.strip():
                span.set_attribute("tool.outcome", "empty")
                return "No text could be extracted from this file."

            span.set_attribute("tool.outcome", "success")
            return text
        except Exception as e:
            span.set_attribute("tool.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(f"Error reading text file {file_path}: {e}")
            return f"Error reading file: {e}"
