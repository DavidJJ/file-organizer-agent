import logging
from pathlib import Path

from langchain_core.tools import tool
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)


@tool
def list_directory(dir_path: str) -> str:
    """List the name and direct contents of a directory. Input must be an absolute directory path."""
    from telemetry import tracer
    with tracer.start_as_current_span("tool.list_directory") as span:
        span.set_attribute("directory.path", dir_path)
        try:
            path = Path(dir_path)
            if not path.is_dir():
                return f"Error: {dir_path} is not a directory."

            children = []
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                kind = "[dir]" if child.is_dir() else "[file]"
                children.append(f"  {kind}  {child.name}")

            count = len(children)
            listing = "\n".join(children) if children else "  (empty)"
            span.set_attribute("tool.outcome", "success")
            span.set_attribute("directory.child_count", count)
            return f"Folder name: {path.name}\nContents ({count} items):\n{listing}"
        except PermissionError as e:
            span.set_attribute("tool.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(f"Permission denied reading directory {dir_path}: {e}")
            return f"Error reading directory: {e}"
        except Exception as e:
            span.set_attribute("tool.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(f"Error listing directory {dir_path}: {e}")
            return f"Error: {e}"
