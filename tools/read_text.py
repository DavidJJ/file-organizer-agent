from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def read_text(file_path: str) -> str:
    """Read a plain text file. Input must be an absolute file path."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(MAX_CHARS)
    except Exception as e:
        return f"Error reading file: {e}"
