from pathlib import Path

from langchain_core.tools import tool


@tool
def list_directory(dir_path: str) -> str:
    """List the name and direct contents of a directory. Input must be an absolute directory path."""
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
        return f"Folder name: {path.name}\nContents ({count} items):\n{listing}"
    except PermissionError as e:
        return f"Error reading directory: {e}"
    except Exception as e:
        return f"Error: {e}"
