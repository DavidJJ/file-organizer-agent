import os
from pathlib import Path
from typing import Iterator

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def scan_files(root_path: str) -> Iterator[Path]:
    """Yield supported file paths under root_path.

    If root_path is a file, yield it directly (bypassing the extension check so
    the user can explicitly target any file for testing).  If it is a directory,
    recurse into it and yield all supported files, skipping hidden entries.
    """
    root = Path(root_path).expanduser().resolve()

    if root.is_file():
        yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path
