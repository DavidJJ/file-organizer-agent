import os
from pathlib import Path
from typing import Iterator

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def scan_files(root_path: str) -> Iterator[Path]:
    """Recursively yield supported file paths, skipping hidden files and directories."""
    root = Path(root_path).expanduser().resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path
