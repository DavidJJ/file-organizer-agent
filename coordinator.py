import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from langchain_classic.agents import AgentExecutor
from opentelemetry.trace import StatusCode

from agent import classify_file
from folder_agent import classify_folder
from output import CSVWriter
from telemetry import tracer

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
PROGRESS_FILE = ".progress.json"


@dataclass
class Progress:
    files: Set[str] = field(default_factory=set)
    skipped_dirs: Set[str] = field(default_factory=set)


def load_progress(progress_file: str = PROGRESS_FILE) -> Progress:
    if not os.path.exists(progress_file):
        return Progress()
    with open(progress_file) as f:
        data = json.load(f)
    # Backward compat: old format was a plain list of file paths
    if isinstance(data, list):
        return Progress(files=set(data))
    return Progress(
        files=set(data.get("files", [])),
        skipped_dirs=set(data.get("skipped_dirs", [])),
    )


def save_progress(progress: Progress, progress_file: str = PROGRESS_FILE) -> None:
    with open(progress_file, "w") as f:
        json.dump(
            {"files": list(progress.files), "skipped_dirs": list(progress.skipped_dirs)},
            f,
        )


class DirectoryCoordinator:
    def __init__(
        self,
        folder_agent: AgentExecutor,
        file_agent: AgentExecutor,
        csv_writer: CSVWriter,
        progress: Progress,
        progress_file: str = PROGRESS_FILE,
    ):
        self.folder_agent = folder_agent
        self.file_agent = file_agent
        self.csv_writer = csv_writer
        self.progress = progress
        self.progress_file = progress_file

    def process(self, directory: Path) -> None:
        """Recursively process a directory: classify folder, then files, then subdirs."""
        with tracer.start_as_current_span("directory.process") as span:
            span.set_attribute("directory.path", str(directory))

            # Resume: skip already-classified directories
            if str(directory) in self.progress.skipped_dirs:
                logger.info(f"Skipping (previously classified): {directory.name}")
                return

            # Ask folder agent: skip or process?
            decision = classify_folder(self.folder_agent, directory)

            if decision.skip:
                logger.info(f"Skipping folder: {directory.name} — {decision.reason}")
                self.progress.skipped_dirs.add(str(directory))
                save_progress(self.progress, self.progress_file)
                return

            # List children — guard against permission errors
            try:
                children = sorted(directory.iterdir())
            except PermissionError as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                logger.warning(f"Cannot read directory {directory}: {e}")
                return

            # Process direct files
            for child in children:
                if child.name.startswith("."):
                    continue
                if not (child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS):
                    continue
                path_str = str(child)
                if path_str in self.progress.files:
                    logger.info(f"Skipping (already processed): {child.name}")
                    continue
                result = classify_file(self.file_agent, child)
                if result and result.is_receipt:
                    self.csv_writer.append_receipt(
                        category=result.category or "Other",
                        original_path=path_str,
                        suggested_path=result.suggested_path
                        or f"~/Documents/Receipts/Other/{child.name}",
                        reason=result.reason,
                    )
                self.progress.files.add(path_str)
                save_progress(self.progress, self.progress_file)

            # Recurse into subdirectories
            for child in children:
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    self.process(child)
