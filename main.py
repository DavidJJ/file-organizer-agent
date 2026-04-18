import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Prevent HuggingFace Hub from making network calls at runtime.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from agent import build_agent
from coordinator import DirectoryCoordinator, load_progress
from folder_agent import build_folder_agent
from output import CSVWriter
from telemetry import setup_telemetry, tracer

logger = logging.getLogger(__name__)


def main() -> None:
    setup_telemetry()

    parser = argparse.ArgumentParser(
        description="Scan files and identify receipts using a local LLM."
    )
    parser.add_argument(
        "--root",
        default=os.getenv("ORGANIZER_ROOT", "~/Downloads"),
        help="Root directory to scan (default: ~/Downloads)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("ORGANIZER_OUTPUT_DIR", "."),
        help="Directory to write the output CSV (default: current directory)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.output_dir) / f"receipts_{timestamp}.csv"

    logger.info(
        "Starting receipt scan",
        extra={"root": args.root, "model": args.model},
    )

    progress = load_progress()
    folder_agent = build_folder_agent(args.ollama_url, args.model)
    file_agent = build_agent(args.ollama_url, args.model)
    csv_writer = CSVWriter(str(csv_path))

    root = Path(args.root).expanduser().resolve()

    with tracer.start_as_current_span("scan.run") as run_span:
        run_span.set_attribute("scan.root", str(root))
        run_span.set_attribute("scan.model", args.model)
        run_span.set_attribute("scan.csv_path", str(csv_path))

        coordinator = DirectoryCoordinator(
            folder_agent=folder_agent,
            file_agent=file_agent,
            csv_writer=csv_writer,
            progress=progress,
        )
        coordinator.process(root)

    logger.info(
        f"Scan complete. Results saved to {csv_path}",
        extra={"scan.csv_path": str(csv_path)},
    )


if __name__ == "__main__":
    main()
