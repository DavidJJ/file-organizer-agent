import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Prevent HuggingFace Hub from making network calls at runtime.
# Models are pre-downloaded via `docling-tools models download`.
# Without this, HF Hub checks for updates on every run and prints
# unauthenticated request warnings even when the cache is fully populated.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from agent import build_agent, classify_file
from output import CSVWriter
from scanner import scan_files
from telemetry import setup_telemetry, tracer

logger = logging.getLogger(__name__)

PROGRESS_FILE = ".progress.json"


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(processed: set) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed), f)


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

    all_files = list(scan_files(args.root))
    total = len(all_files)
    logger.info(f"Found {total} files to process", extra={"file.count": total})

    processed = load_progress()
    skipped = len([f for f in all_files if str(f) in processed])
    agent_executor = build_agent(args.ollama_url, args.model)
    csv_writer = CSVWriter(str(csv_path))

    receipts_found = 0
    errors = 0

    with tracer.start_as_current_span("scan.run") as run_span:
        run_span.set_attribute("scan.root", args.root)
        run_span.set_attribute("scan.model", args.model)
        run_span.set_attribute("scan.total_files", total)
        run_span.set_attribute("scan.already_processed", skipped)
        run_span.set_attribute("scan.csv_path", str(csv_path))

        for i, file_path in enumerate(all_files, 1):
            path_str = str(file_path)

            if path_str in processed:
                logger.info(
                    f"[{i}/{total}] Skipping (already processed): {file_path.name}",
                    extra={"file.path": path_str},
                )
                continue

            logger.info(
                f"[{i}/{total}] Scanning: {file_path.name}",
                extra={
                    "file.path": path_str,
                    "file.type": file_path.suffix.lstrip("."),
                    "scan.progress": f"{i}/{total}",
                },
            )

            result = classify_file(agent_executor, file_path)

            if result is None:
                errors += 1
            elif result.is_receipt:
                receipts_found += 1
                csv_writer.append_receipt(
                    category=result.category or "Other",
                    original_path=path_str,
                    suggested_path=result.suggested_path
                    or f"~/Documents/Receipts/Other/{file_path.name}",
                    reason=result.reason,
                )

            processed.add(path_str)
            save_progress(processed)

            # Keep running totals current on the span so partial runs are useful
            run_span.set_attribute("scan.receipts_found", receipts_found)
            run_span.set_attribute("scan.errors", errors)
            run_span.set_attribute("scan.files_evaluated", i - skipped)

        run_span.set_attribute("scan.complete", True)

    logger.info(
        f"Scan complete. {receipts_found} receipts found in {total - skipped} files. "
        f"Results saved to {csv_path}",
        extra={
            "scan.receipts_found": receipts_found,
            "scan.total_files": total,
            "scan.errors": errors,
            "scan.csv_path": str(csv_path),
        },
    )


if __name__ == "__main__":
    main()
