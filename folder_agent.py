import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama
from opentelemetry.trace import StatusCode

from telemetry import tracer
from tools.list_directory import list_directory

logger = logging.getLogger(__name__)

FOLDER_CLASSIFIER_PROMPT = """You are a directory classifier. You will be given the name and contents of a folder. Decide whether ALL files in it are clearly related to each other and clearly not receipts or financial documents.

Skip the folder (skip: true) when the listing suggests:
- Files share a common naming pattern or project prefix (e.g. "B-29-1828-WingSpars.pdf", "B-29-1829-Fuselage.pdf")
- The folder contains files typical of a git repo or software project (README, LICENSE, node_modules, package.json, .git, src/, etc.)
- The folder contains files typical of a hardware/3D printing/RC component snapshot (e.g. .3mf, .stl, .m3d, .step files)
- The folder name describes a specific project, part, component, or snapshot

Process the folder (skip: false) when:
- Files appear unrelated or the folder name is generic (Downloads, Documents, misc, temp)
- Any file could plausibly be a receipt, invoice, shipping label, or financial document
- The folder is empty

If any file could plausibly be a receipt or financial document, say so in the reason.

Return ONLY a valid JSON object on a single line. No explanation, no markdown, no code fences.

Examples:
{{"skip": false, "reason": "Folder contains a mix of unrelated files."}}
{{"skip": true, "reason": "All files are numbered B-29 aircraft drawings — clearly a build project."}}
{{"skip": true, "reason": "Contains README, LICENSE, and src/ — this is a git repository."}}
{{"skip": false, "reason": "Contains invoice.pdf which could be a receipt."}}

Directory listing:
{listing}

JSON:"""


@dataclass
class FolderClassificationResult:
    skip: bool
    reason: str


def build_folder_agent(ollama_base_url: str, model: str) -> ChatOllama:
    """Construct the LLM used for folder classification."""
    return ChatOllama(model=model, base_url=ollama_base_url)


def classify_folder(
    llm: ChatOllama, dir_path: Path
) -> FolderClassificationResult:
    """Classify a directory: fetch its listing directly, then ask the LLM to decide.

    The listing step is done in Python — not by the LLM — so we never rely on the
    model following a ReAct tool-call loop. Smaller local models skip Action steps
    unreliably; this approach is robust regardless of model size.

    Returns skip=False (process) on any error.
    """
    with tracer.start_as_current_span("folder.classify") as span:
        span.set_attribute("folder.path", str(dir_path))
        span.set_attribute("folder.name", dir_path.name)

        # Step 1: fetch the directory listing directly in Python
        listing = list_directory.invoke(str(dir_path))
        span.set_attribute("folder.listing_chars", len(listing))

        # Step 2: ask the LLM to classify based on the listing
        try:
            prompt = FOLDER_CLASSIFIER_PROMPT.format(listing=listing)
            response = llm.invoke(prompt)
            output = response.content if hasattr(response, "content") else str(response)

            parsed = _parse_json_output(output)

            if parsed is None:
                span.set_attribute("folder.outcome", "parse_error")
                span.set_status(StatusCode.ERROR, "Failed to parse folder classifier JSON output")
                logger.warning(
                    "Failed to parse folder classifier JSON output, defaulting to process",
                    extra={"folder.path": str(dir_path), "output": output[:200]},
                )
                return FolderClassificationResult(skip=False, reason="Parse error — defaulting to process")

            skip = parsed.get("skip", False)
            reason = parsed.get("reason", "")

            span.set_attribute("folder.skip", skip)
            span.set_attribute("folder.reason", reason)
            span.set_attribute("folder.outcome", "skip" if skip else "process")

            logger.info(
                f"{'Skipping' if skip else 'Processing'} folder: {dir_path.name} — {reason}",
                extra={"folder.path": str(dir_path), "folder.skip": skip},
            )

            return FolderClassificationResult(skip=skip, reason=reason)

        except Exception as e:
            span.set_attribute("folder.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(
                f"Folder classifier error: {e}",
                extra={"folder.path": str(dir_path)},
            )
            return FolderClassificationResult(skip=False, reason=f"Error — defaulting to process: {e}")


def _parse_json_output(output: str) -> Optional[dict]:
    """Parse JSON from LLM output string, stripping markdown fences if present."""
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*", "", output.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
