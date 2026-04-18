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

FOLDER_CLASSIFIER_PROMPT = """You are a directory classifier. You will be given a directory listing. Your job is to output a single JSON object.

The JSON object has exactly two fields:
- "skip": true or false
- "reason": one sentence explanation

Set "skip" to true when this folder should be IGNORED — meaning it clearly does NOT contain receipts and all the files are obviously related to each other (e.g. a build project, git repo, 3D model snapshot, RC aircraft plans, software application).

Set "skip" to false when this folder should be SCANNED for receipts — meaning it might contain receipts, invoices, or financial documents, OR the files are a mixed/unrelated collection.

Examples of skip=true folders:
- A folder named "B-29" containing "B-29-1828-WingSpars.pdf", "B-29-1829-Fuselage.pdf" → all related technical drawings
- A folder containing README.md, LICENSE, package.json, node_modules/ or other files commonly associated with a git repo or other project.
- A folder containing R-0904N-KC.m3d, R-0904N-KC.step, R-0904N-KC.png → 3D model snapshot
- A folder containing files that all have similar names. 
- An empty folder. Nothing to do so skip it.

Examples of skip=false folders:
- A folder with a generic name like "Downloads" or "Documents" or "Files" with a mix of unrelated files → could contain receipts
- A folder containing "invoice.pdf", receipt.txt etc that could be a receipt

IMPORTANT: If you believe the folder should be skipped, set "skip" to true. If you believe it should be scanned, set "skip" to false. Do not mix up the values.

Return ONLY valid JSON on a single line. No explanation before or after. No markdown. No code fences.

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
            span.set_attribute("folder.llm_output", output[:500])
            logger.debug(
                f"Folder classifier raw output for {dir_path.name}: {output[:200]}",
                extra={"folder.path": str(dir_path)},
            )

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
