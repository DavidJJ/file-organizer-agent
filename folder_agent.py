import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from opentelemetry.trace import StatusCode

from prompts import FOLDER_REACT_PROMPT
from telemetry import tracer
from tools.list_directory import list_directory

logger = logging.getLogger(__name__)

FOLDER_TOOLS = [list_directory]


@dataclass
class FolderClassificationResult:
    skip: bool
    reason: str


def build_folder_agent(ollama_base_url: str, model: str) -> AgentExecutor:
    """Construct the LangChain ReAct AgentExecutor for folder classification."""
    llm = ChatOllama(model=model, base_url=ollama_base_url)
    agent = create_react_agent(llm, FOLDER_TOOLS, FOLDER_REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=FOLDER_TOOLS,
        verbose=True,
        handle_parsing_errors=(
            "Your response was not in the correct format. "
            "You MUST use list_directory first, then end with 'Final Answer: ' "
            "followed immediately by a JSON object with 'skip' and 'reason' fields."
        ),
        max_iterations=4,
        return_intermediate_steps=True,
    )


def classify_folder(
    agent_executor: AgentExecutor, dir_path: Path
) -> FolderClassificationResult:
    """Run the folder classifier on a directory. Returns skip=False (process) on any error."""
    with tracer.start_as_current_span("folder.classify") as span:
        span.set_attribute("folder.path", str(dir_path))
        span.set_attribute("folder.name", dir_path.name)

        try:
            result = agent_executor.invoke({
                "input": f"Classify this folder: {dir_path}"
            })

            steps = result.get("intermediate_steps", [])
            span.set_attribute("agent.iterations", len(steps))

            output = result.get("output", "")
            parsed = _parse_json_output(output)

            if parsed is None:
                span.set_attribute("folder.outcome", "parse_error")
                span.set_status(StatusCode.ERROR, "Failed to parse folder agent JSON output")
                logger.warning(
                    "Failed to parse folder agent JSON output, defaulting to process",
                    extra={"folder.path": str(dir_path)},
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
                f"Folder agent error: {e}",
                extra={"folder.path": str(dir_path)},
            )
            return FolderClassificationResult(skip=False, reason=f"Error — defaulting to process: {e}")


def _parse_json_output(output: str) -> Optional[dict]:
    """Parse JSON from agent output string, stripping markdown fences if present."""
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*", "", output.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
