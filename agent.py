import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from prompts import REACT_PROMPT
from telemetry import tracer
from tools import read_pdf, read_docx, read_text

logger = logging.getLogger(__name__)

TOOLS = [read_pdf, read_docx, read_text]


@dataclass
class ReceiptClassificationResult:
    is_receipt: bool
    reason: str
    category: Optional[str] = None
    suggested_path: Optional[str] = None


def build_agent(ollama_base_url: str, model: str) -> AgentExecutor:
    """Construct the LangChain ReAct AgentExecutor backed by a local Ollama model."""
    llm = ChatOllama(model=model, base_url=ollama_base_url)
    agent = create_react_agent(llm, TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=(
            "Your response was not in the correct format. "
            "You MUST use a tool first, then end with 'Final Answer: ' "
            "followed immediately by a JSON object. "
            "Do NOT output bare JSON without the 'Final Answer: ' prefix."
        ),
        max_iterations=8,
        return_intermediate_steps=True,
    )


def classify_file(
        agent_executor: AgentExecutor, file_path: Path
) -> Optional[ReceiptClassificationResult]:
    """Run the ReAct agent on a single file. Returns None if the file should be skipped."""
    with tracer.start_as_current_span("file.classify") as span:
        span.set_attribute("file.path", str(file_path))
        span.set_attribute("file.name", file_path.name)
        span.set_attribute("file.extension", file_path.suffix.lstrip(".").lower())
        try:
            file_size = file_path.stat().st_size
            span.set_attribute("file.size_bytes", file_size)
        except OSError:
            pass

        try:
            parent_folder = file_path.parent.name or "(root)"
            result = agent_executor.invoke({
                "input": (
                    f"Classify this file.\n"
                    f"Filename: {file_path.name}\n"
                    f"Parent folder: {parent_folder}\n"
                    f"Full path: {file_path}"
                )
            })

            # Count ReAct iterations from intermediate steps
            steps = result.get("intermediate_steps", [])
            span.set_attribute("agent.iterations", len(steps))

            # Record which tools were called and in what order
            tools_used = [step[0].tool for step in steps if hasattr(step[0], "tool")]
            span.set_attribute("agent.tools_used", ", ".join(tools_used))

            output = result.get("output", "")
            parsed = _parse_json_output(output)

            if parsed is None:
                span.set_attribute("file.outcome", "parse_error")
                span.set_status(StatusCode.ERROR, "Failed to parse agent JSON output")
                logger.warning(
                    "Failed to parse agent JSON output",
                    extra={"file.path": str(file_path)},
                )
                return None

            is_receipt = parsed.get("is_receipt", False)
            category = parsed.get("category")
            reason = parsed.get("reason", "")

            span.set_attribute("file.is_receipt", is_receipt)
            span.set_attribute("file.outcome", "receipt" if is_receipt else "not_receipt")
            if category:
                span.set_attribute("receipt.category", category)
            if reason:
                span.set_attribute("file.reason", reason)

            logger.info(
                f"{'Receipt' if is_receipt else 'Not a receipt'}: {file_path.name}"
                + (f" [{category}]" if category else ""),
                extra={
                    "file.path": str(file_path),
                    "file.is_receipt": is_receipt,
                    "receipt.category": category or "",
                    "agent.iterations": len(steps),
                },
            )

            return ReceiptClassificationResult(
                is_receipt=is_receipt,
                reason=reason,
                category=category,
                suggested_path=parsed.get("suggested_path"),
            )

        except Exception as e:
            span.set_attribute("file.outcome", "error")
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error(
                f"Agent error: {e}",
                extra={"file.path": str(file_path)},
            )
            return None


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
