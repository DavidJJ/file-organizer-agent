import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama

from prompts import REACT_PROMPT
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
        handle_parsing_errors=True,
        max_iterations=5,
    )


def classify_file(
    agent_executor: AgentExecutor, file_path: Path
) -> Optional[ReceiptClassificationResult]:
    """Run the ReAct agent on a single file. Returns None if the file should be skipped."""
    try:
        result = agent_executor.invoke({"input": f"Classify this file: {file_path}"})
        output = result.get("output", "")
        parsed = _parse_json_output(output)
        if parsed is None:
            logger.warning(
                "Failed to parse agent JSON output",
                extra={"file.path": str(file_path)},
            )
            return None
        return ReceiptClassificationResult(
            is_receipt=parsed.get("is_receipt", False),
            reason=parsed.get("reason", ""),
            category=parsed.get("category"),
            suggested_path=parsed.get("suggested_path"),
        )
    except Exception as e:
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
