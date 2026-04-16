# Receipt Organizer Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangChain ReAct agent that scans a directory for files, uses a local Ollama LLM to classify receipts, and writes results to a CSV — with all observability routed through OpenTelemetry to OpenLIT.

**Architecture:** A Python CLI script wires together five independent modules: a file scanner, three LangChain read tools (PDF/DOCX/TXT), a ReAct agent, and a CSV writer. All logging and tracing flows through the OpenTelemetry SDK to a locally-running OpenLIT instance. A `.progress.json` checkpoint file enables resume-on-crash.

**Tech Stack:** Python 3.11+, LangChain, langchain-ollama, pdfplumber, python-docx, openlit, opentelemetry-sdk, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt` | All Python dependencies |
| `telemetry.py` | `setup_telemetry()` — init openlit + wire Python logging → OTel → OpenLIT |
| `scanner.py` | `scan_files(root_path)` — yield `.pdf/.docx/.doc/.txt` paths, skip hidden |
| `tools/__init__.py` | Re-export all three read tools |
| `tools/read_pdf.py` | `read_pdf` LangChain tool — extract text via pdfplumber, truncate to 3000 chars |
| `tools/read_docx.py` | `read_docx` LangChain tool — extract text via python-docx, truncate to 3000 chars |
| `tools/read_text.py` | `read_text` LangChain tool — read plain text file, truncate to 3000 chars |
| `prompts.py` | `REACT_PROMPT` — PromptTemplate for ReAct agent |
| `agent.py` | `ReceiptClassificationResult`, `build_agent()`, `classify_file()`, `_parse_json_output()` |
| `output.py` | `CSVWriter` — incremental CSV writing with 4 columns |
| `main.py` | CLI entrypoint — arg parsing, checkpoint load/save, orchestration loop |
| `tests/test_scanner.py` | Tests for `scan_files` |
| `tests/test_tools.py` | Tests for all three read tools |
| `tests/test_output.py` | Tests for `CSVWriter` |
| `tests/test_agent.py` | Tests for `_parse_json_output` and `classify_file` |

---

## Pre-requisites

Before starting, ensure:
1. Python 3.11+ is installed: `python3 --version`
2. Ollama is running: `ollama list` (should show available models)
3. You have at least one model pulled, e.g.: `ollama pull llama3.2`
4. Docker is running (for OpenLIT)

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tools/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
langchain>=0.2.0
langchain-ollama>=0.1.0
langchain-community>=0.2.0
pdfplumber>=0.11.0
python-docx>=1.1.0
openlit>=1.32.0
opentelemetry-sdk>=1.25.0
opentelemetry-api>=1.25.0
opentelemetry-exporter-otlp-proto-http>=1.25.0
opentelemetry-instrumentation-langchain>=0.37.0
pytest>=8.0.0
pytest-mock>=3.14.0
```

- [ ] **Step 2: Create virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 3: Create package init files**

Create `tests/__init__.py` — empty file:
```python
```

Create `tools/__init__.py` — empty for now, will be populated in Task 7:
```python
```

- [ ] **Step 4: Start OpenLIT via Docker Compose**

```bash
git clone https://github.com/openlit/openlit /tmp/openlit
cd /tmp/openlit
docker compose up -d
```

Expected: Containers start. Verify UI is reachable at `http://localhost:3000`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py tools/__init__.py
git commit -m "chore: project setup and dependencies"
```

---

## Task 2: File Scanner

**Files:**
- Create: `scanner.py`
- Create: `tests/test_scanner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scanner.py`:

```python
from pathlib import Path
import pytest
from scanner import scan_files


def test_scan_finds_pdf_files(tmp_path):
    (tmp_path / "invoice.pdf").write_bytes(b"fake pdf")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "image.png").write_bytes(b"fake png")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert "invoice.pdf" in names
    assert "notes.txt" in names
    assert "image.png" not in names


def test_scan_skips_hidden_files(tmp_path):
    (tmp_path / ".hidden.pdf").write_bytes(b"fake pdf")
    (tmp_path / "visible.pdf").write_bytes(b"fake pdf")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert ".hidden.pdf" not in names
    assert "visible.pdf" in names


def test_scan_skips_hidden_directories(tmp_path):
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "receipt.pdf").write_bytes(b"fake pdf")
    (tmp_path / "visible.pdf").write_bytes(b"fake pdf")

    results = list(scan_files(str(tmp_path)))
    paths = [str(p) for p in results]

    assert not any(".hidden_dir" in p for p in paths)
    assert any("visible.pdf" in p for p in paths)


def test_scan_walks_subdirectories(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.pdf").write_bytes(b"fake pdf")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert "nested.pdf" in names


def test_scan_finds_docx_and_txt(tmp_path):
    (tmp_path / "doc.docx").write_bytes(b"fake docx")
    (tmp_path / "old.doc").write_bytes(b"fake doc")
    (tmp_path / "note.txt").write_text("text")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert "doc.docx" in names
    assert "old.doc" in names
    assert "note.txt" in names


def test_scan_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "file.pdf").write_bytes(b"fake pdf")

    results = list(scan_files("~"))
    names = [p.name for p in results]

    assert "file.pdf" in names
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scanner.py -v
```

Expected: `ModuleNotFoundError: No module named 'scanner'`

- [ ] **Step 3: Implement scanner.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scanner.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scanner.py tests/test_scanner.py
git commit -m "feat: add file scanner"
```

---

## Task 3: CSV Output Writer

**Files:**
- Create: `output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_output.py`:

```python
import csv
from output import CSVWriter

FIELDNAMES = ["Receipt Category", "Original File Location", "Suggested File Location", "Reason"]


def test_csv_writer_creates_file_with_header(tmp_path):
    csv_path = str(tmp_path / "receipts.csv")
    CSVWriter(csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDNAMES


def test_csv_writer_appends_receipt(tmp_path):
    csv_path = str(tmp_path / "receipts.csv")
    writer = CSVWriter(csv_path)

    writer.append_receipt(
        category="Hobby / Radio Control",
        original_path="/Users/david/Downloads/hobbyking.pdf",
        suggested_path="~/Documents/Receipts/Hobby_RC/hobbyking.pdf",
        reason="Invoice from HobbyKing with itemized parts.",
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["Receipt Category"] == "Hobby / Radio Control"
    assert rows[0]["Original File Location"] == "/Users/david/Downloads/hobbyking.pdf"
    assert rows[0]["Suggested File Location"] == "~/Documents/Receipts/Hobby_RC/hobbyking.pdf"
    assert rows[0]["Reason"] == "Invoice from HobbyKing with itemized parts."


def test_csv_writer_appends_multiple_rows(tmp_path):
    csv_path = str(tmp_path / "receipts.csv")
    writer = CSVWriter(csv_path)

    writer.append_receipt("Mortgage", "/path/a.pdf", "~/Documents/Receipts/Mortgage/a.pdf", "Monthly payment.")
    writer.append_receipt("Travel", "/path/b.pdf", "~/Documents/Receipts/Travel/b.pdf", "Flight booking.")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["Receipt Category"] == "Mortgage"
    assert rows[1]["Receipt Category"] == "Travel"


def test_csv_writer_survives_reinit_on_existing_file(tmp_path):
    """Each CSVWriter call creates a fresh file (timestamps make paths unique in practice)."""
    csv_path = str(tmp_path / "receipts.csv")
    writer = CSVWriter(csv_path)
    writer.append_receipt("Mortgage", "/a.pdf", "~/Documents/Receipts/Mortgage/a.pdf", "Payment.")

    # Re-initializing overwrites (in practice, main.py uses a new timestamped path each run)
    CSVWriter(csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_output.py -v
```

Expected: `ModuleNotFoundError: No module named 'output'`

- [ ] **Step 3: Implement output.py**

```python
import csv

FIELDNAMES = ["Receipt Category", "Original File Location", "Suggested File Location", "Reason"]


class CSVWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    def append_receipt(self, category: str, original_path: str, suggested_path: str, reason: str):
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({
                "Receipt Category": category,
                "Original File Location": original_path,
                "Suggested File Location": suggested_path,
                "Reason": reason,
            })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_output.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add output.py tests/test_output.py
git commit -m "feat: add incremental CSV writer"
```

---

## Task 4: Read Tools (PDF, DOCX, TXT)

**Files:**
- Create: `tools/read_pdf.py`
- Create: `tools/read_docx.py`
- Create: `tools/read_text.py`
- Modify: `tools/__init__.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools.py`:

```python
from unittest.mock import patch, MagicMock


def test_read_pdf_returns_text():
    from tools.read_pdf import read_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Order Total: $29.99\nItem: Widget"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("tools.read_pdf.pdfplumber.open", return_value=mock_pdf):
        result = read_pdf.invoke("/fake/path/receipt.pdf")

    assert "Order Total" in result
    assert "Widget" in result


def test_read_pdf_handles_error():
    from tools.read_pdf import read_pdf

    with patch("tools.read_pdf.pdfplumber.open", side_effect=Exception("corrupt file")):
        result = read_pdf.invoke("/fake/path/broken.pdf")

    assert "Error reading PDF" in result


def test_read_pdf_truncates_long_text():
    from tools.read_pdf import read_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "x" * 5000
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("tools.read_pdf.pdfplumber.open", return_value=mock_pdf):
        result = read_pdf.invoke("/fake/path/long.pdf")

    assert len(result) <= 3000


def test_read_pdf_handles_no_text():
    from tools.read_pdf import read_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = None
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("tools.read_pdf.pdfplumber.open", return_value=mock_pdf):
        result = read_pdf.invoke("/fake/path/empty.pdf")

    assert "No text" in result


def test_read_docx_returns_text():
    from tools.read_docx import read_docx

    mock_para1 = MagicMock()
    mock_para1.text = "Invoice #12345"
    mock_para2 = MagicMock()
    mock_para2.text = "Amount Due: $100.00"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]

    with patch("tools.read_docx.Document", return_value=mock_doc):
        result = read_docx.invoke("/fake/path/invoice.docx")

    assert "Invoice #12345" in result
    assert "Amount Due: $100.00" in result


def test_read_docx_handles_error():
    from tools.read_docx import read_docx

    with patch("tools.read_docx.Document", side_effect=Exception("bad file")):
        result = read_docx.invoke("/fake/path/broken.docx")

    assert "Error reading DOCX" in result


def test_read_text_returns_content(tmp_path):
    from tools.read_text import read_text

    test_file = tmp_path / "receipt.txt"
    test_file.write_text("Purchase confirmation\nTotal: $50.00")

    result = read_text.invoke(str(test_file))

    assert "Purchase confirmation" in result
    assert "Total: $50.00" in result


def test_read_text_truncates_long_content(tmp_path):
    from tools.read_text import read_text

    test_file = tmp_path / "long.txt"
    test_file.write_text("x" * 5000)

    result = read_text.invoke(str(test_file))

    assert len(result) <= 3000


def test_read_text_handles_missing_file():
    from tools.read_text import read_text

    result = read_text.invoke("/nonexistent/path/file.txt")

    assert "Error reading file" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.read_pdf'`

- [ ] **Step 3: Implement tools/read_pdf.py**

```python
import pdfplumber
from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file. Input must be an absolute file path."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
                if len(text) >= MAX_CHARS:
                    break
        return text[:MAX_CHARS] if text.strip() else "No text could be extracted from this PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"
```

- [ ] **Step 4: Implement tools/read_docx.py**

```python
from docx import Document
from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def read_docx(file_path: str) -> str:
    """Extract text from a Word document (.docx). Input must be an absolute file path."""
    try:
        doc = Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text[:MAX_CHARS] if text.strip() else "No text could be extracted from this document."
    except Exception as e:
        return f"Error reading DOCX: {e}"
```

- [ ] **Step 5: Implement tools/read_text.py**

```python
from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def read_text(file_path: str) -> str:
    """Read a plain text file. Input must be an absolute file path."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(MAX_CHARS)
    except Exception as e:
        return f"Error reading file: {e}"
```

- [ ] **Step 6: Update tools/__init__.py to export all tools**

```python
from .read_pdf import read_pdf
from .read_docx import read_docx
from .read_text import read_text

__all__ = ["read_pdf", "read_docx", "read_text"]
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add tools/ tests/test_tools.py
git commit -m "feat: add PDF, DOCX, and text reader tools"
```

---

## Task 5: Prompts

**Files:**
- Create: `prompts.py`

No separate test file — the prompt template is validated by the agent tests in Task 6.

- [ ] **Step 1: Implement prompts.py**

```python
from langchain_core.prompts import PromptTemplate

# ReAct agents require exactly these four input variables in the template:
# {tools}, {tool_names}, {input}, {agent_scratchpad}
REACT_TEMPLATE = """You are a file classification assistant. Determine whether a file is a receipt or proof of purchase.

You have access to these tools:
{tools}

Use this EXACT format — do not deviate:

Thought: I need to read the file to determine if it is a receipt
Action: the tool to use, must be one of [{tool_names}]
Action Input: the exact file path
Observation: the file contents returned by the tool
Thought: Based on the contents, I can now classify this file
Final Answer: a single valid JSON object (no markdown, no code fences)

If the file IS a receipt, the JSON must have these fields:
  is_receipt: true
  category: one of [Mortgage, Utilities, Insurance, Groceries, Travel, Event Tickets, Hobby / Radio Control, Subscription, Other] or a new category if clearly warranted
  reason: one sentence explaining why this is a receipt
  suggested_path: the destination path as ~/Documents/Receipts/<Category>/<filename>

If the file is NOT a receipt:
  is_receipt: false
  reason: one sentence explaining why this is not a receipt

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)
```

- [ ] **Step 2: Verify the prompt template renders without error**

```bash
python3 -c "
from prompts import REACT_PROMPT
rendered = REACT_PROMPT.format(
    tools='read_pdf: reads a pdf',
    tool_names='read_pdf',
    input='Classify this file: /tmp/test.pdf',
    agent_scratchpad=''
)
print('OK:', rendered[:80])
"
```

Expected: Prints `OK:` followed by the first 80 characters of the rendered prompt.

- [ ] **Step 3: Commit**

```bash
git add prompts.py
git commit -m "feat: add ReAct agent prompt"
```

---

## Task 6: Agent

**Files:**
- Create: `agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock
from agent import _parse_json_output, classify_file, ReceiptClassificationResult


def test_parse_json_output_valid_receipt():
    output = '{"is_receipt": true, "category": "Hobby / Radio Control", "reason": "HobbyKing invoice.", "suggested_path": "~/Documents/Receipts/Hobby_RC/order.pdf"}'
    result = _parse_json_output(output)

    assert result is not None
    assert result["is_receipt"] is True
    assert result["category"] == "Hobby / Radio Control"
    assert result["reason"] == "HobbyKing invoice."


def test_parse_json_output_not_receipt():
    output = '{"is_receipt": false, "reason": "This is a user manual."}'
    result = _parse_json_output(output)

    assert result is not None
    assert result["is_receipt"] is False
    assert result["reason"] == "This is a user manual."


def test_parse_json_output_strips_markdown_fences():
    output = '```json\n{"is_receipt": false, "reason": "Not a receipt."}\n```'
    result = _parse_json_output(output)

    assert result is not None
    assert result["is_receipt"] is False


def test_parse_json_output_returns_none_on_invalid():
    result = _parse_json_output("Sorry, I cannot determine this from the file.")
    assert result is None


def test_classify_file_returns_receipt_result():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"is_receipt": true, "category": "Travel", "reason": "Flight booking.", "suggested_path": "~/Documents/Receipts/Travel/ticket.pdf"}'
    }

    result = classify_file(mock_executor, Path("/fake/ticket.pdf"))

    assert result is not None
    assert result.is_receipt is True
    assert result.category == "Travel"
    assert result.reason == "Flight booking."
    assert result.suggested_path == "~/Documents/Receipts/Travel/ticket.pdf"


def test_classify_file_returns_non_receipt_result():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"is_receipt": false, "reason": "This is a user manual."}'
    }

    result = classify_file(mock_executor, Path("/fake/manual.pdf"))

    assert result is not None
    assert result.is_receipt is False
    assert result.reason == "This is a user manual."


def test_classify_file_returns_none_on_malformed_json():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {"output": "I cannot determine this."}

    result = classify_file(mock_executor, Path("/fake/unknown.pdf"))

    assert result is None


def test_classify_file_returns_none_on_agent_exception():
    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = Exception("Ollama connection refused")

    result = classify_file(mock_executor, Path("/fake/file.pdf"))

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Implement agent.py**

```python
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add LangChain ReAct agent with JSON output parser"
```

---

## Task 7: Telemetry

**Files:**
- Create: `telemetry.py`

Telemetry requires a live OpenLIT instance to fully verify — the test here is a smoke test only.

- [ ] **Step 1: Implement telemetry.py**

```python
import logging
import os

import openlit
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor


def setup_telemetry() -> None:
    """Initialize OpenTelemetry: auto-instrument LangChain and route all Python
    logging through OTel to the local OpenLIT instance."""
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

    # Auto-instruments LangChain traces + sets up OTLP trace/metrics exporters
    openlit.init(otlp_endpoint=otlp_endpoint)

    # Wire Python logging → OTel log records → OpenLIT
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint))
    )
    set_logger_provider(logger_provider)

    handler = LoggingHandler(logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
```

- [ ] **Step 2: Smoke test — verify telemetry initializes without error**

```bash
python3 -c "
from telemetry import setup_telemetry
import logging
setup_telemetry()
logger = logging.getLogger('smoke')
logger.info('Telemetry smoke test', extra={'test': True})
print('Telemetry setup OK')
"
```

Expected: Prints `Telemetry setup OK` with no exceptions. Check `http://localhost:3000` in OpenLIT — the log record should appear within a few seconds.

- [ ] **Step 3: Commit**

```bash
git add telemetry.py
git commit -m "feat: add OpenTelemetry setup routing all logs to OpenLIT"
```

---

## Task 8: Main Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement main.py**

```python
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from agent import build_agent, classify_file
from output import CSVWriter
from scanner import scan_files
from telemetry import setup_telemetry

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
    agent_executor = build_agent(args.ollama_url, args.model)
    csv_writer = CSVWriter(str(csv_path))

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
            extra={"file.path": path_str, "file.type": file_path.suffix.lstrip(".")},
        )

        result = classify_file(agent_executor, file_path)

        if result and result.is_receipt:
            csv_writer.append_receipt(
                category=result.category or "Other",
                original_path=path_str,
                suggested_path=result.suggested_path
                or f"~/Documents/Receipts/Other/{file_path.name}",
                reason=result.reason,
            )
            logger.info(
                f"Receipt found: {result.category}",
                extra={"file.path": path_str, "receipt.category": result.category},
            )

        processed.add(path_str)
        save_progress(processed)

    logger.info(f"Scan complete. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI help works**

```bash
python3 main.py --help
```

Expected: Prints argument descriptions for `--root`, `--model`, `--ollama-url`, `--output-dir`.

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

```bash
pytest -v
```

Expected: All tests PASS (scanner: 6, output: 4, tools: 9, agent: 8 = 27 total).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point and orchestration loop"
```

---

## Task 9: End-to-End Smoke Test

This task verifies the full pipeline works against a real (small) directory with Ollama running.

- [ ] **Step 1: Create a small test directory with sample files**

```bash
mkdir -p /tmp/receipt_test
# Create a fake receipt text file
cat > /tmp/receipt_test/amazon_order.txt << 'EOF'
Order Confirmation
Order #112-3456789-0123456
Item: USB-C Cable x2
Subtotal: $15.98
Shipping: $0.00
Total: $15.98
Thank you for shopping with Amazon!
EOF

# Create a non-receipt text file
cat > /tmp/receipt_test/readme.txt << 'EOF'
This is the README for my project.
It describes how to set up the development environment.
See the docs folder for more information.
EOF
```

- [ ] **Step 2: Run the agent against the test directory**

```bash
python3 main.py --root /tmp/receipt_test --output-dir /tmp
```

Expected:
- Console shows `[1/2] Scanning: amazon_order.txt` and `[2/2] Scanning: readme.txt`
- A `receipts_<timestamp>.csv` is created in `/tmp`
- The CSV contains one row for `amazon_order.txt` with category similar to "Groceries" or "Other"
- `readme.txt` does not appear in the CSV

- [ ] **Step 3: Verify the CSV**

```bash
cat /tmp/receipts_*.csv
```

Expected output (category may vary):
```
Receipt Category,Original File Location,Suggested File Location,Reason
Other,/tmp/receipt_test/amazon_order.txt,~/Documents/Receipts/Other/amazon_order.txt,Amazon order confirmation showing purchased items and total.
```

- [ ] **Step 4: Open OpenLIT at http://localhost:3000**

Verify you can see:
- Traces for the two LLM calls
- Log records showing `[1/2] Scanning` and `[2/2] Scanning`
- Token counts per LLM invocation

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete receipt organizer agent — all tasks done"
```

---

## Running Against Your Downloads Folder

Once smoke test passes, run against your real Downloads:

```bash
python3 main.py
```

This uses the default `~/Downloads` root. Results appear in `receipts_<timestamp>.csv` in the current directory. If the run is interrupted, re-running picks up where it left off via `.progress.json`.

To scan a different folder:

```bash
python3 main.py --root ~/Documents
```

To use a different Ollama model:

```bash
python3 main.py --model mistral
```
