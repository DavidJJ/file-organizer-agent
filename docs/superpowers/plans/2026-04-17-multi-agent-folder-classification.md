# Multi-Agent Folder Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the receipt organizer into a multi-agent architecture where a FolderClassifierAgent decides per-directory whether to skip the entire subtree, with a DirectoryCoordinator orchestrating both agents recursively.

**Architecture:** A new `DirectoryCoordinator` replaces the flat file loop in `main.py`. At each directory level it calls `FolderClassifierAgent` (a ReAct LLM agent with a `list_directory` tool) to decide skip or process; if skip, the whole subtree is pruned; if process, direct files go to the existing `FileClassifierAgent` and subdirectories are recursed into. Progress tracking is extended to record skipped directories alongside processed files.

**Tech Stack:** Python 3.11+, LangChain (ReAct agent), langchain-ollama, langchain-classic, OpenTelemetry, pytest

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `tools/list_directory.py` | **CREATE** | LangChain tool: list folder name + direct children |
| `prompts.py` | **MODIFY** | Add `FOLDER_REACT_PROMPT` alongside existing `REACT_PROMPT` |
| `folder_agent.py` | **CREATE** | `FolderClassificationResult`, `build_folder_agent()`, `classify_folder()` |
| `coordinator.py` | **CREATE** | `Progress` dataclass, `load_progress()`, `save_progress()`, `DirectoryCoordinator` |
| `agent.py` | **MODIFY** | Add parent folder name to per-file prompt input |
| `main.py` | **MODIFY** | Replace flat scan loop with `DirectoryCoordinator.process()` |
| `tests/test_list_directory.py` | **CREATE** | Tests for `list_directory` tool |
| `tests/test_folder_agent.py` | **CREATE** | Tests for `classify_folder` and `_parse_json_output` |
| `tests/test_coordinator.py` | **CREATE** | Tests for `DirectoryCoordinator`, `load_progress`, `save_progress` |
| `tests/test_agent.py` | **MODIFY** | Update `test_classify_file_*` for new parent folder in prompt input |

---

## Task 1: `list_directory` Tool

**Files:**
- Create: `tools/list_directory.py`
- Create: `tests/test_list_directory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_list_directory.py`:

```python
import pytest
from tools.list_directory import list_directory


def test_list_directory_returns_folder_name_and_contents(tmp_path):
    (tmp_path / "file1.pdf").write_bytes(b"")
    (tmp_path / "file2.txt").write_text("hello")
    subdir = tmp_path / "subproject"
    subdir.mkdir()

    result = list_directory.invoke(str(tmp_path))

    assert tmp_path.name in result
    assert "[file]  file1.pdf" in result
    assert "[file]  file2.txt" in result
    assert "[dir]  subproject" in result


def test_list_directory_excludes_hidden_entries(tmp_path):
    (tmp_path / ".hidden").write_text("")
    (tmp_path / "visible.pdf").write_bytes(b"")

    result = list_directory.invoke(str(tmp_path))

    assert ".hidden" not in result
    assert "visible.pdf" in result


def test_list_directory_handles_empty_dir(tmp_path):
    result = list_directory.invoke(str(tmp_path))
    assert "(empty)" in result


def test_list_directory_handles_nonexistent_path(tmp_path):
    result = list_directory.invoke(str(tmp_path / "nonexistent"))
    assert "Error" in result


def test_list_directory_shows_item_count(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.pdf").write_bytes(b"")

    result = list_directory.invoke(str(tmp_path))
    assert "2 items" in result


def test_list_directory_handles_non_directory_path(tmp_path):
    f = tmp_path / "file.pdf"
    f.write_bytes(b"")

    result = list_directory.invoke(str(f))
    assert "Error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_list_directory.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.list_directory'`

- [ ] **Step 3: Implement `tools/list_directory.py`**

```python
from pathlib import Path

from langchain_core.tools import tool


@tool
def list_directory(dir_path: str) -> str:
    """List the name and direct contents of a directory. Input must be an absolute directory path."""
    try:
        path = Path(dir_path)
        if not path.is_dir():
            return f"Error: {dir_path} is not a directory."

        children = []
        for child in sorted(path.iterdir()):
            if child.name.startswith("."):
                continue
            kind = "[dir] " if child.is_dir() else "[file]"
            children.append(f"  {kind}  {child.name}")

        count = len(children)
        listing = "\n".join(children) if children else "  (empty)"
        return f"Folder name: {path.name}\nContents ({count} items):\n{listing}"
    except PermissionError as e:
        return f"Error reading directory: {e}"
    except Exception as e:
        return f"Error: {e}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_list_directory.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/list_directory.py tests/test_list_directory.py
git commit -m "feat: add list_directory tool for folder classifier agent"
```

---

## Task 2: Folder Classifier Prompt

**Files:**
- Modify: `prompts.py`

No separate test — the prompt is validated by the folder agent tests in Task 3.

- [ ] **Step 1: Add `FOLDER_REACT_PROMPT` to `prompts.py`**

Append to the end of `prompts.py` (after the existing `REACT_PROMPT = ...` line):

```python
FOLDER_REACT_TEMPLATE = """You are a directory classifier. Determine whether a folder's files are all clearly related to each other AND clearly not receipts or financial documents.

You have access to these tools:
{tools}

You MUST follow this EXACT format every time, no exceptions:

Thought: I need to list the directory contents to classify this folder
Action: list_directory
Action Input: /path/to/folder
Observation: <contents returned by the tool>
Thought: Based on the folder name and file names, I can now classify this folder
Final Answer: {{"skip": false, "reason": "Folder contains a mix of unrelated files."}}

Another example where we skip:

Thought: I need to list the directory contents to classify this folder
Action: list_directory
Action Input: /path/to/B-29
Observation: <contents returned by the tool>
Thought: All files share the B-29 prefix and are technical drawings — this is a project folder.
Final Answer: {{"skip": true, "reason": "All files are B-29 model aircraft drawings with sequential numbering."}}

CRITICAL RULES:
- You MUST call list_directory FIRST before giving a Final Answer. Never skip the Action step.
- The tool name must be one of: [{tool_names}]
- The Final Answer line MUST start with exactly "Final Answer: " followed immediately by JSON.
- Do NOT use markdown or code fences.

Skip the folder (skip: true) when ALL of the following are true:
  1. Files share a common naming pattern or project prefix (e.g. "B-29-1828-WingSpars.pdf", "B-29-1829-Fuselage.pdf")
  2. The folder name describes a project, part, component, or snapshot (e.g. "B-29", "9mm-potentiometer.snapshot.5", "arduino-uno-r3")
  3. Files are clearly technical in nature (drawings, 3D models, datasheets, schematics, firmware, build instructions)

Process the folder (skip: false) when ANY of the following are true:
  - Files appear unrelated to each other
  - The folder name is generic (Downloads, Documents, misc, temp, files)
  - Any file could plausibly be a receipt, invoice, or financial document

Return ONLY valid JSON:
  {{"skip": true, "reason": "..."}}  or  {{"skip": false, "reason": "..."}}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

FOLDER_REACT_PROMPT = PromptTemplate.from_template(FOLDER_REACT_TEMPLATE)
```

- [ ] **Step 2: Verify the prompt template renders without error**

```bash
python3 -c "
from prompts import FOLDER_REACT_PROMPT
rendered = FOLDER_REACT_PROMPT.format(
    tools='list_directory: lists a directory',
    tool_names='list_directory',
    input='Classify this folder: /tmp/B-29',
    agent_scratchpad=''
)
print('OK:', rendered[:80])
"
```

Expected: Prints `OK:` followed by the first 80 characters of the rendered prompt.

- [ ] **Step 3: Commit**

```bash
git add prompts.py
git commit -m "feat: add folder classifier ReAct prompt"
```

---

## Task 3: FolderClassifierAgent

**Files:**
- Create: `folder_agent.py`
- Create: `tests/test_folder_agent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_folder_agent.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

from folder_agent import FolderClassificationResult, classify_folder, _parse_json_output


def test_parse_json_output_skip():
    output = '{"skip": true, "reason": "All files are B-29 drawings."}'
    result = _parse_json_output(output)
    assert result is not None
    assert result["skip"] is True
    assert result["reason"] == "All files are B-29 drawings."


def test_parse_json_output_process():
    output = '{"skip": false, "reason": "Mixed files, generic folder name."}'
    result = _parse_json_output(output)
    assert result is not None
    assert result["skip"] is False


def test_parse_json_output_strips_markdown_fences():
    output = '```json\n{"skip": true, "reason": "Project folder."}\n```'
    result = _parse_json_output(output)
    assert result is not None
    assert result["skip"] is True


def test_parse_json_output_returns_none_on_invalid():
    result = _parse_json_output("I cannot determine this.")
    assert result is None


def test_classify_folder_returns_skip_true():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"skip": true, "reason": "Numbered B-29 technical drawings."}',
        "intermediate_steps": [],
    }
    result = classify_folder(mock_executor, Path("/fake/B-29"))
    assert result.skip is True
    assert result.reason == "Numbered B-29 technical drawings."


def test_classify_folder_returns_skip_false():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"skip": false, "reason": "Mixed content, generic folder."}',
        "intermediate_steps": [],
    }
    result = classify_folder(mock_executor, Path("/fake/Downloads"))
    assert result.skip is False
    assert "Mixed content" in result.reason


def test_classify_folder_defaults_to_process_on_malformed_json():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": "I cannot classify this folder.",
        "intermediate_steps": [],
    }
    result = classify_folder(mock_executor, Path("/fake/unknown"))
    assert result.skip is False


def test_classify_folder_defaults_to_process_on_agent_exception():
    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = Exception("Ollama connection refused")
    result = classify_folder(mock_executor, Path("/fake/folder"))
    assert result.skip is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_folder_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'folder_agent'`

- [ ] **Step 3: Implement `folder_agent.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_folder_agent.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add folder_agent.py tests/test_folder_agent.py
git commit -m "feat: add FolderClassifierAgent with skip/process decision"
```

---

## Task 4: DirectoryCoordinator

**Files:**
- Create: `coordinator.py`
- Create: `tests/test_coordinator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coordinator.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent import ReceiptClassificationResult
from coordinator import (
    DirectoryCoordinator,
    Progress,
    load_progress,
    save_progress,
    SUPPORTED_EXTENSIONS,
)
from folder_agent import FolderClassificationResult


def _make_coordinator(folder_decision, file_result, progress=None):
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    if progress is None:
        progress = Progress()
    return (
        DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress),
        csv_writer,
        progress,
    )


def test_coordinator_skips_folder_when_agent_says_skip(tmp_path):
    (tmp_path / "file.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=True, reason="project folder")), \
         patch("coordinator.classify_file") as mock_file, \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)
        mock_file.assert_not_called()

    assert str(tmp_path) in progress.skipped_dirs


def test_coordinator_processes_files_when_not_skipped(tmp_path):
    (tmp_path / "receipt.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()
    receipt = ReceiptClassificationResult(
        is_receipt=True,
        reason="invoice",
        category="Travel",
        suggested_path="~/Documents/Receipts/Travel/receipt.pdf",
    )

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", return_value=receipt), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    csv_writer.append_receipt.assert_called_once()


def test_coordinator_skips_already_processed_files(tmp_path):
    f = tmp_path / "file.pdf"
    f.write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress(files={str(f)})

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file") as mock_file, \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)
        mock_file.assert_not_called()


def test_coordinator_recurses_into_subdirectories(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    folder_call_paths = []

    def fake_classify_folder(agent, path):
        folder_call_paths.append(path)
        return FolderClassificationResult(skip=False, reason="mixed")

    with patch("coordinator.classify_folder", side_effect=fake_classify_folder), \
         patch("coordinator.classify_file", return_value=None), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    assert len(folder_call_paths) == 2
    assert tmp_path in folder_call_paths
    assert subdir in folder_call_paths


def test_coordinator_skips_previously_skipped_dirs(tmp_path):
    progress = Progress(skipped_dirs={str(tmp_path)})
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()

    with patch("coordinator.classify_folder") as mock_folder:
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)
        mock_folder.assert_not_called()


def test_coordinator_ignores_hidden_files(tmp_path):
    (tmp_path / ".hidden.pdf").write_bytes(b"")
    (tmp_path / "visible.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    classified = []

    def fake_classify_file(agent, path):
        classified.append(path.name)
        return None

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", side_effect=fake_classify_file), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    assert "visible.pdf" in classified
    assert ".hidden.pdf" not in classified


def test_coordinator_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "image.png").write_bytes(b"")
    (tmp_path / "receipt.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    classified = []

    def fake_classify_file(agent, path):
        classified.append(path.name)
        return None

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", side_effect=fake_classify_file), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    assert "receipt.pdf" in classified
    assert "image.png" not in classified


def test_load_progress_returns_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("coordinator.PROGRESS_FILE", str(tmp_path / ".progress.json")):
        progress = load_progress()
    assert len(progress.files) == 0
    assert len(progress.skipped_dirs) == 0


def test_load_progress_backward_compat_with_plain_list(tmp_path):
    progress_file = tmp_path / ".progress.json"
    progress_file.write_text(json.dumps(["/path/to/file.pdf"]))

    with patch("coordinator.PROGRESS_FILE", str(progress_file)):
        progress = load_progress()

    assert "/path/to/file.pdf" in progress.files
    assert len(progress.skipped_dirs) == 0


def test_load_progress_new_format(tmp_path):
    progress_file = tmp_path / ".progress.json"
    progress_file.write_text(json.dumps({
        "files": ["/path/to/file.pdf"],
        "skipped_dirs": ["/path/to/B-29"],
    }))

    with patch("coordinator.PROGRESS_FILE", str(progress_file)):
        progress = load_progress()

    assert "/path/to/file.pdf" in progress.files
    assert "/path/to/B-29" in progress.skipped_dirs


def test_save_progress_writes_both_sets(tmp_path):
    progress_file = tmp_path / ".progress.json"
    progress = Progress(
        files={"/path/to/file.pdf"},
        skipped_dirs={"/path/to/B-29"},
    )

    with patch("coordinator.PROGRESS_FILE", str(progress_file)):
        save_progress(progress)

    data = json.loads(progress_file.read_text())
    assert "/path/to/file.pdf" in data["files"]
    assert "/path/to/B-29" in data["skipped_dirs"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_coordinator.py -v
```

Expected: `ModuleNotFoundError: No module named 'coordinator'`

- [ ] **Step 3: Implement `coordinator.py`**

```python
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from langchain_classic.agents import AgentExecutor

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


def load_progress() -> Progress:
    if not os.path.exists(PROGRESS_FILE):
        return Progress()
    with open(PROGRESS_FILE) as f:
        data = json.load(f)
    # Backward compat: old format was a plain list of file paths
    if isinstance(data, list):
        return Progress(files=set(data))
    return Progress(
        files=set(data.get("files", [])),
        skipped_dirs=set(data.get("skipped_dirs", [])),
    )


def save_progress(progress: Progress) -> None:
    with open(PROGRESS_FILE, "w") as f:
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
    ):
        self.folder_agent = folder_agent
        self.file_agent = file_agent
        self.csv_writer = csv_writer
        self.progress = progress

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
                save_progress(self.progress)
                return

            # List children — guard against permission errors
            try:
                children = sorted(directory.iterdir())
            except PermissionError as e:
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
                save_progress(self.progress)

            # Recurse into subdirectories
            for child in children:
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    self.process(child)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_coordinator.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add coordinator.py tests/test_coordinator.py
git commit -m "feat: add DirectoryCoordinator with recursive folder/file orchestration"
```

---

## Task 5: Enhance FileClassifierAgent Prompt

**Files:**
- Modify: `agent.py` (lines 64-70)
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Update the test for `classify_file` to assert parent folder is in the invocation**

In `tests/test_agent.py`, update `test_classify_file_returns_receipt_result` and `test_classify_file_returns_non_receipt_result` to verify the input includes the parent folder:

```python
def test_classify_file_input_includes_parent_folder():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"is_receipt": false, "reason": "Not a receipt."}',
        "intermediate_steps": [],
    }

    classify_file(mock_executor, Path("/fake/Downloads/ticket.pdf"))

    call_args = mock_executor.invoke.call_args[0][0]
    assert "Downloads" in call_args["input"]
    assert "ticket.pdf" in call_args["input"]
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
pytest tests/test_agent.py::test_classify_file_input_includes_parent_folder -v
```

Expected: FAIL — `AssertionError: assert 'Downloads' in ...`

- [ ] **Step 3: Update `agent.py` to include parent folder in the prompt**

In `agent.py`, find the `agent_executor.invoke` call (around line 64) and change:

```python
result = agent_executor.invoke({
    "input": (
        f"Classify this file.\n"
        f"Filename: {file_path.name}\n"
        f"Full path: {file_path}"
    )
})
```

to:

```python
result = agent_executor.invoke({
    "input": (
        f"Classify this file.\n"
        f"Filename: {file_path.name}\n"
        f"Parent folder: {file_path.parent.name}\n"
        f"Full path: {file_path}"
    )
})
```

- [ ] **Step 4: Run all agent tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: All tests PASS (including the new one).

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: include parent folder name in file classifier prompt"
```

---

## Task 6: Update `main.py`

**Files:**
- Modify: `main.py`

No new tests — `main.py` is integration wiring covered by the coordinator tests.

- [ ] **Step 1: Replace the flat scan loop with the coordinator**

Replace the entire contents of `main.py` with:

```python
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
```

- [ ] **Step 2: Verify the CLI help still works**

```bash
python3 main.py --help
```

Expected: Prints argument descriptions for `--root`, `--model`, `--ollama-url`, `--output-dir`.

- [ ] **Step 3: Run the full test suite**

```bash
pytest -v
```

Expected: All tests PASS. (scanner tests still pass — `scanner.py` is not deleted, just unused by `main.py`.)

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire DirectoryCoordinator into main entry point"
```

---

## Task 7: End-to-End Smoke Test

- [ ] **Step 1: Create a test directory tree**

```bash
mkdir -p /tmp/receipt_test/B-29
mkdir -p /tmp/receipt_test/misc

# B-29 project folder — should be skipped
echo "Wing spar dimensions: 24mm x 6mm x 400mm" > /tmp/receipt_test/B-29/B-29-1828-WingSpars1.txt
echo "Fuselage rib count: 18, spacing: 22mm"     > /tmp/receipt_test/B-29/B-29-1829-Fuselage.txt

# Mixed folder — should be processed
cat > /tmp/receipt_test/misc/amazon_order.txt << 'EOF'
Order Confirmation #112-3456789
Item: USB-C Cable x2
Total: $15.98
Thank you for shopping with Amazon!
EOF

echo "This is a README for my project." > /tmp/receipt_test/misc/readme.txt
```

- [ ] **Step 2: Run the agent against the test directory**

```bash
python3 main.py --root /tmp/receipt_test --output-dir /tmp
```

Expected output (order may vary):
```
Processing folder: receipt_test — mixed content
Skipping folder: B-29 — numbered technical project files
Processing folder: misc — mixed content
Receipt: amazon_order.txt [Other]
Not a receipt: readme.txt
```

- [ ] **Step 3: Verify the CSV**

```bash
cat /tmp/receipts_*.csv
```

Expected: One row for `amazon_order.txt`. No rows for any B-29 files or `readme.txt`.

- [ ] **Step 4: Verify `.progress.json` contains both sections**

```bash
cat .progress.json
```

Expected structure:
```json
{
  "files": [
    "/tmp/receipt_test/misc/amazon_order.txt",
    "/tmp/receipt_test/misc/readme.txt"
  ],
  "skipped_dirs": [
    "/tmp/receipt_test/B-29"
  ]
}
```

- [ ] **Step 5: Re-run to verify resume works**

```bash
rm /tmp/receipts_*.csv
python3 main.py --root /tmp/receipt_test --output-dir /tmp
```

Expected: All files logged as "already processed" or "previously classified". New CSV is empty (no rows — all work was already done).

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: complete multi-agent folder classification"
```
