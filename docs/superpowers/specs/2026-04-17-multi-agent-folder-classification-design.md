# Multi-Agent Folder Classification — Design Spec

**Date:** 2026-04-17
**Status:** Approved

---

## Overview

Extend the receipt organizer from a single flat file-processing loop into a multi-agent architecture. A new `FolderClassifierAgent` decides — at each directory level — whether a folder's contents are clearly related and non-receipt in nature, and if so, skips the entire subtree. A `DirectoryCoordinator` orchestrates the two agents recursively. The existing `FileClassifierAgent` (currently `agent.py`) is unchanged except for a minor prompt enhancement.

---

## Problem Being Solved

The current system processes every supported file individually. This causes two issues:

1. **Cohesive project folders are shredded into individual files.** A folder like `B-29/` containing dozens of numbered drawing PDFs, or `9mm-insulated-shaft-potentiometer-vertical-type-1.snapshot.5/` containing component files, gets processed file-by-file. The LLM has no folder-level context and may misclassify technical documents as receipts.

2. **Receipt misclassification.** Without prominent folder context, the LLM relies heavily on file content, which for technical documents can contain numbers, part codes, and totals that superficially resemble receipts.

---

## Goals

- Add a folder-level classification step that skips entire subtrees when files are clearly related and non-financial
- Make the folder name an explicit, prominent signal in file classification
- Preserve existing CSV output format and resume/checkpoint behavior
- Demonstrate a multi-agent coordinator pattern for learning purposes

## Non-Goals

- Reading file content during folder classification (names only)
- Changing the LLM backend or model configuration
- Changing the CSV output format

---

## Architecture

### Agents

| Agent | File | LLM? | Responsibility |
|---|---|---|---|
| `FolderClassifierAgent` | `folder_agent.py` | Yes | Given folder name + child names, decide: skip or process? |
| `FileClassifierAgent` | `agent.py` | Yes | Given a file path + parent folder, decide: receipt or not? |
| `DirectoryCoordinator` | `coordinator.py` | No | Recursive orchestrator — drives the two agents, owns traversal |

### File Changes

| File | Status | Change |
|---|---|---|
| `coordinator.py` | **NEW** | `DirectoryCoordinator` class |
| `folder_agent.py` | **NEW** | `FolderClassifierAgent` + `FolderClassificationResult` |
| `tools/list_directory.py` | **NEW** | LangChain tool: list folder name + direct children |
| `prompts.py` | **UPDATED** | Add `FOLDER_REACT_PROMPT` alongside existing `REACT_PROMPT` |
| `agent.py` | **UPDATED** | Prompt enhanced to include parent folder name explicitly |
| `main.py` | **UPDATED** | Replace flat scan loop with `coordinator.process()` call |
| `scanner.py` | **RETIRED** | Coordinator owns traversal; `SUPPORTED_EXTENSIONS` constant moves to `coordinator.py` |

---

## Component Details

### `tools/list_directory.py`

A LangChain `@tool` that takes a directory path and returns a plain-text listing:

```
Folder name: B-29
Contents (18 items):
  [file] B-29-1828-WingSpars1.pdf
  [file] B-29-1829-WingSpars2.pdf
  [file] B-29-1830-Fuselage.pdf
  [dir]  reference-images
  ...
```

Hidden entries (names starting with `.`) are excluded. If the directory is unreadable, returns an error string.

### `folder_agent.py` — FolderClassifierAgent

A LangChain ReAct agent with one tool (`list_directory`). One LLM call per directory.

**Prompt (skip/process decision):**

```
You are a directory classifier. Given a folder path, use the list_directory
tool to read its name and contents, then decide whether to skip the entire folder.

Skip the folder (skip: true) when ALL of the following are true:
  1. Files share a common naming pattern or project prefix
     (e.g. "B-29-1828-WingSpars.pdf", "B-29-1829-Fuselage.pdf")
  2. The folder name describes a project, part, component, or snapshot
     (e.g. "B-29", "9mm-potentiometer.snapshot.5", "arduino-uno-r3")
  3. Files are clearly technical in nature (drawings, 3D models, datasheets,
     schematics, firmware, build instructions)

Process the folder (skip: false) when ANY of the following are true:
  - Files appear unrelated to each other
  - The folder name is generic (Downloads, Documents, misc, temp)
  - Any file could plausibly be a receipt, invoice, or financial document

Return ONLY valid JSON:
  {"skip": true, "reason": "..."}   or   {"skip": false, "reason": "..."}
```

**Result dataclass:**

```python
@dataclass
class FolderClassificationResult:
    skip: bool
    reason: str
```

### `coordinator.py` — DirectoryCoordinator

Pure orchestration — no LLM. Drives both agents recursively.

**Constructor:**
```python
DirectoryCoordinator(
    folder_agent: FolderClassifierAgent,
    file_agent: AgentExecutor,
    csv_writer: CSVWriter,
    progress: Progress,          # load/save for both files and skipped_dirs
)
```

**`process(directory: Path)` algorithm:**

```
1. If directory is in progress.skipped_dirs → log skip, return
2. List direct children (non-hidden files + subdirs)
3. Ask FolderClassifierAgent(directory)
4. If skip:
     - Add directory to progress.skipped_dirs
     - Save progress
     - Log: "Skipping folder: <dir> — <reason>"
     - Return (entire subtree pruned)
5. If process:
     a. For each direct file with a supported extension:
          - If file in progress.files → skip
          - Call FileClassifierAgent(file, parent_folder=directory.name)
          - If receipt → csv_writer.append_receipt(...)
          - Add file to progress.files, save progress
     b. For each direct subdirectory (non-hidden):
          - Recurse: self.process(subdir)
```

### `prompts.py` updates

Add `FOLDER_REACT_PROMPT` alongside the existing `REACT_PROMPT`. The folder prompt is a focused ReAct template with `list_directory` as its only tool.

### `agent.py` prompt enhancement

The per-file prompt now includes the parent folder name as an explicit field:

```
Classify this file.
Filename: {filename}
Parent folder: {parent_folder}
Full path: {full_path}
```

This makes the folder name a prominent hint rather than buried in the path string.

### `main.py` changes

The file loop is replaced:

```python
# Before
all_files = list(scan_files(args.root))
for file_path in all_files:
    result = classify_file(agent_executor, file_path)

# After
coordinator = DirectoryCoordinator(
    folder_agent=build_folder_agent(args.ollama_url, args.model),
    file_agent=build_agent(args.ollama_url, args.model),
    csv_writer=csv_writer,
    progress=progress,
)
coordinator.process(Path(args.root).expanduser().resolve())
```

Progress loading/saving moves into `coordinator.py`. The `Progress` object holds two sets: `files` (processed file paths) and `skipped_dirs` (skipped directory paths).

---

## Progress Tracking

`.progress.json` gains a second key:

```json
{
  "files": ["/path/to/file.pdf"],
  "skipped_dirs": ["/path/to/B-29"]
}
```

On resume:
- Any path in `skipped_dirs` → skip without LLM call
- Any path in `files` → skip without LLM call

Backward compatibility: if `.progress.json` contains a plain list (old format), it is read as `files` with `skipped_dirs` defaulting to empty.

---

## Observability

New OTel spans:

| Span | Attributes |
|---|---|
| `folder.classify` | `folder.path`, `folder.skip`, `folder.reason`, `folder.child_count` |
| `directory.process` | `directory.path`, `directory.depth` |

Existing `file.classify` spans are unchanged.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `list_directory` fails (permissions) | Folder agent returns `skip: false` with error reason — files are processed individually |
| Folder agent returns malformed JSON | Log warning, treat as `skip: false` — process files individually (safe default) |
| File agent error (unchanged) | Log error, skip file, record in `progress.files` |

The safe default for any folder agent failure is always to process — we never silently skip files due to an agent error.

---

## Example Traversal

Given:
```
Downloads/
  B-29/
    B-29-1828-WingSpars1.pdf
    B-29-1829-WingSpars2.pdf
  another_folder/
    random_receipt.txt
    deeper_folder/
      another_receipt.txt
      project_y/
        related_y1.doc
        related_y2.doc
  invoice.pdf
```

Coordinator trace:
```
process(Downloads/)
  → FolderClassifier: skip=false (generic folder, mixed content)
  → FileClassifier: invoice.pdf
  → process(B-29/)
      → FolderClassifier: skip=true (project folder, numbered technical files)
      → SKIPPED
  → process(another_folder/)
      → FolderClassifier: skip=false (mixed content)
      → FileClassifier: random_receipt.txt
      → process(deeper_folder/)
          → FolderClassifier: skip=false (mixed content + subdir)
          → FileClassifier: another_receipt.txt
          → process(project_y/)
              → FolderClassifier: skip=true (related docs)
              → SKIPPED
```
