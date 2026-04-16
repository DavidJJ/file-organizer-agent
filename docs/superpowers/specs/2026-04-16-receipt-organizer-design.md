# Receipt Organizer Agent — Design Spec

**Date:** 2026-04-16  
**Status:** Approved

---

## Overview

An agentic Python script that scans a directory for files, uses a local LLM (via Ollama) to determine whether each file is a receipt, categorizes it, and produces a CSV inventory. No file content is sent to any cloud service — all inference runs locally.

---

## Goals

- Scan a configurable root directory (default: `~/Downloads`) for PDF, DOCX, and TXT files
- Classify each file as a receipt or not using a local LLM
- For receipts: assign a category, suggest a destination path, and provide a reason
- Output a CSV file with four columns: Receipt Category, Original File Location, Suggested File Location, Reason
- Emit all observability (traces, logs, progress, errors) via OpenTelemetry to a local OpenLIT instance

---

## Non-Goals

- Actually moving or copying files (CSV output only)
- OCR of scanned/image-based PDFs
- Cloud LLM usage of any kind

---

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent framework | LangChain (ReAct agent) |
| LLM backend | Ollama (local) — configurable model, default `llama3.2` |
| PDF extraction | `pdfplumber` |
| DOCX extraction | `python-docx` |
| Observability | OpenTelemetry SDK + OpenLIT (`openlit`) |
| OTel receiver/UI | OpenLIT (Docker Compose, UI at `http://localhost:3000`) |

---

## Project Structure

```
file-organizer/
├── main.py                  # Entry point — wires everything together
├── scanner.py               # Walks directory tree, yields file paths
├── agent.py                 # LangChain ReAct agent definition
├── prompts.py               # System prompt and per-file prompt templates
├── tools/
│   ├── __init__.py
│   ├── read_pdf.py          # LangChain tool: extracts text via pdfplumber
│   ├── read_docx.py         # LangChain tool: extracts text via python-docx
│   └── read_text.py         # LangChain tool: reads plain text files
├── output.py                # Incremental CSV writer
├── telemetry.py             # OpenTelemetry setup (LoggingHandler + openlit.init)
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-16-receipt-organizer-design.md
└── requirements.txt
```

---

## Architecture

### Flow

```
main.py
  └── telemetry.py         # Initialize OTel + openlit before anything else
  └── scanner.py           # Yield file paths from root directory
      └── for each file:
            agent.py       # Invoke ReAct agent with file path
              └── tools/   # Agent selects and calls appropriate read tool
            output.py      # If receipt: append row to CSV
```

### File Scanner (`scanner.py`)

- Accepts a root path (CLI argument, defaults to `~/Downloads`)
- Recursively walks the directory tree
- Yields absolute paths for files with extensions: `.pdf`, `.docx`, `.doc`, `.txt`
- Skips hidden files and directories (names starting with `.`)
- Logs total file count via OTel before processing begins

### Read Tools (`tools/`)

Each tool is a LangChain `@tool`-decorated function. All accept a file path string and return extracted text.

| Tool | Library | Truncation |
|---|---|---|
| `read_pdf` | `pdfplumber` | First ~3000 characters |
| `read_docx` | `python-docx` | First ~3000 characters |
| `read_text` | built-in `open()` | First ~3000 characters |

Truncation at 3000 characters is sufficient for receipt classification — receipts are short. This prevents unnecessarily large LLM context for long documents.

Each tool emits an OTel log record on success and on error.

### Agent (`agent.py`)

- **Type:** LangChain ReAct agent
- **LLM:** `ChatOllama` from `langchain-ollama`, model configurable via environment variable `OLLAMA_MODEL` (default: `llama3.2`)
- **Tools:** `read_pdf`, `read_docx`, `read_text`
- **Output parser:** Expects a JSON object; retries once on malformed JSON before skipping

The agent receives one file path per invocation and returns a structured result.

### Prompts (`prompts.py`)

**System prompt:**
```
You are a file classification assistant. Your job is to determine whether a 
file is a receipt or proof of purchase, and if so, categorize it.

When given a file path, use the appropriate tool to read its contents, then 
return ONLY a valid JSON object with these fields:

- is_receipt: boolean
- category: string — one of the suggested categories below, or a new one if 
  clearly warranted: "Mortgage", "Utilities", "Insurance", "Groceries", 
  "Travel", "Event Tickets", "Hobby / Radio Control", "Subscription", "Other"
- reason: string — one sentence explaining why this is or is not a receipt
- suggested_path: string — full path as ~/Documents/Receipts/<Category>/<filename>
  (only required when is_receipt is true)

If the file is not a receipt, return:
{"is_receipt": false, "reason": "<one sentence explanation>"}

Return ONLY valid JSON. No explanation, no markdown, no code fences.
```

**Per-file prompt:**
```
Classify this file: {file_path}
```

### Output (`output.py`)

- CSV file named `receipts_<YYYYMMDD_HHMMSS>.csv` written to the current working directory
- Columns: `Receipt Category`, `Original File Location`, `Suggested File Location`, `Reason`
- Rows are appended immediately after each receipt is identified (incremental writes)
- Non-receipts are not written to the CSV

### Checkpoint / Resume

- A `.progress.json` file in the working directory tracks processed file paths
- On startup, already-processed paths are loaded and skipped
- On each file completion (receipt or not), the path is added to `.progress.json`
- This allows interrupted runs to resume without reprocessing files

---

## Observability (OpenTelemetry + OpenLIT)

### Infrastructure

OpenLIT runs locally via Docker Compose (cloned from `https://github.com/openlit/openlit`):

```bash
git clone https://github.com/openlit/openlit
cd openlit
docker compose up -d
```

Exposes:
- UI: `http://localhost:3000`
- OTLP HTTP receiver: `http://localhost:4318`

### Python Setup (`telemetry.py`)

```python
import openlit
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
import logging

def setup_telemetry():
    # Auto-instruments LangChain + sets up OTLP trace exporter
    openlit.init(otlp_endpoint="http://127.0.0.1:4318")

    # Wire Python logging → OTel logs → OpenLIT
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint="http://127.0.0.1:4318"))
    )
    set_logger_provider(logger_provider)

    handler = LoggingHandler(logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
```

All `logging.info/warning/error` calls throughout the codebase are routed to OpenLIT via this handler.

### What Gets Observed

| Signal | What you see in OpenLIT |
|---|---|
| Traces | Full ReAct agent reasoning chain, each tool call, each Ollama LLM invocation with latency + token counts |
| Logs | Per-file progress (`[12/347] Scanning: invoice.pdf`), receipts found, files skipped, errors |

Structured attributes attached to log records where relevant:
- `file.path` — absolute path of the file being processed
- `file.type` — `pdf`, `docx`, or `txt`
- `receipt.category` — category assigned (on receipt records)

---

## Error Handling

| Scenario | Behavior |
|---|---|
| File unreadable (corrupt, permissions) | Log OTel warning with `file.path`, skip file, record in `.progress.json` |
| Ollama returns malformed JSON | Retry once; if still malformed, log OTel error and skip |
| Ollama not running at startup | Fail fast with clear OTel error log before processing any files |
| Unknown file extension | Never reached — scanner filters to known extensions only |

---

## Configuration

All configuration via environment variables or CLI arguments:

| Setting | CLI flag | Env var | Default |
|---|---|---|---|
| Root scan path | `--root` | `ORGANIZER_ROOT` | `~/Downloads` |
| Ollama model | `--model` | `OLLAMA_MODEL` | `llama3.2` |
| Ollama base URL | `--ollama-url` | `OLLAMA_BASE_URL` | `http://localhost:11434` |
| OpenLIT endpoint | `--otel-endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` |
| Output directory | `--output-dir` | `ORGANIZER_OUTPUT_DIR` | current working directory |

---

## Requirements

```
langchain
langchain-ollama
langchain-community
pdfplumber
python-docx
openlit
opentelemetry-sdk
opentelemetry-api
opentelemetry-exporter-otlp-proto-http
opentelemetry-instrumentation-langchain
```

---

## Example CSV Output

```csv
Receipt Category,Original File Location,Suggested File Location,Reason
Hobby / Radio Control,/Users/david/Downloads/hobbyking_order.pdf,~/Documents/Receipts/Hobby_RC/hobbyking_order.pdf,Invoice from HobbyKing showing itemized parts and shipping total.
Event Tickets,/Users/david/Downloads/concert_conf.pdf,~/Documents/Receipts/Event_Tickets/concert_conf.pdf,Ticketmaster order confirmation for two concert tickets.
Mortgage,/Users/david/Documents/may_payment.pdf,~/Documents/Receipts/Mortgage/may_payment.pdf,Monthly mortgage payment confirmation from lender.
```
