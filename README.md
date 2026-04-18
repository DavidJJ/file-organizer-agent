# Receipt Organizer Agent

An agentic pipeline that scans your filesystem for receipts and proof-of-purchase documents, classifies them using a local LLM, and produces a CSV of suggested organized file locations — with zero data leaving your machine.

---

## Why This Exists

Years of saving receipts to a PC in a variety of folders, naming conventions, and file formats leaves thousands of documents scattered across a filesystem with no consistent structure. Manually sorting them is exactly the kind of tedious, context-heavy work that an LLM agent should be able to handle.

This project is that agent. It crawls a directory tree, decides at each folder level whether to skip it entirely or process it file-by-file, reads each document, decides whether it is a receipt, categorizes it, and suggests where it should live — hands-free.

It also serves as a **proof-of-concept and evaluation** for several things:

- **Multi-agent LLM pipelines in practice** — a coordinator orchestrating two specialized ReAct agents, built with LangChain against a local model
- **[OpenLIT](https://openlit.io)** — evaluating its LLM observability and tracing capabilities on a real workload
- **[Docling](https://docling.ai)** — evaluating its document parsing and OCR quality as a replacement for traditional PDF/OCR toolchains

---

## How It Works

The system uses three cooperating components:

### 1. Directory Coordinator
The coordinator drives the entire traversal. At each directory level it asks the **Folder Classifier Agent** whether to skip the whole subtree or process it. If a folder is skipped, none of its files or subdirectories are ever visited. If it is processed, direct files go to the **File Classifier Agent** and the coordinator then recurses into each subdirectory.

The scan root (e.g. `~/Downloads`) is always processed — only subdirectories are classified.

### 2. Folder Classifier Agent
A lightweight ReAct agent that decides — from folder name and child names alone, without reading any file contents — whether a directory is a cohesive project that should be skipped. Skip criteria:

- Files share a naming pattern or project prefix (e.g. `B-29-1828-WingSpars.pdf`, `B-29-1829-Fuselage.pdf`)
- The folder name describes a project, part, component, or snapshot (e.g. `B-29`, `9mm-potentiometer.snapshot.5`)
- Files are clearly technical in nature (drawings, 3D models, datasheets, firmware, build instructions)

If even one file could plausibly be a receipt, the folder is processed.

**Why this matters:** Without folder-level classification, a directory containing 40 numbered aircraft drawings would result in 40 individual LLM calls — each potentially misclassified because the model has no context that they are all part of the same build project.

### 3. File Classifier Agent
A ReAct agent that reads each file's content and classifies it as a receipt or not. For receipts it assigns a category and suggests a destination path. The prompt includes the parent folder name as an explicit field — not buried in the full path — so the LLM can use it as a strong contextual hint.

### Flow

```
main.py
  └── DirectoryCoordinator
        ├── [root dir] always process
        ├── FolderClassifierAgent  ←── list_directory tool
        │     skip? ──► prune subtree
        │     process? ──► continue
        ├── FileClassifierAgent    ←── read_pdf / read_docx / read_text tools
        │     receipt? ──► append to CSV
        └── recurse into subdirectories
```

### Example traversal

```
Downloads/                    ← root, always processed
  B-29/                       ← FolderClassifier: skip (numbered technical drawings)
    B-29-1828-WingSpars.pdf   ← never visited
    B-29-1829-Fuselage.pdf    ← never visited
  9mm-potentiometer.snapshot.5/  ← FolderClassifier: skip (component package)
    R-0904N-KC.m3d            ← never visited
  invoices/                   ← FolderClassifier: process (generic folder name)
    amazon_order.pdf          ← FileClassifier: receipt → Electronics
    readme.txt                ← FileClassifier: not a receipt
    project_y/                ← FolderClassifier: skip (related docs)
      related_y1.doc          ← never visited
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LOCAL MACHINE                      │
│                                                         │
│  main.py                                                │
│    └── DirectoryCoordinator (coordinator.py)            │
│          ├── FolderClassifierAgent (folder_agent.py)    │
│          │     └── list_directory tool                  │
│          │           └── Ollama / llama3.2              │
│          └── FileClassifierAgent (agent.py)             │
│                └── read_pdf / read_docx / read_text     │
│                      └── Docling / python-docx          │
│                            └── Ollama / llama3.2        │
│                                                         │
│  output.py ──► receipts_<timestamp>.csv                 │
│  .progress.json ──► resume checkpoint                   │
│                                                         │
│  Telemetry ──► OpenTelemetry ──► OpenLIT                │
└─────────────────────────────────────────────────────────┘
```

All LLM inference, document parsing, and OCR runs locally. No file contents are transmitted to any external service.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Agent framework** | [LangChain](https://python.langchain.com) (`langchain-classic`) | ReAct agent loop, tool execution |
| **Local LLM** | [Ollama](https://ollama.com) + `llama3.2` | On-device inference, no cloud calls |
| **Document parsing** | [Docling](https://docling.ai) | PDF text extraction + OCR for scanned docs |
| **DOCX parsing** | `python-docx` | Word document text extraction |
| **Observability** | [OpenLIT](https://openlit.io) + OpenTelemetry | LLM traces, spans, logs via OTLP |
| **Telemetry backend** | ClickHouse + OpenLIT dashboard | Trace storage and visualization |
| **Package management** | [uv](https://docs.astral.sh/uv/) + `pyproject.toml` | Fast, reproducible Python deps |
| **Infrastructure** | Docker Compose | OpenLIT, ClickHouse, optional Ollama |

---

## Receipt Categories

The file classifier assigns each receipt to one of:

| Category | Examples |
|---|---|
| Mortgage | Mortgage statements, property tax, home loan payments |
| Utilities | Electricity, gas, water, internet, phone, cable |
| Insurance | Health, car, home, life insurance premiums or claims |
| Groceries | Food and household consumables (Costco, grocery stores, etc.) |
| Travel | Flights, hotels, car rentals, ride shares, parking, tolls |
| Clothing & Apparel | Clothing, shoes, accessories from any retailer |
| Home & Appliances | Furniture, appliances, home improvement, garden, tools |
| Electronics & Components | Electronic parts, PCBs, sensors, dev boards (Mouser, DigiKey, Adafruit, etc.) |
| Hobby / Radio Control | RC vehicles, drones, FPV equipment, hobby kits — specifically RC/drone focused |
| Event Tickets | Concerts, sports, cinema, theatre, theme parks |
| Shipping & Postage | USPS, FedEx, UPS paid postage receipts with a total charge |
| Subscription | Recurring software, streaming, memberships, SaaS, domain registrations/renewals, hosting |
| Other | Receipts that don't fit the above |

---

## Output

A timestamped CSV is written to the output directory on each run:

```
receipts_20260417_143022.csv
```

| Column | Description |
|---|---|
| `Receipt Category` | Classified category |
| `Original File Location` | Absolute path to the file as found |
| `Suggested File Location` | Proposed destination under `~/Documents/Receipts/<Category>/` |
| `Reason` | One-sentence explanation from the LLM |

---

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- [Ollama](https://ollama.com) — install and run locally, or use the bundled Docker service
- [Docker](https://www.docker.com) — for OpenLIT observability stack

### Install dependencies

```bash
uv sync
```

### Pull the model

```bash
ollama pull llama3.2
```

A larger model improves classification accuracy significantly. `llama3.3` or `mistral-small` are good alternatives if your hardware supports them:

```bash
ollama pull llama3.3
uv run python main.py --model llama3.3
```

### Start the observability stack

```bash
# Auto-detects whether Ollama is already running on the host.
# Starts the bundled Ollama Docker service only if it isn't.
./start.sh
```

Or manually:

```bash
# Ollama already running on host — start OpenLIT + ClickHouse only
docker compose up -d

# No local Ollama — start everything including the bundled Ollama service
docker compose --profile ollama up -d
```

### Run the agent

```bash
# Scan ~/Downloads (default)
uv run python main.py

# Scan a specific directory
uv run python main.py --root ~/Documents

# Full options
uv run python main.py \
  --root ~/Downloads \
  --model llama3.2 \
  --ollama-url http://localhost:11434 \
  --output-dir ./results
```

### Environment variables

All CLI flags can also be set via environment variables:

| Variable | Default | Description |
|---|---|---|
| `ORGANIZER_ROOT` | `~/Downloads` | Directory to scan |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `ORGANIZER_OUTPUT_DIR` | `.` | Where to write the CSV |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://127.0.0.1:4318` | OpenTelemetry collector |
| `HF_HUB_OFFLINE` | `1` | Prevent HuggingFace Hub update checks (set automatically after `docling-tools models download`) |

### Resume an interrupted run

Progress is automatically saved to `.progress.json` after each file or skipped folder. Rerunning the same command will skip already-processed files and already-classified folders.

To start fresh:

```bash
rm .progress.json
```

---

## Observability

All logs, LLM traces, and spans are exported via OpenTelemetry to the local OpenLIT instance.

Open the dashboard at **[http://localhost:3000](http://localhost:3000)** after starting the stack.

You can inspect:
- Per-folder classification decisions (skip vs. process) with reasons
- Per-file LLM call latency and token counts
- Agent reasoning traces (Thought → Action → Observation chains)
- Error rates and retry counts
- Full structured logs correlated with traces

### OTel spans

| Span | What it covers |
|---|---|
| `scan.run` | The full top-level scan |
| `directory.process` | Each directory visited by the coordinator |
| `folder.classify` | Each folder agent LLM call |
| `file.classify` | Each file agent LLM call |
| `tool.read_pdf` / `tool.read_docx` / `tool.read_text` | Individual file reads |
| `tool.list_directory` | Directory listings for the folder agent |

---

## Project Structure

```
.
├── main.py              # Entry point, CLI, wires coordinator
├── coordinator.py       # DirectoryCoordinator — recursive orchestration
├── folder_agent.py      # FolderClassifierAgent — skip or process?
├── agent.py             # FileClassifierAgent — receipt classification
├── prompts.py           # ReAct prompts for both agents
├── output.py            # CSV writer
├── telemetry.py         # OpenTelemetry + OpenLIT setup
├── tools/
│   ├── list_directory.py  # Lists folder contents (used by folder agent)
│   ├── read_pdf.py        # Docling-based PDF extraction (text + OCR)
│   ├── read_docx.py       # python-docx Word document extraction
│   └── read_text.py       # Plain text file reader
├── tests/               # pytest test suite
├── docker-compose.yml   # OpenLIT, ClickHouse, optional Ollama
├── start.sh             # Smart startup script (auto-detects host Ollama)
└── pyproject.toml       # Dependencies (managed with uv)
```

---

## A Note on Privacy

This agent was deliberately built to run entirely on-device:

- **Ollama** runs the LLM locally — your file contents are never sent to OpenAI, Anthropic, or any cloud provider
- **Docling** downloads model weights once from HuggingFace, then runs inference locally via PyTorch — your documents are never uploaded
- **OpenLIT** runs in a local Docker container — traces and logs stay on your machine

The only network calls are the one-time model weight downloads (Ollama model + Docling layout models). After that, the agent runs fully air-gapped.
