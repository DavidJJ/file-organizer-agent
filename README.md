# Receipt Organizer Agent

An agentic pipeline that scans your filesystem for receipts and proof-of-purchase documents, classifies them using a local LLM, and produces a CSV of suggested organized file locations — with zero data leaving your machine.

---

## Why This Exists

Years of saving receipts to a PC in a variety of folders, naming conventions, and file formats leaves thousands of documents scattered across a filesystem with no consistent structure. Manually sorting them is exactly the kind of tedious, context-heavy work that an LLM agent should be able to handle.

This project is that agent. It crawls a directory tree, reads each document, decides whether it is a receipt, categorizes it, and suggests where it should live — hands-free.

It also serves as a **proof-of-concept and evaluation** for three things:

- **Agentic LLM pipelines in practice** — building a real ReAct agent with LangChain against a local model, not a toy demo
- **[OpenLIT](https://openlit.io)** — evaluating its LLM observability and tracing capabilities on a real workload
- **[Docling](https://docling.ai)** — evaluating its document parsing and OCR quality as a replacement for traditional PDF/OCR toolchains

---

## How It Works

The agent follows a [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning + Acting) loop for each file:

```
Thought → Action (read file) → Observation → Thought → Final Answer (JSON)
```

1. **Scanner** crawls the root directory and yields `.pdf`, `.docx`, and `.txt` files
2. **Agent** invokes the appropriate read tool to extract text from the file
3. **LLM** (running locally via Ollama) classifies the content and assigns a category
4. **CSV writer** appends confirmed receipts with their suggested destination path
5. **Progress file** checkpoints processed paths so interrupted runs resume without reprocessing

```
┌─────────────────────────────────────────────┐
│               LOCAL MACHINE                 │
│                                             │
│  main.py ──► scanner ──► agent ──► tools   │
│                 │           │         │     │
│                 │      LangChain    Docling │
│                 │       ReAct       DOCX   │
│                 │           │       TXT    │
│                 │        Ollama            │
│                 │       llama3.2           │
│                 │                          │
│              CSV out ◄── results           │
│                                            │
│  Telemetry ──► OpenTelemetry ──► OpenLIT   │
└─────────────────────────────────────────────┘
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

The agent classifies each receipt into one of:

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

Progress is automatically saved to `.progress.json` after each file. Rerunning the same command will skip already-processed files.

To start fresh:

```bash
rm .progress.json
```

---

## Observability

All logs, LLM traces, and spans are exported via OpenTelemetry to the local OpenLIT instance.

Open the dashboard at **[http://localhost:3000](http://localhost:3000)** after starting the stack.

You can inspect:
- Per-file LLM call latency and token counts
- Agent reasoning traces (Thought → Action → Observation chains)
- Error rates and retry counts
- Full structured logs correlated with traces

---

## Project Structure

```
.
├── main.py              # Entry point, CLI, orchestration loop
├── agent.py             # LangChain ReAct agent and result parsing
├── prompts.py           # ReAct prompt template with category definitions
├── scanner.py           # Filesystem walker
├── output.py            # CSV writer
├── telemetry.py         # OpenTelemetry + OpenLIT setup
├── tools/
│   ├── read_pdf.py      # Docling-based PDF extraction (text + OCR)
│   ├── read_docx.py     # python-docx Word document extraction
│   └── read_text.py     # Plain text file reader
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
