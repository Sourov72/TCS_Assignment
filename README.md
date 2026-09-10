# Customer Support Copilot

A Generative AI multi-agent system that lets a support executive ("John," per
the assignment's scenario) ask natural-language questions over two kinds of
data:

1. **Structured customer data** - customer profiles and their support ticket
   history, stored in SQLite.
2. **Unstructured policy documents** - any company policy PDF, searchable via
   a vector store.

A ReAct agent reasons about which tool(s) it needs each turn (customer
data, policy documents, both, or neither), remembers the conversation
across turns, and asks for clarification or declines rather than guessing
when it doesn't have enough information.

**Demo video:** _TODO - add the recorded demo link here before submitting._

---

## Table of contents

- [Customer Support Copilot](#customer-support-copilot)
  - [Table of contents](#table-of-contents)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Tech stack](#tech-stack)
  - [Project structure](#project-structure)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Data \& attribution](#data--attribution)
  - [Testing \& evaluation](#testing--evaluation)
  - [Known limitations \& design trade-offs](#known-limitations--design-trade-offs)

---

## Features

- **Ask about a customer:** "Give me an overview of customer Ema's profile
  and past support ticket details." → looks up the customer and their full
  ticket history from SQL, with a natural-language summary.
- **Ask about a policy:** "What is the current refund policy?" → retrieves
  the relevant policy section and answers using only that content, with a
  citation (source document + page number). Requires at least one PDF to
  have been uploaded first - see below.
- **Ask something needing both:** "Is she eligible for a refund on her
  canceled order, based on our policy?" → the agent calls *both* tools,
  reads what each returns, and combines them into one answer.
- **Multi-turn memory:** follow-up questions ("what about **her** last
  ticket?") correctly resolve using the conversation so far.
- **Guardrails:** off-topic questions ("what's the capital of France?") are
  politely declined instead of answered by guesswork; the SQL agent can only
  ever run a read-only `SELECT`, never a data-modifying query.
- **Upload policy PDFs** directly from the UI and have them searchable
  immediately - the app ships with **no policy PDF pre-loaded**, so this is
  how the policy knowledge base gets populated in the first place (see
  [`Example_PDF/`](#data--attribution) for a ready-to-upload sample).
- **MCP server:** the same capabilities are exposed as MCP tools for any
  MCP-compatible client (e.g., another agent).

## Architecture

```
                    ┌─────────────┐
   User ──chat/PDF──▶  Streamlit  │
                    └──────┬──────┘
                           │ MCP (HTTP, auto-started)
                    ┌──────▼──────┐
                    │  MCP Server │  mcp_server/server.py
                    │ (also used  │  same server + tools any external
                    │ by external │  MCP client can connect to
                    │ MCP clients)│  
                    └──────┬──────┘
                    ┌──────▼───────┐
                    │  ReAct Agent │  reasons about which tool(s) it
                    │ (LangGraph + │  needs, calls them, reads results,
                    │ tool-calling)│  decides if it needs another - or
                    └──┬────────┬──┘  asks for clarification instead
              ┌────────┘        └────────┐
      ┌───────▼──────┐          ┌────────▼──────┐
      │  SQL tool    │          │  Policy tool  │
      │ (NL→SQL via  │          │ (retrieve from│
      │  DeepSeek)   │          │  FAISS + gen) │
      └───────┬──────┘          └───────┬───────┘
              │                          │
      ┌───────▼───────┐          ┌───────▼───────┐
      │   SQLite      │          │  One   FAISS  │
      │ customers,    │          │     index     │
      │ tickets       │          │   (all PDFs)  │
      └───────────────┘          └───────────────┘
```

Both the Streamlit UI and any external MCP client  go through the **same** MCP server and the same
underlying agents.

**How a question is answered:**

`app.py` doesn't call the agent directly - it talks to `mcp_server/server.py`
over MCP (HTTP transport), via the client in `agents/mcp_client.py`. The
server auto-starts in the background the first time it's needed (checked by
`app.py` on page load), so running the app is still one command even though
two processes end up running.

`agents/graph.py` builds a **ReAct agent** (via `langchain.agents.create_agent`,
which compiles to a LangGraph graph under the hood) with two tools:

- `query_customer_database` (`agents/tools.py` → `agents/sql_agent.py`) -
  asks the LLM to write a SQLite `SELECT` query, validates it's read-only,
  executes it, and summarizes the results.
- `search_policy_documents` (`agents/tools.py` → `agents/rag_agent.py`) -
  retrieves the top matching chunks from **FAISS index**
  built from every PDF in `data/policies/`, and answers using only those
  sections (with citations). 


## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain + LangGraph | Recommended tech by the assignment; a LangChain ReAct agent (`langchain.agents.create_agent`) compiles to a LangGraph graph under the hood, giving tool-calling reasoning plus per-thread memory via a LangGraph checkpointer |
| LLM | DeepSeek API (`deepseek-chat`) | Fast, inexpensive, OpenAI-compatible API |
| Embeddings | Local `sentence-transformers/all-MiniLM-L6-v2` | DeepSeek has no embeddings endpoint; a local model needs no extra API key and is free |
| Structured data | SQLite | Zero server setup - single file, easy to clone-and-run |
| Vector store | FAISS, **combined index across all policy PDFs** | Simple, faithful to "upload any policy PDF" - no fixed category taxonomy to force documents into |
| Tool server | MCP (`mcp` Python SDK) | Recommended tech by the assignment; exposes the same agents to external MCP clients |
| UI | Streamlit | Chat interface + PDF upload |

## Project structure

```
TCS_Assignment/
├── app.py                      # Streamlit chat UI
├── requirements.txt
├── .env.example                # copy to .env and fill in your DeepSeek key
├── Example_PDF/
│   └── company_policies_western_capital.pdf  # sample PDF to upload and try the RAG side
├── common/
│   ├── config.py                # loads .env
│   └── llm.py                   # shared DeepSeek chat model client
├── db/
│   ├── schema.sql                # customers + support_tickets tables
│   ├── init_db.py                # applies schema.sql to a SQLite file
│   ├── seed_db.py                # synthetic data (Faker + a hand-written "Ema" fixture)
│   └── inspect_db.py             # manual sanity-check script
├── data/
│   └── policies/                  # empty by default - populated by uploading a PDF via the UI
├── rag/
│   ├── embeddings.py              # local embedding model
│   ├── ingest.py                  # PDF -> chunks -> one combined FAISS index
│   ├── retrieve.py                # plain similarity search over that index
│   └── vectorstore/               # persisted FAISS index (git-ignored, rebuild with ingest.py)
├── agents/
│   ├── sql_agent.py               # NL -> SQL -> guarded execution -> NL answer
│   ├── rag_agent.py               # NL -> retrieval -> cited NL answer
│   ├── tools.py                   # LangChain tool wrappers around both agents above
│   ├── graph.py                   # ReAct agent: reasons, calls tools, remembers
│   └── mcp_client.py              # HTTP MCP client app.py uses to reach the server below
├── mcp_server/
│   └── server.py                  # exposes the agents as MCP tools (used by app.py AND external MCP clients)
├── eval/
│   ├── cases.py                   # curated test question set
│   └── run_eval.py                # accuracy / faithfulness / latency report
└── tests/                          # pytest suite (see Testing & evaluation)
```

## Setup

**Prerequisites:** Python 3.11 (or similar), a [DeepSeek API key](https://platform.deepseek.com).


```bash
# 1. Clone and enter the project
git clone https://github.com/Sourov72/TCS_Assignment.git
cd TCS_Assignment

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
copy .env.example .env       # Windows: copy | macOS/Linux: cp
# then edit .env and set DEEPSEEK_API_KEY=...

# 5. Create and seed the database
python db/seed_db.py
```

That's it - the schema and synthetic customer data are ready to go. The
policy knowledge base starts **empty** on purpose (see
[Data & attribution](#data--attribution)) - run the app and upload
`Example_PDF/company_policies_western_capital.pdf` via the sidebar to try
the policy-question features.

## Usage

```bash
# Run the chat UI (main way to use the assistant) - this auto-starts the
# MCP server in the background on first use, no separate step needed
streamlit run app.py

# Ask a question from the terminal, going through the same MCP server
python agents/mcp_client.py "What is the current refund policy?"

# Ask a question from the terminal, bypassing MCP entirely (calls the
# ReAct agent directly - useful for isolating whether an issue is in the
# agent itself or in the MCP layer)
python agents/graph.py "Give me an overview of customer Ema's profile and past support ticket details."

# Run the evaluation suite (accuracy / faithfulness / latency report) -
# auto-ingests Example_PDF/ into the real index first if nothing's there yet
python eval/run_eval.py

# Run the automated test suite
pytest tests/ -v
```

Individual pieces can also be run directly for debugging, e.g.
`python rag/retrieve.py "What is the refund policy?"` or
`python agents/sql_agent.py "How many customers are on the Enterprise plan?"`.

## Data & attribution

**Structured data** (`db/seed_db.py`) is entirely synthetic: most
customers/tickets are randomly generated with the [Faker](https://faker.readthedocs.io)
library (fixed random seed for reproducibility); one customer, "Ema
Thompson," is hand-written with a fixed 4-ticket history so the assignment's
example query has a stable, known-correct answer. **No real personal data is
used anywhere in this project.**

**Policy documents:** `data/policies/` ships **empty** - no PDF is
pre-ingested. The app is meant to start with no policy knowledge base at
all, and get populated by uploading a PDF through the Streamlit UI (this
is also how a real deployment would work: "John uploads company policy PDF
documents to the system").

A ready-to-use sample lives in `Example_PDF/`, separate from
`data/policies/` so it's never auto-ingested:

| File | Pages | Source |
|---|---|---|
| `company_policies_western_capital.pdf` | 6 | [Western Capital Advisors Pvt Limited](https://westerncap.in/pdfs/privacy-policy-cancellation-and-refund-shipping-and-delivery.pdf) - a genuine document covering Privacy Policy, Cancellation & Refund, and Shipping & Delivery all in one file |

Publicly hosted by the organization and included here for
demonstration/educational purposes as part of a job assessment. Upload it
(or any other PDF) via the sidebar to start asking policy questions - new
PDFs can be added at any time this way, or by dropping a file directly in
`data/policies/` and running `rag/ingest.py`, with no code changes required
since the index is built from whatever PDFs are present rather than a
fixed, hardcoded list.

## Testing & evaluation

The test suite has two layers:

- **Fast tests** - no LLM involved, so these are free and always runnable
  without a real API key. They cover the SQL safety guardrail (read-only, no
  data-modifying statements), real queries against the seeded `db/support.db`,
  real FAISS retrieval, citation formatting, and the tool wrappers (mocking
  the SQL/RAG agents' plain dict-returning functions, not an LLM).
- **Live integration tests** - several tests actually call the real DeepSeek
  API, since the agent's core behaviors (asking for clarification on an
  ambiguous question, combining both tools, declining out-of-scope
  questions) are genuine LLM reasoning that can't be meaningfully verified
  without the real model. This includes `tests/test_graph.py`, the MCP server
  test, the Streamlit `AppTest` chat test, and the full evaluation suite.
  All marked `@pytest.mark.skipif` and skip automatically when
  `DEEPSEEK_API_KEY` isn't set, so CI without a key still passes cleanly,
  while a local run with a key gets full end-to-end verification. Tests
  that need real policy content use the `seeded_index` fixture
  (`tests/conftest.py`), which builds an isolated temp index from
  `Example_PDF/` - the real `rag/vectorstore/` (empty by default) is never
  touched by the test suite.

```bash
pytest tests/ -v          # full suite (45 tests)
python eval/run_eval.py   # standalone evaluation report
```

## Known limitations & design trade-offs

- **The agent's reasoning quality is what enforces the guardrails** (asking
  for clarification on an ambiguous pronoun, declining out-of-scope
  questions, combining both tools when needed) - unlike the read-only-SQL
  guardrail, which is enforced in code and can't be bypassed, these
  behaviors depend on the system prompt in `agents/graph.py` and the
  underlying model following it. Live-tested repeatedly and found reliable,
  but not code-enforced the way the SQL safety check is.
- **Duplicate uploads are detected by content hash**, not by more advanced
  similarity - a PDF that's byte-for-byte identical to one already in
  `data/policies/` is caught and skipped, but a re-scanned/re-exported copy
  of the same document (different bytes, same content) would not be.
- **The evaluation script's token/cost figures are approximate** (a
  character-based estimate), not exact billed usage - getting exact usage
  would mean threading a callback/usage object through the agents' internal
  LLM calls.
- **No Docker** anywhere in this project. SQLite and a
  local FAISS index mean everything runs as plain local processes.
- **The MCP server, once auto-started, keeps running** after the Streamlit
  app is closed - it's a separate, independent process by design (so it
  survives Streamlit reruns), but that also means it isn't automatically
  cleaned up. If port `8933` is ever in a bad state, stop the stray
  `mcp_server/server.py` process manually and reload the page.
