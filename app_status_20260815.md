# AI Research Agent v4 — App Review Summary

*Review dated 2026-08-15. Based on inspection of `app.py`, `rag.py`, `requirements.txt`, `README.md`, `user_manual.md`, and scratch files in the working directory.*

---

## 1. What the App Does

**AI Research Agent v4** is a local, privacy-first, multi-stage research and writing assistant that runs entirely on the user's own machine. It takes a user-defined research topic and walks it through a six-stage pipeline — with human review checkpoints between stages — and ends by producing a long-form document in a chosen format.

### The six workflow stages

| Stage | Purpose |
|-------|---------|
| **0 — Research Context** | User picks a perspective/voice (e.g. *Skeptical Scientist*, *Policy Advocate*) and a target audience (e.g. *Funding Agencies*, *Undergraduates*), plus optional framing notes. These get injected into every subsequent prompt. |
| **1 — Research Plan + Initial Research** | A CrewAI "Interpreter" agent turns the topic into a structured JSON plan (main questions, topics, search queries, expected structure). A "Knowledge Researcher" agent then synthesises initial answers from the local LLM's training data. |
| **2 — Gap Analysis** | A "Research Gap Analyst" agent compares the plan against the initial research and outputs prioritised missing-info gaps and queries — but only flags *genuine* gaps, taking any custom resources into account. |
| **3 — Web Research** | A "Web Research Specialist" agent uses the search tool to fill gaps, summarising findings and noting failures. |
| **4 — Output Format & Length** | User picks one of six output formats and a target length (~500 / 1,000 / 2,500 / 5,000+ words or custom). |
| **5 — Write Output** | A format-specific "writer" agent produces the full document. Downloads are offered as `.docx` and `.md`. |

### Key capabilities that distinguish it from a simple chat wrapper

- **Semantic RAG over user-provided resources** — URLs and PDFs uploaded via the sidebar are scraped/extracted, semantically chunked (600 chars / 80 overlap), embedded via Ollama, and stored in ChromaDB. Retrieved per-stage by relevance rather than by fixed truncation.
- **Per-section retrieval for long docs** — When the target is ≥5,000 words, the writer switches to section-by-section mode. The plan, initial research, and web research are embedded into a temporary `rsch_<session>` collection, and each section retrieves its own relevant chunks (so a "Methods" section sees different context than an "Introduction").
- **Web search reranking** — Web results are reranked by embedding similarity to the query before being handed to the agent.
- **Persistent document library** — Resources added with the "Save to persistent library" toggle go into a cross-session `library` collection, retrievable in future research sessions.
- **Dual web search backends** — Selenium-driven headless Chrome/Firefox (Google → Bing fallback) or the DuckDuckGo API, with retry logic.
- **Direct Ollama context priming** — Before the writer starts, the app calls Ollama's `/api/generate` directly to force-load the model with the chosen `num_ctx` (8K–64K), because CrewAI/LiteLLM silently drops `num_ctx` kwargs.
- **Deterministic DOCX export** — A pure-Python markdown→DOCX converter (using `python-docx`) generates styled Word documents with headings, bold/italic/code runs, bullet/numbered lists, horizontal rules, and 1-inch margins — no extra LLM calls.
- **Resource-aware re-runs** — If a URL/PDF is added after research has already advanced, the sidebar offers one-click re-runs from Initial Research, Gap Analysis, Web Research, or Output Writing.

### Output formats

Blog Post · Research Paper · Essay · Executive Summary · NSF-Style Grant Proposal · Explainer — each with a distinct agent role/goal/backstory prompt and a section template.

---

## 2. Tech Stack

### Runtime / Application
- **Python 3.11+** — application language
- **Streamlit** (≥1.30) — web UI, session state, sidebar, file uploader, download buttons
- **Streamlit-Extras** — UI utilities

### Agentic / LLM orchestration
- **CrewAI** (≥0.70) — multi-agent orchestration (`Agent`, `Task`, `Crew`, `Process.sequential`)
- **LiteLLM** (≥1.0, <2.0) — the LLM routing layer CrewAI uses under the hood
- **Ollama** (≥0.1) — local model server, invoked via two paths:
  - `langchain-ollama.OllamaLLM` for the main agents (called as bare `"ollama/<model>"` strings by CrewAI)
  - Direct `requests.post("http://localhost:11434/api/generate")` for context-window priming
- **LangChain** ecosystem — `langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-text-splitters` (for `RecursiveCharacterTextSplitter`)
- **OpenAI SDK** (≥1.70) — pinned but only used as a dummy key to satisfy CrewAI's check (no real OpenAI calls)

### RAG / Vector storage
- **ChromaDB** (≥0.5) — persistent vector store (`chroma_db/` folder, HNSW with cosine distance)
- **Ollama embeddings** via `OllamaEmbeddings` — default `nomic-embed-text`, also supports `mxbai-embed-large`/`all-minilm`/`bge-*` (auto-detected by substring hints in `rag.py:EMBEDDING_MODEL_HINTS`)
- **Custom semantic chunking** — 600-char chunks, 80-char overlap, separators `["\n\n", "\n", ". ", " ", ""]`

### Web search and scraping
- **Selenium** (with `webdriver_manager` for ChromeDriver/GeckoDriver auto-install)
- **duckduckgo-search** (`DDGS`) — API fallback
- **BeautifulSoup4** + **lxml** — HTML parsing
- **requests** (≥2.31) — HTTP for scraping and the direct Ollama REST call
- **pdfplumber** (≥0.10) + **PyMuPDF** (≥1.24) — PDF text extraction

### Document generation
- **python-docx** (≥1.1) — DOCX assembly with `OxmlElement` for horizontal-rule borders
- **markdown** + **Jinja2** + **html5lib** — listed in requirements but the actual writer path uses raw markdown→DOCX; these appear to be optional/legacy

### Data / utilities
- **pandas** (≥2.0) — listed but unused in the current flow
- **pydantic** (≥2.0), **pydantic-settings** (≥2.0) — for CrewAI's typed config
- **tenacity** — retry helpers
- **python-dotenv** — env loading
- **nltk**, **tiktoken**, **jsonschema** — listed, lightly used
- **uuid**, `hashlib`, `subprocess`, `re`, `io`, `datetime`, `logging` — stdlib

### Storage / on-disk state
- `chroma_db/` — ChromaDB persistent store (collections: `session_<id>`, `rsch_<id>`, `library`)
- `.crewai_storage/` — CrewAI scratch (telemetry disabled)
- `.crewai_storage` env var set early in `app.py` before any CrewAI import

### Platform targets
- macOS (Intel and Apple Silicon — special-cased for ARM compatibility)
- Linux
- Windows (WSL recommended per the README)

### Notable design choices in the code
- CrewAI telemetry is explicitly disabled (`CREWAI_TELEMETRY_ENABLED=false`, `OTEL_SDK_DISABLED=true`) and a dummy OpenAI key is set before any CrewAI import — so the app runs without any cloud credentials.
- The model name is passed to CrewAI as a bare string `"ollama/<model>"` so it goes through the LiteLLM path rather than the LangChain path.
- Stage numbers are slightly non-linear (`0 → 1 → 2 → 3 → 4 → 45 → 5`) so that "Output Format" is inserted between Web Research and Write without renumbering everything.
- The writer context window is **primed via a direct Ollama REST call** before any CrewAI calls, because `litellm.drop_params=True` (hardcoded by CrewAI) silently strips `num_ctx` from kwargs — the comment in `_prime_ollama_context` documents this workaround explicitly.

### Project layout
```
research_agentV4/
├── app.py                  # Streamlit UI + CrewAI orchestration (~2,300 lines)
├── rag.py                  # ChromaDB DocumentStore, chunking, retrieval, reranking (~375 lines)
├── requirements.txt        # All Python dependencies
├── README.md               # Short intro
├── user_manual.md          # Full user manual
├── readme.txt              # Scratch notes / example prompts (not part of runtime)
├── temp_append_script.py   # Scratch helper (not part of runtime)
├── nul_file.txt            # Empty file (not part of runtime)
└── chroma_db/              # Persistent vector store (auto-created)
```

---

## 3. Tool Harness Coding — What's There, What Isn't

### What the app actually does

The app registers **exactly one tool** with the agents, and it does so by hand.

1. **`WebSearchTool` — the only registered tool** (`app.py:28` and `app.py:902–995`):

   ```python
   from crewai.tools import BaseTool

   class WebSearchTool(BaseTool):
       name: str = "web_search"
       description: str = "Search the web for information. Input should be a search query string. Returns top search results with titles, links, and snippets."

       def _run(self, query: str) -> str:
           # ...selects backend (Selenium or DuckDuckGo), reranks results via embeddings...
   ```

   It subclasses `crewai.tools.BaseTool` — CrewAI's tool protocol. It defines a `name`, a `description`, and a `_run` method. The `description` is what the LLM reads to decide whether/when to call the tool, and `name="web_search"` is the function name the agent emits in its tool-call.

2. **The tool is registered with one agent** — `app.py:1216` (the Web Research Specialist agent, `step4_web_research`):

   ```python
   web_researcher_agent = Agent(
       role="Web Research Specialist",
       ...
       llm=llm_string,
       tools=[search_tool],          # <-- only here
       verbose=True,
       allow_delegation=False,
       max_iter=10,
   )
   ```

   That's the **only** `tools=[...]` assignment in the whole codebase. Every other agent — the Interpreter, Knowledge Researcher, Gap Analyst, Writer — is created without a `tools` argument, so they cannot emit tool calls.

### What this means in practice

- **The agent's "tool choice" is hard-coded by the developer**, not selected dynamically by a harness. There is no tool router, no registry of multiple tools, no `Tool(...)` factory, no `@tool` decorator usage, no per-stage tool wiring — just one `BaseTool` subclass bound to one agent.
- **The tool itself has a small "choice" of its own backend**: inside `_run`, it reads `st.session_state.search_method` and picks Selenium (Chrome/Firefox, Google → Bing fallback) or DuckDuckGo. That's a UI-driven choice, not an LLM-driven one.
- After the backend returns raw results, the tool optionally reranks them by embedding cosine similarity when `st.session_state.rag_enabled` is true (`app.py:917–927`). Again, that's deterministic branching — the LLM never gets to "choose" whether reranking happens.
- `allow_delegation=False` is set on every agent, so CrewAI's delegation feature (which is itself a kind of tool routing) is **explicitly disabled** across the board.

### What the harness does *not* do

Searching the codebase turns up **no** use of:

- `langchain_core.tools.Tool(...)` (the import exists at `app.py:29` but the symbol is never used)
- the `@tool` decorator pattern
- a tool registry / dispatcher / router
- dynamic tool selection based on context
- `agent_fork`, `subagent`, `workflow`, or `ralph`-style orchestration (these terms only appear in documentation/comments and as the name of the Streamlit pipeline — they're not invoked as functions anywhere)
- per-stage conditional tool attachment

### Bottom line

There **is** a small amount of CrewAI tool-harness code — exactly one `BaseTool` subclass, manually attached to exactly one agent (the Web Research Specialist). But it isn't a *general* tool-choice harness; it's a single fixed tool wired into a single agent, with a deterministic backend selector (Selenium vs. DuckDuckGo) and a deterministic reranker driven by UI state. The LLM never gets to choose between multiple tools, never delegates, and never has tools attached in stages 1, 2, or 5.

---

## 4. Does CrewAI Support Tool Harness Coding?

**Yes — CrewAI supports rich tool-harness coding natively.** Tool selection is one of its core value propositions. The app I just reviewed is using a *very narrow subset* of that capability.

### What CrewAI supports natively

#### 1. Multiple tools per agent
CrewAI agents accept a `tools=[...]` list. When more than one tool is attached, all their `name`/`description` pairs are injected into the agent's system prompt, and the LLM picks which to call via standard function-calling. The single `tools=[search_tool]` line in this app shows the mechanism — CrewAI would happily accept ten tools the same way.

#### 2. Tool construction patterns
Three are officially supported:
- **`BaseTool` subclassing** (what this app uses): define `name`, `description`, Pydantic `args_schema`, and `_run`.
- **`@tool` decorator**: turn a plain function into a tool with a single decorator.
- **`from crewai_tools import ...`**: a large library of pre-built tools (`SerperDevTool`, `ScrapeWebsiteTool`, `FileReadTool`, `CodeInterpreterTool`, `RagTool`, `BrowserbaseLoadTool`, etc.) that handle auth, retries, and result formatting for you.

#### 3. Tool-level input validation
`BaseTool.args_schema` (a Pydantic model) gives structured input validation before `_run` is called. This app doesn't use it — `WebSearchTool._run(query: str)` takes a raw string.

#### 4. Agent delegation
`allow_delegation=True` lets an agent hand a subtask to another agent and receive the result. This is itself a form of routing/orchestration. The app sets `allow_delegation=False` on every agent, so it explicitly opted out.

#### 5. CrewAI "Flows" (since 2024)
A decorator-based stateful orchestration layer on top of agents: `@start()`, `@listen()`, `@router()` for conditional branching, and persistent state across steps. This is closer to "harness code" in the workflow-engine sense.

#### 6. Built-in `human_input` tool
CrewAI ships a tool that pauses and asks the user a clarifying question mid-task. The app instead builds human checkpoints with Streamlit buttons (`st.button("✅ Proceed to Gap Analysis")`) — so its human-in-the-loop is *outside* CrewAI by choice, not by necessity.

#### 7. Memory across calls
`Crew(memory=True)` with short-term, long-term, and entity memory. The app uses `memory=False` on every Crew — so cross-call memory is disabled.

### What would lie *outside* CrewAI

These are things you'd build yourself, using CrewAI as one component among others:

| Concern | Where it lives |
|---|---|
| **Picking the *right* tool from a large catalog** based on embedding similarity / semantic routing / a router LLM | Your code. You'd attach a single "router" tool whose `_run` does the semantic matching, or use a separate classifier agent. |
| **Retry / fallback chains across tools** (try tool A, on failure try tool B, on total failure run tool C as a synthesizer) | Your code. The app implements a tiny version of this inside `WebSearchTool._run` (Google → Bing → DuckDuckGo fallback). |
| **Aggregating results from N parallel tool calls** before returning to the agent | Your code. CrewAI tasks are sequential by default (`Process.sequential`); `Process.hierarchical` adds a manager agent, but parallel fan-out across tools is something you'd implement in a tool's `_run` (or use Flow's async listeners). |
| **Persistent storage of tool inputs/outputs** for audit / replay | Your code (or a wrapper). |
| **Cost / latency budgets** that abort an agent if a tool-call would exceed them | Your code. |
| **UI-side human-in-the-loop gates** with stage approval | Your code. Streamlit handles this in the app; CrewAI's `human_input` tool is an alternative *inside* the framework. |
| **Tool selection across multiple agents** (e.g., a router decides which agent runs, which tools it sees) | This is genuinely outside CrewAI's per-agent model — you'd build it with Flows, or above CrewAI entirely. |

### How this maps back to the app's design choices

| Choice | What it uses | What it skips |
|---|---|---|
| Tools | One `BaseTool` subclass | `crewai_tools` library, `@tool` decorator, multi-tool agents |
| Orchestration | Six `stepN_*` Python functions wired by a Streamlit state machine | `Flows`, delegation, hierarchical process |
| Routing | Deterministic (UI checkboxes select backend) | Semantic tool routing, LLM-driven tool selection |
| Memory | `memory=False` everywhere | Short-term / long-term / entity memory |
| Human-in-the-loop | Streamlit buttons between stages | `human_input` tool, CrewAI Flow state |
| Fallback | Inside one tool (Google → Bing → DDG) | Cross-tool orchestration |

### Bottom line on CrewAI's role

The app doesn't *lack* tool-harness support because of CrewAI's limits — it lacks it because it intentionally uses the simplest possible CrewAI subset and puts all the harness logic in plain Python (`app.py`'s stage machine + `rag.py`'s `DocumentStore`). The "harness" in this codebase is really:

1. The Streamlit stage state machine (`workflow_stage` numbers 0–45–5)
2. The RAG chunking/retrieval layer in `rag.py`
3. The per-section writer loop in `step5_write_output`

None of those are CrewAI concepts. If you wanted richer tool-harness behavior without rewriting, the changes would mostly be *inside* CrewAI's existing APIs (attach more tools, flip `allow_delegation=True`, try `Process.hierarchical`, or migrate the Streamlit stage machine to a `Flow`). You wouldn't need to escape CrewAI to get them — but the *cross-cutting* concerns (semantic routing, caching, budgets, audit) genuinely live above the framework.

---

## 5. Status Snapshot (2026-08-15)

- **App shape:** Local multi-stage research/writing agent, fully offline, runs via Ollama + CrewAI + Streamlit + ChromaDB.
- **Lines of code:** ~2,300 in `app.py`, ~375 in `rag.py`.
- **Tool harness:** Minimal — one `BaseTool` (`WebSearchTool`) attached to one agent (the Web Research Specialist). No tool router, no delegation, no Flows.
- **Human-in-the-loop:** Implemented via Streamlit UI buttons between stages (Stage 0 → 1 → 2 → 3 → 45 → 5), with re-run shortcuts when resources are added late.
- **RAG:** Fully implemented and toggleable; per-section retrieval active when target ≥5,000 words; persistent library collection available.
- **Web search:** Two backends (Selenium + DuckDuckGo), with embedding-based reranking when RAG is on.
- **Export:** DOCX (deterministic, styled) + Markdown.
- **Privacy:** All LLM inference is local via Ollama; no cloud API keys are required (a dummy OpenAI key is set to satisfy CrewAI's import-time check).
- **Primary gaps relative to CrewAI's full capability:** multi-tool agents, delegation, Flows, built-in memory, `human_input` tool, `crewai_tools` library integration.
- **Primary gaps relative to "tool harness" ideals:** semantic tool routing, retry/fallback chains across tools, tool-call caching, cost/latency budgets, parallel fan-out across tool calls, cross-cutting audit.
