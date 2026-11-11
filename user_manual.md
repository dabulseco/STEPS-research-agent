# AI Research Agent v3 — User Manual

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Starting the App](#starting-the-app)
4. [Sidebar Configuration](#sidebar-configuration)
5. [The Research Workflow](#the-research-workflow)
   - [Stage 0: Research Context](#stage-0-research-context)
   - [Stage 1: Research Plan and Initial Research](#stage-1-research-plan-and-initial-research)
   - [Stage 2: Gap Analysis](#stage-2-gap-analysis)
   - [Stage 3: Web Research](#stage-3-web-research)
   - [Stage 4: Output Format and Length](#stage-4-output-format-and-length)
   - [Stage 5: Write and Download](#stage-5-write-and-download)
6. [Research Resources](#research-resources)
   - [Adding URLs](#adding-urls)
   - [Adding PDFs](#adding-pdfs)
   - [The Persistent Document Library](#the-persistent-document-library)
7. [Output Formats](#output-formats)
8. [RAG and Semantic Retrieval](#rag-and-semantic-retrieval)
9. [Web Search Settings](#web-search-settings)
10. [Writer Context Window](#writer-context-window)
11. [Resetting the Workflow](#resetting-the-workflow)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The AI Research Agent is a local, privacy-first tool that automates the research and writing process. It uses a large language model running on your own machine (via Ollama) to plan research, conduct analysis, fill knowledge gaps with web search, and produce polished long-form documents.

The agent works in six sequential stages with a human review checkpoint between each one. You can edit, approve, or redo any stage before moving forward. Nothing is sent to cloud APIs — all inference happens locally.

---

## Installation

### Prerequisites

- **Python 3.11 or higher**
- **Ollama** installed and running — download from [ollama.ai](https://ollama.ai)
- **Chrome or Firefox** (for Selenium web search — optional but recommended)
- At least **8 GB RAM** (16 GB+ recommended for larger models)

### Steps

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull a language model
ollama pull llama3.2              # lightweight, good for most research
ollama pull gemma3:27b            # higher quality, needs ~20 GB RAM

# 4. Pull an embedding model (strongly recommended)
ollama pull nomic-embed-text      # fast, purpose-built for embeddings
```

> **Why a separate embedding model?**
> Embedding models are small (usually ~300 MB) and optimised for producing vector representations of text. Using one means your main language model does not need to be loaded twice, and retrieval quality is better. If you skip this, the agent falls back to using your main language model for embeddings, which works but is slower.

---

## Starting the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser. Ollama must be running in the background before starting the app.

---

## Sidebar Configuration

The sidebar is visible throughout the entire workflow and controls all agent settings.

### Model Selection

The **Select Ollama Model** dropdown lists every model currently installed in Ollama. Select the model you want to use for all research and writing tasks. The model is loaded on demand — you do not need to start it separately.

To add more models:

```bash
ollama pull <model-name>
```

Restart the app after pulling new models for them to appear in the list.

### RAG Settings

| Option | Description |
|--------|-------------|
| **Enable semantic RAG** | Activates ChromaDB-backed retrieval. When enabled, documents you add are fully indexed and retrieved by relevance rather than truncated. Disable to revert to the original fixed-truncation behaviour. |
| **Embedding model** | The Ollama model used to generate embeddings. Dedicated embedding models (`nomic-embed-text`, `mxbai-embed-large`) appear at the top of the list. |

### Writer Context Window

Controls the token budget available to the writer agent. Larger values allow the agent to hold more text in memory at once, which improves quality for very long documents but requires more VRAM.

| Setting | Use case |
|---------|----------|
| 8,192 | Low-memory systems (<8 GB VRAM) |
| 16,384 | Standard outputs up to ~2,500 words |
| **32,768** | **Default — handles most long outputs** |
| 49,152 | Very long documents (10+ pages) |
| 65,536 | Maximum — requires 24+ GB VRAM |

### Web Search Settings

- **Enable Web Search** — uncheck to skip web research entirely and rely only on the model's training knowledge.
- **Search Method** — choose Ollama Web Search (default when `OLLAMA_API_KEY` is set in `.env`), DuckDuckGo API (fast, may be rate-limited), or Selenium (headless browser, slowest). The method you select is tried first; if it fails or returns nothing, the others are tried automatically as fall-backs (Ollama → DuckDuckGo → Selenium).
- **Browser** — Chrome or Firefox for Selenium. On Apple Silicon Macs, Firefox is recommended if Chrome encounters issues.

---

## The Research Workflow

The workflow has six stages. You must complete each stage before proceeding to the next, but you can always go back and redo a stage.

---

### Stage 0: Research Context

Before entering your research topic, configure how the research should be framed.

**Perspective / Voice** — the viewpoint from which content is written. Options include:

- Neutral / Objective
- Skeptical Scientist
- Science Communicator
- Policy Advocate
- Industry Practitioner
- Educator / Teacher
- Investor / Business Strategist
- Custom (type your own)

**Target Audience** — who will read the final output. This shapes the depth of explanation, vocabulary, and assumptions made. Options include:

- General Public
- Graduate / Doctoral Students
- Domain Experts / Researchers
- Policymakers / Government Officials
- Funding Agencies (NSF / NIH / DOE)
- Industry Professionals
- Undergraduate Students
- Custom

**Framing Notes** — optional free-text guidance, for example:
- *"Assume the reader is sceptical of AI claims"*
- *"Emphasise economic impacts over technical details"*
- *"Avoid jargon; use plain language throughout"*

These settings are injected into every agent prompt across all stages. They are the most powerful lever for controlling the tone and focus of your output.

Once configured, enter your research topic in the **Research Topic** text area and click **Start Research**.

---

### Stage 1: Research Plan and Initial Research

**Step 1 — Research Plan**

The agent analyses your topic and produces a structured JSON plan containing:
- Main research questions (3–5)
- Key topics to investigate
- Suggested web search queries
- Expected information structure
- Topics already covered by any custom resources you added

Review the plan in the text area. If it misses the mark, click **Redo Research Plan** to regenerate it.

**Step 2 — Initial Research**

The agent uses the language model's training knowledge to answer the research questions in the plan. This does not involve web search — it is purely from the model's existing knowledge.

Review the initial research. When satisfied, click **Proceed to Gap Analysis**.

> **Tip:** You can add URLs or PDFs to the resource panel at any time. Resources added here will be incorporated into initial research on redo, or into all subsequent stages if you proceed without redoing.

---

### Stage 2: Gap Analysis

The agent compares the research plan against the initial research and identifies:
- Missing information or unanswered questions
- Areas where data may be outdated
- Topics that would benefit from recent statistics or sources
- Specific web search queries to fill each gap, prioritised by importance

Custom resources you have added are taken into account — topics they already cover are not flagged as gaps.

Review the gap analysis. If the gaps identified seem wrong or incomplete, click **Redo Gap Analysis**. Otherwise click **Proceed to Web Research**.

---

### Stage 3: Web Research

The agent uses the web search tool to run the queries identified in gap analysis. For each gap:

1. The agent constructs a targeted search query
2. Up to 5 results are retrieved
3. If RAG is enabled, results are reranked by embedding similarity to the query before the agent processes them
4. The agent synthesises findings and notes any gaps it could not fill

The final web research report includes a list of sources and a note on any searches that failed or returned no useful results.

Review the findings. Click **Redo Web Research** to run the searches again, or **Proceed to Output Format** when ready.

> **Note:** If you disabled web search in the sidebar, this stage is skipped automatically.

---

### Stage 4: Output Format and Length

Choose the structure and target length of your final document.

**Output Formats:**

| Format | Best for |
|--------|----------|
| Blog Post | Public-facing articles, science communication |
| Research Paper | Academic writing, formal analysis |
| Essay | Analytical argument, thesis-driven writing |
| Executive Summary | Decision-makers, policy briefs |
| NSF-Style Grant Proposal | Research funding applications |
| Explainer | Educational content, accessible breakdowns |

**Target Length:**

| Option | Approximate pages |
|--------|-------------------|
| ~500 words | 1 page |
| ~1,000 words | 2–3 pages |
| ~2,500 words | 5 pages |
| ~5,000 words | 10 pages |
| Custom | Type any target (e.g. "3,500 words") |

> **How long outputs are handled:** Targets under 5,000 words are written in a single pass. Targets at or above 5,000 words switch to **section-by-section mode**, where each section of the document is written as a separate agent call. When RAG is enabled, each section receives the most relevant research chunks for its specific content rather than a compressed slice of all research.

Click **Generate Output** to proceed.

---

### Stage 5: Write and Download

The agent writes the complete document. A progress spinner shows which section is being written in section-by-section mode.

Once complete, the document is rendered as a markdown preview. Download options:

- **Download DOCX** — a formatted Word document with proper heading styles, bold/italic runs, bullet lists, and 1-inch margins.
- **Download Markdown** — the raw markdown source.

Other actions:
- **Redo Output** — regenerate the document without redoing the research.
- **Change Format / Length** — go back to Stage 4 to pick a different format or length.
- **Start New Research** — clear everything and begin a fresh research session.

---

## Research Resources

You can attach URLs and PDFs to your research session at any stage. Resources are injected into every remaining step and explicitly cited in the final output.

### Adding URLs

1. Paste a URL into the **Add a URL** field in the sidebar.
2. Optionally check **Save to persistent library** if you want this document available in future sessions.
3. Click **Add URL**.

The agent scrapes the page, strips navigation and boilerplate, and — if RAG is enabled — embeds the full text into ChromaDB. A 4,000-word preview is also stored as a fallback.

### Adding PDFs

1. Click the **Upload a PDF** file picker and select a file.
2. Optionally check **Save to persistent library**.
3. Click **Add PDF**.

The full PDF text is extracted via `pdfplumber` and embedded into ChromaDB. There is no page or word limit — a 300-page document is fully indexed.

### Re-running after adding a resource late

If you add a resource after research has already progressed, the sidebar shows a warning with options to re-run from an earlier stage:

- **Re-run from Output Writing** — rewrites the document incorporating the new resource (fastest).
- **Re-run from Web Research** — repeats web research with the resource available.
- **Re-run from Gap Analysis** — repeats gap analysis and everything after.
- **Re-run from Initial Research** — starts the research pipeline from scratch.

### The Persistent Document Library

The **Document Library** expander in the sidebar shows all documents you have saved with the **Save to persistent library** toggle. These persist in the ChromaDB `library` collection between sessions.

To remove a document from the library, click the **✕** button next to it. This removes all its embedded chunks from ChromaDB.

**When are library documents used?**
On every step that retrieves resources, the agent queries both the current session collection and the persistent library. The most relevant chunks from either source are returned for the current query.

---

## Output Formats

### Blog Post
An engaging article suitable for a broad readership. Structure: title, hook introduction, clearly headed sections, conclusion with key points. Conversational but authoritative tone. Inline source citations.

### Research Paper
A formal academic-style paper. Structure: Abstract, Introduction, Background and Literature Review, Methods and Approach, Findings and Results, Discussion, Conclusion, References. Third-person voice, precise language.

### Essay
A thesis-driven analytical piece. Structure: thesis introduction, evidence-backed body paragraphs, counter-arguments, strong conclusion. Appropriate for humanities and social science topics.

### Executive Summary
A concise decision-oriented brief. Structure: Key Findings, Background, Analysis, Recommendations, Conclusion. Bullet points, short paragraphs, no jargon. Designed for time-constrained readers.

### NSF-Style Grant Proposal
A research funding proposal. Structure: Project Summary, Introduction and Motivation, Background and Related Work, Research Objectives, Proposed Approach, Intellectual Merit and Broader Impacts, Timeline and References.

### Explainer / Educational Article
An accessible educational breakdown. Structure: big-picture opening, concept-by-concept explanation building in complexity, real-world examples and analogies, key takeaways. Avoids unexplained jargon.

---

## RAG and Semantic Retrieval

When **Enable semantic RAG** is on, the agent uses ChromaDB and embedding vectors instead of fixed text truncation for all resource injection.

### How it works

1. **Ingest** — when you add a URL or PDF, the full text is split into overlapping 600-character chunks using `RecursiveCharacterTextSplitter`.
2. **Embed** — each chunk is embedded using your selected Ollama embedding model.
3. **Store** — chunks are stored in ChromaDB with metadata (source URL/filename, title, type, chunk index).
4. **Retrieve** — at each research stage, the agent's current query (topic, research plan excerpt, section name) is embedded and the most similar chunks are retrieved using cosine similarity.
5. **Inject** — retrieved chunks are formatted and injected into the agent prompt, grouped by source.

### Per-section retrieval (section-by-section writing)

For documents of 5,000+ words, at the start of the writing step the research plan, initial research, and web research findings are also embedded into a temporary ChromaDB collection. Each section then retrieves its own relevant subset — a methodology section gets different context than an introduction.

### Disabling RAG

Uncheck **Enable semantic RAG** in the sidebar to fall back to the original behaviour: resources are truncated to 3,000 characters each and concatenated directly into prompts.

---

## Web Search Settings

### Ollama Web Search (default)

Uses Ollama's hosted search API with your `OLLAMA_API_KEY` (set it in `.env`; see `.env.example`). Fast and needs no browser. If the key is not set, this option is hidden and DuckDuckGo is the default.

### Fall-back behaviour

The method you select is tried first. If it errors or returns no results, the remaining methods are tried automatically in the order Ollama → DuckDuckGo → Selenium.

### Selenium (Headless Browser)

The slowest method, but independent of any API. Launches Chrome or Firefox in headless mode, navigates to Google (with Bing as a fallback), and extracts titles, URLs, and snippets from search results.

**macOS Apple Silicon (M1/M2/M3):** Firefox is recommended over Chrome for compatibility. Install the ARM version of Chrome from [google.com/chrome](https://www.google.com/chrome/) if you prefer Chrome.

### DuckDuckGo API

A faster, browser-free alternative using the `duckduckgo-search` library. May be rate-limited if you run many searches in quick succession. The agent automatically retries up to 3 times with a 2-second delay between attempts.

### Disabling web search

Uncheck **Enable Web Search** in the sidebar. Stage 3 (Web Research) is skipped and the agent proceeds using only its training knowledge and any custom resources you provided.

---

## Writer Context Window

The writer context window determines how many tokens the writer agent can process at once (input + output combined). This is set via a direct API call to Ollama before the writer agent starts, which is necessary because the CrewAI/LiteLLM layer does not reliably pass context window settings to Ollama.

| Tokens | Approximate input capacity | Recommended for |
|--------|---------------------------|-----------------|
| 8,192 | ~6,000 words | Short outputs, low-VRAM systems |
| 16,384 | ~12,000 words | Standard outputs |
| 32,768 | ~24,000 words | Long outputs (default) |
| 49,152 | ~36,000 words | Very long documents |
| 65,536 | ~48,000 words | Maximum — 24+ GB VRAM |

If you reduce the context window, previously generated output is cleared so it is regenerated with the new setting.

---

## Resetting the Workflow

Click **Reset Workflow** in the sidebar at any time. This:

1. Clears the session ChromaDB collections (session resources and embedded research content)
2. Clears all Streamlit session state
3. Returns the app to Stage 0

The persistent document library is **not** cleared by a reset.

---

## Troubleshooting

### "No Ollama models found"
Ensure Ollama is running: `ollama serve`. Check installed models: `ollama list`.

### Web search returns no results
- Try switching from Selenium to DuckDuckGo in the sidebar (or vice versa).
- On macOS ARM, switch from Chrome to Firefox.
- Check your internet connection.
- Reduce the number of simultaneous searches by running fewer research topics at once.

### Writer produces truncated output
- Increase the writer context window in the sidebar.
- For very long targets, ensure you are using section-by-section mode (5,000+ words).
- Try a model with a larger native context window (e.g. `llama3.2` supports 128K context).

### Embedding is slow
- Install and select a dedicated embedding model: `ollama pull nomic-embed-text`.
- Using a large language model (e.g. `gemma3:27b`) as the embedding model is significantly slower.

### ChromaDB errors on startup
Delete the `chroma_db/` folder and restart. It will be recreated automatically. This clears all embedded documents including the persistent library.

```bash
rm -rf chroma_db/
```

### "Could not prime Ollama context window"
The app retries up to 3 times. If this warning persists, Ollama may be under load or the model may not support the requested context size. Try reducing the writer context window in the sidebar.

### App is slow / agents time out
- Use a smaller model (e.g. `llama3.2` instead of `gemma3:27b`).
- Disable web search if you do not need current information.
- Reduce the writer context window.
- Close other applications to free RAM/VRAM.
