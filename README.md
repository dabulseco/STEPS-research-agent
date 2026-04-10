# AI Research Agent v3

A local, privacy-first autonomous research agent that plans, investigates, and writes long-form documents — all running on your machine with no cloud API required.

Built with [CrewAI](https://github.com/crewAIInc/crewAI), [Ollama](https://ollama.ai), [ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).

---

## Features

- **Multi-stage research pipeline** — automated planning, initial research, gap analysis, web search, and writing with human review checkpoints at each stage
- **Fully local** — all LLM inference runs via Ollama; no OpenAI or cloud API keys required
- **Semantic RAG** — documents are chunked, embedded, and stored in ChromaDB; retrieval is query-aware rather than relying on fixed truncation
- **Per-section retrieval** — for long documents (5,000+ words), each section receives the most relevant research context rather than a compressed summary of everything
- **Web search reranking** — search results are reordered by embedding similarity to the query before being passed to the agent
- **Persistent document library** — URLs and PDFs can be saved across sessions and retrieved in future research projects
- **Six output formats** — Blog Post, Research Paper, Essay, Executive Summary, NSF-Style Grant Proposal, Explainer
- **Configurable length** — 500 words to 5,000+ words with section-by-section writing for long outputs
- **DOCX and Markdown export**

---

## Requirements

- macOS, Linux, or Windows (WSL recommended on Windows)
- [Ollama](https://ollama.ai) installed and running
- At least one Ollama language model (e.g. `llama3.2`, `gemma3:27b`)
- `nomic-embed-text` recommended for embeddings (see setup below)
- Python 3.11+
- Chrome or Firefox for Selenium-based web search (optional)

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd research_agent_v3

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull a language model and embedding model
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Recommended Models

| Purpose | Model | Pull command |
|---------|-------|-------------|
| Research & writing | `gemma3:27b` | `ollama pull gemma3:27b` |
| Research & writing (lighter) | `llama3.2` | `ollama pull llama3.2` |
| Embeddings | `nomic-embed-text` | `ollama pull nomic-embed-text` |
| Embeddings (alternative) | `mxbai-embed-large` | `ollama pull mxbai-embed-large` |

Any model installed in Ollama will appear in the model selector. The embedding model can be any Ollama model, but dedicated embedding models are faster and produce better results.

---

## Project Structure

```
research_agent_v3/
├── app.py              # Main Streamlit application
├── rag.py              # ChromaDB DocumentStore — chunking, embedding, retrieval
├── requirements.txt    # Python dependencies
├── user_manual.md      # Full user manual
├── README.md           # This file
└── chroma_db/          # ChromaDB persistent store (git-ignored)
```

---

## Configuration

All settings are available in the sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Ollama Model | First installed | Language model for all research and writing |
| Enable semantic RAG | On | Use ChromaDB for retrieval instead of fixed truncation |
| Embedding model | `nomic-embed-text` | Model used to generate document embeddings |
| Writer context window | 32,768 tokens | Token budget for the writer agent (8K–64K) |
| Enable web search | On | Allow the agent to search the web during gap-filling |
| Search method | Selenium | Headless browser (more reliable) or DuckDuckGo API |

---

## License

MIT
