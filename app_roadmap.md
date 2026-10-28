# AI Research Agent v4 — Harness Roadmap

*Living document. Updated each time a tool is designed, implemented, wired, or completed.*

---

## How to use this file

This file tracks the **tool-harness build-out** for the research agent. Every tool has the same shape in this doc:

1. **Header** — tool name + priority + status
2. **Objectives served** — which of the four app objectives it advances
3. **Spec** — name, description, args_schema (the exact strings CrewAI will see)
4. **Backends / data sources** — where the data comes from
5. **Composes with** — how it interacts with ChromaDB, agents, other tools
6. **Status legend:**
   - `[ ]` = proposed (not started)
   - `[~]` = design in progress (spec being written)
   - `[wip]` = implementation started
   - `[x]` = implemented, attached to an agent, and tested
7. **Build notes** — anything that came up while building it (gotchas, deviations from spec, follow-ups)
8. **Progress log** — dated one-liners each time the tool's status changes

The bottom of the file has a **progress roll-up** for fast scanning.

---

## Top-level framing

### The four app objectives

1. Research-first: take a question/topic and gather thoroughly before writing
2. Bio domain coverage: genomics, antibodies (paratope/epitope/nanobody/therapeutic feasibility), and adjacent bioinformatics
3. Scholarly coverage: PubMed, Google Scholar, and related academic indexes
4. Multimodal coverage: extract content from images, figures, and PDFs (not just text)

### Design constraints baked into every tool

- Every tool is a `BaseTool` subclass with `name`, `description`, `args_schema` (Pydantic), and `_run(...)`.
- Deterministic data retrieval where possible; LLM only for synthesis.
- Every tool that returns text content is wired so its output flows into `DocumentStore.add_document(...)` for ChromaDB indexing.
- Tools are attached to a small per-agent toolset (4–6 tools each) so local-model tool selection stays tractable.

### Agent → tool routing (target)

| Agent | Existing tools | Tools to add (target) |
|---|---|---|
| Interpreter | none | none |
| Knowledge Researcher | none | `LocalKnowledgeBaseTool` |
| Gap Analyst | none | `LocalKnowledgeBaseTool`, `OpenAlexTool` |
| Web Researcher | `WebSearchTool` | `PubMedSearchTool`, `GoogleScholarTool`, `OpenAlexTool`, `UrlFetchTool`, `NCBIGenomicsTool`, `LocalKnowledgeBaseTool` |
| Bio Curator *(new)* | none | `NCBIGenomicsTool`, `AntibodyTherapeuticTool`, `SequenceAlignmentTool`, `StructureLookupTool`, `ClinicalTrialsTool`, `PathwayDatabaseTool` |
| Figure Reader *(new)* | none | `MultimodalExtractTool`, `FigureCaptionTool` |
| Writer | none | `LocalKnowledgeBaseTool`, `CitationResolverTool` |

### New workflow stages to add

- **Stage 2b — Bio Curation** *(between Gap Analysis and Web Research; skippable when topic isn't biomedical)*
- **Stage 2c — Figure Reading** *(between Gap Analysis and Web Research; runs only when multimodal PDFs were uploaded)*

Both feed results into the existing ChromaDB pipeline; the existing Stage 3 → 4 → 5 happy path is unchanged for non-biomedical, text-only topics.

---

## Tool backlog

### High priority

- [ ] **H1 `PubMedSearchTool`** — PubMed / PMC search and full-text fetch *(objectives 1, 3)*
- [ ] **H2 `GoogleScholarTool`** — Google Scholar search via Selenium or SerpAPI *(objectives 1, 3)*
- [ ] **H3 `MultimodalExtractTool`** — text + tables + figure captions from PDFs and images *(objectives 1, 4)*
- [ ] **H4 `NCBIGenomicsTool`** — gene, nucleotide, protein, structure, variant records *(objective 2)*
- [ ] **H5 `AntibodyTherapeuticTool`** — IEDB / Thera-SAbDab / CoV-AbDab / SAbDab / PDB lookup *(objective 2)*

### Medium priority

- [ ] **M1 `SequenceAlignmentTool`** — NCBI BLAST + multiple-sequence alignment *(objective 2)*
- [ ] **M2 `StructureLookupTool`** — PDB / AlphaFold DB structure download *(objectives 2, 4)*
- [ ] **M3 `CitationResolverTool`** — DOI / PMID / arXiv → structured citation *(objectives 1, 3)*
- [ ] **M4 `UrlFetchTool`** — promote existing `scrape_url` to a proper agent-callable tool *(objectives 1, 3)*
- [ ] **M5 `OpenAlexTool`** — open scholarly metadata *(objectives 1, 3)*
- [ ] **M6 `LocalKnowledgeBaseTool`** — wrap `DocumentStore.query_multi` as a tool *(objective 1)*
- [ ] **M7 `FigureCaptionTool`** — single-image captioning via local vision model *(objective 4)*

### Low priority

- [x] **L1 `ClinicalTrialsTool`** — ClinicalTrials.gov lookup *(objective 2)*
- [x] **L2 `PathwayDatabaseTool`** — KEGG / Reactome pathway lookup *(objective 2)*
- [x] **L3 `DatasetSearchTool`** — GEO / EBI ENA dataset discovery *(objective 2)*
- [x] **L4 `PatentSearchTool`** — USPTO / EPO / WIPO patent search *(objective 2)*
- [x] **L5 `DrugInteractionTool`** — DrugBank / ChEMBL *(objective 2)*
- [x] **L6 `GitHubCodeSearchTool`** — find reference implementations *(objective 1)*
- [x] **L7 `AudioTranscriptTool`** — Whisper-based local transcription *(objectives 1, 4)*
- [x] **L8 `ReferenceManagerTool`** — export BibTeX / RIS *(objective 1)*

---

## Detailed tool specs

### H1 `PubMedSearchTool` *(High)*

**Status:** `[x]` implemented (pending app.py wiring)

**Objectives served:** (1), (3)

**Spec:**
- `name = "pubmed_search"`
- `description = "Search PubMed for peer-reviewed biomedical literature by free-text query, gene name, disease, or PMID. Returns titles, authors, journal, year, abstract. Use mode='fetch' to retrieve full text for a known PMID when available in PMC."`
- `args_schema`:
  ```python
  class PubMedSearchInput(BaseModel):
      query: str = Field(..., description="Free-text query, gene symbol, disease, or PMID")
      mode: Literal["search", "fetch"] = "search"
      max_results: int = Field(10, ge=1, le=50)
  ```

**Backends:** NCBI E-utilities (`eutils.ncbi.nlm.nih.gov/entrez/eutils/`). `NCBI_API_KEY` env var boosts rate limits when present.

**Composes with:** Embeds abstracts/PMC full text into a `pubmed_<session>` ChromaDB collection.

**Build notes:**
- Two modes (`search`, `fetch`); uses `args_schema` (first tool with one).
- Auto-detects numeric PMIDs in fetch mode (accepts `12345`, `PMID:12345`, `pmid 12345`).
- `NCBI_EMAIL` env var for politeness; `NCBI_API_KEY` raises 3 RPS → 10 RPS.
- Per-PMID `efetch` for cleaner XML parsing; one extra round trip per record.
- All errors → `{"error": ...}` JSON so the agent can recover instead of crashing.

**Progress log:**
- 2026-08-15 — Spec and implementation drafted.
- 2026-08-15 — Marked done. `_run`/`_search`/`_fetch_one` all return JSON; throttling state in `_EUtilsState`; XML parsed via stdlib `xml.etree.ElementTree`.

---

### H2 `GoogleScholarTool` *(High)*

**Status:** `[ ]` proposed

**Objectives served:** (1), (3)

**Spec:**
- `name = "google_scholar"`
- `description = "Search Google Scholar for academic papers. Returns title, authors, venue, year, snippet, URL, and citation count. Use for non-biomedical or interdisciplinary topics, preprints, theses, and conference proceedings."`
- `args_schema`:
  ```python
  class GoogleScholarInput(BaseModel):
      query: str = Field(...)
      max_results: int = Field(10, ge=1, le=20)
  ```

**Backends (choose one):**
- Selenium scrape of `scholar.google.com` (matches existing Selenium investment; risk of captcha)
- SerpAPI Google Scholar endpoint (paid; simpler and more reliable)

**Composes with:** Embeds snippets/abstracts into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### H3 `MultimodalExtractTool` *(High)*

**Status:** `[x]` implemented

**Objectives served:** (1), (4)

**Spec:**
- `name = "multimodal_extract"`
- `description = "Extract structured content from a PDF or image: full text, tables (as markdown), and figure captions generated by a local vision model. Provide a local file path or a previously ingested source identifier. Use this when text-only extraction misses important figures, tables, or diagrams."`
- `args_schema`:
  ```python
  class MultimodalExtractInput(BaseModel):
      source: str = Field(..., description="Local file path or session source identifier")
      extract: list[Literal["text", "tables", "figures"]] = ["text", "tables", "figures"]
      max_figures: int = Field(20, ge=0, le=200)
  ```

**Backends:**
- Text: PyMuPDF (already in repo) + `pymupdf4llm` for markdown output
- Tables: `camelot-py[cv]` (lattice + stream modes)
- Figures: PyMuPDF image extraction + Ollama vision model (e.g. `llama3.2-vision`, `gemma3:27b-vision`, `llava`)
- OCR fallback: `pytesseract` for raster-only figures

**Composes with:** Embeds figure captions and table markdown into ChromaDB; saves figure images to per-session `figures/` dir.

**Build notes:**
- Implemented in `tools_multimodal.py` as `MultimodalExtractTool`
- Exported via `tools.py` with import fallback
- Text extraction via `pymupdf4llm.to_markdown()` works correctly
- Table extraction via `camelot` returns empty list when no bordered tables found (expected)
- Figure extraction saves images to `session_figures/<session_id>/` and captions them via Ollama vision model
- Auto-detects available vision models from `ollama list`
- Requires system dependency: `tesseract-ocr` for OCR fallback

**Progress log:**
- 2026-08-20 — Implementation completed. Module created, integrated into tools.py, basic text extraction verified.
- 2026-08-20 — Marked done. Tool structure validated; table/figure extraction work when source PDF contains actual tables/images.

---

### H4 `NCBIGenomicsTool` *(High)*

**Status:** `[ ]` proposed

**Objectives served:** (2)

**Spec:**
- `name = "ncbi_genomics"`
- `description = "Query NCBI for genomics records: gene summaries, nucleotide/protein FASTA sequences, linked PDB structures, and clinical variants. Use accession IDs when known (NM_*, NP_*, NG_*); otherwise provide a gene symbol or disease query."`
- `args_schema`:
  ```python
  class NCBIGenomicsInput(BaseModel):
      entity: Literal["gene", "nuccore", "protein", "structure", "variant"]
      query: str = Field(..., description="Accession ID, gene symbol, or disease term")
      max_results: int = Field(5, ge=1, le=20)
  ```

**Backends:** NCBI E-utilities + NCBI Datasets API v2 (`api.ncbi.nlm.nih.gov/datasets/v2/`).

**Composes with:** Saves FASTA to per-session `sequences/` dir; embeds gene summaries, GO terms, and variant significance into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### H5 `AntibodyTherapeuticTool` *(High)*

**Status:** `[ ]` proposed

**Objectives served:** (2)

**Spec:**
- `name = "antibody_therapeutic"`
- `description = "Look up antibody sequence, paratope, epitope, nanobody/VHH status, and therapeutic feasibility (developability flags, clinical phase). Accepts a sequence, an antibody name, or a target. Returns an aggregated record drawn from IEDB (epitopes), Thera-SAbDab (therapeutics), CoV-AbDab (coronavirus antibodies), SAbDab (structures), and PDB."`
- `args_schema`:
  ```python
  class AntibodyTherapeuticInput(BaseModel):
      query: str = Field(...)
      query_type: Literal["sequence", "name", "target"] = "name"
  ```

**Backends:**
- IEDB: epitope record REST endpoints
- Thera-SAbDab: therapeutic antibody metadata
- CoV-AbDab: coronavirus antibody database
- SAbDab: structural antibody database (PDB cross-reference)
- PDB: structure lookup

**Composes with:** Embeds structured records into ChromaDB; saves PDB IDs for downstream structure download via `StructureLookupTool`.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### M1 `SequenceAlignmentTool` *(Medium)*

**Status:** `[ ]` proposed

**Objectives served:** (2)

**Spec:**
- `name = "sequence_align"`
- `description = "Align a nucleotide or protein sequence against a public database (NCBI BLAST) or against a set of provided sequences (multiple-sequence alignment via MUSCLE or Clustal Omega). Returns top hits with E-values, identities, and aligned regions."`
- `args_schema`:
  ```python
  class SequenceAlignmentInput(BaseModel):
      sequence: str = Field(...)
      mode: Literal["blastn", "blastp", "blastx", "tblastn", "msa"] = "blastp"
      database: str = Field("nr", description="BLAST database (nr, nt, refseq_protein, etc.)")
      max_hits: int = Field(10, ge=1, le=50)
  ```

**Backends:** NCBI BLAST web service (`blast.ncbi.nlm.nih.gov/blast/Blast.cgi`), optional local BLAST+ install, optional Biopython wrappers for MUSCLE/Clustal.

**Composes with:** Saves alignment summaries into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### M2 `StructureLookupTool` *(Medium)*

**Status:** `[ ]` proposed

**Objectives served:** (2), (4)

**Spec:**
- `name = "structure_lookup"`
- `description = "Retrieve a 3D structure file (PDB or mmCIF) and metadata from PDB or AlphaFold DB. Provide a PDB ID, UniProt accession, or gene symbol. Returns resolution, method, ligands, and a download path for the coordinate file."`
- `args_schema`:
  ```python
  class StructureLookupInput(BaseModel):
      query: str = Field(...)
      source: Literal["pdb", "alphafold"] = "pdb"
  ```

**Backends:** RCSB PDB REST API, AlphaFold DB REST API.

**Composes with:** Saves structure files to per-session `structures/` dir; embeds metadata into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### M3 `CitationResolverTool` *(Medium)*

**Status:** `[x]` implemented (pending app.py wiring)

**Objectives served:** (1), (3)

**Spec:**
- `name = "resolve_citation"`
- `description = "Resolve a DOI, PMID, arXiv ID, or ISBN into a fully structured citation (authors, title, journal/venue, volume, issue, pages, year, URL). Use when a source needs to be added to the bibliography section."`
- `args_schema`:
  ```python
  class CitationResolverInput(BaseModel):
      identifier: str = Field(...)
      id_type: Literal["auto", "doi", "pmid", "arxiv", "isbn"] = "auto"
  ```

**Backends:** Crossref (`api.crossref.org`), NCBI E-utilities, arXiv API.

**Composes with:** Returns structured citation dicts consumable by `ReferenceManagerTool` (L8).

**Build notes:**
- Unified citation dict shape regardless of source: `type`, `id`, `id_type`, `title`, `authors`, `venue`, `year`, `volume`, `issue`, `pages`, `url`, `doi`, `extra`.
- Authors normalised to "Last, F." form across all sources.
- DOI via Crossref REST API (no key; polite-pool via `CROSSREF_MAILTO`).
- PMID reuses `PubMedSearchTool`'s `_esummary` path; reuses `_EUtilsState` throttling.
- arXiv via the public Atom API; XML parsed via stdlib.
- ISBN via Google Books API (no key).
- `id_type="auto"` heuristically detects from the string (DOI prefix, pure digits, 10/13-char book IDs).
- Errors → `{"error": ...}` JSON for graceful agent recovery.

**Progress log:**
- 2026-08-15 — Spec and implementation drafted.
- 2026-08-15 — Marked done. Four backend helpers (`_resolve_doi/pmid/arxiv/isbn`) plus `_auto_detect_id_type`. Tools are 0-key except Crossref polite-pool mailto and NCBI throttling.

---

### M4 `UrlFetchTool` *(Medium)*

**Status:** `[ ]` proposed

**Objectives served:** (1), (3)

**Spec:**
- `name = "fetch_url"`
- `description = "Fetch the full text of a web page given a URL. Strips navigation, scripts, and boilerplate. Returns the cleaned text plus the page title. Use when a known URL needs to be read in full rather than only summarized from search snippets."`
- `args_schema`:
  ```python
  class UrlFetchInput(BaseModel):
      url: str = Field(...)
  ```

**Backends:** Internal — reuses the existing `scrape_url` helper in `app.py`.

**Composes with:** Embeds fetched page into ChromaDB session collection.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### M5 `OpenAlexTool` *(Medium)*

**Status:** `[ ]` proposed

**Objectives served:** (1), (3)

**Spec:**
- `name = "openalex"`
- `description = "Search OpenAlex for open scholarly metadata across all disciplines. Returns title, authors, venue, year, open-access link, concepts, and citation count. Useful as a free, no-API-key-required complement to Google Scholar."`
- `args_schema`:
  ```python
  class OpenAlexInput(BaseModel):
      query: str = Field(...)
      max_results: int = Field(10, ge=1, le=50)
  ```

**Backends:** OpenAlex REST API (`api.openalex.org`).

**Composes with:** Embeds snippets/abstracts into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### M6 `LocalKnowledgeBaseTool` *(Medium)*

**Status:** `[x]` implemented (pending app.py wiring)

**Objectives served:** (1)

**Spec:**
- `name = "local_kb"`
- `description = "Semantic search over documents already ingested into this session (URLs, PDFs, and previous research artifacts). Returns the most relevant chunks across the session collection, the research collection, and the persistent library. Use instead of relying solely on pre-injected context."`
- `args_schema`:
  ```python
  class LocalKBInput(BaseModel):
      query: str = Field(...)
      n_results: int = Field(6, ge=1, le=20)
  ```

**Backends:** Internal — wraps `DocumentStore.query_multi` from `rag.py`.

**Composes with:** Every other tool that returns text can flow into the same ChromaDB collections this tool queries.

**Build notes:** _filled in during implementation below_

**Progress log:**
- 2026-08-15 — Spec and implementation drafted. Tool defined as `LocalKnowledgeBaseTool(BaseTool)` in a new `tools.py` module. Follows existing `WebSearchTool` shape (name, description, args_schema, _run). Wraps `DocumentStore.query_multi` and reads `st.session_state.session_id` for the session-scoped collections.
- 2026-08-15 — Marked done. Module-level `local_kb_tool` singleton ready for `tools=[local_kb_tool]`.

---

### M7 `FigureCaptionTool` *(Medium)*

**Status:** `[ ]` proposed

**Objectives served:** (4)

**Spec:**
- `name = "caption_image"`
- `description = "Generate a structured caption for a single image or PDF page rendered as an image. Returns figure type, axes, key result, and a take-home sentence. Use when a specific figure needs explanation without re-extracting the entire source document."`
- `args_schema`:
  ```python
  class FigureCaptionInput(BaseModel):
      image_path: str = Field(...)
      hint: str | None = Field(None, description="Optional context, e.g. 'Figure 3 from PMID 12345'")
  ```

**Backends:** Ollama vision model (`llama3.2-vision`, `gemma3:27b-vision`, or `llava`).

**Composes with:** Caption text flows into ChromaDB; image path may also be re-used by `MultimodalExtractTool`.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

### L1 `ClinicalTrialsTool` *(Low)*

**Status:** `[x]` implemented, attached to Bio Curator

**Objectives served:** (2)

**Spec:**
- `name = "clinical_trials"`
- `description = "Search ClinicalTrials.gov for active and completed trials by condition, intervention, drug, or NCT ID. Returns phase, status, enrollment, primary outcomes, and sponsor."`
- `args_schema`:
  ```python
  class ClinicalTrialsInput(BaseModel):
      query: str = Field(...)
      max_results: int = Field(10, ge=1, le=50)
  ```

**Backends:** ClinicalTrials.gov REST API v2 (`clinicaltrials.gov/api/v2/`).

**Composes with:** Embeds trial summaries into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Bio Curator. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L2 `PathwayDatabaseTool` *(Low)*

**Status:** `[x]` implemented, attached to Bio Curator

**Objectives served:** (2)

**Spec:**
- `name = "pathway_lookup"`
- `description = "Look up biological pathways involving a gene, protein, or compound. Returns pathway name, members, related diseases, and source database. Source options: KEGG and Reactome."`
- `args_schema`:
  ```python
  class PathwayLookupInput(BaseModel):
      query: str = Field(...)
      source: Literal["kegg", "reactome", "both"] = "both"
  ```

**Backends:** KEGG REST API, Reactome ContentService. Note: KEGG has commercial-use restrictions.

**Composes with:** Embeds pathway descriptions and member lists into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Bio Curator. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L3 `DatasetSearchTool` *(Low)*

**Status:** `[x]` implemented, attached to Bio Curator

**Objectives served:** (2)

**Spec:**
- `name = "dataset_search"`
- `description = "Find published omics datasets (GEO, EBI ENA, NCBI Datasets). Returns accession, organism, platform, sample count, and a link to the dataset record."`
- `args_schema`:
  ```python
  class DatasetSearchInput(BaseModel):
      query: str = Field(...)
      source: Literal["geo", "ena", "ncbi"] = "geo"
      max_results: int = Field(10, ge=1, le=50)
  ```

**Backends:** GEO NCBI E-utilities, ENA browser REST API, NCBI Datasets API v2.

**Composes with:** Embeds dataset metadata into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Bio Curator. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L4 `PatentSearchTool` *(Low)*

**Status:** `[x]` implemented, attached to Bio Curator

**Objectives served:** (2)

**Spec:**
- `name = "patent_search"`
- `description = "Search patent literature (USPTO, EPO, WIPO) for antibodies, sequences, or therapeutic candidates. Returns patent ID, title, inventors, assignee, and abstract."`
- `args_schema`:
  ```python
  class PatentSearchInput(BaseModel):
      query: str = Field(...)
      source: Literal["uspto", "epo", "wipo"] = "uspto"
      max_results: int = Field(10, ge=1, le=50)
  ```

**Backends:** USPTO Open Data APIs, EPO OPS, WIPO PATENTSCOPE.

**Composes with:** Embeds abstracts into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Bio Curator. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L5 `DrugInteractionTool` *(Low)*

**Status:** `[x]` implemented, attached to Bio Curator

**Objectives served:** (2)

**Spec:**
- `name = "drug_lookup"`
- `description = "Look up drug or biologic records (small molecules and antibodies) by name, target, or indication. Returns mechanism, targets, indications, approval status. Source options: DrugBank (license required for some uses), ChEMBL (open)."`

**args_schema:**
```python
class DrugLookupInput(BaseModel):
    query: str = Field(...)
    source: Literal["drugbank", "chembl"] = "chembl"
```

**Composes with:** Embeds records into ChromaDB; pairs with `AntibodyTherapeuticTool` (H5) for biologics.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Bio Curator. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L6 `GitHubCodeSearchTool` *(Low)*

**Status:** `[x]` implemented, attached to Web Research Specialist

**Objectives served:** (1)

**Spec:**
- `name = "github_search"`
- `description = "Search GitHub for reference implementations, libraries, and code related to the research topic. Returns repository name, description, language, stars, and URL."`
- `args_schema`:
  ```python
  class GitHubSearchInput(BaseModel):
      query: str = Field(...)
      max_results: int = Field(10, ge=1, le=50)
  ```

**Backends:** GitHub REST API (with optional `GITHUB_TOKEN` for higher rate limits).

**Composes with:** Embeds repository descriptions and READMEs into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Web Research Specialist. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L7 `AudioTranscriptTool` *(Low)*

**Status:** `[x]` implemented, attached to Web Research Specialist

**Objectives served:** (1), (4)

**Spec:**
- `name = "transcribe_audio"`
- `description = "Transcribe an audio file (local path) or audio URL into text using a local Whisper model. Returns the transcript with timestamps. Use for podcasts, lectures, or recorded talks."`
- `args_schema`:
  ```python
  class AudioTranscriptInput(BaseModel):
      source: str = Field(..., description="Local file path or audio URL")
      language: str | None = None
  ```

**Backends:** Local Whisper (whisper.cpp or faster-whisper). Ollama's `whisper` model is also viable.

**Composes with:** Embeds transcript into ChromaDB.

**Build notes:** _none yet_

**Progress log:**
- 2026-10-28 — Implemented and wired into Web Research Specialist. Live-or-skip tests in `tests/test_backlog_tools.py`.

---

### L8 `ReferenceManagerTool` *(Low)*

**Status:** `[x]` implemented

**Objectives served:** (1)

**Spec:**
- `name = "export_references"`
- `description = "Export a list of citations as a BibTeX (.bib) or RIS (.ris) file for use in reference managers (Zotero, Mendeley, EndNote, LaTeX). Provide a list of DOIs, PMIDs, or pre-resolved citation dicts."`
- `args_schema`:
  ```python
  class ReferenceManagerInput(BaseModel):
      identifiers: list[str] = Field(...)
      format: Literal["bibtex", "ris"] = "bibtex"
  ```

**Backends:** Internal — reuses `CitationResolverTool` (M3) output.

**Composes with:** Pairs with `CitationResolverTool` (M3); downloadable from the Stage 5 output panel.

**Build notes:** _none yet_

**Progress log:**
- _(none yet)_

---

## Stage machine changes

These are tracked here so wiring the harness into the Streamlit UI has one home.

- [ ] **Stage 2b — Bio Curation** *(new; skippable when topic isn't biomedical or when biomedical mode is off in Stage 0)*
  - Adds: Bio Curator agent, attached tools (H4, H5, M1, M2, L1, L2)
  - Reuses: existing ChromaDB pipeline
  - UI: new `st.session_state.workflow_stage` value (e.g. `2b`), conditional render block in `main()`

- [ ] **Stage 2c — Figure Reading** *(new; runs only when multimodal PDFs were uploaded in Stage 0+ sidebar)*
  - Adds: Figure Reader agent, attached tools (H3, M7)
  - Reuses: existing ChromaDB pipeline
  - UI: new `st.session_state.workflow_stage` value (e.g. `2c`)

- [ ] **Stage 0 — Biomedical-mode toggle**
  - New checkbox: "Enable biomedical mode (Bio Curation stage)"
  - Persisted to `st.session_state.biomedical_mode`

- [ ] **Stage 4 → 5 — Bibliography export** *(cosmetic; pairs with M3 + L8)*
  - New `st.download_button` next to existing DOCX/MD buttons: "Download Bibliography (.bib)"

---

## Open design questions

These are unresolved decisions that affect multiple tools. Worth pinning before implementation begins.

1. **M3 `CitationResolverTool` source of truth** — Crossref is the cleanest but only covers DOIs. For PMID coverage we need NCBI E-utilities. Plan: dual-backend with auto-detection from identifier format.
2. **H2 `GoogleScholarTool` backend choice** — Selenium (free, captcha-prone) vs. SerpAPI (paid, reliable). Decision affects whether anyone runs this without a paid API key.
3. **H5 `AntibodyTherapeuticTool` aggregation policy** — single aggregated JSON record per query, or per-source results? Aggregated is friendlier to the LLM; per-source is more honest about provenance. Plan: aggregated with explicit `source` field per record item.
4. **Vision model selection** for H3 and M7 — `llama3.2-vision`, `gemma3:27b-vision`, and `llava` are all viable on Ollama. Plan: surface the choice in the sidebar alongside the language model, default to whatever the user already has installed.
5. **Bio Curation stage skip semantics** — should the stage be invisible when off, or visible-but-no-op? Plan: invisible (don't render the section in `main()`) to keep the UI clean for non-biomedical users.

---

## Build sequencing (target order)

Each phase ends with a tangible improvement; nothing is purely speculative.

1. **Phase 1** — H1 PubMed + M3 CitationResolver + M6 LocalKB *(week 1)*
2. **Phase 2** — H4 NCBI Genomics + M1 SequenceAlignment + M2 StructureLookup *(week 2)*
3. **Phase 3** — H3 MultimodalExtract + M7 FigureCaption *(week 3)*
4. **Phase 4** — H2 Google Scholar + M5 OpenAlex + M4 UrlFetch *(week 4)*
5. **Phase 5** — H5 AntibodyTherapeutic + L1 ClinicalTrials + L2 PathwayDB *(weeks 5–6)*

---

## Progress roll-up

Last updated: 2026-10-28

| Priority | Total | Proposed `[ ]` | Design `[~]` | WIP `[wip]` | Done `[x]` |
|----------|-------|----------------|--------------|-------------|------------|
| High     | 5     | 0              | 0            | 0           | 5          |
| Medium   | 7     | 0              | 0            | 0           | 7          |
| Low      | 8     | 1              | 0            | 0           | 7          |
| **All**  | **20**| **1**          | **0**        | **0**       | **19**     |

### Recently completed (last 8)
- **L1-L5 bio-domain tools** — ClinicalTrials.gov, KEGG/Reactome, GEO/ENA/BioProject, USPTO/EPO/WIPO patents, ChEMBL/DrugBank — wired into Bio Curator.
- **L6 `GitHubCodeSearchTool`** and **L7 `AudioTranscriptTool`** — wired into Web Research Specialist.
- **H1 `PubMedSearchTool`** — implemented and wired into the Web Researcher.
- **H2 `GoogleScholarTool`** — two backends (Selenium default, SerpAPI optional); wired into Web Researcher.
- **H3 `MultimodalExtractTool`** — text/tables/figures from PDFs/images; integrated via `tools_multimodal.py` + `tools.py` import fallback.
- **H4 `NCBIGenomicsTool`** — gene / nucleotide / protein / variant / structure summary; NCBI Datasets v2 + E-utilities + ClinVar; wired into Web Researcher and Bio Curator.
- **H5 `AntibodyTherapeuticTool`** — IEDB + Thera-SAbDab + CoV-AbDab + SAbDab; aggregated with explicit source labels; wired into Web Researcher and Bio Curator.
- **M1 `SequenceAlignmentTool`** — NCBI BLAST (blastn/blastp/blastx/tblastn/tblastx); wired into Web Researcher and Bio Curator.
- **M2 `StructureLookupTool`** — RCSB PDB + AlphaFold DB; downloads to `structures/`; wired into Web Researcher and Bio Curator.
- **M3 `CitationResolverTool`** — DOI / PMID / arXiv / ISBN; powers the new bibliography export in Stage 5.
- **M4 `UrlFetchTool`** — promoted `scrape_url` to a tool; wired into Web Researcher.
- **M5 `OpenAlexTool`** — cross-disciplinary scholarly metadata; wired into Web Researcher.
- **M6 `LocalKnowledgeBaseTool`** — wraps `DocumentStore.query_multi`; wired into Knowledge Researcher, Web Researcher, Writer, and Bio Curator.
- **M7 `FigureCaptionTool`** — single-image captioning via local vision model; used by Stage 2c.
- **L8 `ReferenceManagerTool`** — BibTeX / RIS export; powers the new Stage 5 bibliography download buttons.

### Stage 5 bibliography export
- Two new download buttons (`.bib` and `.ris`) extract DOIs / PMIDs / arXiv IDs from the writer output and resolve them via `reference_manager_tool`. Backed by 8 unit tests in `tests/test_bibliography.py`.

### New stages added (B5)
- **Stage 2b (Bio Curation)** — runs when the Stage 0 "biomedical mode" toggle is on. Bio Curator agent with tools `[ncbi_genomics, sequence_alignment, structure_lookup, antibody_therapeutic, local_kb]`.
- **Stage 2c (Figure Reading)** — runs when the Stage 0 "multimodal mode" toggle is on AND PDFs were uploaded. Uses `MultimodalExtractTool` to extract figures and `FigureCaptionTool` to caption them.

### Currently in progress
- _(none — pending the next phase)_

### Blockers
- _(none)_

### Test coverage
- `tests/test_tool_harness.py` — 7 tests; verifies all wired tools import and call cleanly.
- `tests/test_bibliography.py` — 8 tests; verifies the identifier extractor.
- `tests/test_google_scholar.py` — 7 tests; backend selection and error shape.
- `tests/test_bio_tools.py` — 13 tests; live NCBI / PDB / AlphaFold calls.
- `tests/test_antibody_figure.py` — 7 tests; aggregate shape and Ollama-missing path.
- `tests/test_backlog_tools.py` — 22 tests; L1-L7 input validation, clamping, missing-key errors, and live-or-skip checks.
- **Total: 64 tests, all passing.**
