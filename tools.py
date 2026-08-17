"""Tool harness for the Research Agent.

Each tool is a `BaseTool` subclass that any CrewAI agent can attach via
`tools=[...]`. Tools follow the same shape as the existing `WebSearchTool`
in `app.py`:

- `name`           — function name the LLM emits in its tool-call
- `description`    — what the LLM reads to decide whether to call the tool
- `_run(...)`      — execution; returns a string (JSON-formatted when
                     structured output is helpful)

Tools are kept in this module rather than `app.py` so the harness can
grow without bloating the Streamlit UI code. `app.py` imports the tools
it needs at agent-construction time.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Literal

import requests
import streamlit as st
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag import DocumentStore

logger = logging.getLogger(__name__)

# Process-local cache for DocumentStore instances, keyed by embedding model.
# `app.py` uses Streamlit's `@st.cache_resource` for the same purpose inside
# the Streamlit runtime; tools.py can't reuse that decorator without importing
# streamlit, so we mirror it with a plain dict. The cache is per-process and
# is safe because DocumentStore holds an Ollama HTTP client (not mutable
# session state) and ChromaDB itself is thread-safe enough for our usage.
_doc_store_cache: dict[str, DocumentStore] = {}


def _get_document_store(embed_model: str) -> DocumentStore:
    """Return a cached DocumentStore for *embed_model* (process-local)."""
    cached = _doc_store_cache.get(embed_model)
    if cached is not None:
        return cached
    store = DocumentStore(embedding_model=embed_model)
    _doc_store_cache[embed_model] = store
    return store


# ---------------------------------------------------------------------------
# M6 — LocalKnowledgeBaseTool
# ---------------------------------------------------------------------------
# Wraps the existing `DocumentStore.query_multi(...)` so any agent can
# semantic-search over documents that have already been ingested into this
# session (URLs, PDFs, prior research artifacts) plus the persistent
# cross-session library.
#
# This is the smallest tool in the harness: ~30 lines, zero new dependencies,
# reuses everything already in `rag.py`. Its job is to prove the harness
# pattern before we build the heavier domain tools (PubMed, NCBI Genomics,
# MultimodalExtract, etc.).

class LocalKnowledgeBaseTool(BaseTool):
    """Semantic search over documents already ingested into this session."""

    name: str = "local_kb"
    description: str = (
        "Search the local knowledge base for chunks relevant to a query. "
        "Returns the most semantically similar passages drawn from documents "
        "the user has added to this session (URLs and PDFs in the sidebar), "
        "from the session's prior research artifacts (plan, initial "
        "research, web research), and from the persistent cross-session "
        "library. Use this when you need authoritative primary-source "
        "content already in the database rather than relying on the model's "
        "training data alone."
    )

    def _run(self, query: str, n_results: int = 6) -> str:
        """Return up to *n_results* relevant chunks as a JSON string.

        Parameters
        ----------
        query:
            Natural-language query. The query is embedded with the same
            Ollama embedding model used at ingest time, then matched
            against stored chunks by cosine similarity.
        n_results:
            Maximum number of chunks to return (1–20, default 6).

        Returns
        -------
        str
            JSON-formatted list of result dicts. Each dict has the keys
            ``content``, ``source``, ``title``, ``type``, and ``score``
            (cosine similarity, 0–1, higher is better). Returns an empty
            JSON list when no documents are indexed or retrieval fails.
        """
        # --- Read configuration from Streamlit session state -------------
        # This mirrors the pattern used by WebSearchTool. Tools that run
        # inside a Streamlit app can read session state directly because
        # agents always execute in the same thread as the UI.
        if not st.session_state.get("rag_enabled", True):
            return json.dumps([])

        session_id: str = st.session_state.get("session_id", "default")
        embed_model: str = st.session_state.get("embed_model", "nomic-embed-text")

        # --- Clamp n_results to a safe range ------------------------------
        try:
            n = int(n_results)
        except (TypeError, ValueError):
            n = 6
        n = max(1, min(20, n))

        # --- Query the three relevant collections -------------------------
        # These match the collections that `DocumentStore` uses elsewhere:
        #   - session_<id>   : resources the user added this session
        #   - rsch_<id>      : embedded research artifacts (plan + research)
        #   - library        : persistent cross-session library
        # `query_multi` deduplicates and merges by cosine similarity score.
        collections = [f"session_{session_id}", f"rsch_{session_id}", "library"]
        try:
            doc_store: DocumentStore = _get_document_store(embed_model)
            results: list[dict[str, Any]] = doc_store.query_multi(
                query, collection_names=collections, n_results=n
            )
        except Exception as exc:
            logger.warning("LocalKB query failed: %s", exc)
            return json.dumps([])

        # --- Format the result for the LLM -------------------------------
        # We return JSON rather than freeform text because the LLM is
        # better at extracting structured fields (title, source, score)
        # from JSON than from prose. Indent=2 mirrors WebSearchTool so the
        # agent's context window shows it in a readable form.
        return json.dumps(results, indent=2)


# Module-level instance ready for `tools=[local_kb_tool]`.
local_kb_tool = LocalKnowledgeBaseTool()


# ===========================================================================
# H1 — PubMedSearchTool
# ===========================================================================
# The first tool with a Pydantic `args_schema` (everything else so far took a
# single `query: str`). The schema lets us offer two modes — search and fetch
# — with bounded `max_results`, which makes the tool-call deterministic from
# the LLM's point of view.
#
# Backends: NCBI E-utilities (Entrez). We use the `requests` library directly
# rather than `biopython.Entrez` so the tool has zero new dependencies. The
# three endpoints we touch are:
#
#   - esearch.fcgi   → PMIDs matching a free-text query
#   - esummary.fcgi  → structured metadata for each PMID
#   - efetch.fcgi    → abstracts or full records (XML; we extract a few tags)
#
# Rate limits (per NCBI):
#   - 3 requests/sec without an API key
#   - 10 requests/sec with an API key
# We pace ourselves with a small sleep between calls. If the user has set
# `NCBI_API_KEY`, we use the higher quota.
#
# Configuration (all optional, all from env vars):
#   - NCBI_API_KEY    — boosts rate limit
#   - NCBI_EMAIL      — required by NCBI etiquette (they contact you if your
#                        script misbehaves). Falls back to a placeholder if
#                        unset; NCBI won't reject the call, but it's polite
#                        to set this.

EUTILS_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUTILS_RATE_NO_KEY: float = 0.34   # seconds between calls (~3 RPS)
EUTILS_RATE_WITH_KEY: float = 0.11 # seconds between calls (~10 RPS)
EUTILS_TIMEOUT: int = 15           # seconds per HTTP request


class _EUtilsState:
    """Module-level throttling state for NCBI E-utilities calls.

    NCBI enforces 3 RPS (no key) or 10 RPS (with key). Rather than rely on
    Biopython's throttling, we maintain a single monotonic clock here so
    every call to a PubMed endpoint waits the minimum required interval.
    This is cheap, has no global lock, and works correctly when the tool
    is called multiple times in quick succession by an agent.
    """
    _last_call_ts: float = 0.0

    @classmethod
    def throttle(cls) -> None:
        interval = (
            EUTILS_RATE_WITH_KEY if os.environ.get("NCBI_API_KEY")
            else EUTILS_RATE_NO_KEY
        )
        elapsed = time.monotonic() - cls._last_call_ts
        if elapsed < interval:
            time.sleep(interval - elapsed)
        cls._last_call_ts = time.monotonic()


class PubMedSearchInput(BaseModel):
    """Inputs for PubMedSearchTool."""

    query: str = Field(
        ...,
        description=(
            "Free-text search query (e.g. 'BRCA1 breast cancer'), gene symbol "
            "(e.g. 'TP53'), disease term, or — in fetch mode — a numeric PMID."
        ),
    )
    mode: Literal["search", "fetch"] = Field(
        "search",
        description=(
            "'search' returns a list of matching papers (default). "
            "'fetch' retrieves the full abstract/record for a single PMID; "
            "pass the PMID as the query string."
        ),
    )
    max_results: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of papers to return in search mode (1–50).",
    )


def _eutils_params(extra: dict[str, str]) -> dict[str, str]:
    """Build the standard parameter dict for any E-utilities call.

    NCBI requires ``db``, ``email`` is strongly recommended, and ``api_key``
    lifts the rate limit when present. Tool name/version identifiers are
    courtesy rather than enforced.
    """
    params: dict[str, str] = {
        "db": "pubmed",
        "email": os.environ.get("NCBI_EMAIL", "[email protected]"),
        "tool": "research_agent_v4",
        "version": "1.0",
        **extra,
    }
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def _esearch(query: str, max_results: int) -> list[str]:
    """Return a list of PMIDs matching *query* via esearch.fcgi."""
    _EUtilsState.throttle()
    resp = requests.get(
        f"{EUTILS_BASE}/esearch.fcgi",
        params=_eutils_params({
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }),
        timeout=EUTILS_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    return pmids


def _esummary(pmids: list[str]) -> list[dict[str, Any]]:
    """Return structured metadata for each PMID via esummary.fcgi."""
    if not pmids:
        return []
    _EUtilsState.throttle()
    resp = requests.get(
        f"{EUTILS_BASE}/esummary.fcgi",
        params=_eutils_params({
            "id": ",".join(pmids),
            "retmode": "json",
        }),
        timeout=EUTILS_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json().get("result", {})
    # esummary returns a top-level "uids" list and one entry per UID.
    # Non-UID metadata keys (e.g. "uids") are skipped.
    records: list[dict[str, Any]] = []
    for pmid in pmids:
        rec = result.get(pmid)
        if not isinstance(rec, dict):
            continue
        records.append({
            "pmid": pmid,
            "title": rec.get("title", "").rstrip("."),
            "authors": [
                f"{a.get('name', '')}" for a in rec.get("authors", [])
            ],
            "journal": rec.get("source", ""),
            "year": str(rec.get("pubdate", "")).split(" ")[0],
            "volume": rec.get("volume", ""),
            "issue": rec.get("issue", ""),
            "pages": rec.get("pages", ""),
            "doi": next(
                (
                    artid
                    for artid in rec.get("articleids", [])
                    if artid.get("idtype") == "doi"
                ),
                "",
            ),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract": "",  # filled in by _efetch_abstract below
        })
    return records


def _efetch_abstract(pmid: str) -> str:
    """Fetch the abstract for a single PMID as plain text.

    We parse the efetch XML response for <AbstractText> tags. efetch
    returns full XML by default; we use a tiny XML walker to avoid adding
    lxml as a hard dependency (it is in requirements.txt but we don't
    assume it for tool code).
    """
    import xml.etree.ElementTree as ET
    _EUtilsState.throttle()
    resp = requests.get(
        f"{EUTILS_BASE}/efetch.fcgi",
        params=_eutils_params({
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        }),
        timeout=EUTILS_TIMEOUT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    parts: list[str] = []
    for abstract_text in root.iter("AbstractText"):
        # Section labels (Background/Methods/Results/Conclusions) appear as
        # attributes on structured abstracts. Preserve them when present.
        label = abstract_text.attrib.get("Label")
        text = "".join(abstract_text.itertext()).strip()
        if not text:
            continue
        if label and label.lower() != "abstract":
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


class PubMedSearchTool(BaseTool):
    """Search PubMed and fetch paper metadata + abstracts."""

    name: str = "pubmed_search"
    description: str = (
        "Search PubMed for peer-reviewed biomedical literature, or fetch "
        "the abstract/metadata for a specific PMID. In the default 'search' "
        "mode, returns a JSON list of papers (pmid, title, authors, journal, "
        "year, doi, url, abstract) ranked by NCBI relevance. In 'fetch' "
        "mode, pass a numeric PMID to retrieve its full structured record. "
        "Use this whenever a biomedical claim needs primary-source support."
    )
    args_schema: type[BaseModel] = PubMedSearchInput

    def _run(
        self,
        query: str,
        mode: Literal["search", "fetch"] = "search",
        max_results: int = 10,
    ) -> str:
        """Search PubMed or fetch a single record.

        Returns a JSON string. In *search* mode the result is a list of
        paper dicts (one per PMID); in *fetch* mode a single dict.
        Errors are caught and returned as ``{"error": "..."}`` so the
        agent can recover rather than crashing the whole crew.
        """
        # --- Defensive clamping -----------------------------------------
        # CrewAI passes schema-validated values, but a tool may also be
        # called from Python tests or future UI buttons. Keep the same
        # defensive posture as LocalKB.
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))
        mode_normalized = mode if mode in ("search", "fetch") else "search"

        try:
            if mode_normalized == "fetch":
                return self._fetch_one(query.strip())
            return self._search(query.strip(), n)
        except requests.RequestException as exc:
            logger.warning("PubMed network error: %s", exc)
            return json.dumps({"error": f"PubMed network error: {exc}"})
        except Exception as exc:
            logger.warning("PubMed tool failed: %s", exc)
            return json.dumps({"error": f"PubMed tool failed: {exc}"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str, n: int) -> str:
        """Search mode: query → list of paper records with abstracts."""
        pmids = _esearch(query, max_results=n)
        if not pmids:
            return json.dumps([])
        records = _esummary(pmids)
        # Fetch abstracts one at a time. efetch accepts comma-separated IDs
        # but emits a multi-record XML that's awkward to parse; one-by-one
        # keeps the parsing simple at the cost of an extra round trip per
        # record. At 10 records this stays well under the 3 RPS limit.
        for rec in records:
            try:
                rec["abstract"] = _efetch_abstract(rec["pmid"])
            except Exception as exc:
                # Non-fatal — keep the record with an empty abstract field
                # so the agent knows there was a partial failure.
                logger.warning("efetch failed for PMID %s: %s", rec["pmid"], exc)
                rec["abstract"] = ""
        return json.dumps(records, indent=2)

    def _fetch_one(self, query: str) -> str:
        """Fetch mode: a single PMID (or numeric prefix) → one record."""
        # Accept plain "12345678" or "PMID:12345678" or "pmid 12345678".
        digits = "".join(ch for ch in query if ch.isdigit())
        if not digits:
            return json.dumps({"error": "fetch mode requires a numeric PMID"})
        pmid = digits
        records = _esummary([pmid])
        if not records:
            return json.dumps({"error": f"No PubMed record found for PMID {pmid}"})
        try:
            records[0]["abstract"] = _efetch_abstract(pmid)
        except Exception as exc:
            logger.warning("efetch failed for PMID %s: %s", pmid, exc)
            records[0]["abstract"] = ""
        return json.dumps(records[0], indent=2)


# Module-level instance ready for `tools=[pubmed_search_tool]`.
pubmed_search_tool = PubMedSearchTool()


# ===========================================================================
# M3 — CitationResolverTool
# ===========================================================================
# Resolves a DOI, PMID, arXiv ID, or ISBN into a unified citation dict:
#
#   {
#     "type":   "journal-article" | "book" | "preprint" | ...,
#     "id":     "<as supplied>",
#     "id_type":"doi" | "pmid" | "arxiv" | "isbn",
#     "title":  "...",
#     "authors":["Last, F.", ...],          # always "Last, F." form
#     "venue":  "Journal Name",              # journal/booktitle/publisher
#     "year":   "2024",
#     "volume": "12",
#     "issue":  "3",
#     "pages":  "45-67",
#     "url":    "https://...",
#     "doi":    "10....",                    # only for DOI-originated records
#     "extra":  {...},                       # source-specific leftovers
#   }
#
# The whole point of this tool is to give the Writer agent one canonical
# shape to emit, no matter where the citation came from. It pairs naturally
# with `PubMedSearchTool` (H1) — when the search tool returns PMIDs, the
# writer can pass each one through this tool to get a fully-formatted
# citation without hand-rolling the field names.
#
# Backends:
#   - DOI    → Crossref REST API (api.crossref.org)
#              No key; polite-pool via `mailto=` query param.
#   - PMID   → NCBI E-utilities (reuses `_EUtilsState` throttling)
#   - arXiv  → arXiv API (export.arxiv.org/api/query) — Atom XML
#   - ISBN   → Google Books API (no key needed for basic metadata)

CROSSREF_BASE: str = "https://api.crossref.org/works"
ARXIV_BASE: str = "http://export.arxiv.org/api/query"
GOOGLE_BOOKS_BASE: str = "https://www.googleapis.com/books/v1/volumes"
CITATION_HTTP_TIMEOUT: int = 15


def _normalize_doi(raw: str) -> str:
    """Strip URL prefix, whitespace, and lower-case a DOI for lookup."""
    s = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
    return s.strip().lower()


def _normalize_pmid(raw: str) -> str:
    """Extract the leading run of digits from any PMID-shaped string."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits


def _normalize_arxiv(raw: str) -> str:
    """Normalise an arXiv ID to its canonical form.

    Accepts ``1234.5678``, ``arXiv:1234.5678``, ``https://arxiv.org/abs/1234.5678``,
    and the older ``cs.AI/0102034`` form (returned unchanged after stripping
    any ``arXiv:`` / URL prefix).
    """
    s = raw.strip()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "arxiv:",
        "ARXIV:",
    ):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
    return s.strip()


def _normalize_isbn(raw: str) -> str:
    """Strip whitespace/hyphens from an ISBN."""
    return "".join(ch for ch in raw if ch.isdigit() or ch == "X" or ch == "x")


def _format_authors_crossref(authors: list[dict[str, str]]) -> list[str]:
    """Crossref → "Last, F." form."""
    out: list[str] = []
    for a in authors:
        family = a.get("family", "").strip()
        given = a.get("given", "").strip()
        if family and given:
            initials = "".join(part[0].upper() + "." for part in given.split() if part)
            out.append(f"{family}, {initials}")
        elif family:
            out.append(family)
        elif given:
            out.append(given)
    return out


def _resolve_doi(doi: str) -> dict[str, Any]:
    """Resolve a DOI via the Crossref REST API."""
    _EUtilsState.throttle()  # share throttling clock; harmless for non-NCBI
    # `mailto` opts into Crossref's polite pool — no key needed, higher
    # rate limit than the default pool.
    polite = os.environ.get("CROSSREF_MAILTO", "[email protected]")
    resp = requests.get(
        f"{CROSSREF_BASE}/{doi}",
        params={"mailto": polite},
        timeout=CITATION_HTTP_TIMEOUT,
        headers={"User-Agent": f"research_agent_v4 (+mailto:{polite})"},
    )
    resp.raise_for_status()
    msg = resp.json().get("message", {})
    return {
        "type": msg.get("type", "unknown"),
        "id": doi,
        "id_type": "doi",
        "title": (msg.get("title") or [""])[0],
        "authors": _format_authors_crossref(msg.get("author", [])),
        "venue": (msg.get("container-title") or [""])[0],
        "year": str(msg.get("issued", {}).get("date-parts", [[""]])[0][0] or ""),
        "volume": msg.get("volume", ""),
        "issue": msg.get("issue", ""),
        "pages": msg.get("page", ""),
        "url": msg.get("URL", f"https://doi.org/{doi}"),
        "doi": doi,
        "extra": {
            "publisher": msg.get("publisher", ""),
            "issn": msg.get("ISSN", []),
        },
    }


def _resolve_pmid(pmid: str) -> dict[str, Any]:
    """Resolve a PMID by reusing the PubMed helpers above.

    Wraps the same ``_esummary`` path that ``PubMedSearchTool`` uses, then
    re-projects the record into the unified citation shape. Reuses the
    ``_EUtilsState`` throttling so we stay under NCBI's rate limit even
    when this tool is called alongside ``pubmed_search_tool``.
    """
    records = _esummary([pmid])
    if not records:
        raise ValueError(f"No PubMed record for PMID {pmid}")
    rec = records[0]
    authors_last_first = []
    for a in rec.get("authors", []):
        # PubMed returns "Last F M"; normalise to "Last, F. M."
        parts = a.strip().split()
        if len(parts) >= 2 and "," not in a:
            last = parts[0]
            rest = " ".join(parts[1:])
            initials = "".join(p[0].upper() + "." for p in rest.split() if p)
            authors_last_first.append(f"{last}, {initials}")
        else:
            authors_last_first.append(a)
    return {
        "type": "journal-article",
        "id": pmid,
        "id_type": "pmid",
        "title": rec.get("title", ""),
        "authors": authors_last_first,
        "venue": rec.get("journal", ""),
        "year": rec.get("year", ""),
        "volume": rec.get("volume", ""),
        "issue": rec.get("issue", ""),
        "pages": rec.get("pages", ""),
        "url": rec.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
        "doi": rec.get("doi", ""),
        "extra": {},
    }


def _resolve_arxiv(arxiv_id: str) -> dict[str, Any]:
    """Resolve an arXiv ID via the arXiv API (Atom XML, parsed with stdlib)."""
    import xml.etree.ElementTree as ET
    _EUtilsState.throttle()
    resp = requests.get(
        ARXIV_BASE,
        params={"id_list": arxiv_id, "max_results": "1"},
        timeout=CITATION_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    # arXiv Atom uses the default namespace; tag lookups need to match it.
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entry = root.find("a:entry", ns)
    if entry is None:
        raise ValueError(f"arXiv returned no entry for {arxiv_id}")
    title = "".join(entry.findtext("a:title", default="", namespaces=ns)).strip()
    summary = "".join(entry.findtext("a:summary", default="", namespaces=ns)).strip()
    published = entry.findtext("a:published", default="", namespaces=ns)
    year = published.split("-")[0] if published else ""
    authors = [
        "".join(a.findtext("a:name", default="", namespaces=ns)).strip()
        for a in entry.findall("a:author", ns)
    ]
    # Author names from arXiv are "First Middle Last" — flip to "Last, F.".
    flipped: list[str] = []
    for a in authors:
        parts = a.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = "".join(p[0].upper() + "." for p in parts[:-1] if p)
            flipped.append(f"{last}, {initials}")
        else:
            flipped.append(a)
    url = entry.findtext("a:id", default="", namespaces=ns)
    return {
        "type": "preprint",
        "id": arxiv_id,
        "id_type": "arxiv",
        "title": title,
        "authors": flipped,
        "venue": "arXiv",
        "year": year,
        "volume": "",
        "issue": "",
        "pages": "",
        "url": url,
        "doi": "",
        "extra": {"summary": summary[:500]},
    }


def _resolve_isbn(isbn: str) -> dict[str, Any]:
    """Resolve an ISBN via the Google Books API (no key needed)."""
    _EUtilsState.throttle()
    resp = requests.get(
        GOOGLE_BOOKS_BASE,
        params={"q": f"isbn:{isbn}", "maxResults": "1"},
        timeout=CITATION_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        raise ValueError(f"No Google Books record for ISBN {isbn}")
    info = items[0].get("volumeInfo", {})
    authors_raw = info.get("authors", [])
    authors_last_first: list[str] = []
    for a in authors_raw:
        parts = a.strip().split()
        if len(parts) >= 2 and "," not in a:
            last = parts[-1]
            initials = "".join(p[0].upper() + "." for p in parts[:-1] if p)
            authors_last_first.append(f"{last}, {initials}")
        else:
            authors_last_first.append(a)
    year = str(info.get("publishedDate", "")).split("-")[0]
    return {
        "type": "book",
        "id": isbn,
        "id_type": "isbn",
        "title": info.get("title", ""),
        "authors": authors_last_first,
        "venue": info.get("publisher", ""),
        "year": year,
        "volume": "",
        "issue": "",
        "pages": str(info.get("pageCount", "")),
        "url": info.get("infoLink", ""),
        "doi": "",
        "extra": {"subtitle": info.get("subtitle", "")},
    }


def _auto_detect_id_type(raw: str) -> str:
    """Heuristically guess the identifier type from a free-form string."""
    s = raw.strip()
    lower = s.lower()
    if lower.startswith(("10.", "https://doi.org/", "doi:")):
        return "doi"
    if lower.startswith(("arxiv:", "https://arxiv.org/", "http://arxiv.org/")):
        return "arxiv"
    # Pure-digit short strings are almost always PMIDs.
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits and digits == s.replace(" ", "").replace("PMID", "").replace("pmid", ""):
        return "pmid"
    # ISBNs are 10 or 13 chars after normalisation; Google Books is forgiving
    # so we send anything that looks like a book ID there.
    cleaned = "".join(ch for ch in s if ch.isdigit() or ch in "Xx")
    if len(cleaned) in (10, 13):
        return "isbn"
    # Fall back to DOI (Crossref can resolve a lot of things as text search).
    return "doi"


class CitationResolverInput(BaseModel):
    """Inputs for CitationResolverTool."""

    identifier: str = Field(
        ...,
        description=(
            "DOI (e.g. '10.1038/nature12373'), PMID (e.g. '23842501'), "
            "arXiv ID (e.g. '2301.00001'), or ISBN (10 or 13 digits). "
            "URLs and 'DOI:' / 'arXiv:' / 'PMID:' prefixes are accepted."
        ),
    )
    id_type: Literal["auto", "doi", "pmid", "arxiv", "isbn"] = Field(
        "auto",
        description=(
            "Identifier type. 'auto' (default) detects from the string. "
            "Specify explicitly when auto-detection is ambiguous."
        ),
    )


class CitationResolverTool(BaseTool):
    """Resolve a DOI / PMID / arXiv ID / ISBN into a structured citation."""

    name: str = "resolve_citation"
    description: str = (
        "Resolve a DOI, PMID, arXiv ID, or ISBN into a structured citation "
        "record. Returns JSON with a unified shape — title, authors, "
        "venue/journal, year, volume, issue, pages, URL, DOI — so the "
        "Writer can emit fully formatted citations without ad-hoc field "
        "names. Use this whenever a source needs to appear in the "
        "bibliography section."
    )
    args_schema: type[BaseModel] = CitationResolverInput

    def _run(
        self,
        identifier: str,
        id_type: Literal["auto", "doi", "pmid", "arxiv", "isbn"] = "auto",
    ) -> str:
        """Resolve *identifier* into a unified citation dict.

        Errors are caught and returned as ``{"error": "..."}`` so the
        agent can recover rather than crashing the whole crew.
        """
        if not identifier or not identifier.strip():
            return json.dumps({"error": "empty identifier"})

        kind = id_type if id_type != "auto" else _auto_detect_id_type(identifier)
        try:
            if kind == "doi":
                citation = _resolve_doi(_normalize_doi(identifier))
            elif kind == "pmid":
                pmid = _normalize_pmid(identifier)
                if not pmid:
                    return json.dumps({"error": "no digits in PMID"})
                citation = _resolve_pmid(pmid)
            elif kind == "arxiv":
                citation = _resolve_arxiv(_normalize_arxiv(identifier))
            elif kind == "isbn":
                citation = _resolve_isbn(_normalize_isbn(identifier))
            else:
                return json.dumps({"error": f"unknown id_type: {kind}"})
        except requests.RequestException as exc:
            logger.warning("CitationResolver network error: %s", exc)
            return json.dumps({"error": f"network error: {exc}"})
        except Exception as exc:
            logger.warning("CitationResolver failed for %r: %s", identifier, exc)
            return json.dumps({"error": f"resolution failed: {exc}"})

        return json.dumps(citation, indent=2)


# Module-level instance ready for `tools=[citation_resolver_tool]`.
citation_resolver_tool = CitationResolverTool()
