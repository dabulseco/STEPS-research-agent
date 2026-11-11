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
import re
import time
from typing import Any, Literal, Optional

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


# ===========================================================================
# M5 — OpenAlexTool
# ===========================================================================
# OpenAlex (api.openalex.org) provides free, no-API-key access to 250M+
# scholarly works across all disciplines. It's a powerful complement to
# PubMed/Google Scholar for cross-disciplinary research.
#
# Endpoints we use:
#   - /works        → search works with filters (title, author, concept, year, etc.)
#   - /works/{id}   → fetch a single work by OpenAlex ID or DOI
#   - /authors      → author search
#   - /concepts     → concept/topic search
#
# Rate limit: 100k requests/day (generous), polite pool via `mailto=` param.

OPENALEX_BASE: str = "https://api.openalex.org"
OPENALEX_TIMEOUT: int = 20
OPENALEX_MAILTO: str = os.environ.get("OPENALEX_MAILTO", "[email protected]")


class OpenAlexInput(BaseModel):
    """Inputs for OpenAlexTool."""

    query: str = Field(
        ...,
        description=(
            "Free-text search query (e.g. 'transformer interpretability', "
            "'CRISPR gene editing', 'quantum error correction')."
        ),
    )
    filter: Optional[str] = Field(
            default=None,
            description=(
                "OpenAlex filter string (e.g. 'from_publication_date:2023-01-01,'"
                "cited_by_count:>10,type:journal-article'). See OpenAlex docs."
            ),
        )
    max_results: int = Field(
        10, ge=1, le=50, description="Maximum number of works to return (1–50)."
    )
    sample: int = Field(
        default=0,
        ge=0,
        le=10000,
        description="Random sample of results (OpenAlex 'sample' parameter).",
    )


def _openalex_params(extra: dict[str, str]) -> dict[str, str]:
    """Build standard params for OpenAlex calls."""
    return {
        "mailto": OPENALEX_MAILTO,
        **extra,
    }


def _format_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAlex work dict to our unified citation shape."""
    # Authors
    authors = []
    for auth in work.get("authorships", []):
        author = auth.get("author", {})
        name = author.get("display_name", "")
        if name:
            # OpenAlex gives "First Last" — convert to "Last, F."
            parts = name.strip().split()
            if len(parts) >= 2:
                last = parts[-1]
                initials = "".join(p[0].upper() + "." for p in parts[:-1] if p)
                authors.append(f"{last}, {initials}")
            else:
                authors.append(name)

    # Venue
    venue = ""
    host = work.get("host_venue", {})
    if host:
        venue = host.get("display_name", "")
    elif work.get("primary_location", {}).get("source", {}).get("display_name"):
        venue = work["primary_location"]["source"]["display_name"]

    # Year
    year = str(work.get("publication_year", ""))

    # DOI
    doi = work.get("doi", "").replace("https://doi.org/", "")

    # OpenAlex ID
    oa_id = work.get("id", "").replace("https://openalex.org/", "")

    return {
        "type": work.get("type", "unknown"),
        "id": oa_id,
        "id_type": "openalex",
        "title": work.get("display_name", ""),
        "authors": authors,
        "venue": venue,
        "year": year,
        "volume": work.get("biblio", {}).get("volume", ""),
        "issue": work.get("biblio", {}).get("issue", ""),
        "pages": work.get("biblio", {}).get("first_page", "")
        + ("-" + work.get("biblio", {}).get("last_page", "") if work.get("biblio", {}).get("last_page") else ""),
        "url": work.get("id", ""),
        "doi": doi,
        "cited_by_count": work.get("cited_by_count", 0),
        "is_oa": work.get("open_access", {}).get("is_oa", False),
        "oa_url": work.get("open_access", {}).get("oa_url", ""),
        "concepts": [c.get("display_name", "") for c in work.get("concepts", [])[:5]],
        "extra": {
            "openalex_id": oa_id,
            "mag": work.get("ids", {}).get("mag", ""),
            "pmid": work.get("ids", {}).get("pmid", ""),
            "pmcid": work.get("ids", {}).get("pmcid", ""),
        },
    }


class OpenAlexTool(BaseTool):
    """Search OpenAlex for open scholarly metadata across all disciplines."""

    name: str = "openalex"
    description: str = (
        "Search OpenAlex (api.openalex.org) for scholarly works across all disciplines. "
        "Returns title, authors, venue, year, DOI, OpenAlex ID, citation count, "
        "open-access status/URL, and top concepts. No API key required; "
        "rate limit ~100k/day. Use for cross-disciplinary literature discovery, "
        "finding open-access PDFs, and citation metrics."
    )
    args_schema: type[BaseModel] = OpenAlexInput

    def _run(
        self,
        query: str,
        filter: Optional[str] = None,
        max_results: int = 10,
        sample: int = 0,
    ) -> str:
        """Search OpenAlex and return structured work records."""
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))

        try:
            sample_n = int(sample)
        except (TypeError, ValueError):
            sample_n = 0
        sample_n = max(0, min(10000, sample_n))

        params = _openalex_params({
            "search": query,
            "per_page": str(n),
        })
        if filter:
            params["filter"] = filter
        if sample_n:
            params["sample"] = str(sample_n)

        try:
            resp = requests.get(
                f"{OPENALEX_BASE}/works",
                params=params,
                timeout=OPENALEX_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            formatted = [_format_openalex_work(w) for w in results]
            return json.dumps(formatted, indent=2)
        except requests.RequestException as exc:
            logger.warning("OpenAlex network error: %s", exc)
            return json.dumps({"error": f"OpenAlex network error: {exc}"})
        except Exception as exc:
            logger.warning("OpenAlex tool failed: %s", exc)
            return json.dumps({"error": f"OpenAlex tool failed: {exc}"})


openalex_tool = OpenAlexTool()


# ===========================================================================
# M4 — UrlFetchTool
# ===========================================================================
# Promote the existing `scrape_url` helper from app.py to a proper agent-callable tool.
# This lets any agent fetch the full text of a known URL (not just search snippets).

class UrlFetchInput(BaseModel):
    """Inputs for UrlFetchTool."""

    url: str = Field(..., description="URL to fetch and extract full text from.")


class UrlFetchTool(BaseTool):
    """Fetch the full text of a web page given a URL."""

    name: str = "fetch_url"
    description: str = (
        "Fetch the full text of a web page given a URL. Strips navigation, scripts, "
        "and boilerplate. Returns the cleaned text plus the page title. Use when a "
        "known URL needs to be read in full rather than only summarized from search snippets."
    )
    args_schema: type[BaseModel] = UrlFetchInput

    def _run(self, url: str) -> str:
        """Fetch and extract text from a URL."""
        if not url or not url.strip():
            return json.dumps({"error": "empty url"})

        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
            resp = requests.get(url.strip(), headers=headers, timeout=20)
            resp.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            full_text = soup.get_text(separator="\n", strip=True)

            # Limit to ~15k chars for context window
            if len(full_text) > 15000:
                full_text = full_text[:15000] + "\n[... content truncated ...]"

            result = {
                "url": url,
                "title": title or url,
                "content": full_text,
                "word_count": len(full_text.split()),
            }
            return json.dumps(result, indent=2)

        except requests.RequestException as exc:
            logger.warning("UrlFetch network error: %s", exc)
            return json.dumps({"error": f"UrlFetch network error: {exc}"})
        except Exception as exc:
            logger.warning("UrlFetch failed for %s: %s", url, exc)
            return json.dumps({"error": f"UrlFetch failed: {exc}"})


url_fetch_tool = UrlFetchTool()


# ===========================================================================
# L8 — ReferenceManagerTool
# ===========================================================================
# Export a list of citations as BibTeX (.bib) or RIS (.ris) for use in
# reference managers (Zotero, Mendeley, EndNote, LaTeX). Reuses CitationResolverTool
# output format.

class ReferenceManagerInput(BaseModel):
    """Inputs for ReferenceManagerTool."""

    identifiers: list[str] = Field(
        ...,
        description=(
            "List of DOIs, PMIDs, arXiv IDs, ISBNs, or OpenAlex IDs to export. "
            "Each will be resolved via CitationResolverTool / OpenAlexTool."
        ),
    )
    format: Literal["bibtex", "ris"] = Field(
        "bibtex",
        description="Output format: 'bibtex' for LaTeX/BibTeX, 'ris' for RIS (Zotero, Mendeley, EndNote).",
    )
    resolve: bool = Field(
        True,
        description="If true, resolve each identifier to full citation first. If false, assume identifiers are already full citation dicts (JSON strings).",
    )


def _citation_to_bibtex(citation: dict[str, Any]) -> str:
    """Convert a unified citation dict to a BibTeX entry."""
    cite_type = citation.get("type", "article")
    cite_id = citation.get("id", "").replace("https://openalex.org/", "").replace("https://doi.org/", "")
    # Sanitize for BibTeX key
    bib_key = re.sub(r"[^a-zA-Z0-9_-]", "_", cite_id)[:50]

    # Map type to BibTeX entry type
    type_map = {
        "journal-article": "article",
        "journal": "article",
        "book": "book",
        "book-chapter": "incollection",
        "proceedings-article": "inproceedings",
        "preprint": "article",
        "dissertation": "phdthesis",
        "report": "techreport",
        "dataset": "misc",
    }
    bib_type = type_map.get(cite_type, "misc")

    lines = [f"@{bib_type}{{{bib_key},"]

    fields = [
        ("title", citation.get("title", "")),
        ("author", " and ".join(citation.get("authors", []))),
        ("journal", citation.get("venue", "")),
        ("year", citation.get("year", "")),
        ("volume", citation.get("volume", "")),
        ("number", citation.get("issue", "")),
        ("pages", citation.get("pages", "")),
        ("doi", citation.get("doi", "")),
        ("url", citation.get("url", "")),
    ]

    for field, value in fields:
        if value:
            lines.append(f"  {field} = {{{value}}},")

    lines.append("}")
    return "\n".join(lines)


def _citation_to_ris(citation: dict[str, Any]) -> str:
    """Convert a unified citation dict to a RIS entry."""
    cite_type = citation.get("type", "article")
    # RIS type mapping
    ris_type_map = {
        "journal-article": "JOUR",
        "journal": "JOUR",
        "book": "BOOK",
        "book-chapter": "CHAP",
        "proceedings-article": "CONF",
        "preprint": "JOUR",
        "dissertation": "THES",
        "report": "RPRT",
        "dataset": "DATA",
    }
    ris_type = ris_type_map.get(cite_type, "GEN")

    lines = [f"TY  - {ris_type}"]

    if citation.get("title"):
        lines.append(f"TI  - {citation['title']}")
    for author in citation.get("authors", []):
        lines.append(f"AU  - {author}")
    if citation.get("venue"):
        lines.append(f"JO  - {citation['venue']}")
    if citation.get("year"):
        lines.append(f"PY  - {citation['year']}")
    if citation.get("volume"):
        lines.append(f"VL  - {citation['volume']}")
    if citation.get("issue"):
        lines.append(f"IS  - {citation['issue']}")
    if citation.get("pages"):
        lines.append(f"SP  - {citation['pages'].split('-')[0]}")
        if "-" in citation.get("pages", ""):
            lines.append(f"EP  - {citation['pages'].split('-')[1]}")
    if citation.get("doi"):
        lines.append(f"DO  - {citation['doi']}")
    if citation.get("url"):
        lines.append(f"UR  - {citation['url']}")
    if citation.get("abstract"):
        lines.append(f"AB  - {citation['abstract'][:500]}")

    lines.append("ER  -")
    return "\n".join(lines)


class ReferenceManagerTool(BaseTool):
    """Export citations as BibTeX (.bib) or RIS (.ris)."""

    name: str = "export_references"
    description: str = (
        "Export a list of citations as a BibTeX (.bib) or RIS (.ris) file for use in "
        "reference managers (Zotero, Mendeley, EndNote, LaTeX). Provide a list of "
        "DOIs, PMIDs, arXiv IDs, ISBNs, or OpenAlex IDs. The tool resolves each to a "
        "full citation and formats the output. Use this when the bibliography section "
        "needs to be generated for the final document."
    )
    args_schema: type[BaseModel] = ReferenceManagerInput

    def _run(
        self,
        identifiers: list[str],
        format: Literal["bibtex", "ris"] = "bibtex",
        resolve: bool = True,
    ) -> str:
        """Resolve identifiers and export as BibTeX or RIS."""
        if not identifiers:
            return json.dumps({"error": "empty identifiers list"})

        # Import resolvers
        from tools import citation_resolver_tool, openalex_tool

        entries = []
        for ident in identifiers:
            ident = ident.strip()
            if not ident:
                continue

            if resolve:
                # Try to detect and resolve
                citation_json = ""
                kind = _auto_detect_id_type(ident)
                if kind == "openalex" or ident.startswith("https://openalex.org/") or ident.startswith("W"):
                    # Use OpenAlex tool
                    citation_json = openalex_tool._run(ident, max_results=1)
                else:
                    citation_json = citation_resolver_tool._run(ident, id_type="auto")

                try:
                    citation = json.loads(citation_json)
                    if "error" not in citation:
                        entries.append(citation)
                    else:
                        logger.warning("Failed to resolve %s: %s", ident, citation.get("error"))
                        # Include as minimal entry
                        entries.append({"id": ident, "title": ident, "error": citation.get("error")})
                except json.JSONDecodeError:
                    logger.warning("Could not parse citation for %s", ident)
                    entries.append({"id": ident, "title": ident})
            else:
                # Assume ident is already a JSON citation dict
                try:
                    citation = json.loads(ident)
                    entries.append(citation)
                except json.JSONDecodeError:
                    entries.append({"id": ident, "title": ident})

        if not entries:
            return json.dumps({"error": "no valid citations to export"})

        if format == "bibtex":
            output = "\n\n".join(_citation_to_bibtex(e) for e in entries)
        else:
            output = "\n\n".join(_citation_to_ris(e) for e in entries)

        return output


reference_manager_tool = ReferenceManagerTool()


# ===========================================================================
# H2 — GoogleScholarTool
# ===========================================================================
# Searches Google Scholar for academic papers — for non-biomedical topics,
# preprints, theses, conference proceedings, and any cross-disciplinary
# literature that PubMed does not cover.
#
# Two backends are supported, chosen at call time via the
# `GOOGLE_SCHOLAR_BACKEND` env var (or the `backend` arg of `_run`):
#
#   - "selenium" (default): free, no key, but Scholar will rate-limit and
#       captcha-block after a small number of requests.  Reuses the same
#       headless Chrome / Firefox infrastructure the existing
#       `WebSearchTool` uses, so it gets the same platform workarounds
#       (macOS ARM, etc.) for free.
#
#   - "serpapi" (optional): paid but reliable.  Reads `SERPAPI_API_KEY`
#       from env.  Falls back to a JSON error if the key is missing, so
#       users without an account are never silently mis-routed to the
#       Selenium path.
#
# Output shape matches the other tools' conventions — a JSON list of
# result dicts (title, link, snippet, year, venue, cited_by, authors,
# source), or an `{"error": "..."}` dict on failure.

SCHOLAR_HTTP_TIMEOUT: int = 20


class GoogleScholarInput(BaseModel):
    """Inputs for GoogleScholarTool."""

    query: str = Field(
        ...,
        description=(
            "Free-text search query (e.g. 'transformer interpretability', "
            "'CRISPR gene editing', 'quantum error correction')."
        ),
    )
    max_results: int = Field(
        10, ge=1, le=20, description="Maximum number of papers to return (1–20)."
    )
    backend: Optional[Literal["selenium", "serpapi"]] = Field(
        default=None,
        description=(
            "Override the backend for this call. 'selenium' is the free "
            "default but is captcha-prone. 'serpapi' is paid and reliable; "
            "requires SERPAPI_API_KEY. If omitted, falls back to the "
            "GOOGLE_SCHOLAR_BACKEND env var, then 'selenium'."
        ),
    )


def _resolve_scholar_backend(override: Optional[str]) -> str:
    """Pick the backend: arg override → env var → 'selenium'."""
    if override in ("selenium", "serpapi"):
        return override
    env = os.environ.get("GOOGLE_SCHOLAR_BACKEND", "selenium").lower()
    return env if env in ("selenium", "serpapi") else "selenium"


def _scholar_serpapi(query: str, max_results: int, api_key: str) -> str:
    """Search via SerpAPI's Google Scholar endpoint.

    SerpAPI is a paid service that proxies Google Scholar and returns
    clean JSON.  Free tier gives 100 searches/month.  The endpoint URL
    is documented at https://serpapi.com/google-scholar-api.
    """
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_scholar",
            "q": query,
            "num": max_results,
            "api_key": api_key,
        },
        timeout=SCHOLAR_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results: list[dict[str, Any]] = []
    for item in data.get("organic_results", [])[:max_results]:
        pub = item.get("publication_info", {}) or {}
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "year": (pub.get("summary") or "").split(" - ")[0].strip() if pub.get("summary") else "",
            "venue": (pub.get("summary") or "").split(" - ")[-1].strip() if pub.get("summary") else "",
            "authors": [
                a.get("name", "") for a in pub.get("authors", []) if a.get("name")
            ],
            "cited_by": item.get("cited_by", {}).get("value", 0)
            if isinstance(item.get("cited_by"), dict) else 0,
            "source": "serpapi",
        })
    return json.dumps(results, indent=2)


def _scholar_selenium(query: str, max_results: int) -> str:
    """Search via Selenium headless browser.

    Reuses the SeleniumSearcher class that the app's WebSearchTool uses.
    Returns a list of paper dicts with title / link / snippet / year /
    venue / cited_by, parsed from the Scholar results page.  Scholar
    aggressively detects bots, so this is best-effort and may captcha-block
    on the first hit — the tool returns a JSON error in that case so the
    agent can fall back to `openalex` or `web_search`.
    """
    # Lazy import so tools.py is importable in environments without
    # Streamlit / Selenium already loaded.
    try:
        from app import SeleniumSearcher  # type: ignore
    except Exception as exc:
        return json.dumps({
            "error": f"selenium backend unavailable: {exc}; "
                     "try backend='serpapi' or use openalex / pubmed_search instead."
        })

    searcher = SeleniumSearcher(browser_type="chrome")
    try:
        # Scholar's HTML structure: each result is a <div class="gs_ri"> with
        # an h3.gs_rt (title + link), div.gs_a (authors/venue/year), and
        # div.gs_rs (snippet).  We use a slightly forgiving CSS to handle
        # layout drift.
        if not searcher._init_driver():
            return json.dumps({"error": "failed to initialise Chrome for Scholar"})

        url = f"https://scholar.google.com/scholar?q={requests.utils.quote(query)}&hl=en"
        searcher.driver.get(url)
        time.sleep(2)  # let the page settle; Scholar's anti-bot checks the
                       # first second of activity
        html = searcher.driver.page_source
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        results: list[dict[str, Any]] = []
        for div in soup.find_all("div", class_="gs_ri")[:max_results]:
            try:
                h3 = div.find("h3", class_="gs_rt")
                if not h3:
                    continue
                a = h3.find("a")
                title = h3.get_text(strip=True)
                link = a.get("href", "") if a else ""
                meta = div.find("div", class_="gs_a")
                meta_text = meta.get_text(" ", strip=True) if meta else ""
                # Meta line is typically "Authors - Venue, Year - Publisher".
                parts = [p.strip() for p in meta_text.split(" - ")]
                authors = parts[0] if parts else ""
                venue = parts[1] if len(parts) > 1 else ""
                year = parts[2] if len(parts) > 2 else ""
                snippet_el = div.find("div", class_="gs_rs")
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

                # Cited-by count is in a sibling footer <div class="gs_fl">;
                # it's not always present, so we tolerate its absence.
                cited_by = 0
                fl = div.find("div", class_="gs_fl")
                if fl:
                    cited_link = fl.find("a", string=_re.compile(r"Cited by \d+"))
                    if cited_link:
                        m = _re.search(r"(\d+)", cited_link.get_text())
                        if m:
                            cited_by = int(m.group(1))

                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "authors": authors,
                    "venue": venue,
                    "year": year,
                    "cited_by": cited_by,
                    "source": "scholar_selenium",
                })
            except Exception:
                continue

        if not results:
            return json.dumps({
                "error": (
                    "no Scholar results extracted (likely a captcha or "
                    "anti-bot block). Try backend='serpapi' or use "
                    "openalex / pubmed_search instead."
                )
            })
        return json.dumps(results, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Scholar (selenium) failed: {exc}"})
    finally:
        try:
            searcher.close()
        except Exception:
            pass


class GoogleScholarTool(BaseTool):
    """Search Google Scholar for academic papers."""

    name: str = "google_scholar"
    description: str = (
        "Search Google Scholar for academic papers, theses, preprints, and "
        "conference proceedings. Returns a JSON list of paper records (title, "
        "link, snippet, authors, venue, year, cited_by). Use for "
        "non-biomedical or cross-disciplinary topics where PubMed doesn't "
        "apply, and as a complement to openalex. The default backend is "
        "Selenium (free, captcha-prone); the paid SerpAPI backend is more "
        "reliable and is enabled by setting SERPAPI_API_KEY and "
        "GOOGLE_SCHOLAR_BACKEND=serpapi in the environment."
    )
    args_schema: type[BaseModel] = GoogleScholarInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        backend: Optional[Literal["selenium", "serpapi"]] = None,
    ) -> str:
        """Search Google Scholar and return structured paper records."""
        # Defensive clamping — same posture as the other tools.
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(20, n))
        if not query or not query.strip():
            return json.dumps({"error": "empty query"})

        chosen = _resolve_scholar_backend(backend)
        if chosen == "serpapi":
            api_key = os.environ.get("SERPAPI_API_KEY")
            if not api_key:
                return json.dumps({
                    "error": (
                        "backend='serpapi' requires SERPAPI_API_KEY. "
                        "Either set the env var or omit `backend` to use "
                        "the free Selenium path (with captcha risk)."
                    )
                })
            try:
                return _scholar_serpapi(query.strip(), n, api_key)
            except requests.RequestException as exc:
                return json.dumps({"error": f"SerpAPI network error: {exc}"})
            except Exception as exc:
                return json.dumps({"error": f"SerpAPI failed: {exc}"})

        # Default: Selenium.
        return _scholar_selenium(query.strip(), n)


google_scholar_tool = GoogleScholarTool()


# ===========================================================================
# M1 — SequenceAlignmentTool
# ===========================================================================
# Wraps NCBI BLAST web service (blast.ncbi.nlm.nih.gov) for sequence
# similarity search.  No API key required; NCBI throttles anonymous
# usage at ~3 RPS, which `_EUtilsState.throttle()` already enforces.
#
# Five modes correspond to the five BLAST programs:
#   - blastn    : nucleotide vs nucleotide
#   - blastp    : protein vs protein
#   - blastx    : nucleotide (translated) vs protein
#   - tblastn   : protein vs nucleotide (translated)
#   - tblastx   : nucleotide (translated) vs nucleotide (translated)
#
# A sixth mode, "msa", is a placeholder for future Biopython-MUSCLE
# integration. For now it returns an explicit error so the agent
# doesn't silently no-op.

BLAST_BASE: str = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
BLAST_TIMEOUT: int = 30


class SequenceAlignmentInput(BaseModel):
    """Inputs for SequenceAlignmentTool."""

    sequence: str = Field(
        ...,
        description=(
            "The query sequence as a FASTA string or a bare sequence. "
            "Nucleotides for blastn/blastx/tblastx, amino acids for "
            "blastp/tblastn. 20+ characters recommended."
        ),
    )
    mode: Literal["blastn", "blastp", "blastx", "tblastn", "tblastx", "msa"] = Field(
        "blastp",
        description=(
            "BLAST program. 'blastn' is nucleotide; 'blastp' is protein. "
            "'msa' is reserved for future multiple-sequence alignment and "
            "currently returns an error."
        ),
    )
    database: str = Field(
        "nr",
        description=(
            "BLAST database short name (e.g. 'nr', 'nt', 'refseq_protein', "
            "'refseq_rna', 'pdbnt', 'pdbaa', 'swissprot'). See "
            "https://ncbi.github.io/blast-cloud/ for the full list."
        ),
    )
    max_hits: int = Field(
        10, ge=1, le=50, description="Maximum number of hits to return (1–50)."
    )


def _submit_blast(sequence: str, program: str, database: str) -> str:
    """Submit a BLAST job and return the Request ID (RID)."""
    _EUtilsState.throttle()
    resp = requests.post(
        BLAST_BASE,
        data={
            "CMD": "Put",
            "PROGRAM": program,
            "DATABASE": database,
            "QUERY": sequence,
            "MEGABLAST": "on" if program == "blastn" else "",
            "FORMAT_TYPE": "XML",
            "NCBI_GI": "",
            "TOOL": "research_agent_v4",
        },
        timeout=BLAST_TIMEOUT,
    )
    resp.raise_for_status()
    # NCBI returns the RID and estimated time-to-ready in the HTML body.
    text = resp.text
    rid = _re.search(r"RID = (\S+)", text)
    if not rid:
        raise ValueError("BLAST submit did not return a Request ID")
    return rid.group(1)


def _poll_blast(rid: str, max_wait_s: int = 60) -> str:
    """Poll BLAST until the job is ready; return the XML body.

    Polls every 3 seconds (NCBI's recommended interval) up to max_wait_s.
    A real run of BLAST against nr typically completes in 15–45 seconds
    for short queries; longer sequences or larger DBs may need more.
    """
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        _EUtilsState.throttle()
        time.sleep(3)
        resp = requests.get(
            BLAST_BASE,
            params={
                "CMD": "Get",
                "FORMAT_TYPE": "XML",
                "RID": rid,
                "TOOL": "research_agent_v4",
            },
            timeout=BLAST_TIMEOUT,
        )
        resp.raise_for_status()
        if "Status=READY" in resp.text:
            return resp.text
        if "Status=FAILED" in resp.text:
            raise ValueError(f"BLAST job {rid} failed")
    raise TimeoutError(f"BLAST job {rid} did not complete within {max_wait_s}s")


def _parse_blast_xml(xml_text: str, max_hits: int) -> list[dict[str, Any]]:
    """Extract a list of hits with title, accession, e-value, score, identity."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    hits: list[dict[str, Any]] = []
    for iteration in root.iter("Iteration"):
        for hit in iteration.iter("Hit")[:max_hits]:
            try:
                hit_def = hit.findtext("Hit_def", default="", namespaces={}).strip()
                hit_acc = hit.findtext("Hit_accession", default="", namespaces={})
                hsps = hit.findall("Hsp")
                if not hsps:
                    continue
                # First HSP (best) — extract the headline numbers.
                hsp = hsps[0]
                hits.append({
                    "title": hit_def,
                    "accession": hit_acc,
                    "score": float(hsp.findtext("Hsp_score", default="0", namespaces={}) or 0),
                    "evalue": float(hsp.findtext("Hsp_evalue", default="0", namespaces={}) or 0),
                    "identity_pct": float(hsp.findtext("Hsp_identity", default="0", namespaces={}) or 0)
                                   / max(1, float(hsp.findtext("Hsp_align-len", default="1", namespaces={}) or 1))
                                   * 100.0,
                    "alignment_length": int(hsp.findtext("Hsp_align-len", default="0", namespaces={}) or 0),
                })
            except Exception:
                continue
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break
    return hits


class SequenceAlignmentTool(BaseTool):
    """Run NCBI BLAST or future MSA against a query sequence."""

    name: str = "sequence_align"
    description: str = (
        "Align a nucleotide or protein sequence against a public database "
        "via NCBI BLAST. Returns the top hits with E-values, scores, and "
        "percent identity. Use 'blastp' for protein queries (default), "
        "'blastn' for nucleotide, 'blastx'/'tblastn'/'tblastx' for "
        "translated searches. 'msa' (multiple-sequence alignment) is "
        "reserved and currently returns an error. Requires 20+ character "
        "queries for sensible results."
    )
    args_schema: type[BaseModel] = SequenceAlignmentInput

    def _run(
        self,
        sequence: str,
        mode: Literal["blastn", "blastp", "blastx", "tblastn", "tblastx", "msa"] = "blastp",
        database: str = "nr",
        max_hits: int = 10,
    ) -> str:
        """Run BLAST and return a JSON list of top hits."""
        if mode == "msa":
            return json.dumps({
                "error": "MSA mode is not yet implemented; use 'blastp' or 'blastn' instead."
            })
        if not sequence or len(sequence.strip()) < 10:
            return json.dumps({
                "error": "sequence is too short; provide at least 10 characters."
            })
        try:
            n = int(max_hits)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))
        try:
            rid = _submit_blast(sequence.strip(), mode, database)
            xml = _poll_blast(rid, max_wait_s=60)
            hits = _parse_blast_xml(xml, n)
            return json.dumps({
                "rid": rid,
                "program": mode,
                "database": database,
                "hit_count": len(hits),
                "hits": hits,
            }, indent=2)
        except requests.RequestException as exc:
            return json.dumps({"error": f"BLAST network error: {exc}"})
        except TimeoutError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            return json.dumps({"error": f"BLAST failed: {exc}"})


sequence_alignment_tool = SequenceAlignmentTool()


# ===========================================================================
# M2 — StructureLookupTool
# ===========================================================================
# Looks up 3D structure files (PDB or mmCIF) and metadata from RCSB PDB
# or the AlphaFold DB.  Two sources, one tool:
#
#   - "pdb"        → RCSB PDB REST API (data.rcsb.org)
#   - "alphafold"  → AlphaFold DB REST API (alphafold.ebi.ac.uk)
#
# For each, the tool returns:
#   - metadata (resolution, method, title, etc.)
#   - a download path under per-session "structures/<id>.<ext>"
#   - the raw URL the agent can mention in citations

PDB_BASE: str = "https://data.rcsb.org/rest/v1/core"
ALPHAFOLD_BASE: str = "https://alphafold.ebi.ac.uk/api"
STRUCTURES_DIR: str = "structures"
STRUCTURE_HTTP_TIMEOUT: int = 30


class StructureLookupInput(BaseModel):
    """Inputs for StructureLookupTool."""

    query: str = Field(
        ...,
        description=(
            "Structure identifier. For source='pdb', a 4-character PDB ID "
            "(e.g. '1A2B', '6VXX'). For source='alphafold', a UniProt "
            "accession (e.g. 'P0DTC2') or an AlphaFold DB ID (e.g. "
            "'AF-P0DTC2-F1')."
        ),
    )
    source: Literal["pdb", "alphafold"] = Field(
        "pdb",
        description="Structure source: 'pdb' for RCSB, 'alphafold' for EBI AlphaFold DB.",
    )


def _lookup_pdb(pdb_id: str) -> dict[str, Any]:
    """Fetch PDB metadata; download the mmCIF to structures/<id>.cif."""
    pdb_id = pdb_id.upper().strip()
    if not _re.match(r"^[1-9][A-Z0-9]{3}$", pdb_id):
        raise ValueError(f"invalid PDB id format: {pdb_id!r}")
    meta_resp = requests.get(f"{PDB_BASE}/entry/{pdb_id}", timeout=STRUCTURE_HTTP_TIMEOUT)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    # Try to find a method/resolution summary.
    exptl = (meta.get("exptl") or [{}])[0]
    method = exptl.get("method", "")
    resolution = ""
    refine = (meta.get("refine") or [{}])
    if refine and isinstance(refine, list) and refine[0]:
        for entry in refine:
            for line in entry.get("ls_d_res_high", []) if isinstance(entry.get("ls_d_res_high"), list) else []:
                resolution = line.strip()
                break
            if resolution:
                break

    # Download mmCIF.
    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    cif_resp = requests.get(cif_url, timeout=STRUCTURE_HTTP_TIMEOUT)
    cif_resp.raise_for_status()
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    out_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.cif")
    with open(out_path, "wb") as fh:
        fh.write(cif_resp.content)

    return {
        "source": "pdb",
        "id": pdb_id,
        "title": (meta.get("struct") or {}).get("title", ""),
        "method": method,
        "resolution_angstrom": resolution,
        "released": (meta.get("rcsb_accession_info") or {}).get("initial_release_date", "")[:10],
        "url": f"https://www.rcsb.org/structure/{pdb_id}",
        "mmcif_path": out_path,
        "mmcif_url": cif_url,
    }


def _lookup_alphafold(uniprot_or_af_id: str) -> dict[str, Any]:
    """Fetch AlphaFold metadata; download the PDB-format model."""
    q = uniprot_or_af_id.strip()
    if q.startswith("AF-") and q.endswith("-F1"):
        af_id = q
        uniprot = q[3:].rsplit("-", 1)[0]
    else:
        af_id = f"AF-{q}-F1"
        uniprot = q

    url = f"{ALPHAFOLD_BASE}/prediction/{af_id}"
    resp = requests.get(url, timeout=STRUCTURE_HTTP_TIMEOUT)
    if resp.status_code == 404:
        # Try the bare UniProt endpoint as a fallback.
        url2 = f"{ALPHAFOLD_BASE}/prediction/{uniprot}"
        resp = requests.get(url2, timeout=STRUCTURE_HTTP_TIMEOUT)
    resp.raise_for_status()
    meta = resp.json()
    if isinstance(meta, list) and meta:
        meta = meta[0]
    pdb_url = meta.get("pdbUrl") or meta.get("cifUrl") or ""
    if not pdb_url:
        raise ValueError(f"no model URL in AlphaFold response for {af_id}")

    pdb_resp = requests.get(pdb_url, timeout=STRUCTURE_HTTP_TIMEOUT)
    pdb_resp.raise_for_status()
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    ext = "pdb" if pdb_url.endswith(".pdb") else "cif"
    out_path = os.path.join(STRUCTURES_DIR, f"{af_id}.{ext}")
    with open(out_path, "wb") as fh:
        fh.write(pdb_resp.content)

    return {
        "source": "alphafold",
        "id": af_id,
        "uniprot": uniprot,
        "organism": meta.get("organismScientificName", ""),
        "uniprot_description": meta.get("uniprotDescription", ""),
        "global_pLDDT": meta.get("globalMetricValue", ""),
        "model_created": (meta.get("modelCreatedDate") or "")[:10],
        "url": f"https://alphafold.ebi.ac.uk/entry/{af_id}",
        "model_path": out_path,
        "model_url": pdb_url,
    }


class StructureLookupTool(BaseTool):
    """Look up a 3D structure from PDB or AlphaFold DB."""

    name: str = "structure_lookup"
    description: str = (
        "Retrieve a 3D structure file (PDB mmCIF or AlphaFold model) and "
        "metadata from RCSB PDB or AlphaFold DB. For source='pdb', provide "
        "a 4-character PDB ID (e.g. '6VXX'). For source='alphafold', "
        "provide a UniProt accession (e.g. 'P0DTC2') or AlphaFold ID "
        "(e.g. 'AF-P0DTC2-F1'). Returns the metadata, a local path to the "
        "downloaded coordinate file under structures/, and the source URL."
    )
    args_schema: type[BaseModel] = StructureLookupInput

    def _run(self, query: str, source: Literal["pdb", "alphafold"] = "pdb") -> str:
        if not query or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            if source == "pdb":
                return json.dumps(_lookup_pdb(query), indent=2)
            return json.dumps(_lookup_alphafold(query), indent=2)
        except requests.RequestException as exc:
            return json.dumps({"error": f"structure lookup network error: {exc}"})
        except Exception as exc:
            return json.dumps({"error": f"structure lookup failed: {exc}"})


structure_lookup_tool = StructureLookupTool()


# ===========================================================================
# H4 — NCBIGenomicsTool
# ===========================================================================
# One tool, five entity types, all backed by NCBI E-utilities (the same
# E-utilities endpoints `PubMedSearchTool` uses) plus the NCBI Datasets
# v2 API for gene records.  This keeps the implementation zero-new-deps:
#
#   entity='gene'       → NCBI Datasets v2 (api.ncbi.nlm.nih.gov/datasets/v2/)
#   entity='nuccore'    → E-utilities efetch (returns a FASTA snippet)
#   entity='protein'    → E-utilities efetch
#   entity='structure'  → Cross-ref to StructureLookupTool (summary only)
#   entity='variant'    → NCBI ClinVar E-utilities (esearch + esummary)
#
# Where the response is large, we truncate the FASTA to 5 KB — enough for
# the agent to confirm what it got, with the accession number preserved
# so the user can fetch the full record downstream.

NCBI_DATASETS_BASE: str = "https://api.ncbi.nlm.nih.gov/datasets/v2"
CLINVAR_ESearch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CLINVAR_ESummary: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class NCBIGenomicsInput(BaseModel):
    """Inputs for NCBIGenomicsTool."""

    entity: Literal["gene", "nuccore", "protein", "structure", "variant"] = Field(
        ...,
        description=(
            "Type of record. 'gene' for gene summaries, 'nuccore' for a "
            "nucleotide FASTA, 'protein' for a protein FASTA, 'structure' "
            "for a PDB structure summary, 'variant' for a ClinVar variant "
            "record."
        ),
    )
    query: str = Field(
        ...,
        description=(
            "Accession or identifier. Examples: gene symbol 'TP53', RefSeq "
            "nucleotide 'NM_000546', RefSeq protein 'NP_000537', PDB ID "
            "'6VXX', or a ClinVar variation ID like '12345'."
        ),
    )
    max_results: int = Field(
        5, ge=1, le=20, description="Maximum number of records to return (1–20)."
    )


def _gene_lookup(query: str) -> dict[str, Any]:
    """NCBI Datasets v2 gene report — returns gene metadata + RefSeq links."""
    resp = requests.get(
        f"{NCBI_DATASETS_BASE}/gene/symbol/{requests.utils.quote(query)}",
        params={"page_size": "1"},
        timeout=STRUCTURE_HTTP_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    payload = resp.json()
    reports = payload.get("reports", [])
    if not reports:
        raise ValueError(f"no gene report found for {query!r}")
    rep = reports[0]
    gene = rep.get("gene", {})
    return {
        "entity": "gene",
        "symbol": gene.get("symbol", query),
        "name": gene.get("description", ""),
        "chromosome": gene.get("chromosomes", [""])[0] if gene.get("chromosomes") else "",
        "map_location": gene.get("map_location", ""),
        "type": gene.get("type", ""),
        "refseq_nucleotide": [
            t.get("accession_version", "") for t in rep.get("transcripts", [])
            if t.get("accession_version", "").startswith("NM_")
        ][:5],
        "refseq_protein": [
            t.get("accession_version", "") for t in rep.get("transcripts", [])
            if t.get("accession_version", "").startswith("NP_")
        ][:5],
        "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene.get('gene_id', '')}",
        "source": "ncbi_datasets_v2",
    }


def _efetch_fasta(db: str, accession: str) -> str:
    """Fetch the first 5 KB of a FASTA record for *accession* in *db*."""
    _EUtilsState.throttle()
    resp = requests.get(
        f"{EUTILS_BASE}/efetch.fcgi",
        params=_eutils_params({
            "id": accession,
            "db": db,
            "rettype": "fasta",
            "retmode": "text",
        }),
        timeout=STRUCTURE_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.text
    return text[:5120] + ("\n[... truncated ...]" if len(text) > 5120 else "")


def _variant_lookup(query: str) -> dict[str, Any]:
    """ClinVar variant record by variation ID."""
    # Query may be a numeric variation ID or a free-text term.
    if query.strip().isdigit():
        var_ids = [query.strip()]
    else:
        _EUtilsState.throttle()
        resp = requests.get(
            CLINVAR_ESearch,
            params=_eutils_params({
                "db": "clinvar",
                "term": query,
                "retmax": "5",
                "retmode": "json",
            }),
            timeout=STRUCTURE_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        var_ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not var_ids:
        raise ValueError(f"no ClinVar variants found for {query!r}")
    _EUtilsState.throttle()
    resp = requests.get(
        CLINVAR_ESummary,
        params=_eutils_params({
            "db": "clinvar",
            "id": ",".join(var_ids),
            "retmode": "json",
        }),
        timeout=STRUCTURE_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json().get("result", {})
    records = []
    for vid in var_ids:
        rec = result.get(vid, {})
        if not isinstance(rec, dict):
            continue
        germ = rec.get("germline_classification", {}) or {}
        records.append({
            "variation_id": vid,
            "title": rec.get("title", ""),
            "clinical_significance": germ.get("description", ""),
            "review_status": germ.get("review_status", ""),
            "trait": (rec.get("trait_set") or [{}])[0].get("trait_name", "")
                      if rec.get("trait_set") else "",
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{vid}/",
        })
    return {"entity": "variant", "source": "clinvar", "records": records}


class NCBIGenomicsTool(BaseTool):
    """Look up gene / nucleotide / protein / structure / variant records."""

    name: str = "ncbi_genomics"
    description: str = (
        "Query NCBI for genomics records. 'gene' returns gene metadata and "
        "RefSeq links (NCBI Datasets v2). 'nuccore' returns a FASTA snippet "
        "for a RefSeq nucleotide accession (NM_/NR_/XM_). 'protein' returns "
        "a FASTA snippet for a RefSeq protein accession (NP_/XP_). "
        "'structure' returns a PDB structure summary. 'variant' returns a "
        "ClinVar variant record by variation ID or free-text query."
    )
    args_schema: type[BaseModel] = NCBIGenomicsInput

    def _run(
        self,
        entity: Literal["gene", "nuccore", "protein", "structure", "variant"],
        query: str,
        max_results: int = 5,
    ) -> str:
        if not query or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(20, n))

        try:
            if entity == "gene":
                return json.dumps(_gene_lookup(query.strip()), indent=2)
            if entity == "nuccore":
                fasta = _efetch_fasta("nuccore", query.strip())
                return json.dumps({
                    "entity": "nuccore",
                    "accession": query.strip(),
                    "fasta_snippet": fasta,
                }, indent=2)
            if entity == "protein":
                fasta = _efetch_fasta("protein", query.strip())
                return json.dumps({
                    "entity": "protein",
                    "accession": query.strip(),
                    "fasta_snippet": fasta,
                }, indent=2)
            if entity == "structure":
                # Structure summaries piggyback on the structure_lookup
                # tool so we don't fork the PDB code.
                try:
                    summary = _lookup_pdb(query.strip())
                    # Drop the heavy mmcif_path/cif for the summary view.
                    summary = {k: v for k, v in summary.items()
                               if k not in ("mmcif_path",)}
                    return json.dumps(summary, indent=2)
                except Exception as exc:
                    return json.dumps({
                        "entity": "structure",
                        "error": f"PDB lookup failed: {exc}; "
                                 "the structure_lookup tool handles full file download."
                    })
            if entity == "variant":
                return json.dumps(_variant_lookup(query.strip()), indent=2)
            return json.dumps({"error": f"unknown entity: {entity}"})
        except requests.RequestException as exc:
            return json.dumps({"error": f"NCBI genomics network error: {exc}"})
        except Exception as exc:
            return json.dumps({"error": f"NCBI genomics failed: {exc}"})


ncbi_genomics_tool = NCBIGenomicsTool()


# ===========================================================================
# H5 — AntibodyTherapeuticTool
# ===========================================================================
# Aggregates antibody-specific data from five sources, returning one
# unified JSON record per query:
#
#   - IEDB           → epitope records (iedb.org)
#   - Thera-SAbDab   → therapeutic antibody metadata (opig.stats.ox.ac.uk)
#   - CoV-AbDab      → coronavirus antibody database
#   - SAbDab         → structural antibody database
#   - PDB            → cross-referenced structure entries
#
# The aggregation policy: collect whatever each source returns, label
# each item with a `source` field, and merge them under the query.  The
# agent sees a single record with provenance for every line, so it can
# cite correctly without guessing where a fact came from.
#
# This tool deliberately returns a flat list rather than a deep tree —
# flat lists are easier for the LLM to scan and reason about.

THERASABDAB_BASE: str = "https://opig.stats.ox.ac.uk/webapps/therasabdab"
COVABDAB_BASE: str = "http://opig.stats.ox.ac.uk/webapps/covabdab"
SABDAB_BASE: str = "http://opig.stats.ox.ac.uk/webapps/sabdab"
ANTIBODY_HTTP_TIMEOUT: int = 20


class AntibodyTherapeuticInput(BaseModel):
    """Inputs for AntibodyTherapeuticTool."""

    query: str = Field(
        ...,
        description=(
            "Antibody name (e.g. 'pembrolizumab', 'casirivimab'), target "
            "(e.g. 'PD-1', 'SARS-CoV-2 spike'), or INN / brand name. "
            "Free-text matching across all five sources."
        ),
    )
    query_type: Literal["name", "target"] = Field(
        "name",
        description=(
            "'name' for an antibody identifier, 'target' for an antigen "
            "name. Affects which fields are searched in each source."
        ),
    )


def _therasabdab_lookup(query: str) -> list[dict[str, Any]]:
    """Thera-SAbDab — therapeutic antibodies in clinical development."""
    try:
        resp = requests.get(
            f"{THERASABDAB_BASE}/search/{requests.utils.quote(query)}",
            headers={"Accept": "application/json"},
            timeout=ANTIBODY_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data if isinstance(data, list) else data.get("results", [])
        out = []
        for r in results[:10]:
            out.append({
                "source": "therasabdab",
                "name": r.get("name", ""),
                "inn": r.get("inn", ""),
                "target": r.get("target", ""),
                "format": r.get("format", ""),
                "phase": r.get("highest_clinical_phase", ""),
                "isotype": r.get("isotype", ""),
                "url": f"https://opig.stats.ox.ac.uk/webapps/therasabdab/therasabdab/{r.get('name', '')}",
            })
        return out
    except Exception:
        return []


def _covabdab_lookup(query: str) -> list[dict[str, Any]]:
    """CoV-AbDab — coronavirus neutralising antibodies and binding data."""
    try:
        resp = requests.get(
            f"{COVABDAB_BASE}/search",
            params={"search": query, "format": "json"},
            timeout=ANTIBODY_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        rows = resp.json()
        if not isinstance(rows, list):
            return []
        out = []
        for r in rows[:10]:
            out.append({
                "source": "covabdab",
                "name": r.get("Name", r.get("name", "")),
                "origin": r.get("Origin", ""),
                "v_gene": r.get("VHgene", ""),
                "j_gene": r.get("JHgene", ""),
                "target": ", ".join(r.get("Sources", []) or []) if isinstance(r.get("Sources"), list) else "",
                "neutralising": r.get("Neutralising vs", ""),
                "url": f"http://opig.stats.ox.ac.uk/webapps/covabdab/{r.get('Name', '')}",
            })
        return out
    except Exception:
        return []


def _sabdab_lookup(query: str) -> list[dict[str, Any]]:
    """SAbDab — structural antibody database (PDB cross-references)."""
    try:
        resp = requests.get(
            f"{SABDAB_BASE}/search",
            params={"search": query, "format": "json"},
            timeout=ANTIBODY_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        rows = resp.json()
        if not isinstance(rows, list):
            return []
        out = []
        for r in rows[:10]:
            out.append({
                "source": "sabdab",
                "pdb": r.get("pdb", ""),
                "name": r.get("antibody_name", ""),
                "heavy_species": r.get("heavy_species", ""),
                "light_species": r.get("light_species", ""),
                "resolution": r.get("resolution", ""),
                "method": r.get("method", ""),
                "antigen": r.get("antigen_name", ""),
                "url": f"https://www.ebi.ac.uk/pdbe/entry/pdb/{r.get('pdb', '')}",
            })
        return out
    except Exception:
        return []


def _iedb_lookup(query: str) -> list[dict[str, Any]]:
    """IEDB — Immune Epitope Database, epitope records."""
    try:
        # IEDB's search API is open and returns JSON. The exact endpoint
        # is documented at https://www.iedb.org/database_export_v3.html.
        resp = requests.get(
            "https://query-api.iedb.org/epitope_search",
            params={
                "query": query,
                "select": "epitope_id,epitope_type,source_organism,source_antigen,structure_type",
                "limit": "10",
            },
            timeout=ANTIBODY_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        rows = resp.json() if isinstance(resp.json(), list) else []
        return [
            {
                "source": "iedb",
                "epitope_id": r.get("epitope_id", ""),
                "epitope_type": r.get("epitope_type", ""),
                "source_organism": r.get("source_organism", ""),
                "antigen": r.get("source_antigen", ""),
                "structure_type": r.get("structure_type", ""),
                "url": f"https://www.iedb.org/epitope/{r.get('epitope_id', '')}",
            }
            for r in rows
        ]
    except Exception:
        return []


class AntibodyTherapeuticTool(BaseTool):
    """Look up antibody therapeutic and structural records."""

    name: str = "antibody_therapeutic"
    description: str = (
        "Look up antibody sequence, paratope, epitope, and therapeutic "
        "feasibility data by aggregating IEDB (epitopes), Thera-SAbDab "
        "(clinical-stage therapeutics), CoV-AbDab (coronavirus antibodies), "
        "SAbDab (PDB cross-references), and PDB structure entries. Accepts "
        "an antibody name (e.g. 'pembrolizumab') or target (e.g. 'PD-1'). "
        "Returns a unified record with explicit `source` field per item."
    )
    args_schema: type[BaseModel] = AntibodyTherapeuticInput

    def _run(
        self,
        query: str,
        query_type: Literal["name", "target"] = "name",
    ) -> str:
        if not query or not query.strip():
            return json.dumps({"error": "empty query"})
        q = query.strip()
        # Run all five lookups; each one tolerates its own failure and
        # returns an empty list rather than raising. We aggregate whatever
        # comes back.
        thera = _therasabdab_lookup(q)
        cov = _covabdab_lookup(q)
        sab = _sabdab_lookup(q)
        iedb = _iedb_lookup(q)
        total = len(thera) + len(cov) + len(sab) + len(iedb)
        if total == 0:
            return json.dumps({
                "query": q,
                "query_type": query_type,
                "result_count": 0,
                "records": [],
                "message": "no antibody records found across the five sources; "
                           "try a different name or target.",
            }, indent=2)
        return json.dumps({
            "query": q,
            "query_type": query_type,
            "result_count": total,
            "therasabdab": thera,
            "covabdab": cov,
            "sabdab": sab,
            "iedb": iedb,
        }, indent=2)


antibody_therapeutic_tool = AntibodyTherapeuticTool()


# ===========================================================================
# M7 — FigureCaptionTool
# ===========================================================================
# Generates a structured caption for a single image file (PNG, JPG, TIFF)
# or a PDF page rendered as an image.  Uses a local Ollama vision model
# (llama3.2-vision, gemma3:27b-vision, or llava) and returns a small
# JSON record with figure type, axes, key result, and a take-home sentence.
#
# The vision-model selection mirrors the same auto-detect pattern the
# multimodal extract tool uses — see the Ollama vision model hint
# priority in `_OLLAMA_VISION_HINTS`.

import base64 as _base64
from pathlib import Path as _Path

_OLLAMA_VISION_HINTS = (
    "llama3.2-vision",
    "gemma3:27b-vision", "gemma3-vision",
    "llava", "llava-phi3",
    "minicpm-v",
)
OLLAMA_GENERATE_URL: str = "http://localhost:11434/api/generate"


class FigureCaptionInput(BaseModel):
    """Inputs for FigureCaptionTool."""

    image_path: str = Field(
        ...,
        description="Local path to an image file (PNG, JPG, JPEG, TIFF) "
                    "or a single-page PDF to render and caption."
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional context to bias the caption, e.g. "
                    "'Figure 3 from PMID 12345' or 'Results panel B'."
    )


def _detect_vision_model() -> Optional[str]:
    """Return the first installed Ollama vision model, or None."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return None
    for hint in _OLLAMA_VISION_HINTS:
        for m in models:
            if m.startswith(hint):
                return m
    return None


def _ensure_image_png(path: str) -> bytes:
    """Return PNG bytes for *path*. Renders the first page if it's a PDF."""
    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".pdf":
        # Render first page via PyMuPDF (already a project dependency).
        import fitz  # type: ignore
        doc = fitz.open(path)
        if not doc:
            raise ValueError(f"empty PDF: {path}")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        return pix.tobytes("png")
    # Otherwise trust the file as-is.
    return p.read_bytes()


def _caption_with_ollama(image_bytes: bytes, hint: Optional[str], model: str) -> str:
    """POST a base64 image to Ollama's /api/generate with a vision prompt."""
    img_b64 = _base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "You are a scientific figure captioner. Describe the image in a "
        "structured way. Return ONLY valid JSON with these keys: "
        "`figure_type` (chart / micrograph / diagram / photo / other), "
        "`axes` (list of axis labels, or []), "
        "`key_result` (one sentence), "
        "`take_home` (one sentence a non-expert would understand)."
    )
    if hint:
        prompt += f" Context: {hint}."
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        return json.dumps({"error": f"Ollama vision call failed: {exc}"})


class FigureCaptionTool(BaseTool):
    """Caption a single image using a local vision model."""

    name: str = "caption_image"
    description: str = (
        "Generate a structured caption for a single image or a single PDF "
        "page rendered as an image. Uses a local Ollama vision model "
        "(llama3.2-vision, gemma3:27b-vision, or llava) and returns a JSON "
        "record with figure_type, axes, key_result, and take_home fields. "
        "Use when a specific figure needs explanation without re-extracting "
        "the entire source document."
    )
    args_schema: type[BaseModel] = FigureCaptionInput

    def _run(
        self,
        image_path: str,
        hint: Optional[str] = None,
    ) -> str:
        if not image_path or not image_path.strip():
            return json.dumps({"error": "empty image_path"})
        try:
            image_bytes = _ensure_image_png(image_path.strip())
        except FileNotFoundError:
            return json.dumps({"error": f"file not found: {image_path}"})
        except Exception as exc:
            return json.dumps({"error": f"could not load image: {exc}"})

        model = _detect_vision_model()
        if not model:
            return json.dumps({
                "error": (
                    "no Ollama vision model found. Pull one with "
                    "'ollama pull llama3.2-vision' (or gemma3:27b-vision / llava) "
                    "and ensure Ollama is running at localhost:11434."
                )
            })

        raw = _caption_with_ollama(image_bytes, hint, model)
        # Try to parse the model's JSON; if it returned a string, wrap it.
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw_response": raw}
        parsed["image_path"] = image_path
        parsed["model"] = model
        return json.dumps(parsed, indent=2)


figure_caption_tool = FigureCaptionTool()

# ===========================================================================
# L1 — ClinicalTrialsTool
# ===========================================================================
# ClinicalTrials.gov API v2 (no API key required).

CLINICAL_TRIALS_BASE: str = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_TIMEOUT: int = 20


class ClinicalTrialsInput(BaseModel):
    """Inputs for ClinicalTrialsTool."""

    query: str = Field(
        ..., description="Condition, intervention, drug name, or NCT ID to search for."
    )
    max_results: int = Field(
        10, ge=1, le=50, description="Maximum number of trials to return (1-50)."
    )


def _format_trial(study: dict[str, Any]) -> dict[str, Any]:
    section = study.get("protocolSection", {}) or {}
    ident = section.get("identificationModule", {}) or {}
    status = section.get("statusModule", {}) or {}
    design = section.get("designModule", {}) or {}
    sponsor = (section.get("sponsorCollaboratorsModule", {}) or {}).get("leadSponsor", {}) or {}
    outcomes = section.get("outcomesModule", {}) or {}
    primary = outcomes.get("primaryOutcomes", []) or []
    enrollment = design.get("enrollmentInfo", {}) or {}
    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus", ""),
        "phase": design.get("phases", []),
        "enrollment": enrollment.get("count", ""),
        "sponsor": sponsor.get("name", ""),
        "primary_outcomes": [o.get("measure", "") for o in primary][:5],
        "url": f"https://clinicaltrials.gov/study/{ident.get('nctId', '')}",
    }


class ClinicalTrialsTool(BaseTool):
    """Search ClinicalTrials.gov for active and completed trials."""

    name: str = "clinical_trials"
    description: str = (
        "Search ClinicalTrials.gov for active and completed trials by condition, "
        "intervention, drug name, or NCT ID. Returns phase, status, enrollment, "
        "primary outcomes, and sponsor. No API key required."
    )
    args_schema: type[BaseModel] = ClinicalTrialsInput

    def _run(self, query: str, max_results: int = 10) -> str:
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))

        try:
            resp = requests.get(
                CLINICAL_TRIALS_BASE,
                params={"query.term": query.strip(), "pageSize": str(n)},
                timeout=CLINICAL_TRIALS_TIMEOUT,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            studies = resp.json().get("studies", []) or []
            return json.dumps([_format_trial(s) for s in studies], indent=2)
        except requests.RequestException as exc:
            logger.warning("ClinicalTrials network error: %s", exc)
            return json.dumps({"error": f"ClinicalTrials network error: {exc}"})
        except Exception as exc:
            logger.warning("ClinicalTrials tool failed: %s", exc)
            return json.dumps({"error": f"ClinicalTrials tool failed: {exc}"})


clinical_trials_tool = ClinicalTrialsTool()


# ===========================================================================
# L2 — PathwayDatabaseTool
# ===========================================================================
# KEGG REST (rest.kegg.jp) + Reactome ContentService — both no-key, public.
# KEGG has commercial-use restrictions; this tool is read-only lookup, which
# is within KEGG's free-use terms for individual/academic use.

KEGG_BASE: str = "https://rest.kegg.jp"
REACTOME_BASE: str = "https://reactome.org/ContentService"
PATHWAY_TIMEOUT: int = 20


class PathwayLookupInput(BaseModel):
    """Inputs for PathwayDatabaseTool."""

    query: str = Field(
        ..., description="Gene symbol, protein name, compound, or pathway keyword."
    )
    source: Literal["kegg", "reactome", "both"] = Field(
        "both", description="Pathway database to query: 'kegg', 'reactome', or 'both'."
    )


def _kegg_pathways(query: str) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{KEGG_BASE}/find/pathway/{requests.utils.quote(query)}",
        timeout=PATHWAY_TIMEOUT,
    )
    resp.raise_for_status()
    results = []
    for line in resp.text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        pid, name = parts
        clean_id = pid.replace("path:", "")
        results.append({
            "source": "kegg",
            "id": clean_id,
            "name": name,
            "url": f"https://www.kegg.jp/pathway/{clean_id}",
        })
    return results


def _reactome_pathways(query: str) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{REACTOME_BASE}/search/query",
        params={"query": query, "species": "Homo sapiens", "types": "Pathway", "cluster": "true"},
        timeout=PATHWAY_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for group in data.get("results", []) or []:
        for entry in group.get("entries", []) or []:
            results.append({
                "source": "reactome",
                "id": entry.get("stId", ""),
                "name": re.sub(r"<[^>]+>", "", entry.get("name", "") or ""),
                "species": ", ".join(entry.get("species", []) or []),
                "url": f"https://reactome.org/content/detail/{entry.get('stId', '')}",
            })
    return results


class PathwayDatabaseTool(BaseTool):
    """Look up biological pathways involving a gene, protein, or compound."""

    name: str = "pathway_lookup"
    description: str = (
        "Look up biological pathways involving a gene, protein, or compound. "
        "Returns pathway name, ID, and source database. Source options: "
        "'kegg', 'reactome', or 'both'. No API key required."
    )
    args_schema: type[BaseModel] = PathwayLookupInput

    def _run(self, query: str, source: Literal["kegg", "reactome", "both"] = "both") -> str:
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "empty query"})
        q = query.strip()
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        if source in ("kegg", "both"):
            try:
                results.extend(_kegg_pathways(q))
            except Exception as exc:
                errors.append(f"KEGG: {exc}")
        if source in ("reactome", "both"):
            try:
                results.extend(_reactome_pathways(q))
            except Exception as exc:
                errors.append(f"Reactome: {exc}")
        if not results and errors:
            return json.dumps({"error": "; ".join(errors)})
        payload: dict[str, Any] = {"results": results[:50]}
        if errors:
            payload["partial_errors"] = errors
        return json.dumps(payload, indent=2)


pathway_lookup_tool = PathwayDatabaseTool()


# ===========================================================================
# L3 — DatasetSearchTool
# ===========================================================================
# 'geo' and 'ncbi' both reuse the existing NCBI E-utilities helpers
# (`_EUtilsState`, `_eutils_params`, `EUTILS_BASE`) defined earlier in this
# module for PubMedSearchTool — 'geo' queries db=gds, 'ncbi' queries
# db=bioproject. 'ena' hits the EBI ENA Portal API directly (no key).

ENA_PORTAL_BASE: str = "https://www.ebi.ac.uk/ena/portal/api/search"
DATASET_HTTP_TIMEOUT: int = 20


class DatasetSearchInput(BaseModel):
    """Inputs for DatasetSearchTool."""

    query: str = Field(..., description="Free-text query: organism, assay type, gene, or disease.")
    source: Literal["geo", "ena", "ncbi"] = Field(
        "geo",
        description="'geo' for NCBI GEO datasets, 'ena' for ENA read studies, 'ncbi' for NCBI BioProject.",
    )
    max_results: int = Field(
        10, ge=1, le=50, description="Maximum number of records to return (1-50)."
    )


def _eutils_dataset_search(query: str, n: int, db: str) -> list[dict[str, Any]]:
    _EUtilsState.throttle()
    resp = requests.get(
        f"{EUTILS_BASE}/esearch.fcgi",
        params=_eutils_params({"db": db, "term": query, "retmax": str(n), "retmode": "json"}),
        timeout=DATASET_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    _EUtilsState.throttle()
    resp = requests.get(
        f"{EUTILS_BASE}/esummary.fcgi",
        params=_eutils_params({"db": db, "id": ",".join(ids), "retmode": "json"}),
        timeout=DATASET_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json().get("result", {})
    records = []
    for rid in ids:
        rec = result.get(rid, {})
        if not isinstance(rec, dict):
            continue
        records.append({
            "source": "geo" if db == "gds" else "ncbi_bioproject",
            "accession": rec.get("accession", rec.get("project_acc", rid)),
            "title": rec.get("title", rec.get("project_title", "")),
            "organism": rec.get("taxon", rec.get("organism_name", "")),
            "summary": (rec.get("summary", rec.get("project_description", "")) or "")[:500],
            "url": (
                f"https://www.ncbi.nlm.nih.gov/gds/?term={rid}" if db == "gds"
                else f"https://www.ncbi.nlm.nih.gov/bioproject/{rid}"
            ),
        })
    return records


def _ena_dataset_search(query: str, n: int) -> list[dict[str, Any]]:
    # ENA's query language is Lucene-like: a literal `"` would break out of the
    # quoted wildcard, breaking the request (HTTP 400).
    #
    # A bare `-` inside the quoted wildcard silently zeroes out results (e.g.
    # "SARS-CoV-2") because ENA tokenizes on `-` and a wildcard match can't
    # bridge that token boundary. Backslash-escaping `-` (the usual Lucene
    # escape) does NOT fix this -- confirmed live, `\-` still returns zero
    # results. What does work live is replacing `-` with `*`, turning the
    # hyphen itself into a wildcard gap between tokens.
    sanitized_query = query.replace('"', "").replace("-", "*")
    resp = requests.get(
        ENA_PORTAL_BASE,
        params={
            "result": "study",
            "query": f'study_title="*{sanitized_query}*"',
            "fields": "study_accession,study_title,tax_id",
            "format": "json",
            "limit": str(n),
        },
        timeout=DATASET_HTTP_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json() if resp.text.strip() else []
    records = []
    for rec in data:
        records.append({
            "source": "ena",
            "accession": rec.get("study_accession", ""),
            "title": rec.get("study_title", ""),
            "organism": rec.get("tax_id", ""),
            "url": f"https://www.ebi.ac.uk/ena/browser/view/{rec.get('study_accession', '')}",
        })
    return records


class DatasetSearchTool(BaseTool):
    """Find published omics datasets (GEO, ENA, NCBI BioProject)."""

    name: str = "dataset_search"
    description: str = (
        "Find published omics datasets. 'geo' searches NCBI GEO (expression/sequencing "
        "datasets), 'ena' searches the EBI European Nucleotide Archive (read studies), "
        "'ncbi' searches NCBI BioProject. Returns accession, title, organism, and a link. "
        "No API key required."
    )
    args_schema: type[BaseModel] = DatasetSearchInput

    def _run(self, query: str, source: Literal["geo", "ena", "ncbi"] = "geo", max_results: int = 10) -> str:
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))

        try:
            if source == "geo":
                records = _eutils_dataset_search(query.strip(), n, db="gds")
            elif source == "ncbi":
                records = _eutils_dataset_search(query.strip(), n, db="bioproject")
            else:
                records = _ena_dataset_search(query.strip(), n)
            return json.dumps(records, indent=2)
        except requests.RequestException as exc:
            logger.warning("DatasetSearch network error: %s", exc)
            return json.dumps({"error": f"DatasetSearch network error: {exc}"})
        except Exception as exc:
            logger.warning("DatasetSearch tool failed: %s", exc)
            return json.dumps({"error": f"DatasetSearch tool failed: {exc}"})


dataset_search_tool = DatasetSearchTool()


# ===========================================================================
# L4 — PatentSearchTool
# ===========================================================================
# Three backends, all requiring a credential this harness does not ship
# with — each gated the same way GoogleScholarTool gates SERPAPI_API_KEY:
# missing key -> a clear {"error": "... requires <ENV_VAR> ..."}, not a crash.

PATENTSVIEW_BASE: str = "https://search.patentsview.org/api/v1/patent/"
EPO_OPS_AUTH_URL: str = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_OPS_SEARCH_BASE: str = "https://ops.epo.org/3.2/rest-services/published-data/search"
WIPO_PATENTSCOPE_BASE: str = "https://patentscope.wipo.int/search/rest"
PATENT_TIMEOUT: int = 20


class PatentSearchInput(BaseModel):
    """Inputs for PatentSearchTool."""

    query: str = Field(..., description="Free-text query: invention keywords, assignee, or inventor.")
    source: Literal["uspto", "epo", "wipo"] = Field("uspto", description="Patent office to search.")
    max_results: int = Field(
        10, ge=1, le=50, description="Maximum number of patents to return (1-50)."
    )


def _uspto_patentsview_search(query: str, n: int) -> list[dict[str, Any]]:
    api_key = os.environ.get("PATENTSVIEW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "USPTO search requires PATENTSVIEW_API_KEY (free key from "
            "https://patentsview.org/apis/keyrequest)"
        )
    q = json.dumps({"_text_any": {"patent_title": query}})
    f = json.dumps([
        "patent_id", "patent_title", "patent_date",
        "assignees.assignee_organization",
    ])
    o = json.dumps({"size": n})
    resp = requests.get(
        PATENTSVIEW_BASE,
        params={"q": q, "f": f, "o": o},
        headers={"X-Api-Key": api_key},
        timeout=PATENT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    records = []
    for p in data.get("patents", []) or []:
        assignees = p.get("assignees") or []
        records.append({
            "source": "uspto",
            "patent_id": p.get("patent_id", ""),
            "title": p.get("patent_title", ""),
            "date": p.get("patent_date", ""),
            "assignee": assignees[0].get("assignee_organization", "") if assignees else "",
            "url": f"https://patents.google.com/patent/US{p.get('patent_id', '')}",
        })
    return records


def _epo_search(query: str, n: int) -> list[dict[str, Any]]:
    key = os.environ.get("EPO_OPS_KEY")
    secret = os.environ.get("EPO_OPS_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "EPO search requires EPO_OPS_KEY and EPO_OPS_SECRET "
            "(register at developers.epo.org)"
        )
    token_resp = requests.post(
        EPO_OPS_AUTH_URL,
        data={"grant_type": "client_credentials"},
        auth=(key, secret),
        timeout=PATENT_TIMEOUT,
    )
    token_resp.raise_for_status()
    token = token_resp.json().get("access_token", "")
    resp = requests.get(
        EPO_OPS_SEARCH_BASE,
        params={"q": query},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=PATENT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    docs = (
        data.get("ops:world-patent-data", {})
        .get("ops:biblio-search", {})
        .get("ops:search-result", {})
        .get("ops:publication-reference", [])
    )
    if isinstance(docs, dict):
        docs = [docs]
    records = []
    for d in docs[:n]:
        doc_id = d.get("document-id") or [{}]
        if isinstance(doc_id, list):
            doc_id = doc_id[0] if doc_id else {}
        country = (doc_id.get("country") or {}).get("$", "")
        number = (doc_id.get("doc-number") or {}).get("$", "")
        records.append({
            "source": "epo",
            "patent_id": f"{country}{number}",
            "url": f"https://worldwide.espacenet.com/patent/search?q={country}{number}",
        })
    return records


def _wipo_search(query: str, n: int) -> list[dict[str, Any]]:
    key = os.environ.get("WIPO_PATENTSCOPE_KEY")
    if not key:
        raise RuntimeError(
            "WIPO PATENTSCOPE search requires WIPO_PATENTSCOPE_KEY "
            "(register at patentscope.wipo.int)"
        )
    try:
        resp = requests.get(
            f"{WIPO_PATENTSCOPE_BASE}/search",
            params={"q": query, "key": key, "size": n},
            timeout=PATENT_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(str(exc).replace(key, "***")) from None
    data = resp.json()
    records = []
    for r in (data.get("results") or [])[:n]:
        records.append({
            "source": "wipo",
            "patent_id": r.get("documentId", ""),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
        })
    return records


class PatentSearchTool(BaseTool):
    """Search patent literature for antibodies, sequences, or therapeutics."""

    name: str = "patent_search"
    description: str = (
        "Search patent literature for antibodies, sequences, or therapeutic candidates. "
        "'uspto' uses the PatentsView API (requires PATENTSVIEW_API_KEY), 'epo' uses "
        "EPO OPS (requires EPO_OPS_KEY/EPO_OPS_SECRET), 'wipo' uses PATENTSCOPE "
        "(requires WIPO_PATENTSCOPE_KEY). Returns patent ID, title, date, and assignee."
    )
    args_schema: type[BaseModel] = PatentSearchInput

    def _run(self, query: str, source: Literal["uspto", "epo", "wipo"] = "uspto", max_results: int = 10) -> str:
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))

        try:
            if source == "uspto":
                records = _uspto_patentsview_search(query.strip(), n)
            elif source == "epo":
                records = _epo_search(query.strip(), n)
            else:
                records = _wipo_search(query.strip(), n)
            return json.dumps(records, indent=2)
        except requests.RequestException as exc:
            logger.warning("PatentSearch network error: %s", exc)
            return json.dumps({"error": f"PatentSearch network error: {exc}"})
        except Exception as exc:
            logger.warning("PatentSearch tool failed: %s", exc)
            return json.dumps({"error": str(exc)})


patent_search_tool = PatentSearchTool()

# ===========================================================================
# L5 — DrugInteractionTool
# ===========================================================================
# 'chembl' is open, no-key (EBI ChEMBL REST API). 'drugbank' requires a
# commercial license key, gated the same way L4's backends are.

CHEMBL_BASE: str = "https://www.ebi.ac.uk/chembl/api/data"
DRUG_TIMEOUT: int = 20


class DrugLookupInput(BaseModel):
    """Inputs for DrugInteractionTool."""

    query: str = Field(..., description="Drug or biologic name, target, or indication.")
    source: Literal["drugbank", "chembl"] = Field("chembl", description="Drug database to query.")


def _chembl_lookup(query: str) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{CHEMBL_BASE}/molecule/search.json",
        params={"q": query},
        timeout=DRUG_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    records = []
    for m in (data.get("molecules") or [])[:10]:
        records.append({
            "source": "chembl",
            "chembl_id": m.get("molecule_chembl_id", ""),
            "name": m.get("pref_name", ""),
            "molecule_type": m.get("molecule_type", ""),
            "max_phase": m.get("max_phase", ""),
            "url": f"https://www.ebi.ac.uk/chembl/explore/compound/{m.get('molecule_chembl_id', '')}",
        })
    return records


def _drugbank_lookup(query: str) -> list[dict[str, Any]]:
    api_key = os.environ.get("DRUGBANK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DrugBank search requires DRUGBANK_API_KEY (commercial license; "
            "see go.drugbank.com)"
        )
    resp = requests.get(
        "https://api.drugbank.com/v1/drug_names",
        params={"q": query},
        headers={"Authorization": api_key},
        timeout=DRUG_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    records = []
    for d in (data or [])[:10]:
        records.append({
            "source": "drugbank",
            "drugbank_id": d.get("drugbank_id", ""),
            "name": d.get("name", ""),
        })
    return records


class DrugInteractionTool(BaseTool):
    """Look up drug or biologic records by name, target, or indication."""

    name: str = "drug_lookup"
    description: str = (
        "Look up drug or biologic records (small molecules and antibodies) by name, "
        "target, or indication. 'chembl' is open and requires no key; 'drugbank' "
        "requires DRUGBANK_API_KEY (commercial license). Returns molecule type, "
        "development phase, and identifiers."
    )
    args_schema: type[BaseModel] = DrugLookupInput

    def _run(self, query: str, source: Literal["drugbank", "chembl"] = "chembl") -> str:
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            if source == "chembl":
                records = _chembl_lookup(query.strip())
            else:
                records = _drugbank_lookup(query.strip())
            return json.dumps(records, indent=2)
        except requests.RequestException as exc:
            logger.warning("DrugLookup network error: %s", exc)
            return json.dumps({"error": f"DrugLookup network error: {exc}"})
        except Exception as exc:
            logger.warning("DrugLookup tool failed: %s", exc)
            return json.dumps({"error": str(exc)})


drug_interaction_tool = DrugInteractionTool()

# ===========================================================================
# L6 — GitHubCodeSearchTool
# ===========================================================================
# GitHub REST search API. Works unauthenticated at a lower rate limit;
# GITHUB_TOKEN (optional) raises the limit the same way NCBI_API_KEY does
# for PubMedSearchTool.

GITHUB_SEARCH_BASE: str = "https://api.github.com/search/repositories"
GITHUB_TIMEOUT: int = 20


class GitHubSearchInput(BaseModel):
    """Inputs for GitHubCodeSearchTool."""

    query: str = Field(..., description="Search terms: library name, algorithm, or topic.")
    max_results: int = Field(
        10, ge=1, le=50, description="Maximum number of repositories to return (1-50)."
    )


class GitHubCodeSearchTool(BaseTool):
    """Search GitHub for reference implementations and libraries."""

    name: str = "github_search"
    description: str = (
        "Search GitHub for reference implementations, libraries, and code related to "
        "the research topic. Returns repository name, description, language, stars, "
        "and URL. Works without a token at a lower rate limit; set GITHUB_TOKEN for "
        "a higher rate limit."
    )
    args_schema: type[BaseModel] = GitHubSearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "empty query"})
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 10
        n = max(1, min(50, n))

        headers = {"Accept": "application/vnd.github+json"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = requests.get(
                GITHUB_SEARCH_BASE,
                params={"q": query.strip(), "per_page": str(n), "sort": "stars", "order": "desc"},
                headers=headers,
                timeout=GITHUB_TIMEOUT,
            )
            resp.raise_for_status()
            items = resp.json().get("items", []) or []
            records = [{
                "name": repo.get("full_name", ""),
                "description": repo.get("description", "") or "",
                "language": repo.get("language", "") or "",
                "stars": repo.get("stargazers_count", 0),
                "url": repo.get("html_url", ""),
            } for repo in items]
            return json.dumps(records, indent=2)
        except requests.RequestException as exc:
            logger.warning("GitHubSearch network error: %s", exc)
            return json.dumps({"error": f"GitHubSearch network error: {exc}"})
        except Exception as exc:
            logger.warning("GitHubSearch tool failed: %s", exc)
            return json.dumps({"error": f"GitHubSearch tool failed: {exc}"})


github_code_search_tool = GitHubCodeSearchTool()

# ===========================================================================
# L7 — AudioTranscriptTool
# ===========================================================================
# Local transcription via faster-whisper. Imported lazily inside _run so
# tools.py keeps importing cleanly when the (optional, not-in-requirements)
# package isn't installed — degrades to a JSON {"error": ...} instead of
# an ImportError at module load time.

AUDIO_HTTP_TIMEOUT: int = 60
_whisper_model_cache: dict[str, Any] = {}


class AudioTranscriptInput(BaseModel):
    """Inputs for AudioTranscriptTool."""

    source: str = Field(..., description="Local audio file path or a direct audio URL (mp3/wav/m4a).")
    language: Optional[str] = Field(None, description="Optional ISO-639-1 language hint, e.g. 'en'.")


def _resolve_audio_source(source: str) -> tuple[str, bool]:
    """Return a local file path for *source* and a boolean indicating if it's a temp file.

    If source is a URL, downloads it to a temp file and returns (path, True).
    If source is a local file, returns (path, False).
    Downloads are the caller's responsibility to clean up.
    """
    if source.startswith("http://") or source.startswith("https://"):
        import tempfile
        resp = requests.get(source, timeout=AUDIO_HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()
        suffix = os.path.splitext(source.split("?")[0])[-1] or ".mp3"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return path, True
    if not os.path.isfile(source):
        raise FileNotFoundError(f"audio file not found: {source}")
    return source, False


def _get_whisper_model():
    """Process-local cache for the faster-whisper model (loading it is expensive)."""
    from faster_whisper import WhisperModel
    cached = _whisper_model_cache.get("base")
    if cached is not None:
        return cached
    model = WhisperModel("base", device="cpu", compute_type="int8")
    _whisper_model_cache["base"] = model
    return model


class AudioTranscriptTool(BaseTool):
    """Transcribe an audio file or URL into text using a local Whisper model."""

    name: str = "transcribe_audio"
    description: str = (
        "Transcribe an audio file (local path) or audio URL into text using a local "
        "Whisper model (faster-whisper). Returns the transcript with per-segment "
        "timestamps. Use for podcasts, lectures, or recorded talks. Requires the "
        "faster-whisper package to be installed."
    )
    args_schema: type[BaseModel] = AudioTranscriptInput

    def _run(self, source: str, language: Optional[str] = None) -> str:
        if not isinstance(source, str) or not source.strip():
            return json.dumps({"error": "empty source"})
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            return json.dumps({
                "error": "faster-whisper is not installed. Install it with "
                         "'pip install faster-whisper' to enable audio transcription."
            })

        path = None
        is_temp = False
        try:
            path, is_temp = _resolve_audio_source(source.strip())
            model = _get_whisper_model()
            segments, info = model.transcribe(path, language=language)
            seg_list = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments
            ]
            transcript = " ".join(s["text"] for s in seg_list)
            return json.dumps({
                "source": source,
                "language": info.language,
                "transcript": transcript,
                "segments": seg_list,
            }, indent=2)
        except requests.RequestException as exc:
            logger.warning("AudioTranscript network error: %s", exc)
            return json.dumps({"error": f"AudioTranscript network error: {exc}"})
        except Exception as exc:
            logger.warning("AudioTranscript tool failed: %s", exc)
            return json.dumps({"error": f"AudioTranscript tool failed: {exc}"})
        finally:
            # Clean up downloaded temp files, but not local files the user provided.
            if is_temp and path:
                try:
                    os.remove(path)
                except Exception as cleanup_exc:
                    logger.warning("Failed to clean up temp audio file %s: %s", path, cleanup_exc)


audio_transcript_tool = AudioTranscriptTool()


# ---------------------------------------------------------------------------
# Web search: Ollama hosted search API + fallback chain
# ---------------------------------------------------------------------------
# `WebSearchTool` in app.py calls these.  They live here (not in app.py) so
# they can be unit-tested without importing the whole Streamlit app.

OLLAMA_SEARCH_PATH = "/api/web_search"
OLLAMA_SEARCH_TIMEOUT = 20
OLLAMA_SEARCH_MAX_RESULTS = 10  # API-side cap

# Placeholder titles that the app's search backends return (instead of
# raising) when they find nothing or fail.  Used to detect "this backend
# did not really work, try the next one".
_SEARCH_FAILURE_TITLES = {
    "No results found",
    "Search error",
    "Search temporarily unavailable",
}


def ollama_web_search(
    query: str,
    api_key: str,
    base_url: str = "https://ollama.com",
    max_results: int = 5,
) -> str:
    """Search via Ollama's hosted web search API (POST /api/web_search).

    Returns a JSON string: a list of ``{title, link, snippet}`` dicts, the
    same shape the other web-search backends produce.  Raises
    ``RuntimeError`` on a missing key, HTTP/network error, or malformed
    response; the message never contains the API key.
    """
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set")
    max_results = max(1, min(int(max_results), OLLAMA_SEARCH_MAX_RESULTS))
    try:
        resp = requests.post(
            base_url.rstrip("/") + OLLAMA_SEARCH_PATH,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": max_results},
            timeout=OLLAMA_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        raise RuntimeError(f"Ollama web search returned HTTP {status}") from None
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Ollama web search failed: {type(e).__name__}") from None

    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Ollama web search returned an unexpected response")
    results = [
        {
            "title": it.get("title") or "No title",
            "link": it.get("url") or "",
            "snippet": it.get("content") or "No snippet available",
        }
        for it in items[:max_results]
        if isinstance(it, dict)
    ]
    return json.dumps(results, indent=2)


def search_result_is_failure(raw: str) -> bool:
    """True if a backend's JSON output is empty or a failure placeholder."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return True
    if not isinstance(data, list) or not data:
        return True
    first = data[0]
    return isinstance(first, dict) and first.get("title") in _SEARCH_FAILURE_TITLES


def search_with_fallback(query: str, backends: list[tuple[str, Any]]) -> str:
    """Try each ``(name, callable)`` backend in order; return the first good result.

    A backend fails if it raises or returns an empty/placeholder result.
    If every backend fails, returns a single "No results found" placeholder
    that names the backends tried.
    """
    tried: list[str] = []
    for name, fn in backends:
        tried.append(name)
        try:
            raw = fn(query)
        except Exception as e:  # noqa: BLE001 - any backend failure -> next
            logger.warning("web search backend %s failed: %s", name, type(e).__name__)
            continue
        if not search_result_is_failure(raw):
            return raw
        logger.info("web search backend %s returned no usable results", name)
    return json.dumps([{
        "title": "No results found",
        "link": "",
        "snippet": (
            f'No web results found for "{query}" '
            f"(tried: {', '.join(tried) or 'none'}). Try a different query."
        ),
    }], indent=2)
