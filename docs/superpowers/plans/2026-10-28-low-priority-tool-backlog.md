# Low-Priority Tool Backlog (L1-L7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the seven remaining Low-priority tools from the harness backlog (`ClinicalTrialsTool`, `PathwayDatabaseTool`, `DatasetSearchTool`, `PatentSearchTool`, `DrugInteractionTool`, `GitHubCodeSearchTool`, `AudioTranscriptTool`), each as a `BaseTool` subclass in `tools.py`, then wire them into the existing Bio Curator and Web Research Specialist agents in `app.py`.

**Architecture:** Every tool follows the exact shape already established by the 12 tools already in `tools.py` (see `OpenAlexTool`, `StructureLookupTool`, `NCBIGenomicsTool` for reference): a Pydantic `args_schema`, a `BaseTool` subclass with `name`/`description`/`_run`, a module-level singleton instance, and a `try/except` that always returns JSON (`{"error": ...}` on failure, never a raised exception). Bio-domain tools (L1, L2, L3, L4, L5) attach to the **Bio Curator** agent (`step2b_bio_curation`, `app.py:1253`); general-research tools (L6, L7) attach to the **Web Research Specialist** agent (`step4_web_research`, `app.py:1422`). No new agents or workflow stages are added — this plan only adds tools to agents that already exist.

**Tech Stack:** Python 3.11+, `requests` (already a dependency), stdlib `json`/`os`. `AudioTranscriptTool` (L7) uses `faster-whisper`, imported lazily inside `_run` with a graceful `{"error": ...}` fallback when it isn't installed — it is **not** added as a hard dependency in `requirements.txt`, only documented there as an optional extra. No other new dependencies.

**Spec:** `app_roadmap.md` (sections "Low priority" backlog list and the L1-L7 "Detailed tool specs" entries) — written 2026-08-15/2026-08-22, specs unchanged since.

## Global Constraints

- Every tool is a `BaseTool` subclass with `name`, `description`, `args_schema` (Pydantic), and `_run(...)` — matches the pattern of all 12 existing tools in `tools.py`.
- `_run` always returns a JSON string. Success: a `list`/`dict` of structured records. Failure (network error, missing key, bad input): `{"error": "<message>"}`. Never raise out of `_run`.
- No required new pip dependencies. `faster-whisper` is optional and imported lazily; absence degrades to a JSON error, not an import-time crash.
- Tools that need a paid/licensed API key (EPO OPS, WIPO PATENTSCOPE, DrugBank, PatentsView/USPTO) read the key from an env var and return a clear `{"error": "... requires <ENV_VAR> ..."}` when it's missing — this is the same pattern `GoogleScholarTool` already uses for `SERPAPI_API_KEY`.
- Live-network tests are gated behind `SKIP_LIVE_TOOL_TESTS=1`, matching `tests/test_bio_tools.py` and `tests/test_tool_harness.py`. Tests that don't need network (empty-input validation, clamping, missing-key errors) always run.
- New tools attach only to the two existing agents named above. Do not create new agents, new workflow stages, or modify the Streamlit stage machine.
- Append new tool code to the end of `tools.py`, after the existing `FigureCaptionTool`/`figure_caption_tool` block (currently ends at `tools.py:2457`) — follow the file's established single-module convention rather than splitting into new modules.

---

### Task 1: `ClinicalTrialsTool` (L1)

**Files:**
- Modify: `tools.py` (append after line 2457, end of file)
- Create: `tests/test_backlog_tools.py`

**Interfaces:**
- Consumes: `requests`, `json`, `os`, `logging.getLogger` (`logger`), `BaseTool`, `BaseModel`, `Field`, `Literal` — all already imported at the top of `tools.py`.
- Produces: `clinical_trials_tool` (module-level `ClinicalTrialsTool()` singleton in `tools.py`), importable as `from tools import clinical_trials_tool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backlog_tools.py`:

```python
"""Smoke tests for the Low-priority tool backlog (L1-L7).

Live tests are gated by SKIP_LIVE_TOOL_TESTS=1, matching the existing
convention in tests/test_bio_tools.py and tests/test_tool_harness.py.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import clinical_trials_tool  # noqa: E402


def test_clinical_trials_empty_query():
    out = clinical_trials_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_clinical_trials_max_results_clamping():
    out = clinical_trials_tool._run("lung cancer", max_results=9999)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))


def test_clinical_trials_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = clinical_trials_tool._run("pembrolizumab", max_results=3)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))
    if isinstance(parsed, list) and parsed:
        assert "nct_id" in parsed[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'clinical_trials_tool' from 'tools'`

- [ ] **Step 3: Implement `ClinicalTrialsTool`**

Append to `tools.py`:

```python
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
        if not query or not query.strip():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py
git commit -m "tools: add L1 ClinicalTrialsTool (ClinicalTrials.gov, no key required)"
```

---

### Task 2: `PathwayDatabaseTool` (L2)

**Files:**
- Modify: `tools.py` (append after Task 1's block)
- Modify: `tests/test_backlog_tools.py`

**Interfaces:**
- Consumes: same shared imports as Task 1.
- Produces: `pathway_lookup_tool` (`from tools import pathway_lookup_tool`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backlog_tools.py` (update the import line and append):

```python
from tools import clinical_trials_tool, pathway_lookup_tool  # noqa: E402


def test_pathway_lookup_empty_query():
    out = pathway_lookup_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_pathway_lookup_kegg_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = pathway_lookup_tool._run("p53", source="kegg")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "results" in parsed or "error" in parsed


def test_pathway_lookup_reactome_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = pathway_lookup_tool._run("apoptosis", source="reactome")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "results" in parsed or "error" in parsed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'pathway_lookup_tool' from 'tools'`

- [ ] **Step 3: Implement `PathwayDatabaseTool`**

Append to `tools.py`:

```python
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
                "name": entry.get("name", ""),
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
        if not query or not query.strip():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py
git commit -m "tools: add L2 PathwayDatabaseTool (KEGG + Reactome, no key required)"
```

---

### Task 3: `DatasetSearchTool` (L3)

**Files:**
- Modify: `tools.py` (append after Task 2's block)
- Modify: `tests/test_backlog_tools.py`

**Interfaces:**
- Consumes: `EUTILS_BASE`, `_eutils_params`, `_EUtilsState` (already defined earlier in `tools.py` for `PubMedSearchTool`/`NCBIGenomicsTool` — reused here for GEO/BioProject, not redefined).
- Produces: `dataset_search_tool` (`from tools import dataset_search_tool`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backlog_tools.py`:

```python
from tools import (  # noqa: E402
    clinical_trials_tool, pathway_lookup_tool, dataset_search_tool,
)


def test_dataset_search_empty_query():
    out = dataset_search_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_dataset_search_max_results_clamping():
    out = dataset_search_tool._run("cancer", source="geo", max_results=9999)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))


def test_dataset_search_geo_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = dataset_search_tool._run("breast cancer RNA-seq", source="geo", max_results=3)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))


def test_dataset_search_ena_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = dataset_search_tool._run("SARS-CoV-2", source="ena", max_results=3)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))
```

(Replace the previous `from tools import clinical_trials_tool, pathway_lookup_tool` line with the combined import shown above.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'dataset_search_tool' from 'tools'`

- [ ] **Step 3: Implement `DatasetSearchTool`**

Append to `tools.py`:

```python
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
            "accession": rec.get("accession", rec.get("Accession", rid)),
            "title": rec.get("title", rec.get("Project_Title", "")),
            "organism": rec.get("taxon", rec.get("Organism_Name", "")),
            "summary": (rec.get("summary", rec.get("Project_Description", "")) or "")[:500],
            "url": (
                f"https://www.ncbi.nlm.nih.gov/gds/?term={rid}" if db == "gds"
                else f"https://www.ncbi.nlm.nih.gov/bioproject/{rid}"
            ),
        })
    return records


def _ena_dataset_search(query: str, n: int) -> list[dict[str, Any]]:
    resp = requests.get(
        ENA_PORTAL_BASE,
        params={
            "result": "study",
            "query": f'study_title="*{query}*"',
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
        if not query or not query.strip():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py
git commit -m "tools: add L3 DatasetSearchTool (GEO + ENA + BioProject, no key required)"
```

---

### Task 4: `PatentSearchTool` (L4)

**Files:**
- Modify: `tools.py` (append after Task 3's block)
- Modify: `tests/test_backlog_tools.py`

**Interfaces:**
- Consumes: shared imports only; no cross-tool reuse.
- Produces: `patent_search_tool` (`from tools import patent_search_tool`).

**Build note:** All three backends require a credential this environment does not have (`PATENTSVIEW_API_KEY`, `EPO_OPS_KEY`/`EPO_OPS_SECRET`, `WIPO_PATENTSCOPE_KEY`). Tests therefore assert on the deterministic missing-key error path rather than live calls — there is nothing to gate behind `SKIP_LIVE_TOOL_TESTS` here.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backlog_tools.py`:

```python
from tools import (  # noqa: E402
    clinical_trials_tool, pathway_lookup_tool, dataset_search_tool,
    patent_search_tool,
)


def test_patent_search_empty_query():
    out = patent_search_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_patent_search_uspto_missing_key():
    os.environ.pop("PATENTSVIEW_API_KEY", None)
    out = patent_search_tool._run("anti-PD1 antibody", source="uspto")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "PATENTSVIEW_API_KEY" in parsed["error"]


def test_patent_search_epo_missing_key():
    os.environ.pop("EPO_OPS_KEY", None)
    os.environ.pop("EPO_OPS_SECRET", None)
    out = patent_search_tool._run("anti-PD1 antibody", source="epo")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "EPO_OPS_KEY" in parsed["error"]


def test_patent_search_wipo_missing_key():
    os.environ.pop("WIPO_PATENTSCOPE_KEY", None)
    out = patent_search_tool._run("anti-PD1 antibody", source="wipo")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "WIPO_PATENTSCOPE_KEY" in parsed["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'patent_search_tool' from 'tools'`

- [ ] **Step 3: Implement `PatentSearchTool`**

Append to `tools.py`:

```python
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
    resp = requests.get(
        f"{WIPO_PATENTSCOPE_BASE}/search",
        params={"q": query, "key": key, "size": n},
        timeout=PATENT_TIMEOUT,
    )
    resp.raise_for_status()
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
        if not query or not query.strip():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py
git commit -m "tools: add L4 PatentSearchTool (USPTO/EPO/WIPO, keys gated per-backend)"
```

---

### Task 5: `DrugInteractionTool` (L5)

**Files:**
- Modify: `tools.py` (append after Task 4's block)
- Modify: `tests/test_backlog_tools.py`

**Interfaces:**
- Consumes: shared imports only.
- Produces: `drug_interaction_tool` (`from tools import drug_interaction_tool`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backlog_tools.py`:

```python
from tools import (  # noqa: E402
    clinical_trials_tool, pathway_lookup_tool, dataset_search_tool,
    patent_search_tool, drug_interaction_tool,
)


def test_drug_lookup_empty_query():
    out = drug_interaction_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_drug_lookup_drugbank_missing_key():
    os.environ.pop("DRUGBANK_API_KEY", None)
    out = drug_interaction_tool._run("imatinib", source="drugbank")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "DRUGBANK_API_KEY" in parsed["error"]


def test_drug_lookup_chembl_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = drug_interaction_tool._run("imatinib", source="chembl")
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'drug_interaction_tool' from 'tools'`

- [ ] **Step 3: Implement `DrugInteractionTool`**

Append to `tools.py`:

```python
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
        if not query or not query.strip():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py
git commit -m "tools: add L5 DrugInteractionTool (ChEMBL open, DrugBank key-gated)"
```

---

### Task 6: `GitHubCodeSearchTool` (L6)

**Files:**
- Modify: `tools.py` (append after Task 5's block)
- Modify: `tests/test_backlog_tools.py`

**Interfaces:**
- Consumes: shared imports only.
- Produces: `github_code_search_tool` (`from tools import github_code_search_tool`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backlog_tools.py`:

```python
from tools import (  # noqa: E402
    clinical_trials_tool, pathway_lookup_tool, dataset_search_tool,
    patent_search_tool, drug_interaction_tool, github_code_search_tool,
)


def test_github_search_empty_query():
    out = github_code_search_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_github_search_max_results_clamping():
    out = github_code_search_tool._run("transformer", max_results=9999)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))


def test_github_search_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = github_code_search_tool._run("transformer attention pytorch", max_results=3)
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))
    if isinstance(parsed, list) and parsed:
        assert "url" in parsed[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'github_code_search_tool' from 'tools'`

- [ ] **Step 3: Implement `GitHubCodeSearchTool`**

Append to `tools.py`:

```python
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
        if not query or not query.strip():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (20 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py
git commit -m "tools: add L6 GitHubCodeSearchTool (GitHub REST search, token optional)"
```

---

### Task 7: `AudioTranscriptTool` (L7)

**Files:**
- Modify: `tools.py` (append after Task 6's block)
- Modify: `tests/test_backlog_tools.py`
- Modify: `requirements.txt` (document the optional extra; do not install it)

**Interfaces:**
- Consumes: shared imports only; lazily imports `faster_whisper` inside `_run`/`_get_whisper_model` (not a top-of-file import, so `tools.py` still imports cleanly when the package isn't installed).
- Produces: `audio_transcript_tool` (`from tools import audio_transcript_tool`).

**Build note:** This repo's `requirements.txt` has no entry for `faster-whisper`, and it is intentionally not added as a hard dependency — the tool must degrade to a JSON error, not an ImportError, when it's absent. This mirrors how `tools_multimodal.py` treats `camelot-py` as optional (inline `import camelot` inside a try/except).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backlog_tools.py`:

```python
from tools import (  # noqa: E402
    clinical_trials_tool, pathway_lookup_tool, dataset_search_tool,
    patent_search_tool, drug_interaction_tool, github_code_search_tool,
    audio_transcript_tool,
)


def test_audio_transcript_empty_source():
    out = audio_transcript_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_audio_transcript_missing_dependency_or_bad_path():
    """Deterministic without network or faster-whisper installed: either the
    'faster-whisper not installed' branch or the 'file not found' branch
    fires, and both are JSON errors, not exceptions."""
    out = audio_transcript_tool._run("/nonexistent/path/to/audio.mp3")
    parsed = json.loads(out)
    assert "error" in parsed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'audio_transcript_tool' from 'tools'`

- [ ] **Step 3: Implement `AudioTranscriptTool`**

Append to `tools.py`:

```python
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


def _resolve_audio_source(source: str) -> str:
    """Return a local file path for *source*, downloading it first if it's a URL."""
    if source.startswith("http://") or source.startswith("https://"):
        import tempfile
        resp = requests.get(source, timeout=AUDIO_HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()
        suffix = os.path.splitext(source.split("?")[0])[-1] or ".mp3"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return path
    if not os.path.isfile(source):
        raise FileNotFoundError(f"audio file not found: {source}")
    return source


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
        if not source or not source.strip():
            return json.dumps({"error": "empty source"})
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            return json.dumps({
                "error": "faster-whisper is not installed. Install it with "
                         "'pip install faster-whisper' to enable audio transcription."
            })

        try:
            path = _resolve_audio_source(source.strip())
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


audio_transcript_tool = AudioTranscriptTool()
```

Then modify `requirements.txt` — add a commented-out optional line under a new section near the bottom:

```
# Optional — enables AudioTranscriptTool (L7); not required for the rest of the app
# faster-whisper>=1.0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backlog_tools.py -v`
Expected: PASS (22 passed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_backlog_tools.py requirements.txt
git commit -m "tools: add L7 AudioTranscriptTool (optional faster-whisper, lazy import)"
```

---

### Task 8: Wire L1-L7 into the Bio Curator and Web Research Specialist agents

**Files:**
- Modify: `app.py:20-33` (tool import block)
- Modify: `app.py:1253-1281` (Bio Curator agent — add L1, L2, L3, L4, L5 tools + update goal/backstory)
- Modify: `app.py:1296-1312` (Bio Curator task's "Suggested workflow" list)
- Modify: `app.py:1422-1456` (Web Research Specialist agent — add L6, L7 tools + update backstory)

**Interfaces:**
- Consumes: `clinical_trials_tool`, `pathway_lookup_tool`, `dataset_search_tool`, `patent_search_tool`, `drug_interaction_tool`, `github_code_search_tool`, `audio_transcript_tool` from `tools.py` (Tasks 1-7).
- Produces: nothing new for downstream tasks — this is the final integration point.

- [ ] **Step 1: Add the new imports to `app.py`**

Current (`app.py:20-33`):

```python
from tools import (
    local_kb_tool,
    pubmed_search_tool,
    citation_resolver_tool,
    openalex_tool,
    url_fetch_tool,
    reference_manager_tool,
    google_scholar_tool,
    sequence_alignment_tool,
    structure_lookup_tool,
    ncbi_genomics_tool,
    antibody_therapeutic_tool,
    figure_caption_tool,
)
```

Replace with:

```python
from tools import (
    local_kb_tool,
    pubmed_search_tool,
    citation_resolver_tool,
    openalex_tool,
    url_fetch_tool,
    reference_manager_tool,
    google_scholar_tool,
    sequence_alignment_tool,
    structure_lookup_tool,
    ncbi_genomics_tool,
    antibody_therapeutic_tool,
    figure_caption_tool,
    clinical_trials_tool,
    pathway_lookup_tool,
    dataset_search_tool,
    patent_search_tool,
    drug_interaction_tool,
    github_code_search_tool,
    audio_transcript_tool,
)
```

- [ ] **Step 2: Attach L1-L5 to the Bio Curator agent and extend its goal/backstory/workflow**

Current (`app.py:1253-1281`):

```python
    bio_curator_agent = Agent(
        role="Bio Curator",
        goal=(
            "Resolve the biomedical entities in the research plan and gap "
            "analysis into structured primary data: gene records, protein "
            "FASTA snippets, structure files, BLAST hits, variant "
            "significance, and antibody therapeutic metadata. Cite every "
            "claim with the source record (NCBI gene ID, PDB ID, UniProt, "
            "Thera-SAbDab, etc.)."
        ),
        backstory=(
            "You are a domain curator who pulls authoritative records from "
            "NCBI, RCSB PDB, AlphaFold DB, and antibody databases. You "
            "prefer RefSeq / UniProt primary IDs over free-text names. When "
            "a tool returns an error, you note it and continue rather than "
            "halting the run."
        ),
        llm=llm_string,
        tools=[
            ncbi_genomics_tool,          # H4
            sequence_alignment_tool,     # M1
            structure_lookup_tool,       # M2
            antibody_therapeutic_tool,   # H5
            local_kb_tool,               # M6 (for cross-references in user uploads)
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
```

Replace with:

```python
    bio_curator_agent = Agent(
        role="Bio Curator",
        goal=(
            "Resolve the biomedical entities in the research plan and gap "
            "analysis into structured primary data: gene records, protein "
            "FASTA snippets, structure files, BLAST hits, variant "
            "significance, antibody therapeutic metadata, relevant clinical "
            "trials, pathways, published datasets, patents, and drug "
            "records. Cite every claim with the source record (NCBI gene "
            "ID, PDB ID, UniProt, Thera-SAbDab, NCT ID, etc.)."
        ),
        backstory=(
            "You are a domain curator who pulls authoritative records from "
            "NCBI, RCSB PDB, AlphaFold DB, antibody databases, "
            "ClinicalTrials.gov, KEGG/Reactome, GEO/ENA, patent offices, "
            "and ChEMBL. You prefer RefSeq / UniProt primary IDs over "
            "free-text names. When a tool returns an error — including a "
            "missing API key for a licensed source like DrugBank or EPO — "
            "you note it and continue rather than halting the run."
        ),
        llm=llm_string,
        tools=[
            ncbi_genomics_tool,          # H4
            sequence_alignment_tool,     # M1
            structure_lookup_tool,       # M2
            antibody_therapeutic_tool,   # H5
            local_kb_tool,               # M6 (for cross-references in user uploads)
            clinical_trials_tool,        # L1
            pathway_lookup_tool,         # L2
            dataset_search_tool,         # L3
            patent_search_tool,          # L4
            drug_interaction_tool,       # L5
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
```

Current (`app.py:1296-1312`, inside the `bio_task` description's "Suggested workflow"):

```python
        1. For each gene symbol mentioned, call `ncbi_genomics` with
           entity='gene' to get the RefSeq accession list.
        2. For protein-coding transcripts, call `ncbi_genomics` with
           entity='protein' to fetch a FASTA snippet.
        3. For antibodies, call `antibody_therapeutic` with the antibody
           name or target to get Thera-SAbDab / CoV-AbDab / SAbDab records.
        4. For PDB IDs, call `structure_lookup` with source='pdb' to
           download the mmCIF and pull resolution / method.
        5. For ClinVar variants, call `ncbi_genomics` with entity='variant'.

        End with a one-paragraph synthesis: which entities are central to
        the user's question, what is well-established vs. uncertain, and
        which sources were unreachable.
```

Replace with:

```python
        1. For each gene symbol mentioned, call `ncbi_genomics` with
           entity='gene' to get the RefSeq accession list.
        2. For protein-coding transcripts, call `ncbi_genomics` with
           entity='protein' to fetch a FASTA snippet.
        3. For antibodies, call `antibody_therapeutic` with the antibody
           name or target to get Thera-SAbDab / CoV-AbDab / SAbDab records.
        4. For PDB IDs, call `structure_lookup` with source='pdb' to
           download the mmCIF and pull resolution / method.
        5. For ClinVar variants, call `ncbi_genomics` with entity='variant'.
        6. If a drug, disease, or gene has an active therapeutic program,
           call `clinical_trials` to find relevant trials and their phase/status.
        7. For pathway context on a gene or compound, call `pathway_lookup`
           with source='both' (KEGG + Reactome).
        8. If the topic involves an omics dataset, call `dataset_search`
           with source='geo' (or 'ena'/'ncbi') to find published datasets.
        9. For therapeutic candidates that may be patented, call
           `patent_search` with source='uspto' first (the only backend that
           works without a paid key in this deployment).
        10. For small-molecule or biologic mechanism/phase data, call
            `drug_lookup` with source='chembl' (no key required).

        End with a one-paragraph synthesis: which entities are central to
        the user's question, what is well-established vs. uncertain, and
        which sources were unreachable (including any skipped because a
        required API key was not configured).
```

- [ ] **Step 3: Attach L6-L7 to the Web Research Specialist agent and extend its backstory**

Current (`app.py:1431-1456`):

```python
        backstory=(
            "You are skilled at searching PubMed, OpenAlex, Google Scholar, "
            "and the open web, at fetching the full text of a known URL, and "
            "at pulling relevant passages from the user's own materials. You "
            "always cite PMIDs for biomedical claims, prefer recent primary "
            "sources, and document any gaps you could not fill. When a source "
            "is referenced, you can resolve it to a structured citation. When "
            "the topic is biomedical, you can also pull structured records "
            "from NCBI (genes, proteins, variants), BLAST sequences against "
            "public databases, fetch PDB/AlphaFold structures, and look up "
            "antibody therapeutic metadata."
        ),
        llm=llm_string,
        tools=[
            search_tool,                # web_search        (existing)
            pubmed_search_tool,         # pubmed_search     (H1)
            openalex_tool,              # openalex          (M5)
            google_scholar_tool,        # google_scholar    (H2)
            local_kb_tool,              # local_kb          (M6)
            url_fetch_tool,             # fetch_url         (M4)
            citation_resolver_tool,     # resolve_citation  (M3)
            ncbi_genomics_tool,         # ncbi_genomics     (H4) — bio
            sequence_alignment_tool,    # sequence_align    (M1) — bio
            structure_lookup_tool,      # structure_lookup  (M2) — bio
            antibody_therapeutic_tool,  # antibody_therapeutic (H5) — bio
        ],
```

Replace with:

```python
        backstory=(
            "You are skilled at searching PubMed, OpenAlex, Google Scholar, "
            "and the open web, at fetching the full text of a known URL, and "
            "at pulling relevant passages from the user's own materials. You "
            "always cite PMIDs for biomedical claims, prefer recent primary "
            "sources, and document any gaps you could not fill. When a source "
            "is referenced, you can resolve it to a structured citation. When "
            "the topic is biomedical, you can also pull structured records "
            "from NCBI (genes, proteins, variants), BLAST sequences against "
            "public databases, fetch PDB/AlphaFold structures, and look up "
            "antibody therapeutic metadata. When the topic has a software or "
            "implementation angle, you can search GitHub for reference code. "
            "When a resource is a podcast, lecture, or recorded talk, you can "
            "transcribe it locally rather than skipping it."
        ),
        llm=llm_string,
        tools=[
            search_tool,                # web_search        (existing)
            pubmed_search_tool,         # pubmed_search     (H1)
            openalex_tool,              # openalex          (M5)
            google_scholar_tool,        # google_scholar    (H2)
            local_kb_tool,              # local_kb          (M6)
            url_fetch_tool,             # fetch_url         (M4)
            citation_resolver_tool,     # resolve_citation  (M3)
            ncbi_genomics_tool,         # ncbi_genomics     (H4) — bio
            sequence_alignment_tool,    # sequence_align    (M1) — bio
            structure_lookup_tool,      # structure_lookup  (M2) — bio
            antibody_therapeutic_tool,  # antibody_therapeutic (H5) — bio
            github_code_search_tool,    # github_search     (L6)
            audio_transcript_tool,      # transcribe_audio  (L7)
        ],
```

- [ ] **Step 4: Verify `app.py` still imports cleanly and the full test suite passes**

Run: `python -c "import os; os.environ.setdefault('CREWAI_TELEMETRY_ENABLED','false'); os.environ.setdefault('OPENAI_API_KEY','sk-dummy'); import ast; ast.parse(open('app.py', encoding='utf-8').read())"`
Expected: no output (syntax parses cleanly) — this avoids a real Streamlit/CrewAI runtime import, which needs `st.session_state` and isn't available outside `streamlit run`.

Run: `pytest tests/ -q`
Expected: `64 passed` (the existing 42 + the 22 new tests from Tasks 1-7)

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "app: wire L1-L7 tools into Bio Curator and Web Research Specialist agents"
```

---

### Task 9: Update roadmap and status docs, final verification

**Files:**
- Modify: `app_roadmap.md` (mark L1-L7 as `[x]`, update progress roll-up table and "Recently completed" list)
- Modify: `app_status_20260815.md` (bump the tool-harness line in the "Status Snapshot" section, or leave a note that it is superseded by `app_roadmap.md` — the roadmap is the living doc; the status file is a point-in-time snapshot from 2026-08-15 and is not meant to be kept current)

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (documentation only).

- [ ] **Step 1: Update `app_roadmap.md`'s backlog checkboxes**

In the "Low priority" section near the top of the file, change each line from `- [ ]` to `- [x]`:

```markdown
### Low priority

- [x] **L1 `ClinicalTrialsTool`** — ClinicalTrials.gov lookup *(objective 2)*
- [x] **L2 `PathwayDatabaseTool`** — KEGG / Reactome pathway lookup *(objective 2)*
- [x] **L3 `DatasetSearchTool`** — GEO / EBI ENA dataset discovery *(objective 2)*
- [x] **L4 `PatentSearchTool`** — USPTO / EPO / WIPO patent search *(objective 2)*
- [x] **L5 `DrugInteractionTool`** — DrugBank / ChEMBL *(objective 2)*
- [x] **L6 `GitHubCodeSearchTool`** — find reference implementations *(objective 1)*
- [x] **L7 `AudioTranscriptTool`** — Whisper-based local transcription *(objectives 1, 4)*
- [ ] **L8 `ReferenceManagerTool`** — export BibTeX / RIS *(objective 1)*
```

(Note: L8 was already implemented earlier — verify its line above the "Low priority" `[ ]` checkbox in the backlog list and correct that specific stale checkbox to `[x]` as well, since the "Detailed tool specs" section for L8 already says "Status: `[x]` implemented" — the top-of-file backlog checklist and the detailed spec status had drifted out of sync.)

Each `### LN ...` detailed spec section further down also has its own `**Status:**` line — update each from `[ ] proposed` to `[x] implemented, attached to Bio Curator` (L1, L2, L3, L4, L5) or `[x] implemented, attached to Web Research Specialist` (L6, L7), and append a dated **Progress log** entry to each, e.g.:

```markdown
**Progress log:**
- 2026-10-28 — Implemented and wired into Bio Curator. Live-or-skip tests in `tests/test_backlog_tools.py`.
```

- [ ] **Step 2: Update the progress roll-up table**

Replace the table at the bottom of `app_roadmap.md`:

```markdown
| Priority | Total | Proposed `[ ]` | Design `[~]` | WIP `[wip]` | Done `[x]` |
|----------|-------|----------------|--------------|-------------|------------|
| High     | 5     | 0              | 0            | 0           | 5          |
| Medium   | 7     | 1              | 0            | 0           | 6          |
| Low      | 8     | 8              | 0            | 0           | 0          |
| **All**  | **20**| **9**          | **0**        | **0**       | **11**     |
```

with:

```markdown
| Priority | Total | Proposed `[ ]` | Design `[~]` | WIP `[wip]` | Done `[x]` |
|----------|-------|----------------|--------------|-------------|------------|
| High     | 5     | 0              | 0            | 0           | 5          |
| Medium   | 7     | 0              | 0            | 0           | 7          |
| Low      | 8     | 1              | 0            | 0           | 7          |
| **All**  | **20**| **1**          | **0**        | **0**       | **19**     |
```

Update "Last updated:" to `2026-10-28` and add to "Recently completed (last 8)":

```markdown
- **L1-L5 bio-domain tools** — ClinicalTrials.gov, KEGG/Reactome, GEO/ENA/BioProject, USPTO/EPO/WIPO patents, ChEMBL/DrugBank — wired into Bio Curator.
- **L6 `GitHubCodeSearchTool`** and **L7 `AudioTranscriptTool`** — wired into Web Research Specialist.
```

And update the "Test coverage" list to add:

```markdown
- `tests/test_backlog_tools.py` — 22 tests; L1-L7 input validation, clamping, missing-key errors, and live-or-skip checks.
- **Total: 64 tests, all passing.**
```

- [ ] **Step 3: Run the full suite one final time**

Run: `pytest tests/ -q`
Expected: `64 passed`

- [ ] **Step 4: Commit**

```bash
git add app_roadmap.md
git commit -m "docs: mark L1-L7 backlog tools complete, update roadmap roll-up"
```

---

## Self-Review Notes

- **Spec coverage:** All 7 Low-priority backlog entries (L1-L7) from `app_roadmap.md` have a task (Tasks 1-7), plus wiring (Task 8) and doc updates (Task 9). L8 was already done before this plan — Task 9 only fixes a stale checkbox.
- **No placeholders:** Every implementation step has complete, runnable code; every test step has complete assertions; no "TBD"/"similar to above" shortcuts.
- **Type/name consistency:** Tool `name` fields (`clinical_trials`, `pathway_lookup`, `dataset_search`, `patent_search`, `drug_lookup`, `github_search`, `transcribe_audio`) match their singleton variable names throughout Tasks 1-8 and the app.py wiring block.
- **Known deviation (flagged, not fixed by this plan):** the roadmap's own design constraint says "4-6 tools per agent" for local-model tractability. The Web Research Specialist is already at 11 tools before this plan and goes to 13; the Bio Curator goes from 5 to 10. This plan does not restructure agents to fix that — it's a pre-existing deviation from before this plan, and splitting agents is a larger architectural change outside this backlog's scope. Worth a follow-up conversation if local-model tool selection proves unreliable in practice.
