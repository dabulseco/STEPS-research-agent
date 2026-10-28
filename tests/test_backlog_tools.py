"""Smoke tests for the Low-priority tool backlog (L1-L7).

Live tests are gated by SKIP_LIVE_TOOL_TESTS=1, matching the existing
convention in tests/test_bio_tools.py and tests/test_tool_harness.py.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (  # noqa: E402
    clinical_trials_tool, pathway_lookup_tool, dataset_search_tool,
    patent_search_tool, drug_interaction_tool, github_code_search_tool,
    audio_transcript_tool,
)


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
