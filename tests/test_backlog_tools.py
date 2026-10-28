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
