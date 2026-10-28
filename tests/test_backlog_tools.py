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
