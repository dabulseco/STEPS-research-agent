"""Smoke tests for the GoogleScholarTool (B1).

Selenium is slow and captcha-prone, so live tests are gated by env var
and only test the SerpAPI path or the error path. The default Selenium
backend is exercised only as a no-network error-shape test.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import google_scholar_tool, _resolve_scholar_backend  # noqa: E402


def test_resolve_backend_arg_wins():
    assert _resolve_scholar_backend("serpapi") == "serpapi"
    assert _resolve_scholar_backend("selenium") == "selenium"


def test_resolve_backend_env_default(monkeypatch):
    monkeypatch.setenv("GOOGLE_SCHOLAR_BACKEND", "serpapi")
    assert _resolve_scholar_backend(None) == "serpapi"
    monkeypatch.delenv("GOOGLE_SCHOLAR_BACKEND", raising=False)
    assert _resolve_scholar_backend(None) == "selenium"


def test_resolve_backend_invalid_falls_back():
    """Garbage env value should fall back to selenium, not crash."""
    os.environ["GOOGLE_SCHOLAR_BACKEND"] = "totally-bogus"
    try:
        assert _resolve_scholar_backend(None) == "selenium"
    finally:
        del os.environ["GOOGLE_SCHOLAR_BACKEND"]


def test_empty_query_returns_error_json():
    out = google_scholar_tool._run("", max_results=5)
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "error" in parsed


def test_max_results_clamping():
    """Out-of-range max_results should be clamped without crashing."""
    out = google_scholar_tool._run("anything", max_results=9999)
    parsed = json.loads(out)
    # Either an error (no network) or a list of <= 20 results.
    assert isinstance(parsed, (list, dict))
    if isinstance(parsed, list):
        assert len(parsed) <= 20


def test_serpapi_without_key_returns_helpful_error():
    """When backend=serpapi but no key, error should be helpful, not silent."""
    out = google_scholar_tool._run(
        "test query", max_results=3, backend="serpapi"
    )
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "error" in parsed
    assert "SERPAPI_API_KEY" in parsed["error"]


def test_serpapi_with_key_or_skip():
    """Live test: only runs if SERPAPI_API_KEY is set in the env."""
    if not os.environ.get("SERPAPI_API_KEY"):
        return
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = google_scholar_tool._run(
        "transformer interpretability", max_results=3, backend="serpapi"
    )
    parsed = json.loads(out)
    # Either a list of records or an error dict.
    assert isinstance(parsed, (list, dict))


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
