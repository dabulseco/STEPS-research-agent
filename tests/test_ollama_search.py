"""Tests for the Ollama web search backend and the fallback chain.

All network access is mocked; nothing here hits ollama.com.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

# CRITICAL: env vars must be set before crewai is imported anywhere.
os.environ.setdefault('CREWAI_TELEMETRY_ENABLED', 'false')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
os.environ.setdefault('OPENAI_API_KEY', 'sk-dummy-key-for-local-llm')
os.environ.setdefault('CREWAI_DISABLE_TELEMETRY_PROMPT', 'true')
os.environ.setdefault('CREWAI_STORAGE_DIR', './.crewai_storage')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (  # noqa: E402
    ollama_web_search,
    search_result_is_failure,
    search_with_fallback,
)

SECRET = "test-secret-key-123"


def _response(json_body=None, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


GOOD_BODY = {"results": [
    {"title": "Ollama", "url": "https://ollama.com", "content": "Run LLMs locally."},
    {"title": "Docs", "url": "https://docs.ollama.com", "content": "Documentation."},
]}


# --- ollama_web_search ------------------------------------------------------

def test_maps_fields_and_sends_auth():
    with patch("tools.requests.post", return_value=_response(GOOD_BODY)) as post:
        out = json.loads(ollama_web_search("what is ollama", SECRET))
    assert out[0] == {"title": "Ollama", "link": "https://ollama.com",
                      "snippet": "Run LLMs locally."}
    args, kwargs = post.call_args
    assert args[0] == "https://ollama.com/api/web_search"
    assert kwargs["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert kwargs["json"] == {"query": "what is ollama", "max_results": 5}


def test_custom_base_url_trailing_slash():
    with patch("tools.requests.post", return_value=_response(GOOD_BODY)) as post:
        ollama_web_search("q", SECRET, base_url="https://example.test/")
    assert post.call_args[0][0] == "https://example.test/api/web_search"


def test_max_results_clamped():
    with patch("tools.requests.post", return_value=_response(GOOD_BODY)) as post:
        ollama_web_search("q", SECRET, max_results=99)
        assert post.call_args[1]["json"]["max_results"] == 10
        ollama_web_search("q", SECRET, max_results=0)
        assert post.call_args[1]["json"]["max_results"] == 1


def test_missing_key_raises_without_network():
    with patch("tools.requests.post") as post:
        with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
            ollama_web_search("q", "")
    post.assert_not_called()


def test_http_error_raises_without_leaking_key():
    with patch("tools.requests.post", return_value=_response(status=401)):
        with pytest.raises(RuntimeError) as exc:
            ollama_web_search("q", SECRET)
    assert "401" in str(exc.value)
    assert SECRET not in str(exc.value)


def test_network_error_raises_without_leaking_key():
    err = requests.ConnectionError(f"boom Bearer {SECRET}")
    with patch("tools.requests.post", side_effect=err):
        with pytest.raises(RuntimeError) as exc:
            ollama_web_search("q", SECRET)
    assert SECRET not in str(exc.value)


def test_malformed_response_raises():
    for body in ({"nope": []}, ["x"], None):
        with patch("tools.requests.post", return_value=_response(body)):
            with pytest.raises(RuntimeError, match="unexpected"):
                ollama_web_search("q", SECRET)


# --- search_result_is_failure ----------------------------------------------

@pytest.mark.parametrize("raw", [
    "[]", "not json", "{}", None,
    json.dumps([{"title": "No results found", "link": "", "snippet": "x"}]),
    json.dumps([{"title": "Search error", "link": "", "snippet": "x"}]),
    json.dumps([{"title": "Search temporarily unavailable", "link": "", "snippet": "x"}]),
])
def test_failure_detection_true(raw):
    assert search_result_is_failure(raw) is True


def test_failure_detection_false_for_real_results():
    raw = json.dumps([{"title": "Real", "link": "https://a.b", "snippet": "s"}])
    assert search_result_is_failure(raw) is False


# --- search_with_fallback ---------------------------------------------------

GOOD = json.dumps([{"title": "Real", "link": "https://a.b", "snippet": "s"}])
BAD = json.dumps([{"title": "No results found", "link": "", "snippet": "x"}])


def test_first_good_backend_wins_and_later_not_called():
    second = MagicMock(return_value=GOOD)
    out = search_with_fallback("q", [("a", lambda q: GOOD), ("b", second)])
    assert out == GOOD
    second.assert_not_called()


def test_falls_back_when_first_raises():
    def boom(q):
        raise RuntimeError("down")
    out = search_with_fallback("q", [("a", boom), ("b", lambda q: GOOD)])
    assert out == GOOD


def test_falls_back_when_first_returns_placeholder():
    out = search_with_fallback("q", [("a", lambda q: BAD), ("b", lambda q: GOOD)])
    assert out == GOOD


def test_all_fail_returns_placeholder_naming_backends():
    def boom(q):
        raise RuntimeError("down")
    out = json.loads(search_with_fallback("my query", [("ollama", boom), ("ddg", lambda q: BAD)]))
    assert out[0]["title"] == "No results found"
    assert "ollama, ddg" in out[0]["snippet"]
    assert "my query" in out[0]["snippet"]


def test_no_backends_returns_placeholder():
    out = json.loads(search_with_fallback("q", []))
    assert out[0]["title"] == "No results found"
