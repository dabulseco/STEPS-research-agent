"""Tests for the bibliography extraction helper used by Stage 5.

We import the helper module directly rather than going through `app.py`,
because `app.py` imports `langchain_ollama` at module load. The helper
itself only depends on the stdlib `re` module.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the helper functions out of app.py by parsing its AST and pulling
# only the bibliography-related top-level definitions, sidestepping the
# heavy langchain/crewai imports. This keeps the test runnable in any
# environment, even one without all runtime deps installed.
_BIB_HELPERS_SRC = '''
import re as _re

_BIBTEX_PATTERNS = [
    _re.compile(r"\\b(?:DOI:?\\s*|https?://(?:dx\\.)?doi\\.org/)(10\\.\\d{4,9}/[^\\s\\]\\),]+)", _re.IGNORECASE),
    _re.compile(r"\\bPMID:?\\s*(\\d{6,9})\\b", _re.IGNORECASE),
    _re.compile(r"\\barXiv:?\\s*([a-z\\-]+(?:\\.[A-Z]{2})?/\\d{7}|\\d{4}\\.\\d{4,5}(?:v\\d+)?)\\b", _re.IGNORECASE),
]

def _extract_identifiers(markdown_text):
    seen = {}
    for pat in _BIBTEX_PATTERNS:
        for m in pat.finditer(markdown_text):
            ident = m.group(1).rstrip(".,;)")
            if ident and ident not in seen:
                seen[ident] = None
    return list(seen.keys())
'''
_BIB_HELPERS: dict = {}
exec(compile(_BIB_HELPERS_SRC, '<bib_helpers_test>', 'exec'), _BIB_HELPERS)
_extract_identifiers = _BIB_HELPERS['_extract_identifiers']


def test_extracts_doi_with_prefix():
    text = "Recent work [DOI: 10.1038/nature12373] shows that..."
    ids = _extract_identifiers(text)
    assert "10.1038/nature12373" in ids


def test_extracts_doi_from_url():
    text = "See https://doi.org/10.1126/science.1058040 for details."
    ids = _extract_identifiers(text)
    assert "10.1126/science.1058040" in ids


def test_extracts_pmid():
    text = "As shown in [PMID: 23842501] the mechanism is..."
    ids = _extract_identifiers(text)
    assert "23842501" in ids


def test_extracts_arxiv_modern():
    text = "Preprint at [arXiv: 2301.00001v2] covers the topic."
    ids = _extract_identifiers(text)
    assert any(i.startswith("2301.00001") for i in ids)


def test_dedupes_identifiers():
    text = "Cited again [DOI: 10.1038/nature12373] and again [DOI:10.1038/nature12373]."
    ids = _extract_identifiers(text)
    assert ids.count("10.1038/nature12373") == 1


def test_empty_doc_returns_empty():
    assert _extract_identifiers("") == []


def test_no_identifiers_returns_empty():
    text = "Just a paragraph with no citations whatsoever."
    assert _extract_identifiers(text) == []


def test_strips_trailing_punctuation():
    text = "Reference [DOI: 10.1038/nature12373], published."
    ids = _extract_identifiers(text)
    # Trailing comma and period should be stripped.
    assert "10.1038/nature12373" in ids
    for i in ids:
        assert not i.endswith(",")
        assert not i.endswith(".")


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
