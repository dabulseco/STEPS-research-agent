"""Tests for the markdown -> sanitized standalone HTML export."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_html import markdown_to_html, sanitize_html  # noqa: E402


def test_standalone_document_structure():
    out = markdown_to_html("# Hello\n\nWorld", title="My Paper")
    assert out.startswith("<!DOCTYPE html>")
    assert "<title>My Paper</title>" in out
    assert '<meta charset="utf-8">' in out
    assert "Content-Security-Policy" in out
    assert "<h1>Hello</h1>" in out
    assert "<p>World</p>" in out


def test_title_is_escaped():
    out = markdown_to_html("x", title="<script>alert(1)</script>")
    assert "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>" in out


def test_renders_common_constructs():
    md = (
        "## Section\n\n**bold** and *italic* and `code`\n\n"
        "- one\n- two\n\n1. a\n2. b\n\n"
        "| h1 | h2 |\n|----|----|\n| c1 | c2 |\n\n"
        "```\nblock\n```\n\n> quoted\n\n[link](https://example.com)"
    )
    out = markdown_to_html(md)
    for fragment in ("<h2>Section</h2>", "<strong>bold</strong>", "<em>italic</em>",
                     "<code>code</code>", "<ul>", "<ol>", "<table>", "<th>h1</th>",
                     "<pre>", "<blockquote>", 'href="https://example.com"'):
        assert fragment in out, fragment


def test_empty_and_none_input():
    assert "<main>" in markdown_to_html("")
    assert "<main>" in markdown_to_html(None)


def test_unicode_preserved():
    assert "café — naïve ✓" in markdown_to_html("café — naïve ✓")


@pytest.mark.parametrize("payload", [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<iframe src='https://evil.test'></iframe>",
    "<svg onload=alert(1)></svg>",
    "<style>body{display:none}</style>",
    "<form action='https://evil.test'><input name=x></form>",
    "<object data='x'></object>",
    "<a href='javascript:alert(1)' onclick='alert(1)'>x</a>",
    "<a href='java\tscript:alert(1)'>x</a>",
    "<a href=' JaVaScRiPt:alert(1)'>x</a>",
    "<a href='data:text/html,<script>alert(1)</script>'>x</a>",
    "<div style='background:url(javascript:alert(1))'>x</div>",
    "<!-- <script>alert(1)</script> -->",
    "[x](javascript:alert(1))",
    "![x](javascript:alert(1))",
])
def test_dangerous_payloads_neutralised(payload):
    out = markdown_to_html(f"before\n\n{payload}\n\nafter")
    body = out.split("<main>")[1].split("</main>")[0].lower()
    assert "<script" not in body
    assert "<iframe" not in body
    assert "<style" not in body
    assert "<form" not in body
    assert "<object" not in body
    assert "<svg" not in body
    assert "onerror" not in body and "onclick" not in body and "onload" not in body
    assert "javascript:" not in body.replace("\t", "").replace(" ", "")
    assert "style=" not in body
    assert "data:text/html" not in body
    assert "before" in body and "after" in body  # surrounding content survives


def test_safe_links_kept_and_hardened():
    out = sanitize_html('<a href="https://example.com" target="_blank">x</a>')
    assert 'href="https://example.com"' in out
    assert 'rel="noopener noreferrer"' in out
    assert "target" not in out


def test_mailto_and_relative_and_fragment_links_kept():
    for href in ("mailto:a@b.co", "/path", "#section", "page.html"):
        assert f'href="{href}"' in sanitize_html(f'<a href="{href}">x</a>')


def test_images_http_https_only():
    assert 'src="https://a.b/i.png"' in sanitize_html('<img src="https://a.b/i.png" alt="a">')
    assert "<img" not in sanitize_html('<img src="data:image/svg+xml,<svg onload=alert(1)>">')
    assert "<img" not in sanitize_html('<img src="file:///etc/passwd">')


def test_unknown_tag_unwrapped_text_kept():
    out = sanitize_html("<custom-tag>keep me</custom-tag>")
    assert "keep me" in out and "custom-tag" not in out
