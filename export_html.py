"""Markdown -> standalone, sanitized HTML export for research output.

The research output is LLM-written and can contain text scraped from the
web, so raw HTML in it is untrusted.  `markdown_to_html` therefore renders
the markdown, then runs the result through an allowlist sanitizer
(BeautifulSoup) before wrapping it in a self-contained page that also
carries a restrictive Content-Security-Policy as defence in depth.

Kept out of `app.py` so it can be unit-tested without importing Streamlit.
"""
from __future__ import annotations

from html import escape
from urllib.parse import urlparse

import markdown
from bs4 import BeautifulSoup, Comment

_ALLOWED_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "hr", "blockquote",
    "ul", "ol", "li", "strong", "b", "em", "i", "u", "s", "del", "sub", "sup",
    "code", "pre", "a", "img", "table", "thead", "tbody", "tfoot", "tr", "th",
    "td", "span", "div", "dl", "dt", "dd",
}
# Tags whose *content* must be dropped, not just the tag itself.
_DROP_WITH_CONTENT = {
    "script", "style", "iframe", "object", "embed", "form", "textarea",
    "select", "option", "button", "noscript", "template", "svg", "math",
    "head", "title", "meta", "link", "base", "frame", "frameset",
}
_ALLOWED_ATTRS = {
    "a": {"href", "title"},
    "img": {"src", "alt", "title"},
    "th": {"align"},
    "td": {"align"},
    "code": {"class"},
}
_SAFE_LINK_SCHEMES = {"http", "https", "mailto"}
_SAFE_IMG_SCHEMES = {"http", "https"}

_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; "
    "img-src https: http: data:; base-uri 'none'; form-action 'none'"
)

_CSS = """
:root { color-scheme: light dark; }
body { font-family: Georgia, 'Times New Roman', serif; line-height: 1.6;
       max-width: 46rem; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
       background: #fff; }
h1, h2, h3, h4 { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
                 line-height: 1.25; margin-top: 2rem; }
h1 { border-bottom: 1px solid #ccc; padding-bottom: .3rem; }
a { color: #0b57d0; }
code { background: rgba(127,127,127,.15); padding: .1em .3em; border-radius: 3px;
       font-size: .9em; }
pre { background: rgba(127,127,127,.12); padding: .8rem; overflow-x: auto;
      border-radius: 4px; }
pre code { background: none; padding: 0; }
blockquote { margin: 1rem 0; padding: .1rem 1rem; border-left: 4px solid #bbb;
             color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #bbb; padding: .4rem .6rem; text-align: left; }
th { background: rgba(127,127,127,.15); }
img { max-width: 100%; height: auto; }
@media (prefers-color-scheme: dark) {
  body { background: #161616; color: #e6e6e6; }
  a { color: #8ab4f8; }
  blockquote { color: #aaa; border-left-color: #666; }
  th, td, h1 { border-color: #555; }
}
@media print { body { max-width: none; margin: 0; } }
"""


def _scheme_ok(url: str, allowed: set[str]) -> bool:
    """True for relative/fragment URLs and URLs with an allowlisted scheme."""
    # Browsers ignore embedded whitespace/control chars in schemes
    # ("java\tscript:"), so strip them before parsing.
    cleaned = "".join(ch for ch in url if ch.isprintable() and not ch.isspace())
    scheme = urlparse(cleaned).scheme.lower()
    return scheme == "" or scheme in allowed


def sanitize_html(fragment: str) -> str:
    """Return *fragment* reduced to a safe allowlist of tags/attributes."""
    soup = BeautifulSoup(fragment, "html.parser")

    for node in soup.find_all(string=lambda t: isinstance(t, Comment)):
        node.extract()

    for tag in soup.find_all(True):
        if tag.name is None:  # already removed along with a dropped ancestor
            continue
        name = tag.name.lower()
        if name in _DROP_WITH_CONTENT:
            tag.decompose()
            continue
        if name not in _ALLOWED_TAGS:
            tag.unwrap()  # keep the text, drop the unknown wrapper
            continue

        allowed = _ALLOWED_ATTRS.get(name, set())
        for attr in list(tag.attrs):
            if attr.lower() not in allowed:
                del tag.attrs[attr]
        if name == "a":
            href = tag.attrs.get("href")
            if href is not None and not _scheme_ok(str(href), _SAFE_LINK_SCHEMES):
                del tag.attrs["href"]
            elif href is not None:
                tag.attrs["rel"] = "noopener noreferrer"
        elif name == "img":
            src = tag.attrs.get("src")
            if src is None or not _scheme_ok(str(src), _SAFE_IMG_SCHEMES):
                tag.decompose()
    return str(soup)


def markdown_to_html(markdown_text: str, title: str = "Research Output") -> str:
    """Render *markdown_text* to a complete, standalone, sanitized HTML page."""
    body = markdown.markdown(
        markdown_text or "",
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )
    body = sanitize_html(body)
    safe_title = escape(title)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">\n'
        f"<title>{safe_title}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        f"<main>\n{body}\n</main>\n"
        "</body>\n</html>\n"
    )
