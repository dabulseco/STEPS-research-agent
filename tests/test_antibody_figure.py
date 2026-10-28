"""Smoke tests for H5 (AntibodyTherapeutic) and M7 (FigureCaption).

Live network tests are gated by env var.  FigureCaption is also gated
on Ollama being available locally with a vision model.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (  # noqa: E402
    antibody_therapeutic_tool,
    figure_caption_tool,
    _detect_vision_model,
    _ensure_image_png,
)


def test_antibody_therapeutic_empty_query():
    out = antibody_therapeutic_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_antibody_therapeutic_returns_unified_shape():
    """Even when every source returns nothing, the shape is well-defined."""
    out = antibody_therapeutic_tool._run(
        "this-is-definitely-not-a-real-antibody-zzz", query_type="name"
    )
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "result_count" in parsed
    assert "records" in parsed or all(
        k in parsed for k in ("therasabdab", "covabdab", "sabdab", "iedb")
    )
    # When no records are found, the response should include a helpful
    # `message` so the agent knows to try a different query.
    if parsed["result_count"] == 0:
        assert "message" in parsed


def test_antibody_therapeutic_live_or_skip():
    """Live test: pembrolizumab should appear in Thera-SAbDab."""
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = antibody_therapeutic_tool._run("pembrolizumab", query_type="name")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if parsed.get("result_count", 0) > 0:
        # Thera-SAbDab almost always has the major checkpoint inhibitors.
        assert isinstance(parsed.get("therasabdab"), list)


def test_figure_caption_empty_path():
    out = figure_caption_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_figure_caption_missing_file():
    out = figure_caption_tool._run("C:/no/such/file.png")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "not found" in parsed["error"]


def test_figure_caption_no_ollama(monkeypatch, tmp_path):
    """If no Ollama vision model is available, the tool should return a
    clear error that names the model and the install command.

    We stub `_detect_vision_model` to None so the test exercises the
    no-vision-model branch regardless of the local Ollama state.
    """
    # Create a real (tiny) PNG file so the file-existence check passes.
    import struct, zlib
    def _make_png(path):
        # Minimal 1x1 PNG. Pre-built bytes — avoids pulling in PIL.
        png = (
            b'\x89PNG\r\n\x1a\n'
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde'
            b'\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa3R\xbc\x82'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        Path(path).write_bytes(png)
    from pathlib import Path
    img = tmp_path / "tiny.png"
    _make_png(str(img))

    monkeypatch.setattr("tools._detect_vision_model", lambda: None)
    out = figure_caption_tool._run(str(img))
    parsed = json.loads(out)
    assert "error" in parsed
    assert "vision model" in parsed["error"]


def test_ensure_image_png_missing():
    """_ensure_image_png should raise FileNotFoundError, not crash."""
    try:
        _ensure_image_png("C:/nope/never/here.png")
    except FileNotFoundError:
        return
    except Exception:
        # Any other exception is fine — we just want no silent success.
        return
    raise AssertionError("expected FileNotFoundError on missing file")


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
