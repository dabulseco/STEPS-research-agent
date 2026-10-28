"""Multimodal extraction tool for the Research Agent.

Extracts text, tables, and figure captions from PDFs and images.
Uses PyMuPDF for text/tables, Ollama vision models for figure captioning,
and Tesseract OCR as fallback for raster-only figures.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import fitz  # PyMuPDF
import pymupdf4llm
import requests
import streamlit as st
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_VISION_MODELS = ["llama3.2-vision", "gemma3:27b-vision", "llava", "llava:13b", "llava:34b"]
FIGURES_DIR = Path("session_figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_installed_vision_models() -> list[str]:
    """Fetch list of locally installed Ollama models that support vision."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return DEFAULT_VISION_MODELS

        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    # Check if it's a known vision model
                    if any(v in model_name.lower() for v in ["vision", "llava", "gemma3"]):
                        models.append(model_name)
        return models if models else DEFAULT_VISION_MODELS
    except Exception:
        return DEFAULT_VISION_MODELS


def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_ollama_vision(model: str, prompt: str, image_b64: str) -> str:
    """Call Ollama vision model with an image and prompt."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 512},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("Ollama vision call failed: %s", exc)
        return f"[Vision model error: {exc}]"


def _ensure_figures_dir(session_id: str) -> Path:
    """Create and return the per-session figures directory."""
    fig_dir = FIGURES_DIR / session_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ---------------------------------------------------------------------------
# Pydantic Input Schema
# ---------------------------------------------------------------------------


class MultimodalExtractInput(BaseModel):
    """Inputs for MultimodalExtractTool."""

    source: str = Field(
        ...,
        description=(
            "Local file path to a PDF or image, or a session source identifier "
            "(e.g., a URL or PDF filename previously added in the sidebar)"
        ),
    )
    extract: list[Literal["text", "tables", "figures"]] = Field(
        default=["text", "tables", "figures"],
        description="Which content types to extract",
    )
    max_figures: int = Field(
        default=20, ge=0, le=200, description="Maximum number of figures to process"
    )
    vision_model: Optional[str] = Field(
        default=None,
        description="Ollama vision model to use (auto-detected if not specified)",
    )


# ---------------------------------------------------------------------------
# Core Extraction Functions
# ---------------------------------------------------------------------------


def _extract_text_pymupdf(pdf_path: str) -> str:
    """Extract full text from PDF using PyMuPDF + pymupdf4llm for markdown."""
    try:
        # Use pymupdf4llm for better markdown output
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except Exception as exc:
        logger.warning("pymupdf4llm failed, falling back to basic PyMuPDF: %s", exc)
        # Fallback to basic text extraction
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)


def _extract_tables_camelot(pdf_path: str) -> list[dict[str, Any]]:
    """Extract tables from PDF using camelot-py."""
    tables = []
    try:
        import camelot

        # Try lattice mode first (for bordered tables)
        lattice_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        for i, table in enumerate(lattice_tables):
            if table.df.shape[0] > 1 and table.df.shape[1] > 1:  # non-trivial
                tables.append(
                    {
                        "page": table.page,
                        "method": "lattice",
                        "accuracy": table.parsing_report.get("accuracy", 0),
                        "markdown": table.df.to_markdown(index=False),
                        "shape": list(table.df.shape),
                    }
                )

        # Try stream mode (for borderless tables)
        stream_tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for i, table in enumerate(stream_tables):
            if table.df.shape[0] > 1 and table.df.shape[1] > 1:
                tables.append(
                    {
                        "page": table.page,
                        "method": "stream",
                        "accuracy": table.parsing_report.get("accuracy", 0),
                        "markdown": table.df.to_markdown(index=False),
                        "shape": list(table.df.shape),
                    }
                )

    except Exception as exc:
        logger.warning("Camelot table extraction failed: %s", exc)

    return tables


def _extract_figures_pymupdf(
    pdf_path: str, figures_dir: Path, max_figures: int
) -> list[dict[str, Any]]:
    """Extract figure images from PDF using PyMuPDF."""
    figures = []
    try:
        doc = fitz.open(pdf_path)
        figure_count = 0

        for page_num in range(len(doc)):
            if figure_count >= max_figures:
                break

            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                if figure_count >= max_figures:
                    break

                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Skip very small images (likely decorative)
                    if pix.width < 100 or pix.height < 100:
                        continue

                    # Convert to RGB if needed
                    if pix.n > 4:  # CMYK or similar
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Save figure
                    fig_filename = f"page{page_num+1}_fig{img_idx+1}.png"
                    fig_path = figures_dir / fig_filename
                    pix.save(str(fig_path))

                    # Get figure caption from nearby text (simple heuristic)
                    caption = _find_caption_near_image(page, img_info)

                    figures.append(
                        {
                            "page": page_num + 1,
                            "index": img_idx + 1,
                            "path": str(fig_path),
                            "width": pix.width,
                            "height": pix.height,
                            "caption_guess": caption,
                        }
                    )
                    figure_count += 1

                except Exception as exc:
                    logger.warning("Failed to extract figure %d on page %d: %s", img_idx, page_num, exc)
                    continue

        doc.close()
    except Exception as exc:
        logger.warning("PyMuPDF figure extraction failed: %s", exc)

    return figures


def _find_caption_near_image(page: fitz.Page, img_info: tuple) -> str:
    """Heuristic: find text near an image that might be a caption."""
    try:
        # Get image bbox
        xref = img_info[0]
        img_rect = page.get_image_bbox(img_info)

        # Search for text below the image (typical caption position)
        # Expand search area below the image
        search_rect = fitz.Rect(
            img_rect.x0 - 50,
            img_rect.y1,
            img_rect.x1 + 50,
            min(img_rect.y1 + 100, page.rect.height),
        )

        text = page.get_text("text", clip=search_rect).strip()

        # Filter for caption-like text
        if text and len(text) > 5 and len(text) < 500:
            # Check for caption keywords
            lower = text.lower()
            if any(kw in lower for kw in ["figure", "fig.", "fig ", "table", "chart", "diagram", "image"]):
                return text[:500]

        return text[:500] if text else ""
    except Exception:
        return ""


def _caption_figure_with_vision(
    figure_path: str,
    vision_model: str,
    context_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a structured caption for a figure using an Ollama vision model."""
    prompt = """Analyze this figure/image from a research document. Provide a structured description with:

1. **Figure Type**: (e.g., line chart, bar chart, scatter plot, microscopy image, diagram, schematic, photograph, table screenshot, flowchart)
2. **Axes/Labels**: What are the axes, labels, or key visual elements?
3. **Key Result**: What is the main finding or data shown?
4. **Take-home Sentence**: One sentence summarizing the figure's significance.

Be specific and concise. If the image is not a data figure (e.g., decorative, logo), note that."""

    if context_hint:
        prompt += f"\n\nContext hint: {context_hint}"

    image_b64 = _encode_image_to_base64(figure_path)
    response = _call_ollama_vision(vision_model, prompt, image_b64)

    # Parse structured response
    figure_type = ""
    axes = ""
    key_result = ""
    take_home = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("figure type") or line.lower().startswith("1."):
            figure_type = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("axes") or line.lower().startswith("2."):
            axes = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("key result") or line.lower().startswith("3."):
            key_result = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("take-home") or line.lower().startswith("4."):
            take_home = line.split(":", 1)[-1].strip()

    return {
        "caption": response,
        "figure_type": figure_type,
        "axes_labels": axes,
        "key_result": key_result,
        "take_home": take_home,
    }


def _ocr_image_tesseract(image_path: str) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as exc:
        logger.warning("Tesseract OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------


class MultimodalExtractTool(BaseTool):
    """Extract structured content from a PDF or image: text, tables, and figure captions."""

    name: str = "multimodal_extract"
    description: str = (
        "Extract structured content from a PDF or image file: full text (as markdown), "
        "tables (as markdown), and figure captions generated by a local vision model. "
        "Provide a local file path or a previously ingested source identifier. "
        "Use this when text-only extraction misses important figures, tables, or diagrams."
    )
    args_schema: type[BaseModel] = MultimodalExtractInput

    def _run(
        self,
        source: str,
        extract: list[Literal["text", "tables", "figures"]] = ["text", "tables", "figures"],
        max_figures: int = 20,
        vision_model: Optional[str] = None,
    ) -> str:
        """Extract multimodal content from the source."""
        # Resolve source: could be a local path or a session resource identifier
        session_id = st.session_state.get("session_id", "default")
        resolved_path = self._resolve_source(source, session_id)

        if not resolved_path or not Path(resolved_path).exists():
            return json.dumps(
                {"error": f"Source not found: {source}. Provide a valid local file path or session resource identifier."}
            )

        # Auto-detect vision model if not provided
        if vision_model is None:
            vision_models = _get_installed_vision_models()
            vision_model = vision_models[0] if vision_models else "llama3.2-vision"

        # Prepare figures directory
        figures_dir = _ensure_figures_dir(session_id)

        results: dict[str, Any] = {
            "source": source,
            "resolved_path": resolved_path,
            "vision_model_used": vision_model,
        }

        # --- Text extraction ---
        if "text" in extract:
            if resolved_path.lower().endswith(".pdf"):
                results["text"] = _extract_text_pymupdf(resolved_path)
            else:
                # For images, use OCR
                results["text"] = _ocr_image_tesseract(resolved_path)

        # --- Table extraction ---
        if "tables" in extract and resolved_path.lower().endswith(".pdf"):
            tables = _extract_tables_camelot(resolved_path)
            results["tables"] = tables
            results["table_count"] = len(tables)
        elif "tables" in extract:
            results["tables"] = []
            results["table_count"] = 0

        # --- Figure extraction + captioning ---
        if "figures" in extract and resolved_path.lower().endswith(".pdf"):
            figures = _extract_figures_pymupdf(resolved_path, figures_dir, max_figures)

            # Caption each figure with vision model
            captioned_figures = []
            for fig in figures:
                caption_result = _caption_figure_with_vision(
                    fig["path"], vision_model, fig.get("caption_guess")
                )
                fig.update(caption_result)
                captioned_figures.append(fig)

            results["figures"] = captioned_figures
            results["figure_count"] = len(captioned_figures)
        elif "figures" in extract and not resolved_path.lower().endswith(".pdf"):
            # Single image - caption it directly
            caption_result = _caption_figure_with_vision(resolved_path, vision_model)
            results["figures"] = [
                {
                    "path": resolved_path,
                    "caption": caption_result["caption"],
                    "figure_type": caption_result["figure_type"],
                    "axes_labels": caption_result["axes_labels"],
                    "key_result": caption_result["key_result"],
                    "take_home": caption_result["take_home"],
                }
            ]
            results["figure_count"] = 1
        else:
            results["figures"] = []
            results["figure_count"] = 0

        # Return JSON string for the agent
        return json.dumps(results, indent=2, default=str)

    def _resolve_source(self, source: str, session_id: str) -> Optional[str]:
        """Resolve a source identifier to a local file path."""
        # If it's already a valid local path, return it
        if Path(source).exists():
            return str(Path(source).resolve())

        # Check if it's a session resource (URL or PDF added in sidebar)
        custom_resources = st.session_state.get("custom_resources", [])
        for resource in custom_resources:
            if resource.get("source") == source or resource.get("title") == source:
                # For URLs, we'd need to re-download; for PDFs, check if we have the file
                # For now, return None to indicate we can't resolve it automatically
                # In a full implementation, we'd cache downloaded files
                pass

        # Check in session figures directory
        fig_dir = FIGURES_DIR / session_id
        if fig_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
                candidate = fig_dir / (source + ext)
                if candidate.exists():
                    return str(candidate)

        return None


# Module-level instance
multimodal_extract_tool = MultimodalExtractTool()


# ---------------------------------------------------------------------------
# Testing helper (can be run standalone)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        tool = MultimodalExtractTool()
        result = tool._run(test_file, extract=["text", "tables", "figures"], max_figures=5)
        print(result)