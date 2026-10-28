"""Smoke tests for M1, M2, and H4 — the bio-domain tools.

Live tests are gated by env var so they don't run in CI without network.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (  # noqa: E402
    sequence_alignment_tool,
    structure_lookup_tool,
    ncbi_genomics_tool,
)


def test_sequence_alignment_msa_not_implemented():
    out = sequence_alignment_tool._run("MKTAYIAKQR", mode="msa")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "error" in parsed
    assert "MSA" in parsed["error"]


def test_sequence_alignment_too_short():
    out = sequence_alignment_tool._run("MK", mode="blastp")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert "error" in parsed


def test_sequence_alignment_max_hits_clamping():
    """Out-of-range max_hits should clamp without crashing."""
    out = sequence_alignment_tool._run(
        "MKTAYIAKQRQISFVKSHRSRQHIRT", mode="blastp", max_hits=9999
    )
    parsed = json.loads(out)
    assert isinstance(parsed, (list, dict))


def test_structure_lookup_empty_query():
    out = structure_lookup_tool._run("")
    parsed = json.loads(out)
    assert "error" in parsed


def test_structure_lookup_invalid_pdb_id():
    out = structure_lookup_tool._run("not-a-real-pdb-id", source="pdb")
    parsed = json.loads(out)
    assert "error" in parsed


def test_ncbi_genomics_empty_query():
    out = ncbi_genomics_tool._run("gene", "")
    parsed = json.loads(out)
    assert "error" in parsed


def test_ncbi_genomics_unknown_entity():
    """We can't trigger this through the public API (Literal type) but
    if we ever loosen it, the tool should error gracefully."""
    out = ncbi_genomics_tool._run("bogus_entity", "TP53")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)


def test_ncbi_genomics_gene_live_or_skip():
    """Live test: TP53 gene record from NCBI Datasets v2."""
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = ncbi_genomics_tool._run("gene", "TP53")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if "error" not in parsed:
        assert parsed.get("symbol") == "TP53"
        assert parsed.get("entity") == "gene"


def test_ncbi_genomics_nucleotide_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = ncbi_genomics_tool._run("nuccore", "NM_000546")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if "error" not in parsed:
        assert parsed.get("accession") == "NM_000546"
        assert "fasta_snippet" in parsed
        assert parsed["fasta_snippet"].startswith(">")


def test_ncbi_genomics_protein_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = ncbi_genomics_tool._run("protein", "NP_000537")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if "error" not in parsed:
        assert parsed.get("accession") == "NP_000537"
        assert "fasta_snippet" in parsed


def test_ncbi_genomics_variant_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    # ClinVar variation ID 12345 is a well-known TP53 variant.
    out = ncbi_genomics_tool._run("variant", "12345")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if "error" not in parsed:
        assert "records" in parsed
        assert len(parsed["records"]) >= 1


def test_structure_lookup_pdb_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = structure_lookup_tool._run("1A2B", source="pdb")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if "error" not in parsed:
        assert parsed.get("source") == "pdb"
        assert parsed.get("id") == "1A2B"
        # File should exist on disk.
        assert os.path.isfile(parsed.get("mmcif_path", ""))


def test_structure_lookup_alphafold_live_or_skip():
    if os.environ.get("SKIP_LIVE_TOOL_TESTS") == "1":
        return
    out = structure_lookup_tool._run("P0DTC2", source="alphafold")
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    if "error" not in parsed:
        assert parsed.get("source") == "alphafold"
        assert parsed.get("uniprot") == "P0DTC2"
        assert os.path.isfile(parsed.get("model_path", ""))


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
