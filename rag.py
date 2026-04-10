"""RAG module for Research Agent v3.

Provides ChromaDB-backed document storage, semantic chunking,
embedding-based retrieval, and web result reranking.

Collections:
  session_<id>  — per-session documents (URLs / PDFs added in sidebar)
  rsch_<id>     — per-session research content (plan + research + web findings)
  library       — persistent cross-session document library
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_N_RESULTS = 6

# Hints that identify dedicated embedding models in the Ollama model list
EMBEDDING_MODEL_HINTS = ["nomic-embed", "mxbai-embed", "all-minilm", "embed", "bge-"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _safe_collection_name(name: str) -> str:
    """Sanitise a string so it is a valid ChromaDB collection name.

    Rules: 3-63 chars, start/end alphanumeric, only [a-zA-Z0-9_-].
    """
    import re
    sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Ensure starts with alphanumeric
    if sanitised and not sanitised[0].isalnum():
        sanitised = "c" + sanitised
    # Truncate to 63 chars
    sanitised = sanitised[:63]
    # Pad to minimum 3 chars
    if len(sanitised) < 3:
        sanitised = sanitised.ljust(3, "0")
    return sanitised


def classify_embedding_models(all_models: list[str]) -> list[str]:
    """Return models best suited for embeddings, with dedicated models first."""
    dedicated = [
        m for m in all_models
        if any(hint in m.lower() for hint in EMBEDDING_MODEL_HINTS)
    ]
    others = [m for m in all_models if m not in dedicated]
    return dedicated + others


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------

class DocumentStore:
    """ChromaDB-backed store for semantic chunking, retrieval, and reranking."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embeddings = None
        self._client: Optional[object] = None  # chromadb.ClientAPI
        self._splitter = None

    # ------------------------------------------------------------------
    # Lazy-initialisation properties
    # ------------------------------------------------------------------

    @property
    def splitter(self):
        if self._splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._splitter

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_ollama import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url="http://localhost:11434",
            )
        return self._embeddings

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _collection(self, name: str):
        """Get or create a ChromaDB collection (cosine distance metric)."""
        safe = _safe_collection_name(name)
        return self.client.get_or_create_collection(
            name=safe,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        content: str,
        source: str,
        title: str,
        doc_type: str,
        collection_name: str = "session",
    ) -> int:
        """Chunk *content*, embed, and upsert into *collection_name*.

        Returns the number of chunks stored (0 on failure).
        """
        chunks = self.splitter.split_text(content)
        if not chunks:
            return 0
        try:
            embeddings = self.embeddings.embed_documents(chunks)
        except Exception as exc:
            logger.warning("Embedding failed for '%s': %s", source, exc)
            return 0

        coll = self._collection(collection_name)
        ids = [
            hashlib.md5(f"{source}||chunk_{i}".encode()).hexdigest()
            for i in range(len(chunks))
        ]
        metadatas = [
            {
                "source": source,
                "title": title,
                "type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def embed_research_content(
        self,
        research_plan: str,
        initial_research: str,
        web_research: str,
        session_id: str,
    ) -> int:
        """Embed all three research artefacts into the per-session research collection.

        Called once at the start of step 5 to enable per-section retrieval.
        Returns total chunks stored.
        """
        coll_name = f"rsch_{session_id}"
        total = 0

        sources = [
            (research_plan,    "research_plan",    "Research Plan",    "research"),
            (initial_research, "initial_research", "Initial Research", "research"),
            (web_research,     "web_research",     "Web Research",     "research"),
        ]
        for content, source, title, dtype in sources:
            if content and content.strip():
                total += self.add_document(content, source, title, dtype, coll_name)
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        collection_name: str = "session",
        n_results: int = DEFAULT_N_RESULTS,
    ) -> list[dict]:
        """Return top-N relevant chunks from *collection_name*.

        Each result is a dict with keys:
          content, source, title, type, score  (score is cosine similarity 0-1)
        """
        try:
            coll = self._collection(collection_name)
            count = coll.count()
            if count == 0:
                return []
            n = min(n_results, count)
            query_emb = self.embeddings.embed_query(query_text)
            res = coll.query(
                query_embeddings=[query_emb],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            results = []
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                results.append(
                    {
                        "content": doc,
                        "source": meta.get("source", ""),
                        "title": meta.get("title", ""),
                        "type": meta.get("type", ""),
                        "score": round(1.0 - dist, 4),
                    }
                )
            return results
        except Exception as exc:
            logger.warning("Query failed on '%s': %s", collection_name, exc)
            return []

    def query_multi(
        self,
        query_text: str,
        collection_names: list[str],
        n_results: int = DEFAULT_N_RESULTS,
    ) -> list[dict]:
        """Query multiple collections, merge, deduplicate, and return top-N results."""
        all_results: list[dict] = []
        seen: set[str] = set()
        for name in collection_names:
            for r in self.query(query_text, collection_name=name, n_results=n_results):
                key = r["content"][:120]
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:n_results]

    # ------------------------------------------------------------------
    # Web result reranking
    # ------------------------------------------------------------------

    def rerank_web_results(
        self,
        query: str,
        results: list[dict],
        snippet_key: str = "snippet",
    ) -> list[dict]:
        """Rerank *results* by embedding cosine similarity between *query*
        and each result's snippet (or title).

        Returns results sorted best-first. Falls back to original order on error.
        """
        if len(results) <= 1:
            return results
        try:
            query_emb = self.embeddings.embed_query(query)
            texts = [r.get(snippet_key) or r.get("title", "") for r in results]
            text_embs = self.embeddings.embed_documents(texts)
            scored = [
                (r, _cosine_similarity(query_emb, emb))
                for r, emb in zip(results, text_embs)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [r for r, _ in scored]
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            return results

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def clear_collection(self, collection_name: str) -> None:
        """Delete *collection_name* entirely (no-op if it does not exist)."""
        try:
            safe = _safe_collection_name(collection_name)
            self.client.delete_collection(safe)
        except Exception:
            pass

    def clear_session(self, session_id: str) -> None:
        """Remove both the session resource and research collections."""
        self.clear_collection(f"session_{session_id}")
        self.clear_collection(f"rsch_{session_id}")

    def list_documents(self, collection_name: str = "library") -> list[dict]:
        """Return unique source documents stored in *collection_name*."""
        try:
            coll = self._collection(collection_name)
            if coll.count() == 0:
                return []
            res = coll.get(include=["metadatas"])
            seen: dict[str, dict] = {}
            for meta in res["metadatas"]:
                src = meta.get("source", "")
                if src and src not in seen:
                    seen[src] = {
                        "source": src,
                        "title": meta.get("title", src),
                        "type": meta.get("type", "unknown"),
                    }
            return list(seen.values())
        except Exception:
            return []

    def remove_document(
        self, source: str, collection_name: str = "library"
    ) -> None:
        """Remove all chunks belonging to *source* from *collection_name*."""
        try:
            coll = self._collection(collection_name)
            res = coll.get(where={"source": source})
            if res["ids"]:
                coll.delete(ids=res["ids"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Availability probe
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if ChromaDB and embeddings are reachable."""
        try:
            _ = self.client  # triggers PersistentClient init
            _ = self.embeddings  # triggers OllamaEmbeddings init
            return True
        except Exception:
            return False
