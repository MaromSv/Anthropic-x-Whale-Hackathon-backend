"""Retrieval layer for the RAG mockup.

Reads documents straight from the pack registry — same source the FastAPI
endpoints serve — embeds them with a small sentence-transformer, and returns
the top-K most relevant chunks for a user query. Embeddings are cached to
disk so re-runs are fast.

This mirrors what the on-device app will eventually do: filter by tags first
(fast lane), fall back to the full set if the top hit is weak.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from packs.registry import REGISTRY

# Load .env once on import so HF token / other secrets are available
# without leaking into the repo. Silent if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    # transformers + huggingface_hub both look at HF_TOKEN.
    if "HUGGING_FACE_HUB_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
except ImportError:
    pass

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 90 MB, CPU-fast


@dataclass
class IndexedDoc:
    pack_id: str
    title: str
    category: str
    content: str
    tags: list[str]
    priority: str
    source: Optional[str]


@dataclass
class SearchHit:
    doc: IndexedDoc
    score: float


def _collect_docs() -> list[IndexedDoc]:
    """Pull every doc from every pack, via the same builders the API uses."""
    out: list[IndexedDoc] = []
    for pack_id, spec in REGISTRY.items():
        try:
            docs = spec.builder()
        except FileNotFoundError as e:
            print(f"  skipping {pack_id} (data not built yet): {e}")
            continue
        for d in docs:
            out.append(IndexedDoc(
                pack_id=pack_id,
                title=d.title,
                category=d.category,
                content=d.content,
                tags=list(d.tags),
                priority=d.priority,
                source=d.source,
            ))
    return out


def _content_hash(docs: Iterable[IndexedDoc]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.pack_id.encode())
        h.update(d.title.encode())
        h.update(d.content.encode())
    return h.hexdigest()[:16]


class RAGIndex:
    def __init__(self) -> None:
        print("Loading embedding model...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._docs: list[IndexedDoc] = []
        self._embeddings: np.ndarray | None = None

    def build(self) -> None:
        """Compute (or load cached) embeddings for every doc in every pack."""
        print("Collecting documents from packs...")
        self._docs = _collect_docs()
        print(f"  {len(self._docs)} documents loaded")
        if not self._docs:
            raise RuntimeError("No documents available. Run the build scripts first.")

        cache_key = _content_hash(self._docs)
        emb_file = CACHE_DIR / f"emb_{cache_key}.npy"
        meta_file = CACHE_DIR / f"meta_{cache_key}.json"

        if emb_file.exists() and meta_file.exists():
            print(f"  cache hit: {emb_file.name}")
            self._embeddings = np.load(emb_file)
            return

        print(f"  cache miss — embedding {len(self._docs)} docs...")
        texts = [f"{d.title}\n\n{d.content}" for d in self._docs]
        self._embeddings = self._model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(emb_file, self._embeddings)
        meta_file.write_text(json.dumps({"count": len(self._docs)}))
        print(f"  cached → {emb_file.name}")

    def search(
        self,
        query: str,
        k: int = 5,
        tag_filter: Optional[str] = None,
        pack_filter: Optional[str] = None,
    ) -> list[SearchHit]:
        if self._embeddings is None:
            raise RuntimeError("Call .build() first.")

        # Optional pre-filter (the 'fast lane' the app would use)
        candidate_idx = np.arange(len(self._docs))
        if tag_filter:
            candidate_idx = np.array([
                i for i in candidate_idx if tag_filter in self._docs[i].tags
            ])
        if pack_filter:
            candidate_idx = np.array([
                i for i in candidate_idx if self._docs[i].pack_id == pack_filter
            ])
        if len(candidate_idx) == 0:
            return []

        q_emb = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings[candidate_idx] @ q_emb  # cosine, since normalized
        top = np.argsort(-sims)[:k]
        return [SearchHit(doc=self._docs[candidate_idx[i]], score=float(sims[i])) for i in top]
