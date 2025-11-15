from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import FAISS_DIR


@dataclass
class Chunk:
    id: str
    text: str
    url: Optional[str] = None


class SimpleRAGIndexer:
    """
    Minimal RAG backbone for ATLAS:
    - Splits a document into chunks
    - Embeds chunks with sentence-transformers
    - Stores them in a FAISS index in memory
    - Can retrieve top-k chunks for a query
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Cosine similarity via inner product on normalized vectors
        self.index = faiss.IndexFlatIP(self.dim)

        self.chunks: List[Chunk] = []

    # --------- Chunking ---------
    def _chunk_text(
        self,
        text: str,
        url: Optional[str] = None,
        min_chunk_chars: int = 200,
    ) -> List[Chunk]:
        # Very simple: split on double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[Chunk] = []
        for p in paragraphs:
            if len(p) < min_chunk_chars:
                continue

            cid = self._make_chunk_id(p, url)
            chunks.append(Chunk(id=cid, text=p, url=url))

        # Fallback: if everything was too short, keep whole text
        if not chunks and text.strip():
            cid = self._make_chunk_id(text, url)
            chunks.append(Chunk(id=cid, text=text.strip(), url=url))

        return chunks

    @staticmethod
    def _make_chunk_id(text: str, url: Optional[str]) -> str:
        raw = (url or "") + text[:200]
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # --------- Index building ---------
    def index_document(self, text: str, url: Optional[str] = None) -> None:
        """
        Split `text` into chunks, embed them, and add to FAISS index.
        """
        new_chunks = self._chunk_text(text, url=url)
        if not new_chunks:
            return

        texts = [c.text for c in new_chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # for cosine similarity
        ).astype("float32")

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

    # --------- Retrieval ---------
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(q_vec, k)
        idxs = indices[0]
        scs = scores[0]

        results: List[Dict[str, Any]] = []
        for rank, (i, score) in enumerate(zip(idxs, scs), start=1):
            if i < 0 or i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "id": chunk.id,
                    "text": chunk.text,
                    "url": chunk.url,
                }
            )

        return results
