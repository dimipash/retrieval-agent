from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb
from rich.console import Console

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk

console = Console()


@dataclass
class SearchResult:
    text: str
    source: str
    score: float
    metadata: dict


def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.source}::{chunk.chunk_index}::{chunk.text[:64]}"
    return hashlib.md5(key.encode()).hexdigest()


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        client = chromadb.PersistentClient(path=persist_dir)
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        console.print(
            f"[green]vector store[/green] '{collection_name}' "
            f"— {self._collection.count()} chunks"
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        self._collection.upsert(
            ids=[_chunk_id(c) for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )
        console.print(
            f"[green]stored[/green] {len(chunks)} chunks → total {self.count()}"
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return [
            SearchResult(
                text=text,
                source=meta.get("source", ""),
                score=round(1 - distance, 4),
                metadata=meta,
            )
            for text, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        return self._collection.count()
