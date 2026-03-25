from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from src.ingestion.loaders import Document

_TOKENISER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    return len(_TOKENISER.encode(text))


def _decode_tokens(token_ids: list[int]) -> str:
    return _TOKENISER.decode(token_ids)


def _split_into_segments(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    segments: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        segments.extend(s.strip() for s in sentences if s.strip())

    return segments


def _build_chunks(
    segments: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    chunks: list[str] = []
    current_tokens: list[int] = []

    for segment in segments:
        segment_tokens = _TOKENISER.encode(segment)

        if len(current_tokens) + len(segment_tokens) <= chunk_size:
            current_tokens.extend(segment_tokens)
            continue

        if current_tokens:
            chunks.append(_decode_tokens(current_tokens))

        if len(segment_tokens) > chunk_size:
            for start in range(0, len(segment_tokens), chunk_size - chunk_overlap):
                window = segment_tokens[start : start + chunk_size]
                chunks.append(_decode_tokens(window))
            current_tokens = []
        else:
            overlap_start = max(0, len(current_tokens) - chunk_overlap)
            current_tokens = current_tokens[overlap_start:] + segment_tokens

    if current_tokens:
        chunks.append(_decode_tokens(current_tokens))

    return [c for c in chunks if c.strip()]


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    segments = _split_into_segments(doc.content)
    raw_chunks = _build_chunks(segments, chunk_size, chunk_overlap)

    return [
        Chunk(
            text=text,
            source=doc.source,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            metadata={**doc.metadata, "doc_type": doc.doc_type},
        )
        for i, text in enumerate(raw_chunks)
    ]


def chunk_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    return [
        chunk
        for doc in docs
        for chunk in chunk_document(doc, chunk_size, chunk_overlap)
    ]
