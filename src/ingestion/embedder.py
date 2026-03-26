from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import track

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk

console = Console()


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def _loaded_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            console.print(f"[dim]loading embedding model {self.model_name}...[/dim]")
            self._model = SentenceTransformer(self.model_name)
            console.print(f"[green]embedding model ready[/green]")
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        embeddings: list[list[float]] = []

        for batch in track(batches, description="embedding..."):
            vectors = self._loaded_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.extend(vectors.tolist())

        return embeddings

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int = 32
    ) -> list[list[float]]:
        return self.embed_texts([c.text for c in chunks], batch_size)

    def embed_query(self, query: str) -> list[float]:
        vectors = self._loaded_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()
