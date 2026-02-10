"""FAISS-based embedding store for similarity search."""

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class EmbeddingStore:
    """
    Store and search embeddings using FAISS.

    Uses L2 distance for similarity search with optional normalization
    for cosine similarity.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_cosine: bool = True,
    ):
        """
        Initialize the embedding store.

        Args:
            embedding_dim: Dimension of embeddings
            use_cosine: If True, normalize embeddings for cosine similarity
        """
        self.embedding_dim = embedding_dim
        self.use_cosine = use_cosine

        # Initialize FAISS index
        # For cosine similarity, we use IndexFlatIP (inner product) with normalized vectors
        # For L2 distance, we use IndexFlatL2
        if use_cosine:
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)

        # Metadata storage
        self.page_titles: list[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        page_titles: list[str],
    ) -> None:
        """
        Add embeddings to the store.

        Args:
            embeddings: Embedding matrix [num_pages, embedding_dim]
            page_titles: List of page titles corresponding to embeddings
        """
        if len(embeddings) != len(page_titles):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) doesn't match "
                f"number of page titles ({len(page_titles)})"
            )

        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        if self.use_cosine:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Prevent division by zero
            embeddings = embeddings / norms

        # Store raw embeddings and titles
        self.embeddings = embeddings
        self.page_titles = list(page_titles)

        # Add to FAISS index
        self.index.add(embeddings)

        print(f"Added {len(embeddings)} embeddings to the store")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Search for similar pages.

        Args:
            query_embedding: Query embedding [embedding_dim] or [1, embedding_dim]
            k: Number of results to return
            exclude_self: If True, exclude exact matches (for querying by page title)

        Returns:
            List of (page_title, similarity_score) tuples
        """
        query = query_embedding.astype(np.float32)

        # Ensure 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize for cosine similarity
        if self.use_cosine:
            norm = np.linalg.norm(query, keepdims=True)
            query = query / max(norm.item(), 1e-8)

        # Search
        # Request extra results if we need to exclude self
        search_k = k + 1 if exclude_self else k
        distances, indices = self.index.search(query, search_k)

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            page_title = self.page_titles[idx]

            # Convert distance to similarity score
            if self.use_cosine:
                # Inner product is already a similarity score for normalized vectors
                similarity = float(dist)
            else:
                # Convert L2 distance to similarity
                similarity = 1.0 / (1.0 + float(dist))

            # Skip exact matches if requested
            if exclude_self and similarity > 0.9999:
                continue

            results.append((page_title, similarity))

            if len(results) >= k:
                break

        return results

    def search_by_title(
        self,
        page_title: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find similar pages by page title.

        Args:
            page_title: Title of the query page
            k: Number of results to return

        Returns:
            List of (page_title, similarity_score) tuples
        """
        try:
            idx = self.page_titles.index(page_title)
        except ValueError:
            raise ValueError(f"Page '{page_title}' not found in store")

        query_embedding = self.embeddings[idx]
        return self.search(query_embedding, k=k, exclude_self=True)

    def get_embedding(self, page_title: str) -> np.ndarray:
        """
        Get the embedding for a page.

        Args:
            page_title: Title of the page

        Returns:
            Embedding vector [embedding_dim]
        """
        try:
            idx = self.page_titles.index(page_title)
        except ValueError:
            raise ValueError(f"Page '{page_title}' not found in store")

        return self.embeddings[idx]

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """
        Save the embedding store to disk.

        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata (titles, embeddings)
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata = {
            "page_titles": self.page_titles,
            "embeddings": self.embeddings,
            "embedding_dim": self.embedding_dim,
            "use_cosine": self.use_cosine,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"Saved FAISS index to {index_path}")
        print(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "EmbeddingStore":
        """
        Load an embedding store from disk.

        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the metadata

        Returns:
            Loaded EmbeddingStore
        """
        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create store
        store = cls(
            embedding_dim=metadata["embedding_dim"],
            use_cosine=metadata["use_cosine"],
        )

        # Load FAISS index
        store.index = faiss.read_index(str(index_path))
        store.page_titles = metadata["page_titles"]
        store.embeddings = metadata["embeddings"]

        print(f"Loaded {len(store.page_titles)} embeddings from {index_path}")
        return store

    def search_pages(self, query: str, limit: int = 10) -> list[str]:
        """
        Search for pages by title substring.

        Args:
            query: Search query (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of matching page titles
        """
        query_lower = query.lower()
        matches = [
            title for title in self.page_titles if query_lower in title.lower()
        ]
        return matches[:limit]

    def __len__(self) -> int:
        return len(self.page_titles)

    def __contains__(self, page_title: str) -> bool:
        return page_title in self.page_titles
