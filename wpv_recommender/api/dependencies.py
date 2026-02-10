"""FastAPI dependencies for loading models and stores."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..config import get_config, PathConfig
from ..model.embedding_store import EmbeddingStore


class AppState:
    """Application state holding loaded models and stores."""

    _instance: Optional["AppState"] = None
    _embedding_store: Optional[EmbeddingStore] = None
    _is_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def embedding_store(self) -> Optional[EmbeddingStore]:
        return self._embedding_store

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, paths: Optional[PathConfig] = None) -> None:
        """
        Load the embedding store.

        Args:
            paths: Path configuration (uses default if None)
        """
        if self._is_loaded:
            return

        if paths is None:
            paths = get_config().paths

        index_path = paths.embedding_index_path
        metadata_path = paths.embedding_metadata_path

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Embedding files not found. Expected:\n"
                f"  - {index_path}\n"
                f"  - {metadata_path}\n"
                f"Please run extract_embeddings.py first."
            )

        self._embedding_store = EmbeddingStore.load(index_path, metadata_path)
        self._is_loaded = True

    def unload(self) -> None:
        """Unload the embedding store."""
        self._embedding_store = None
        self._is_loaded = False


@lru_cache()
def get_app_state() -> AppState:
    """Get the application state singleton."""
    return AppState()


def get_embedding_store() -> EmbeddingStore:
    """
    Dependency to get the embedding store.

    Raises:
        RuntimeError: If the store is not loaded
    """
    state = get_app_state()
    if not state.is_loaded or state.embedding_store is None:
        raise RuntimeError(
            "Embedding store not loaded. "
            "Please ensure the application started correctly."
        )
    return state.embedding_store
