"""Model components for the Transformer Autoencoder."""

from .transformer_ae import TransformerAutoencoder
from .positional_encoding import PositionalEncoding

__all__ = ["TransformerAutoencoder", "PositionalEncoding", "EmbeddingStore"]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "EmbeddingStore":
        from .embedding_store import EmbeddingStore
        return EmbeddingStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
