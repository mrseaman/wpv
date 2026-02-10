"""Model components for the CNN Autoencoder."""

from .cnn_ae import CNNAutoencoder

__all__ = ["CNNAutoencoder", "EmbeddingStore"]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "EmbeddingStore":
        from .embedding_store import EmbeddingStore
        return EmbeddingStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
