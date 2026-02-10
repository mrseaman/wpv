"""FastAPI application for serving recommendations."""

__all__ = ["app"]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "app":
        from .main import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
