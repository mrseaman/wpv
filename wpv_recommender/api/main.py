"""FastAPI application for Wikipedia page recommendations."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_app_state
from .routes import router
from ..config import get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print("Starting Wikipedia Pageview Recommender API...")
    state = get_app_state()
    config = get_config()

    try:
        state.load(config.paths)
        print(f"Loaded {len(state.embedding_store)} page embeddings")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("API will start in degraded mode.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("API will start in degraded mode.")

    yield

    # Shutdown
    print("Shutting down...")
    state.unload()


# Create FastAPI app
app = FastAPI(
    title="Wikipedia Pageview Recommender",
    description=(
        "A recommendation API that finds similar Wikipedia pages based on "
        "their pageview patterns using a Transformer Autoencoder."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs."""
    return {
        "message": "Wikipedia Pageview Recommender API",
        "docs": "/docs",
        "health": "/health",
    }
