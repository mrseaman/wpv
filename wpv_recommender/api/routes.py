"""API route definitions."""

from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import get_embedding_store, get_app_state
from .schemas import (
    HealthResponse,
    RecommendationsResponse,
    RecommendationItem,
    SearchResponse,
    SearchResult,
    PageEmbeddingResponse,
    PaginatedPagesResponse,
)
from ..model.embedding_store import EmbeddingStore


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the service status, whether the model is loaded,
    and the number of pages in the index.
    """
    state = get_app_state()

    if not state.is_loaded or state.embedding_store is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            num_pages=0,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        num_pages=len(state.embedding_store),
    )


@router.get(
    "/recommendations/{page_title:path}",
    response_model=RecommendationsResponse,
    tags=["Recommendations"],
)
async def get_recommendations(
    page_title: str,
    k: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    store: EmbeddingStore = Depends(get_embedding_store),
):
    """
    Get recommendations for a Wikipedia page.

    Returns k pages most similar to the given page based on their
    pageview patterns.

    - **page_title**: Wikipedia page title (URL-encoded if necessary)
    - **k**: Number of recommendations to return (1-100)
    """
    if page_title not in store:
        raise HTTPException(
            status_code=404,
            detail=f"Page '{page_title}' not found in the index",
        )

    try:
        results = store.search_by_title(page_title, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    recommendations = [
        RecommendationItem(page_title=title, similarity=score)
        for title, score in results
    ]

    return RecommendationsResponse(
        query_page=page_title,
        recommendations=recommendations,
    )


@router.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_pages(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum results"),
    store: EmbeddingStore = Depends(get_embedding_store),
):
    """
    Search for pages by title.

    Performs a case-insensitive substring search on page titles.

    - **q**: Search query string
    - **limit**: Maximum number of results to return (1-100)
    """
    matches = store.search_pages(q, limit=limit)

    results = [SearchResult(page_title=title) for title in matches]

    return SearchResponse(
        query=q,
        results=results,
        total=len(results),
    )


@router.get(
    "/page/{page_title:path}",
    response_model=PageEmbeddingResponse,
    tags=["Pages"],
)
async def get_page_embedding(
    page_title: str,
    store: EmbeddingStore = Depends(get_embedding_store),
):
    """
    Get the embedding vector for a Wikipedia page.

    Returns the learned embedding representation of the page's
    pageview pattern.

    - **page_title**: Wikipedia page title (URL-encoded if necessary)
    """
    if page_title not in store:
        raise HTTPException(
            status_code=404,
            detail=f"Page '{page_title}' not found in the index",
        )

    try:
        embedding = store.get_embedding(page_title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PageEmbeddingResponse(
        page_title=page_title,
        embedding=embedding.tolist(),
        embedding_dim=len(embedding),
    )


@router.get("/pages", response_model=PaginatedPagesResponse, tags=["Pages"])
async def list_pages(
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of pages"),
    store: EmbeddingStore = Depends(get_embedding_store),
):
    """
    List all pages in the index with pagination.

    Returns a paginated list of all Wikipedia page titles in the index.

    - **offset**: Starting position in the list
    - **limit**: Number of pages to return (1-1000)
    """
    total = len(store)
    pages = store.page_titles[offset : offset + limit]
    has_more = offset + limit < total

    return PaginatedPagesResponse(
        pages=pages,
        total=total,
        offset=offset,
        limit=limit,
        has_more=has_more,
    )
