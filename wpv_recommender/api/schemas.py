"""Pydantic models for API request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    num_pages: int = Field(..., description="Number of pages in the index")


class RecommendationItem(BaseModel):
    """A single recommendation item."""

    page_title: str = Field(..., description="Wikipedia page title")
    similarity: float = Field(..., description="Similarity score (0-1)")


class RecommendationsResponse(BaseModel):
    """Response for recommendations endpoint."""

    query_page: str = Field(..., description="The queried page title")
    recommendations: list[RecommendationItem] = Field(
        ..., description="List of similar pages"
    )


class SearchResult(BaseModel):
    """A single search result."""

    page_title: str = Field(..., description="Wikipedia page title")


class SearchResponse(BaseModel):
    """Response for search endpoint."""

    query: str = Field(..., description="The search query")
    results: list[SearchResult] = Field(..., description="List of matching pages")
    total: int = Field(..., description="Total number of results returned")


class PageEmbeddingResponse(BaseModel):
    """Response for page embedding endpoint."""

    page_title: str = Field(..., description="Wikipedia page title")
    embedding: list[float] = Field(..., description="Page embedding vector")
    embedding_dim: int = Field(..., description="Embedding dimension")


class PaginatedPagesResponse(BaseModel):
    """Response for paginated pages list."""

    pages: list[str] = Field(..., description="List of page titles")
    total: int = Field(..., description="Total number of pages")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Number of pages per page")
    has_more: bool = Field(..., description="Whether there are more pages")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
