"""
Pydantic schemas for request/response validation.

TODO: Complete the schema definitions below.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# =============================================================================
# TODO 1: Define PredictionRequest schema
# =============================================================================
# Requirements:
# - user_id: string, required, example "196"
# - movie_id: string, required, example "242"
#
# Hint: Use Field(..., example="value") for required fields with examples

class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    # TODO: Define fields here
    pass


# =============================================================================
# TODO 2: Define PredictionResponse schema
# =============================================================================
# Requirements:
# - user_id: string
# - movie_id: string
# - predicted_rating: float (the predicted rating 1.0-5.0)
# - model_version: string

class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    # TODO: Define fields here
    pass


# =============================================================================
# TODO 3: Define HealthResponse schema
# =============================================================================
# Requirements:
# - status: string (e.g., "healthy", "unhealthy")
# - model_loaded: boolean

class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    # TODO: Define fields here
    pass


# =============================================================================
# BONUS: Define BatchPredictionRequest and BatchPredictionResponse
# =============================================================================
# For batch predictions (optional)

class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""
    user_id: str
    movie_id: str


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    predictions: List[PredictionItem]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    predictions: List[PredictionResponse]
    total_count: int
