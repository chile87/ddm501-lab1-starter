"""
Pydantic schemas for request/response validation.

These models define the public contract of the API and drive the Swagger
documentation that FastAPI generates at /docs. Every field carries a
description and every model carries an example, so the generated docs are
usable without reading the source.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field

# Maximum number of pairs accepted by /predict/batch. Keeps a single request
# from tying up the worker for an unbounded amount of time.
MAX_BATCH_SIZE = 100

# Rating bounds of the MovieLens dataset the model was trained on.
MIN_RATING = 1.0
MAX_RATING = 5.0


# =============================================================================
# Prediction
# =============================================================================
class PredictionRequest(BaseModel):
    """Request schema for the /predict endpoint."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"user_id": "196", "movie_id": "242"}}
    )

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="ID of the user to predict a rating for.",
    )
    movie_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="ID of the movie being rated.",
    )


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    model_config = ConfigDict(
        protected_namespaces=(),
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "user_id": "196",
                "movie_id": "242",
                "predicted_rating": 3.64,
                "model_version": "1.0.0",
            }
        },
    )

    user_id: str = Field(..., description="The user ID from the request.")
    movie_id: str = Field(..., description="The movie ID from the request.")
    predicted_rating: float = Field(
        ...,
        ge=MIN_RATING,
        le=MAX_RATING,
        description="Predicted rating, always between 1.0 and 5.0.",
    )
    model_version: str = Field(..., description="Version of the model that served this prediction.")


# =============================================================================
# Batch prediction
# =============================================================================
class PredictionItem(BaseModel):
    """A single user-movie pair inside a batch request."""

    user_id: str = Field(..., min_length=1, max_length=64, description="ID of the user.")
    movie_id: str = Field(..., min_length=1, max_length=64, description="ID of the movie.")


class BatchPredictionRequest(BaseModel):
    """Request schema for the /predict/batch endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {"user_id": "196", "movie_id": "242"},
                    {"user_id": "186", "movie_id": "302"},
                ]
            }
        }
    )

    predictions: List[PredictionItem] = Field(
        ...,
        min_length = 1,
        max_length=MAX_BATCH_SIZE,
        description=f"User-movie pairs to score (at most {MAX_BATCH_SIZE} per request).",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for the /predict/batch endpoint."""

    predictions: List[PredictionResponse] = Field(
        ..., description="One prediction per item in the request, in the same order."
    )
    total_count: int = Field(..., ge=0, description="Number of predictions returned.")


# =============================================================================
# Health & info
# =============================================================================
class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={"example": {"status": "healthy", "model_loaded": True}},
    )

    status: str = Field(..., description='"healthy" when the model is loaded, otherwise "unhealthy".')
    model_loaded: bool = Field(..., description="Whether the model is loaded in memory.")


class ModelInfoResponse(BaseModel):
    """Response schema for the /model/info endpoint."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "model_version": "1.0.0",
                "model_type": "SVD (Collaborative Filtering)",
                "is_loaded": True,
            }
        },
    )

    model_version: str = Field(..., description="Version of the deployed model.")
    model_type: str = Field(..., description="Algorithm behind the model.")
    is_loaded: bool = Field(..., description="Whether the model is loaded in memory.")


class ErrorResponse(BaseModel):
    """Body returned for handled error responses (503, 500)."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"detail": "Model not loaded"}}
    )

    detail: str = Field(..., description="Human-readable description of the error.")
