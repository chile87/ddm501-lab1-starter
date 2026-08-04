"""
FastAPI application for Movie Rating Prediction.

Exposes the trained collaborative-filtering (SVD) model over REST:
    GET  /health        - liveness/readiness probe used by the Docker healthcheck
    POST /predict       - rating for one user-movie pair
    POST /predict/batch - ratings for many pairs in one round trip
    GET  /model/info    - deployed model version and type
    GET  /docs          - Swagger UI
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_DESCRIPTION, API_TITLE, API_VERSION, MODEL_VERSION
from app.model import MovieRatingModel
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Model lifecycle
# =============================================================================
# Loaded once at startup and shared by every request: unpickling the SVD model
# takes far longer than a prediction, so doing it per request would dominate
# response latency.
model: Optional[MovieRatingModel] = None

MODEL_TYPE = "SVD (Collaborative Filtering)"

# Reusable Swagger documentation for the failure modes of the predict endpoints.
PREDICT_ERROR_RESPONSES = {
    422: {"description": "Request body failed validation."},
    500: {"model": ErrorResponse, "description": "Unexpected error during prediction."},
    503: {"model": ErrorResponse, "description": "Model is not loaded."},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and release it on shutdown."""
    global model
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        # Starting without a model is deliberate: the container comes up and
        # /health reports "unhealthy" instead of crash-looping with no
        # diagnosable endpoint.
        logger.error(f"Failed to load model: {e}")
        model = None

    yield

    model = None


def _require_model() -> MovieRatingModel:
    """Return the loaded model or raise a 503."""
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


# =============================================================================
# Initialize FastAPI app
# =============================================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Liveness and readiness checks."},
        {"name": "Prediction", "description": "Rating prediction endpoints."},
        {"name": "Info", "description": "API and model metadata."},
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoint
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Always returns 200 so the probe can distinguish "process is up but the
    model failed to load" from "process is down". Inspect `model_loaded` to
    tell the two apart.
    """
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded(),
    )


# =============================================================================
# Prediction Endpoints
# =============================================================================
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict the rating one user would give one movie",
    responses=PREDICT_ERROR_RESPONSES,
)
async def predict(request: PredictionRequest):
    """
    Predict movie rating for a user.

    Unknown IDs are handled gracefully - the model falls back to the global
    mean rather than failing - so any well-formed request yields a rating
    between 1.0 and 5.0.

    Args:
        request: PredictionRequest with user_id and movie_id

    Returns:
        PredictionResponse with the predicted rating

    Raises:
        HTTPException: 503 if the model is not loaded, 500 on an unexpected error.
    """
    active_model = _require_model()

    try:
        rating = active_model.predict(request.user_id, request.movie_id)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        user_id=request.user_id,
        movie_id=request.movie_id,
        predicted_rating=rating,
        model_version=MODEL_VERSION,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict ratings for many user-movie pairs at once",
    responses=PREDICT_ERROR_RESPONSES,
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict movie ratings for multiple user-movie pairs.

    Cheaper than one request per pair: the model is loaded once and the HTTP
    overhead is paid once.

    Args:
        request: BatchPredictionRequest with a list of user-movie pairs

    Returns:
        BatchPredictionResponse with one prediction per pair, in request order

    Raises:
        HTTPException: 503 if the model is not loaded, 500 on an unexpected error.
    """
    active_model = _require_model()

    try:
        pairs = [(item.user_id, item.movie_id) for item in request.predictions]
        ratings = active_model.predict_batch(pairs)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    results = [
        PredictionResponse(
            user_id=item.user_id,
            movie_id=item.movie_id,
            predicted_rating=rating,
            model_version=MODEL_VERSION,
        )
        for item, rating in zip(request.predictions, ratings)
    ]
    return BatchPredictionResponse(predictions=results, total_count=len(results))


# =============================================================================
# Info Endpoints
# =============================================================================
@app.get("/", tags=["Info"], summary="API metadata")
async def root():
    """Root endpoint with API information and links to the docs."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Info"],
    summary="Deployed model version and type",
)
async def model_info():
    """Get information about the loaded model."""
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        is_loaded=model is not None and model.is_loaded(),
    )


# =============================================================================
# Run with uvicorn (for development)
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
