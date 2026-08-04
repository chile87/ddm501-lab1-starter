"""
FastAPI application for Movie Rating Prediction.

TODO: Complete the API endpoints below.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, MODEL_VERSION
from app.model import MovieRatingModel
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Initialize FastAPI app
# =============================================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
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
# Load model at startup
# =============================================================================
model: MovieRatingModel = None


@app.on_event("startup")
async def startup_event():
    """Load model when application starts."""
    global model
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Model will be None, health check will report unhealthy


# =============================================================================
# Health Check Endpoint (PROVIDED - DO NOT MODIFY)
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded()
    )


# =============================================================================
# TODO 1: Implement the /predict endpoint
# =============================================================================
# Requirements:
# - Method: POST
# - Path: /predict
# - Request body: PredictionRequest (user_id, movie_id)
# - Response: PredictionResponse (user_id, movie_id, predicted_rating, model_version)
# - Handle exceptions and return appropriate HTTP errors
#
# Hints:
# - Use model.predict(request.user_id, request.movie_id)
# - Wrap in try-except to catch errors
# - Raise HTTPException(status_code=500, detail=str(e)) for errors
# - Use MODEL_VERSION from config for the model_version field

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict movie rating for a user.
    
    Args:
        request: PredictionRequest with user_id and movie_id
        
    Returns:
        PredictionResponse with predicted rating
    """
    # TODO: Implement this endpoint
    #
    # Check if model is loaded
    # if model is None or not model.is_loaded():
    #     raise HTTPException(status_code=503, detail="Model not loaded")
    #
    # try:
    #     rating = model.predict(???, ???)
    #     return PredictionResponse(
    #         user_id=???,
    #         movie_id=???,
    #         predicted_rating=???,
    #         model_version=MODEL_VERSION
    #     )
    # except Exception as e:
    #     logger.error(f"Prediction error: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))
    pass


# =============================================================================
# TODO 2: Implement the /predict/batch endpoint (BONUS)
# =============================================================================
# Requirements:
# - Method: POST
# - Path: /predict/batch
# - Request body: BatchPredictionRequest
# - Response: BatchPredictionResponse
# - Process multiple predictions in one request

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict movie ratings for multiple user-movie pairs.
    
    Args:
        request: BatchPredictionRequest with list of predictions
        
    Returns:
        BatchPredictionResponse with all predicted ratings
    """
    # TODO: Implement this endpoint (BONUS)
    #
    # if model is None or not model.is_loaded():
    #     raise HTTPException(status_code=503, detail="Model not loaded")
    #
    # try:
    #     results = []
    #     for item in request.predictions:
    #         rating = model.predict(item.user_id, item.movie_id)
    #         results.append(PredictionResponse(...))
    #     return BatchPredictionResponse(predictions=results, total_count=len(results))
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    pass


# =============================================================================
# Root endpoint
# =============================================================================
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Model info endpoint
# =============================================================================
@app.get("/model/info", tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": "SVD (Collaborative Filtering)",
        "is_loaded": model is not None and model.is_loaded(),
    }


# =============================================================================
# Run with uvicorn (for development)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
