"""
ML model wrapper for movie rating prediction.

Wraps the pickled Surprise SVD model produced by scripts/train_model.py and
exposes a small, API-friendly surface: load once, predict one, predict many.
"""

import logging
import pickle
from typing import List, Optional, Tuple

from app.config import MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model is available."""


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)

        Raises:
            FileNotFoundError: The model file does not exist.
            ValueError: The file exists but could not be unpickled.
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the trained model from disk.

        Raises:
            FileNotFoundError: The model file does not exist.
            ValueError: The file exists but is not a usable pickled model.
        """
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            logger.error(
                f"Model file not found: {self.model_path}. "
                "Run `python scripts/train_model.py` to create it."
            )
            raise
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            logger.error(f"Model file at {self.model_path} could not be loaded: {e}")
            raise ValueError(f"Corrupted or incompatible model file: {e}") from e

        if not hasattr(self.model, "predict"):
            self.model = None
            raise ValueError(
                f"Object loaded from {self.model_path} has no predict() method."
            )

        logger.info(f"Model loaded successfully from {self.model_path}")

    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Unknown user or movie IDs are not an error: the underlying SVD model
        falls back to the global mean, so the result stays within 1.0-5.0.

        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)

        Returns:
            Predicted rating (float between 1.0 and 5.0)

        Raises:
            ModelNotLoadedError: The model is not loaded.
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded; cannot predict.")

        prediction = self.model.predict(user_id, movie_id)
        return round(prediction.est, 2)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings, in the same order as the input

        Raises:
            ModelNotLoadedError: The model is not loaded.
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded; cannot predict.")

        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# =============================================================================
# Singleton instance
# =============================================================================
_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance
