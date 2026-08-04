"""
ML Model wrapper for movie rating prediction.

TODO: Complete the model loading and prediction functions.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from app.config import MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    # =========================================================================
    # TODO 1: Implement _load_model method
    # =========================================================================
    # Requirements:
    # - Load the pickle file from self.model_path
    # - Store the loaded model in self.model
    # - Log success message
    # - Handle FileNotFoundError gracefully
    #
    # Hint: Use pickle.load() with 'rb' mode
    
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        # TODO: Implement this method
        # 
        # try:
        #     with open(self.model_path, 'rb') as f:
        #         self.model = ???
        #     logger.info(f"Model loaded successfully from {self.model_path}")
        # except FileNotFoundError:
        #     logger.error(f"Model file not found: {self.model_path}")
        #     raise
        pass
    
    # =========================================================================
    # TODO 2: Implement predict method
    # =========================================================================
    # Requirements:
    # - Use self.model.predict(user_id, movie_id) to get prediction
    # - The prediction object has an 'est' attribute with the estimated rating
    # - Round the result to 2 decimal places
    # - Return the predicted rating as a float
    #
    # Hint: prediction = self.model.predict(uid, iid); return prediction.est
    
    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.
        
        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)
            
        Returns:
            Predicted rating (float between 1.0 and 5.0)
        """
        # TODO: Implement this method
        #
        # prediction = self.model.predict(???, ???)
        # return round(prediction.???, 2)
        pass
    
    # =========================================================================
    # TODO 3: Implement predict_batch method
    # =========================================================================
    # Requirements:
    # - Take a list of (user_id, movie_id) tuples
    # - Return a list of predicted ratings
    # - Use the predict method for each pair
    #
    # Hint: Use list comprehension
    
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            pairs: List of (user_id, movie_id) tuples
            
        Returns:
            List of predicted ratings
        """
        # TODO: Implement this method
        #
        # return [self.predict(???, ???) for ???, ??? in pairs]
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# =============================================================================
# Singleton instance (optional pattern)
# =============================================================================
# You can use this pattern to ensure only one model instance exists

_model_instance: Optional[MovieRatingModel] = None

def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance
