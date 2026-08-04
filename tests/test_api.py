"""
Unit tests for Movie Rating Prediction API.

TODO: Complete the test cases below.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Create test client
client = TestClient(app)


# =============================================================================
# Health Check Tests (PROVIDED)
# =============================================================================
class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_format(self):
        """Test that health response has correct format."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)


# =============================================================================
# Root Endpoint Tests (PROVIDED)
# =============================================================================
class TestRootEndpoint:
    """Tests for the / endpoint."""
    
    def test_root_returns_200(self):
        """Test that root endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_api_info(self):
        """Test that root response contains API information."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


# =============================================================================
# TODO 1: Prediction Endpoint Tests
# =============================================================================
class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    # -------------------------------------------------------------------------
    # TODO: Implement test_predict_valid_input
    # -------------------------------------------------------------------------
    # Requirements:
    # - Send POST request to /predict with valid user_id and movie_id
    # - Assert status code is 200
    # - Assert response contains "predicted_rating"
    # - Assert predicted_rating is between 1.0 and 5.0
    
    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        # TODO: Implement this test
        #
        # response = client.post(
        #     "/predict",
        #     json={"user_id": "196", "movie_id": "242"}
        # )
        # assert response.status_code == 200
        # data = response.json()
        # assert "predicted_rating" in data
        # assert 1.0 <= data["predicted_rating"] <= 5.0
        pass
    
    # -------------------------------------------------------------------------
    # TODO: Implement test_predict_response_format
    # -------------------------------------------------------------------------
    # Requirements:
    # - Assert response contains all required fields:
    #   user_id, movie_id, predicted_rating, model_version
    
    def test_predict_response_format(self):
        """Test that prediction response has correct format."""
        # TODO: Implement this test
        #
        # response = client.post(
        #     "/predict",
        #     json={"user_id": "196", "movie_id": "242"}
        # )
        # data = response.json()
        # 
        # assert "user_id" in data
        # assert "movie_id" in data
        # assert "predicted_rating" in data
        # assert "model_version" in data
        pass
    
    # -------------------------------------------------------------------------
    # TODO: Implement test_predict_missing_user_id
    # -------------------------------------------------------------------------
    # Requirements:
    # - Send request without user_id
    # - Assert status code is 422 (Validation Error)
    
    def test_predict_missing_user_id(self):
        """Test prediction with missing user_id."""
        # TODO: Implement this test
        #
        # response = client.post(
        #     "/predict",
        #     json={"movie_id": "242"}  # Missing user_id
        # )
        # assert response.status_code == 422
        pass
    
    # -------------------------------------------------------------------------
    # TODO: Implement test_predict_missing_movie_id
    # -------------------------------------------------------------------------
    # Requirements:
    # - Send request without movie_id
    # - Assert status code is 422 (Validation Error)
    
    def test_predict_missing_movie_id(self):
        """Test prediction with missing movie_id."""
        # TODO: Implement this test
        pass
    
    # -------------------------------------------------------------------------
    # TODO: Implement test_predict_empty_body
    # -------------------------------------------------------------------------
    # Requirements:
    # - Send request with empty JSON body
    # - Assert status code is 422 (Validation Error)
    
    def test_predict_empty_body(self):
        """Test prediction with empty request body."""
        # TODO: Implement this test
        #
        # response = client.post("/predict", json={})
        # assert response.status_code == 422
        pass


# =============================================================================
# TODO 2: Edge Case Tests (BONUS)
# =============================================================================
class TestEdgeCases:
    """Edge case tests."""
    
    def test_predict_unknown_user(self):
        """Test prediction with unknown user ID."""
        # The model should still return a prediction (with default rating)
        # or handle gracefully
        # TODO: Implement this test
        pass
    
    def test_predict_unknown_movie(self):
        """Test prediction with unknown movie ID."""
        # TODO: Implement this test
        pass
    
    def test_predict_special_characters_in_id(self):
        """Test prediction with special characters in IDs."""
        # TODO: Implement this test
        pass


# =============================================================================
# TODO 3: Model Info Endpoint Tests
# =============================================================================
class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""
    
    def test_model_info_returns_200(self):
        """Test that model info endpoint returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200
    
    def test_model_info_contains_version(self):
        """Test that model info contains version."""
        # TODO: Implement this test
        pass


# =============================================================================
# Batch Prediction Tests (BONUS)
# =============================================================================
class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint (BONUS)."""
    
    def test_batch_predict_multiple_items(self):
        """Test batch prediction with multiple items."""
        # TODO: Implement this test (BONUS)
        pass
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        # TODO: Implement this test (BONUS)
        pass


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
