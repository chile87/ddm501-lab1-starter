"""
API tests for Movie Rating Prediction API.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.schemas import MAX_BATCH_SIZE

# Create test client
client = TestClient(app)


# =============================================================================
# Startup fixture
# =============================================================================
# TestClient only triggers the app's lifespan when it is used as a context
# manager. Without this fixture the model is never loaded and every prediction
# endpoint answers 503 "Model not loaded".
@pytest.fixture(scope="session", autouse=True)
def run_app_lifespan():
    """Run the FastAPI lifespan once for the whole test session."""
    with client:
        yield


# =============================================================================
# Health Check Tests
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

    def test_health_check_reports_model_loaded(self):
        """Test that the model is actually loaded during the test session."""
        data = client.get("/health").json()

        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_reports_unhealthy_without_model(self, monkeypatch):
        """Health must report unhealthy - not crash - when the model is missing."""
        monkeypatch.setattr(main_module, "model", None)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "unhealthy", "model_loaded": False}


# =============================================================================
# Root Endpoint Tests
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
# Prediction Endpoint Tests (happy path)
# =============================================================================
class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "predicted_rating" in data
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_response_format(self):
        """Test that prediction response has correct format."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"},
        )
        data = response.json()

        assert "user_id" in data
        assert "movie_id" in data
        assert "predicted_rating" in data
        assert "model_version" in data

        assert data["user_id"] == "196"
        assert data["movie_id"] == "242"
        assert isinstance(data["predicted_rating"], float)
        assert isinstance(data["model_version"], str)

    def test_predict_is_deterministic(self):
        """The same request must produce the same rating on a frozen model."""
        payload = {"user_id": "196", "movie_id": "242"}

        first = client.post("/predict", json=payload).json()["predicted_rating"]
        second = client.post("/predict", json=payload).json()["predicted_rating"]

        assert first == second


# =============================================================================
# Input Validation Tests
# =============================================================================
class TestPredictValidation:
    """Input validation for the /predict endpoint."""

    def test_predict_missing_user_id(self):
        """Test prediction with missing user_id."""
        response = client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422

    def test_predict_missing_movie_id(self):
        """Test prediction with missing movie_id."""
        response = client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422

    def test_predict_empty_body(self):
        """Test prediction with empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_wrong_type(self):
        """Test prediction with a non-string id."""
        response = client.post(
            "/predict",
            json={"user_id": {"nested": "object"}, "movie_id": "242"},
        )
        assert response.status_code == 422

    def test_predict_empty_string_ids(self):
        """Empty IDs are meaningless and must be rejected by validation."""
        response = client.post("/predict", json={"user_id": "", "movie_id": ""})
        assert response.status_code == 422

    def test_predict_oversized_id(self):
        """IDs longer than the declared max_length must be rejected."""
        response = client.post(
            "/predict",
            json={"user_id": "9" * 65, "movie_id": "242"},
        )
        assert response.status_code == 422

    def test_predict_malformed_json(self):
        """A body that is not valid JSON must produce 422, not a 500."""
        response = client.post(
            "/predict",
            content="{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# =============================================================================
# Edge Case Tests
# =============================================================================
class TestEdgeCases:
    """Edge case tests."""

    def test_predict_unknown_user(self):
        """Test prediction with unknown user ID."""
        # SVD falls back to the global mean for unknown users, so the request
        # must still succeed with a rating inside the valid range.
        response = client.post(
            "/predict",
            json={"user_id": "999999", "movie_id": "242"},
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_unknown_movie(self):
        """Test prediction with unknown movie ID."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "999999"},
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_both_ids_unknown(self):
        """Both IDs unknown still falls back to the global mean."""
        response = client.post(
            "/predict",
            json={"user_id": "888888", "movie_id": "999999"},
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_special_characters_in_id(self):
        """IDs are opaque strings - injection-looking input is just an unknown ID."""
        response = client.post(
            "/predict",
            json={"user_id": "'; DROP TABLE users;--", "movie_id": "<script>"},
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_unicode_ids(self):
        """Non-ASCII IDs must not break encoding or crash the model."""
        response = client.post(
            "/predict",
            json={"user_id": "người-dùng-196", "movie_id": "phim-242"},
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_returns_503_when_model_missing(self, monkeypatch):
        """Without a model the endpoint must answer 503, not 500."""
        monkeypatch.setattr(main_module, "model", None)

        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_predict_returns_500_on_unexpected_model_error(self, monkeypatch):
        """An unexpected model failure is surfaced as a 500, not an unhandled crash."""

        class BrokenModel:
            def is_loaded(self):
                return True

            def predict(self, user_id, movie_id):
                raise RuntimeError("boom")

        monkeypatch.setattr(main_module, "model", BrokenModel())

        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"},
        )

        assert response.status_code == 500
        assert "boom" in response.json()["detail"]


# =============================================================================
# Model Info Endpoint Tests
# =============================================================================
class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info_returns_200(self):
        """Test that model info endpoint returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_version(self):
        """Test that model info contains version."""
        data = client.get("/model/info").json()

        assert "model_version" in data
        assert "model_type" in data
        assert "is_loaded" in data
        assert isinstance(data["model_version"], str)
        assert data["is_loaded"] is True


# =============================================================================
# Batch Prediction Tests
# =============================================================================
class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_multiple_items(self):
        """Test batch prediction with multiple items."""
        payload = {
            "predictions": [
                {"user_id": "196", "movie_id": "242"},
                {"user_id": "186", "movie_id": "302"},
                {"user_id": "22", "movie_id": "377"},
            ]
        }
        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3
        for item in data["predictions"]:
            assert 1.0 <= item["predicted_rating"] <= 5.0
            assert "model_version" in item

    def test_batch_predict_preserves_order(self):
        """Responses must line up with the requested pairs, in order."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        payload = {
            "predictions": [{"user_id": u, "movie_id": m} for u, m in pairs]
        }

        data = client.post("/predict/batch", json=payload).json()

        returned = [(p["user_id"], p["movie_id"]) for p in data["predictions"]]
        assert returned == pairs

    def test_batch_matches_single_predictions(self):
        """A batch result must equal the single-prediction result for each pair."""
        pairs = [("196", "242"), ("186", "302")]

        batch = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": u, "movie_id": m} for u, m in pairs]},
        ).json()

        for (user_id, movie_id), item in zip(pairs, batch["predictions"]):
            single = client.post(
                "/predict", json={"user_id": user_id, "movie_id": movie_id}
            ).json()
            assert item["predicted_rating"] == single["predicted_rating"]

    def test_batch_predict_single_item(self):
        """A batch of one is valid."""
        response = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "196", "movie_id": "242"}]},
        )

        assert response.status_code == 200
        assert response.json()["total_count"] == 1

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list returns 422 validation error."""
        response = client.post("/predict/batch", json={"predictions": []})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_batch_predict_at_size_limit(self):
        """A batch exactly at the cap is accepted."""
        payload = {
            "predictions": [
                {"user_id": "196", "movie_id": str(i)} for i in range(MAX_BATCH_SIZE)
            ]
        }
        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        assert response.json()["total_count"] == MAX_BATCH_SIZE

    def test_batch_predict_over_size_limit(self):
        """A batch over the cap is rejected by validation."""
        payload = {
            "predictions": [
                {"user_id": "196", "movie_id": str(i)}
                for i in range(MAX_BATCH_SIZE + 1)
            ]
        }
        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 422

    def test_batch_predict_missing_field(self):
        """Test batch prediction with a malformed item."""
        response = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "196"}]},  # Missing movie_id
        )
        assert response.status_code == 422

    def test_batch_predict_empty_string_id(self):
        """Empty IDs inside a batch item are rejected too."""
        response = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "", "movie_id": "242"}]},
        )
        assert response.status_code == 422

    def test_batch_predict_missing_predictions_key(self):
        """The predictions key is required."""
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422

    def test_batch_predict_returns_503_when_model_missing(self, monkeypatch):
        """Batch endpoint must also answer 503 without a model."""
        monkeypatch.setattr(main_module, "model", None)

        response = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "196", "movie_id": "242"}]},
        )

        assert response.status_code == 503


# =============================================================================
# Swagger / OpenAPI Documentation Tests
# =============================================================================
class TestDocumentation:
    """The generated API documentation is a graded deliverable - verify it."""

    def test_swagger_ui_is_served(self):
        """Swagger UI must be reachable at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_lists_all_endpoints(self):
        """Every implemented endpoint must appear in the OpenAPI schema."""
        schema = client.get("/openapi.json").json()

        for path in ["/health", "/predict", "/predict/batch", "/model/info", "/"]:
            assert path in schema["paths"]

    def test_request_schema_has_example(self):
        """Pydantic models must carry examples so /docs is self-explanatory."""
        schema = client.get("/openapi.json").json()
        request_schema = schema["components"]["schemas"]["PredictionRequest"]

        assert "example" in request_schema
        assert request_schema["example"]["user_id"] == "196"

    def test_endpoints_have_descriptions(self):
        """Docstrings must reach the generated docs."""
        schema = client.get("/openapi.json").json()

        assert schema["paths"]["/predict"]["post"]["description"]
        assert schema["paths"]["/health"]["get"]["description"]


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
