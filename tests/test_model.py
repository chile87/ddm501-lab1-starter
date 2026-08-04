"""
Unit tests for the model wrapper (app/model.py).

These exercise MovieRatingModel directly, without going through HTTP, so a
failure here points at the model layer rather than the API layer.
"""

import pickle

import pytest

import app.model as model_module
from app.config import MODEL_PATH
from app.model import ModelNotLoadedError, MovieRatingModel, get_model


@pytest.fixture(scope="module")
def loaded_model():
    """The real trained model, loaded once for this module."""
    return MovieRatingModel()


def _unloaded_model() -> MovieRatingModel:
    """Build a wrapper with no model attached, bypassing __init__."""
    instance = object.__new__(MovieRatingModel)
    instance.model_path = "unused"
    instance.model = None
    return instance


# =============================================================================
# Loading
# =============================================================================
class TestModelLoading:
    """Tests for loading the model from disk."""

    def test_model_loads_from_default_path(self, loaded_model):
        """The trained model ships with the repo and loads without arguments."""
        assert loaded_model.is_loaded() is True
        assert loaded_model.model is not None

    def test_model_path_defaults_to_config(self, loaded_model):
        """The wrapper uses MODEL_PATH from config when none is given."""
        assert loaded_model.model_path == MODEL_PATH

    def test_missing_file_raises_file_not_found(self):
        """A missing model file must raise FileNotFoundError, not hang or pass."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="models/does_not_exist.pkl")

    def test_corrupted_file_raises_value_error(self, tmp_path):
        """A file that is not a valid pickle must be reported clearly."""
        bad_file = tmp_path / "corrupted.pkl"
        bad_file.write_bytes(b"this is definitely not a pickle")

        with pytest.raises(ValueError):
            MovieRatingModel(model_path=str(bad_file))

    def test_pickle_without_predict_raises_value_error(self, tmp_path):
        """A valid pickle that isn't a model must be rejected at load time."""
        wrong_file = tmp_path / "wrong_object.pkl"
        wrong_file.write_bytes(pickle.dumps({"not": "a model"}))

        with pytest.raises(ValueError, match="predict"):
            MovieRatingModel(model_path=str(wrong_file))

    def test_is_loaded_false_without_model(self):
        """is_loaded reflects the absence of a model."""
        assert _unloaded_model().is_loaded() is False


# =============================================================================
# Single prediction
# =============================================================================
class TestPredict:
    """Tests for MovieRatingModel.predict."""

    def test_predict_returns_float(self, loaded_model):
        """Predictions are plain floats, ready for JSON serialization."""
        assert isinstance(loaded_model.predict("196", "242"), float)

    def test_predict_within_rating_scale(self, loaded_model):
        """Predictions must respect the 1-5 MovieLens rating scale."""
        assert 1.0 <= loaded_model.predict("196", "242") <= 5.0

    def test_predict_rounded_to_two_decimals(self, loaded_model):
        """The wrapper rounds to 2 decimals before returning."""
        rating = loaded_model.predict("196", "242")
        assert rating == round(rating, 2)

    def test_predict_is_deterministic(self, loaded_model):
        """A trained model is frozen - same input, same output."""
        assert loaded_model.predict("196", "242") == loaded_model.predict("196", "242")

    @pytest.mark.parametrize(
        "user_id,movie_id",
        [
            ("999999", "242"),  # unknown user
            ("196", "999999"),  # unknown movie
            ("999999", "888888"),  # both unknown
            ("'; DROP TABLE users;--", "<script>"),  # opaque garbage IDs
        ],
    )
    def test_predict_handles_unknown_ids(self, loaded_model, user_id, movie_id):
        """Unknown IDs fall back to the global mean instead of raising."""
        rating = loaded_model.predict(user_id, movie_id)
        assert 1.0 <= rating <= 5.0

    def test_predict_without_model_raises(self):
        """Predicting before the model is loaded is a clear, typed error."""
        with pytest.raises(ModelNotLoadedError):
            _unloaded_model().predict("196", "242")


# =============================================================================
# Batch prediction
# =============================================================================
class TestPredictBatch:
    """Tests for MovieRatingModel.predict_batch."""

    def test_batch_returns_one_rating_per_pair(self, loaded_model):
        """Output length matches input length."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        assert len(loaded_model.predict_batch(pairs)) == len(pairs)

    def test_batch_matches_individual_predictions(self, loaded_model):
        """Batching is an optimization, not a different computation."""
        pairs = [("196", "242"), ("186", "302")]

        batched = loaded_model.predict_batch(pairs)
        individual = [loaded_model.predict(u, m) for u, m in pairs]

        assert batched == individual

    def test_batch_all_within_rating_scale(self, loaded_model):
        """Every batched rating respects the 1-5 scale."""
        pairs = [(str(u), str(m)) for u in range(1, 11) for m in range(1, 11)]

        assert all(1.0 <= r <= 5.0 for r in loaded_model.predict_batch(pairs))

    def test_batch_empty_list(self, loaded_model):
        """An empty batch is valid and returns an empty list."""
        assert loaded_model.predict_batch([]) == []

    def test_batch_without_model_raises(self):
        """Batch predicting before the model is loaded is a clear, typed error."""
        with pytest.raises(ModelNotLoadedError):
            _unloaded_model().predict_batch([("196", "242")])


# =============================================================================
# Singleton accessor
# =============================================================================
class TestGetModel:
    """Tests for the get_model singleton helper."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self, monkeypatch):
        """Keep the module-level singleton out of other tests."""
        monkeypatch.setattr(model_module, "_model_instance", None)
        yield

    def test_get_model_returns_loaded_instance(self):
        """The singleton hands back a ready-to-use model."""
        assert get_model().is_loaded() is True

    def test_get_model_reuses_the_same_instance(self):
        """Unpickling is expensive - it must happen only once."""
        assert get_model() is get_model()
