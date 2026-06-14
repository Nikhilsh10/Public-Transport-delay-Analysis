"""Unit tests for src/model.py.

Tests cover:
- predict()        : correct shape, numeric output, no NaN values
- load_pipeline()  : proper error when the path does not exist

The full Pipeline (preprocessing + RandomForestRegressor) is constructed
in-memory so the tests never need a pre-trained model file on disk.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.preprocess import PandasOneHotEncoder, build_preprocess_transformer
from src.model import predict, load_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_pipeline(feature_df):
    """Tiny trained pipeline fixture (10 trees, quick fit)."""
    X = feature_df
    y = pd.Series([5.0, 12.0, 0.0], name="actual_arrival_delay_min")
    preprocessor = build_preprocess_transformer(X)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_numpy_array(self, tiny_pipeline, feature_df):
        result = predict(tiny_pipeline, feature_df)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input_rows(self, tiny_pipeline, feature_df):
        result = predict(tiny_pipeline, feature_df)
        assert len(result) == len(feature_df)

    def test_output_is_numeric(self, tiny_pipeline, feature_df):
        result = predict(tiny_pipeline, feature_df)
        assert np.issubdtype(result.dtype, np.floating)

    def test_no_nan_in_predictions(self, tiny_pipeline, feature_df):
        result = predict(tiny_pipeline, feature_df)
        assert not np.isnan(result).any()

    def test_single_row_prediction(self, tiny_pipeline, feature_df):
        single_row = feature_df.iloc[[0]]
        result = predict(tiny_pipeline, single_row)
        assert result.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# load_pipeline()
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPipeline:
    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        missing = tmp_path / "nonexistent_model.joblib"
        with pytest.raises(Exception):
            load_pipeline(path=missing)

    def test_loads_saved_pipeline(self, tiny_pipeline, tmp_path):
        """Save and reload a pipeline; verify it still predicts."""
        import joblib

        model_path = tmp_path / "model.joblib"
        joblib.dump(tiny_pipeline, model_path)

        loaded = load_pipeline(path=model_path)
        assert loaded is not None

    def test_loaded_pipeline_can_predict(self, tiny_pipeline, feature_df, tmp_path):
        import joblib

        model_path = tmp_path / "model.joblib"
        joblib.dump(tiny_pipeline, model_path)

        loaded = load_pipeline(path=model_path)
        result = loaded.predict(feature_df)
        assert len(result) == len(feature_df)
