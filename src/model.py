# src/model.py
"""Utility for loading and using the trained regression pipeline.
The pipeline contains both preprocessing (ColumnTransformer) and the
RandomForestRegressor model, saved as a single ``joblib`` file.
"""
import joblib
from pathlib import Path

from .config import MODEL_PATH


def load_pipeline(path: Path = MODEL_PATH):
    """Load the saved sklearn Pipeline.
    Returns the pipeline object ready for ``predict``.
    """
    return joblib.load(path)


def predict(pipeline, X):
    """Generate predictions using the loaded pipeline.
    ``X`` should be a DataFrame with the same raw feature columns as the
    training data (preprocessing will be applied inside the pipeline).
    """
    return pipeline.predict(X)
