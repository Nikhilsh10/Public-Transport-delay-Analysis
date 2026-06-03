# src/preprocess.py
"""Data loading and preprocessing utilities for the Transport Delay regression project.
The module provides functions to:
- Load the raw CSV dataset.
- Clean missing values (specifically `event_type`).
- Split the dataframe into features (X) and target (y).
- Build a single sklearn ``ColumnTransformer`` for one‑hot encoding of categorical
  columns while passing through numeric columns.
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# ColumnTransformer will be imported lazily inside build_preprocess_transformer

from .config import DATA_PATH, TARGET_COLUMN


def load_data(csv_name: str = "public_transport_delays.csv") -> pd.DataFrame:
    """Load the raw CSV from the ``data`` folder.

    Parameters
    ----------
    csv_name: str
        Filename located under ``data/``.
    """
    return pd.read_csv(DATA_PATH / csv_name)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform minimal cleaning required by the original notebook.
    - Fill missing ``event_type`` values with the placeholder ``"None"``.
    """
    df["event_type"] = df["event_type"].fillna("None")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate features and the regression target.

    The target column is defined in ``config.TARGET_COLUMN`` (``actual_arrival_delay_min``).
    Columns that are not useful for modeling are dropped, mirroring the original
    notebook's preprocessing steps.
    """
    y = df[TARGET_COLUMN]
    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "delayed",  # not needed for regression
            "trip_id",
            "origin_station",
            "destination_station",
            "date",
            "time",
            "scheduled_departure",
            "scheduled_arrival",
            "route_id",
        ]
    )
    return X, y


def build_preprocess_transformer(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ``ColumnTransformer`` that one‑hot encodes categorical columns.
    Numeric columns are passed through unchanged.
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return preprocessor
