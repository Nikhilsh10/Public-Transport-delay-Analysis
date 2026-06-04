from __future__ import annotations

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
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

from .config import DATA_PATH, TARGET_COLUMN

def load_data(csv_name: str = "public_transport_delays.csv") -> pd.DataFrame:
    """Load the raw CSV from the configured data path.

    Parameters
    ----------
    csv_name: str
        Filename located under ``data/`` or an absolute path.
    """
    # If the default filename is requested, use the pre‑computed DATA_PATH.
    if csv_name == "public_transport_delays.csv":
        path = DATA_PATH
    else:
        potential_path = Path(csv_name)
        if potential_path.is_absolute():
            path = potential_path
        else:
            # Resolve relative to the project root.
            path = Path(csv_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Ensure the CSV file is present at the expected location."
        )
    return pd.read_csv(path)





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


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_preprocess_transformer(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that one‑hot encodes categorical columns.
    It uses ``OneHotEncoder`` with ``drop='first'`` to avoid multicollinearity and
    ``sparse=False`` so the output is a dense NumPy array, which works with RandomForest.
    The transformer also passes through numeric columns unchanged.
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    # Define the transformer: one‑hot encode categoricals, passthrough numerics
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough",
    )
    return preprocessor
