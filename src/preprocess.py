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
from pathlib import Path
# ColumnTransformer will be imported lazily inside build_preprocess_transformer

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


def build_preprocess_transformer(X: pd.DataFrame) -> "ColumnTransformer":
    """Create a transformer that one‑hot encodes categorical columns using pandas.

    This avoids reliance on internal sklearn attributes that changed between versions.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    
    class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.categorical_cols = []
            self.feature_names = []
        def fit(self, X, y=None):
            self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            # Perform one‑hot on a sample to capture feature names
            X_enc = pd.get_dummies(X[self.categorical_cols], drop_first=True)
            self.feature_names = X_enc.columns.tolist()
            return self
        def transform(self, X):
            X_num = X.drop(columns=self.categorical_cols)
            X_cat = pd.get_dummies(X[self.categorical_cols], drop_first=True)
            # Align columns with those seen during fit
            X_cat = X_cat.reindex(columns=self.feature_names, fill_value=0)
            return pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        def get_feature_names_out(self, input_features=None):
            return self.feature_names
    
    return PandasOneHotEncoder()
