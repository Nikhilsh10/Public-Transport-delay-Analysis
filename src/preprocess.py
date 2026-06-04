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


class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    """A simple one‑hot encoder using pandas.

    This transformer fits on a DataFrame, records the categorical columns and the
    generated dummy column names, and transforms new DataFrames to the same layout.
    It is defined at module level so it can be pickled by joblib.
    """
    def __init__(self):
        self.categorical_cols = []
        self.feature_names = []
    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        X_enc = pd.get_dummies(X[self.categorical_cols], drop_first=True)
        self.feature_names = X_enc.columns.tolist()
        return self
    def transform(self, X):
        X_num = X.drop(columns=self.categorical_cols)
        X_cat = pd.get_dummies(X[self.categorical_cols], drop_first=True)
        X_cat = X_cat.reindex(columns=self.feature_names, fill_value=0)
        return pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    def get_feature_names_out(self, input_features=None):
        return self.feature_names

def build_preprocess_transformer(X: pd.DataFrame) -> "PandasOneHotEncoder":
    """Return a pandas‑based one‑hot encoder.
    The function signature remains unchanged; the returned object is a
    ``PandasOneHotEncoder`` instance that can be used in an sklearn ``Pipeline``.
    """
    return PandasOneHotEncoder()
