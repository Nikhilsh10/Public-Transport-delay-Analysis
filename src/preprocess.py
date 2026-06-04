from __future__ import annotations

"""Data loading and preprocessing utilities for the Transport Delay regression project.
Provides functions to load data, clean, split, and a deterministic one‑hot encoder.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

from .config import DATA_PATH, TARGET_COLUMN


def load_data(csv_name: str = "public_transport_delays.csv") -> pd.DataFrame:
    """Load CSV from the data path or a given location."""
    if csv_name == "public_transport_delays.csv":
        path = DATA_PATH
    else:
        path = Path(csv_name)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``event_type`` with the placeholder ``"None"``."""
    df["event_type"] = df["event_type"].fillna("None")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate features (X) and target (y)."""
    y = df[TARGET_COLUMN]
    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "delayed",
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
    """Deterministic one‑hot encoder based on pandas ``get_dummies``.

    The encoder learns the unique categories for each object column during ``fit``
    and ensures the same column order during ``transform``. Missing values are
    filled with the placeholder ``"None"`` before encoding.
    """

    def __init__(self, drop_first: bool = True, placeholder: str = "None"):
        self.drop_first = drop_first
        self.placeholder = placeholder
        self._categories: dict[str, list] = {}

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        """Record categories for each categorical column.
        ``X`` is expected to be a ``DataFrame``.
        """
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            series = X[col].fillna(self.placeholder).astype(str)
            cats = series.unique().tolist()
            if self.drop_first and len(cats) > 1:
                # drop the first category to mimic OneHotEncoder(drop='first')
                cats = cats[1:]
            self._categories[col] = cats
        return self

    def transform(self, X: pd.DataFrame):  # noqa: D401
        """Apply one‑hot encoding using the stored categories.
        Returns a ``DataFrame`` with a deterministic column layout.
        """
        X_filled = X.copy()
        for col in self._categories:
            X_filled[col] = X_filled[col].fillna(self.placeholder).astype(str)
        # Use pandas get_dummies without dropping, then manually drop first category if needed
        X_enc = pd.get_dummies(X_filled, columns=self._categories.keys(), drop_first=False)
        # Ensure only the learned categories are present
        cols_to_keep = []
        for col, cats in self._categories.items():
            for cat in cats:
                col_name = f"{col}_{cat}"
                cols_to_keep.append(col_name)
        # Add any numeric columns that were passed through untouched
        numeric_cols = X_filled.select_dtypes(exclude=["object"]).columns.tolist()
        cols_to_keep = numeric_cols + cols_to_keep
        # Reindex to guarantee ordering and fill missing columns with zeros
        X_enc = X_enc.reindex(columns=cols_to_keep, fill_value=0)
        return X_enc


def build_preprocess_transformer(X: pd.DataFrame) -> PandasOneHotEncoder:
    """Return a deterministic pandas one‑hot encoder.
    The transformer does not need to learn additional parameters beyond the
    categories captured in ``fit``.
    """
    return PandasOneHotEncoder(drop_first=True, placeholder="None")
