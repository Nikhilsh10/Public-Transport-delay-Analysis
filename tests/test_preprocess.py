"""Unit tests for src/preprocess.py.

Covers:
- clean_data        : null-filling behaviour for event_type
- split_features_target : correct column removal and target extraction
- PandasOneHotEncoder  : fit / transform / consistency / edge-cases
- build_preprocess_transformer : factory function return type
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocess import (
    clean_data,
    split_features_target,
    PandasOneHotEncoder,
    build_preprocess_transformer,
)
from src.config import TARGET_COLUMN


# ─────────────────────────────────────────────────────────────────────────────
# clean_data
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanData:
    def test_null_event_type_filled(self, raw_df):
        result = clean_data(raw_df.copy())
        assert result["event_type"].isna().sum() == 0

    def test_null_event_type_replaced_with_none_string(self, raw_df):
        result = clean_data(raw_df.copy())
        # Rows that originally had NaN event_type must now be "None"
        originally_null_mask = raw_df["event_type"].isna()
        assert (result.loc[originally_null_mask, "event_type"] == "None").all()

    def test_non_null_event_type_unchanged(self, raw_df):
        result = clean_data(raw_df.copy())
        originally_set_mask = raw_df["event_type"].notna()
        pd.testing.assert_series_equal(
            result.loc[originally_set_mask, "event_type"].reset_index(drop=True),
            raw_df.loc[originally_set_mask, "event_type"].reset_index(drop=True),
        )

    def test_returns_dataframe(self, raw_df):
        assert isinstance(clean_data(raw_df.copy()), pd.DataFrame)

    def test_row_count_unchanged(self, raw_df):
        result = clean_data(raw_df.copy())
        assert len(result) == len(raw_df)


# ─────────────────────────────────────────────────────────────────────────────
# split_features_target
# ─────────────────────────────────────────────────────────────────────────────

_COLUMNS_TO_DROP = [
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


class TestSplitFeaturesTarget:
    def test_returns_tuple_of_two(self, cleaned_df):
        result = split_features_target(cleaned_df)
        assert len(result) == 2

    def test_target_series_name_matches_config(self, cleaned_df):
        _, y = split_features_target(cleaned_df)
        assert y.name == TARGET_COLUMN

    def test_target_length_matches_input(self, cleaned_df):
        X, y = split_features_target(cleaned_df)
        assert len(y) == len(cleaned_df)

    def test_dropped_columns_absent_from_X(self, cleaned_df):
        X, _ = split_features_target(cleaned_df)
        for col in _COLUMNS_TO_DROP:
            assert col not in X.columns, f"Column '{col}' should have been dropped"

    def test_feature_df_not_empty(self, cleaned_df):
        X, _ = split_features_target(cleaned_df)
        assert not X.empty

    def test_row_count_preserved(self, cleaned_df):
        X, y = split_features_target(cleaned_df)
        assert len(X) == len(cleaned_df)


# ─────────────────────────────────────────────────────────────────────────────
# PandasOneHotEncoder
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_cat_df():
    """Tiny DataFrame with one categorical and one numeric column."""
    return pd.DataFrame(
        {
            "color": ["red", "blue", "green", "blue"],
            "size": [1, 2, 3, 4],
        }
    )


class TestPandasOneHotEncoder:
    # ── fit ────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        returned = enc.fit(simple_cat_df)
        assert returned is enc

    def test_fit_records_categories(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        assert "color" in enc._categories
        assert set(enc._categories["color"]) == {"red", "blue", "green"}

    def test_fit_drop_first_removes_one_category(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=True)
        enc.fit(simple_cat_df)
        assert len(enc._categories["color"]) == 2  # 3 values → 2 after drop

    def test_fit_numeric_columns_ignored(self, simple_cat_df):
        enc = PandasOneHotEncoder()
        enc.fit(simple_cat_df)
        assert "size" not in enc._categories

    def test_fit_deterministic_ordering(self, simple_cat_df):
        """Two fits on the same data must produce the same category order."""
        enc1 = PandasOneHotEncoder(drop_first=False)
        enc2 = PandasOneHotEncoder(drop_first=False)
        enc1.fit(simple_cat_df)
        enc2.fit(simple_cat_df)
        assert enc1._categories == enc2._categories

    # ── transform ──────────────────────────────────────────────────────────

    def test_transform_returns_dataframe(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        result = enc.transform(simple_cat_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_numeric_column_preserved(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        result = enc.transform(simple_cat_df)
        assert "size" in result.columns

    def test_transform_creates_indicator_columns(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        result = enc.transform(simple_cat_df)
        for cat in enc._categories["color"]:
            assert f"color_{cat}" in result.columns

    def test_transform_no_original_cat_column(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        result = enc.transform(simple_cat_df)
        assert "color" not in result.columns

    def test_transform_consistent_shape_on_repeated_calls(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        r1 = enc.transform(simple_cat_df)
        r2 = enc.transform(simple_cat_df)
        assert r1.shape == r2.shape
        assert list(r1.columns) == list(r2.columns)

    def test_transform_unseen_category_filled_with_zero(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        new_df = pd.DataFrame({"color": ["purple"], "size": [10]})
        result = enc.transform(new_df)
        # All known color columns should be 0 for an unseen category
        for cat in enc._categories["color"]:
            assert result[f"color_{cat}"].iloc[0] == 0

    def test_transform_null_values_filled_before_encoding(self):
        train = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
        enc = PandasOneHotEncoder(drop_first=False, placeholder="MISSING")
        enc.fit(train)
        test_with_null = pd.DataFrame({"cat": [None, "a"], "num": [4, 5]})
        # Should not raise
        result = enc.transform(test_with_null)
        assert isinstance(result, pd.DataFrame)

    # ── sklearn compatibility ──────────────────────────────────────────────

    def test_fit_transform_pipeline(self, feature_df):
        """Encoder must work inside a fit→transform pipeline without errors."""
        enc = PandasOneHotEncoder(drop_first=True)
        enc.fit(feature_df)
        result = enc.transform(feature_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(feature_df)

    def test_no_nan_in_output(self, feature_df):
        enc = PandasOneHotEncoder(drop_first=True)
        enc.fit(feature_df)
        result = enc.transform(feature_df)
        assert not result.isnull().any().any()

    def test_output_dtype_numeric(self, simple_cat_df):
        enc = PandasOneHotEncoder(drop_first=False)
        enc.fit(simple_cat_df)
        result = enc.transform(simple_cat_df)
        indicator_cols = [c for c in result.columns if c.startswith("color_")]
        for col in indicator_cols:
            assert pd.api.types.is_numeric_dtype(result[col])


# ─────────────────────────────────────────────────────────────────────────────
# build_preprocess_transformer
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPreprocessTransformer:
    def test_returns_pandas_one_hot_encoder(self, feature_df):
        transformer = build_preprocess_transformer(feature_df)
        assert isinstance(transformer, PandasOneHotEncoder)

    def test_drop_first_enabled_by_default(self, feature_df):
        transformer = build_preprocess_transformer(feature_df)
        assert transformer.drop_first is True

    def test_placeholder_is_none_string(self, feature_df):
        transformer = build_preprocess_transformer(feature_df)
        assert transformer.placeholder == "None"
