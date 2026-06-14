"""Shared pytest fixtures for the Public Transport Delay prediction project."""

import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Minimal sample data mirroring the real dataset schema
# ---------------------------------------------------------------------------
_SAMPLE_ROWS = [
    {
        "trip_id": "T001",
        "route_id": "R1",
        "date": "2024-01-01",
        "time": "08:00",
        "origin_station": "A",
        "destination_station": "B",
        "scheduled_departure": "08:00",
        "scheduled_arrival": "09:00",
        "transport_type": "bus",
        "weather_condition": "clear",
        "event_type": None,
        "actual_arrival_delay_min": 5.0,
        "delayed": 1,
    },
    {
        "trip_id": "T002",
        "route_id": "R2",
        "date": "2024-01-01",
        "time": "09:00",
        "origin_station": "B",
        "destination_station": "C",
        "scheduled_departure": "09:00",
        "scheduled_arrival": "10:00",
        "transport_type": "train",
        "weather_condition": "rain",
        "event_type": "concert",
        "actual_arrival_delay_min": 12.0,
        "delayed": 1,
    },
    {
        "trip_id": "T003",
        "route_id": "R1",
        "date": "2024-01-02",
        "time": "10:00",
        "origin_station": "A",
        "destination_station": "B",
        "scheduled_departure": "10:00",
        "scheduled_arrival": "11:00",
        "transport_type": "bus",
        "weather_condition": "clear",
        "event_type": None,
        "actual_arrival_delay_min": 0.0,
        "delayed": 0,
    },
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Return a small raw DataFrame matching the dataset schema."""
    return pd.DataFrame(_SAMPLE_ROWS)


@pytest.fixture
def cleaned_df(raw_df) -> pd.DataFrame:
    """Return the sample DataFrame after clean_data() has been applied."""
    from src.preprocess import clean_data
    return clean_data(raw_df.copy())


@pytest.fixture
def feature_df(cleaned_df) -> pd.DataFrame:
    """Return only the feature columns (X) of the cleaned sample data."""
    from src.preprocess import split_features_target
    X, _ = split_features_target(cleaned_df)
    return X
