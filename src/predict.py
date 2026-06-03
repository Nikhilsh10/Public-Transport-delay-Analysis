# src/predict.py
"""Inference script for the Transport Delay regression model.
It loads the saved sklearn ``Pipeline`` (preprocessing + RandomForestRegressor)
and predicts the arrival delay minutes for a new CSV file.
"""
import argparse
import pandas as pd
from pathlib import Path
import joblib

from .preprocess import load_data, clean_data, split_features_target
from .config import MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Predict arrival delay minutes using the trained pipeline")
    parser.add_argument("--input", type=str, required=True, help="CSV file name in the data folder")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to the saved pipeline")
    args = parser.parse_args()

    # Load raw data
    df_raw = load_data(args.input)
    df = clean_data(df_raw)

    # Split features (target not needed for inference)
    X, _ = split_features_target(df)

    # Load pipeline
    pipeline = joblib.load(args.model)

    # Predict
    preds = pipeline.predict(X)
    df_raw["predicted_arrival_delay_min"] = preds
    # Show first few predictions
    print(df_raw[["predicted_arrival_delay_min"]].head())

    # Optionally, save predictions next to input file
    output_path = Path(args.input).with_name(Path(args.input).stem + "_predicted.csv")
    df_raw.to_csv(Path("data") / output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
