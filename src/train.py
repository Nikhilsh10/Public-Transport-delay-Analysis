# src/train.py
"""Training script for the Transport Delay regression model.
It builds a single sklearn ``Pipeline`` that includes preprocessing and a
``RandomForestRegressor``. After fitting, the pipeline is persisted to
``models/model.joblib`` using ``joblib``.
During training the script prints three regression metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score
"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .preprocess import load_data, clean_data, split_features_target, build_preprocess_transformer
from .config import MODEL_PATH, DATA_PATH, N_ESTIMATORS, RANDOM_STATE, TEST_SIZE, STRATIFY, TARGET_COLUMN


def main():
    parser = argparse.ArgumentParser(description="Train RandomForestRegressor for transport delay prediction")
    parser.add_argument("--data", type=str, default=str(DATA_PATH / "public_transport_delays.csv"), help="CSV file path (absolute or relative)")
    parser.add_argument("--output", type=str, default=str(MODEL_PATH), help="Path to save the trained pipeline")
    args = parser.parse_args()

    # Load and clean data
    df = load_data(args.data)
    df = clean_data(df)

    # Split features and target
    X, y = split_features_target(df)

    # Build preprocessing transformer
    preprocessor = build_preprocess_transformer(X)

    # Assemble full pipeline (preprocess + model)
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # Train / test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if STRATIFY else None
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")

    # Save pipeline
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pipeline, output_path)
    print(f"Pipeline saved to {output_path}")

if __name__ == "__main__":
    main()
