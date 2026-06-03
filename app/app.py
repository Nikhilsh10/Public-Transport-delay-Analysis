# app/app.py
"""Streamlit dashboard for Transport Delay regression model.
Features:
- Input form for all model features (categorical as selectbox, numeric as number_input).
- Predicts arrival delay minutes.
- Shows delay severity (Low / Moderate / High) based on thresholds.
- Visualizes feature importance (bar chart).
- Shows overall delay distribution histogram from the training data.
"""
import streamlit as st
import pandas as pd
import joblib
import pathlib
import matplotlib.pyplot as plt

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "public_transport_delays.csv"

# Load pipeline
# Load pipeline
@st.cache_resource
def load_pipeline():
    """Load the serialized sklearn pipeline.
    If loading fails (missing file, unpickling error, or version mismatch),
    we delete the corrupted model file, retrain a fresh model, and load it again.
    This ensures the Streamlit app always starts.
    """
    import os
    # If the model file does not exist, train a new one
    if not MODEL_PATH.is_file():
        st.info("Model file not found. Training a new model…")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        from src.train import main as train_main
        train_main()
        return joblib.load(MODEL_PATH)
    # Try loading the existing model
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Model loading failed ({e}). Deleting corrupted file and retraining…")
        # Remove corrupted file
        try:
            os.remove(MODEL_PATH)
        except OSError:
            pass
        # Retrain
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        from src.train import main as train_main
        train_main()
        return joblib.load(MODEL_PATH)





pipeline = load_pipeline()
preprocess = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# Helper to get feature names after one‑hot encoding
feature_names = preprocess.get_feature_names_out()

# Load a single row to infer column types for UI generation
sample_df = pd.read_csv(DATA_PATH).head(1)
# Apply same cleaning as in preprocessing
sample_df["event_type"] = sample_df["event_type"].fillna("None")
# Drop columns that were removed during training
cols_to_drop = [
    "actual_arrival_delay_min",
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
X_raw = sample_df.drop(columns=cols_to_drop, errors="ignore")

st.title("🚆 Transport Arrival Delay Predictor")
st.write("Enter trip details to predict the expected arrival delay (in minutes).")

# Build input form
input_data = {}
for col in X_raw.columns:
    if X_raw[col].dtype == "object":
        # categorical – show unique values from the full dataset for better coverage
        options = pd.read_csv(DATA_PATH)[col].dropna().unique().tolist()
        # Ensure the placeholder "None" is present for event_type
        if col == "event_type" and "None" not in options:
            options.append("None")
        input_data[col] = st.selectbox(col, options)
    else:
        # numeric – use the sample value as default
        default_val = float(X_raw[col].iloc[0]) if not X_raw[col].empty else 0.0
        input_data[col] = st.number_input(col, value=default_val)

if st.button("Predict Delay"):
    df_input = pd.DataFrame([input_data])
    pred = pipeline.predict(df_input)[0]
    st.success(f"**Predicted arrival delay:** {pred:.2f} minutes")

    # Determine severity
    if pred < 5:
        severity = "Low"
    elif pred < 15:
        severity = "Moderate"
    else:
        severity = "High"
    st.info(f"**Delay severity:** {severity}")

    # Feature importance chart
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    importance_series = pd.Series(importances, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False).head(20)  # top 20
    st.bar_chart(importance_series)

    # Delay distribution histogram (from training data)
    st.subheader("Historical Arrival Delay Distribution")
    target_series = pd.read_csv(DATA_PATH)["actual_arrival_delay_min"].dropna()
    fig, ax = plt.subplots()
    ax.hist(target_series, bins=30, edgecolor="black")
    ax.set_xlabel("Arrival delay (minutes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
