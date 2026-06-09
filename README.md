# Public Transport Delay Analysis

**A production‑ready machine learning pipeline for predicting public‑transport arrival delays.**

**Live Demo:** (https://public-transport-delay-analysis-st4p559gagmrxp5yjf8scr.streamlit.app/)

- Built reusable training and inference modules.
- Implemented a single sklearn preprocessing pipeline (one‑hot encoding + numeric passthrough).
- Trained a `RandomForestRegressor` on the full Kaggle dataset.
- Evaluation metrics: MAE = 4.73 min, RMSE = 7.12 min, R² = 0.31.
---

## Project Overview
A production‑ready machine‑learning workflow that predicts **arrival delay minutes** for public‑transport trips using features such as transport type, weather, events, and traffic congestion.

## Dataset Description
- **Source:** Kaggle – *Public Transport Delays with Weather and Events*.
- **Rows:** 2 000
- **Columns:** 25 (including target `actual_arrival_delay_min`).

## Model Architecture
- **Preprocessing:** `ColumnTransformer` → one‑hot encode categorical columns, passthrough numeric columns.
- **Model:** `RandomForestRegressor` (200 trees, `random_state=42`).
- **Pipeline:** Single sklearn `Pipeline` containing preprocessing + model, saved with `joblib`.

## Evaluation Metrics
| Metric | Value |
|--------|-------|
| MAE (minutes) | **4.73** |
| RMSE (minutes) | **7.12** |
| R² | **0.31** |

## Local Setup
```bash
# Clone the repo
git clone https://github.com/Nikhilsh10/Public-Transport-delay-Analysis.git
cd Public-Transport-delay-Analysis

# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install -r requirements.txt

# Ensure the dataset is present at data/public_transport_delays.csv
python -m src.train   # trains and saves models/model.joblib
python -m src.predict --input public_transport_delays.csv   # inference example
```

## Streamlit Dashboard
```bash
streamlit run app/app.py
```
- Input trip features → predicted arrival‑delay minutes.
- Delay severity (Low / Moderate / High) shown.
- Feature‑importance bar chart and historical delay histogram displayed.

### Deploy to Streamlit Community Cloud
1. Push the repository to GitHub (`main` branch).
2. In Streamlit Cloud, create a new app:
   - **Repository:** `Nikhilsh10/Public-Transport-delay-Analysis`
   - **Branch:** `main`
   - **Main file:** `app/app.py`
3. Streamlit will install dependencies from `requirements.txt` and launch the app.

## Future Improvements
- Explore richer temporal features (hour of day, day of week).
- Test alternative models (Gradient Boosting, XGBoost, Neural Networks).
- Incorporate live weather/event APIs for up‑to‑date predictions.
- Perform cross‑validation and uncertainty quantification.

---

© 2026 Nikhil Sharma – released under the MIT License.
