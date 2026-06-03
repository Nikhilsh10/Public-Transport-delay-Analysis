# Central configuration for the Transport Delay regression project
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

# Hyper‑parameters (can be overridden via CLI args)
RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.2
# No stratification for regression targets
STRATIFY = False
TARGET_COLUMN = "actual_arrival_delay_min"
