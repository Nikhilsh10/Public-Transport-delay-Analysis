# Central configuration for the Transport Delay regression project
import pathlib
import os
import tempfile

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "public_transport_delays.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
# Determine a writable location for the model file. On deployment platforms (e.g., Streamlit Cloud) the repository directory may be read‑only.
# Attempt to use the 'models' folder; if it is not writable, fall back to the system temporary directory.
_default_model_path = BASE_DIR / "models" / "model.joblib"
try:
    _default_model_path.parent.mkdir(parents=True, exist_ok=True)
    # Try writing a dummy file to test write permission
    test_path = _default_model_path.with_suffix('.tmp')
    with open(test_path, 'w') as f:
        f.write('test')
    os.remove(test_path)
    MODEL_PATH = _default_model_path
except Exception:
    # Fallback to temp directory which is always writable
    MODEL_PATH = pathlib.Path(tempfile.gettempdir()) / "model.joblib"

# Hyper‑parameters (can be overridden via CLI args)
RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.2
# No stratification for regression targets
STRATIFY = False
TARGET_COLUMN = "actual_arrival_delay_min"
