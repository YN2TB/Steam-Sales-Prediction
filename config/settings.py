"""config/settings.py"""

from pathlib import Path

DATA_RAW_DIR = Path("data/raw")
DATA_PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("data/result")
MODELS_DIR = Path("models")

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": 42,
}

MIN_SAMPLES_TO_TRAIN = 5
DEFAULT_N_PREDICTIONS = 3
