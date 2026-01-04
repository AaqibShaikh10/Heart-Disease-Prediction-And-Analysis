"""
Configuration constants for Heart Disease Prediction project.
"""
from pathlib import Path

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Random seed for reproducibility
RANDOM_SEED = 42

# File paths
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Column definitions
# Numeric columns (continuous or count-like variables)
NUMERIC_COLS = [
    "age",
    "resting bp s",
    "cholesterol",
    "max heart rate",
    "oldpeak"
]

# Categorical columns (discrete categories)
CATEGORICAL_COLS = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting ecg",
    "exercise angina",
    "ST slope"
]

# Target column
TARGET_COL = "target"

# All feature columns
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Default classification threshold
THRESHOLD = 0.50

# Model training parameters
TEST_SIZE = 0.20
CV_FOLDS = 5

# Columns where zero values should be treated as missing
ZERO_AS_MISSING_COLS = ["resting bp s", "cholesterol"]
