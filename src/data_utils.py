"""
Data loading, validation, and cleaning utilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from .config import (
    NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL, FEATURE_COLS,
    ZERO_AS_MISSING_COLS
)


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load CSV data from the given path.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Raw DataFrame
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    df = pd.read_csv(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by stripping whitespace and converting to lowercase
    for matching, but keep original format for consistency.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame has all required columns.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = set(FEATURE_COLS + [TARGET_COL])
    found_cols = set(df.columns)
    
    # Case-insensitive matching for comparison
    found_cols_lower = {c.lower().strip() for c in found_cols}
    required_cols_lower = {c.lower().strip() for c in required_cols}
    
    missing = required_cols_lower - found_cols_lower
    
    if missing:
        # Find the original case versions for the error message
        missing_original = [c for c in required_cols if c.lower().strip() in missing]
        raise ValueError(
            f"Missing required columns.\n"
            f"Expected: {sorted(required_cols)}\n"
            f"Found: {sorted(found_cols)}\n"
            f"Missing: {sorted(missing_original)}"
        )


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Clean the dataset:
    - Drop duplicates
    - Convert columns to appropriate types
    - Treat resting bp s==0 and cholesterol==0 as missing (NaN)
    - Separate features and target
    - Ensure target is binary (0/1)
    
    Args:
        df: Input DataFrame (already validated)
        
    Returns:
        Tuple of (X features DataFrame, y target Series, cleaning stats dict)
    """
    stats = {}
    df = df.copy()
    
    # Record initial shape
    initial_rows = len(df)
    
    # Drop duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    stats["duplicates_removed"] = duplicates_removed
    
    # Convert numeric columns to numeric type (coerce errors to NaN)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert categorical columns to numeric type
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Treat zeros as missing for specific columns
    for col in ZERO_AS_MISSING_COLS:
        if col in df.columns:
            zeros_count = (df[col] == 0).sum()
            stats[f"{col}_zeros_as_nan"] = int(zeros_count)
            df.loc[df[col] == 0, col] = np.nan
    
    # Count missing values per column
    missing_counts = df.isnull().sum().to_dict()
    stats["missing_values"] = {k: int(v) for k, v in missing_counts.items() if v > 0}
    
    # Separate features and target
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Ensure target is binary (0/1)
    # If any value > 1, map to 1 (assuming positive class)
    y = (y > 0).astype(int)
    
    stats["final_rows"] = len(df)
    stats["target_distribution"] = y.value_counts().to_dict()
    
    return X, y, stats


def get_feature_names_from_transformer(preprocessor, numeric_cols, categorical_cols):
    """
    Extract feature names after ColumnTransformer transformation.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    # Numeric features (keep original names)
    feature_names.extend(numeric_cols)
    
    # Categorical features (get one-hot encoded names)
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    except Exception:
        # Fallback if structure is different
        for col in categorical_cols:
            feature_names.append(col)
    
    return feature_names
