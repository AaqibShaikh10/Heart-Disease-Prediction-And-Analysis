"""
Prediction script for Heart Disease Prediction model.

Usage:
    python -m src.predict --model models/best_model.joblib --json '{"age": 55, ...}'
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import joblib

from .config import MODEL_PATH, FEATURE_COLS, THRESHOLD


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions with the trained heart disease model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(MODEL_PATH),
        help='Path to the trained model (.joblib)'
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='JSON string with feature values (keys must match dataset column names exactly)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=THRESHOLD,
        help=f'Classification threshold (default: {THRESHOLD})'
    )
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run training first: python -m src.train")
        sys.exit(1)
    
    pipeline = joblib.load(model_path)
    
    # Parse input JSON
    try:
        input_data = json.loads(args.json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        sys.exit(1)
    
    # Validate that all required features are present
    missing_features = []
    for col in FEATURE_COLS:
        # Check for exact match or case-insensitive match
        found = False
        for key in input_data.keys():
            if key.strip().lower() == col.lower():
                found = True
                break
        if not found:
            missing_features.append(col)
    
    if missing_features:
        print(f"ERROR: Missing required features: {missing_features}")
        print(f"Required features: {FEATURE_COLS}")
        sys.exit(1)
    
    # Create DataFrame with exact column names expected by pipeline
    row_data = {}
    for col in FEATURE_COLS:
        # Find matching key in input (case-insensitive)
        for key, value in input_data.items():
            if key.strip().lower() == col.lower():
                row_data[col] = value
                break
    
    df = pd.DataFrame([row_data])
    
    # Make prediction
    try:
        proba = pipeline.predict_proba(df)[0, 1]
        prediction = 1 if proba >= args.threshold else 0
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        sys.exit(1)
    
    # Output result
    result = {
        "prediction": int(prediction),
        "probability": round(float(proba), 4),
        "threshold": args.threshold,
        "label": "Heart Disease" if prediction == 1 else "Normal"
    }
    
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
