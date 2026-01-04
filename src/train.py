"""
Training script for Heart Disease Prediction model.

Usage:
    python -m src.train --data data/raw/heart_disease.csv
"""
import argparse
import sys
from pathlib import Path
import joblib

from .config import DEFAULT_DATA_PATH, MODEL_PATH, FIG_DIR, RANDOM_SEED, TEST_SIZE
from .data_utils import load_data, standardize_columns, validate_schema, clean_data
from .modeling import (
    compare_models, tune_model, build_pipeline, get_models,
    train_test_split
)


def main():
    parser = argparse.ArgumentParser(
        description='Train heart disease prediction model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help='Path to the heart disease CSV dataset'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Heart Disease Prediction - Model Training")
    print("=" * 60)
    
    # Load and validate data
    print(f"\n[1/6] Loading data from: {args.data}")
    try:
        df = load_data(args.data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the dataset file exists at the specified path.")
        sys.exit(1)
    
    df = standardize_columns(df)
    
    print(f"[2/6] Validating schema...")
    try:
        validate_schema(df)
        print("      Schema validation passed.")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"[3/6] Cleaning data...")
    X, y, stats = clean_data(df)
    print(f"      - Duplicates removed: {stats['duplicates_removed']}")
    print(f"      - Final dataset size: {stats['final_rows']} rows")
    print(f"      - Target distribution: {stats['target_distribution']}")
    if stats.get('missing_values'):
        print(f"      - Missing values (including zeros as NaN): {stats['missing_values']}")
    
    # Train/test split
    print(f"\n[4/6] Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Training set: {len(X_train)} samples")
    print(f"      Test set: {len(X_test)} samples")
    
    # Model comparison
    print(f"\n[5/6] Comparing models with {5}-fold cross-validation...")
    results = compare_models(X_train, y_train)
    print("\nModel Comparison Results (sorted by ROC-AUC):")
    print("-" * 60)
    print(results.to_string(index=False))
    print("-" * 60)
    
    # Select top 2 models for tuning (excluding DummyClassifier)
    top_models = results[results['Model'] != 'DummyClassifier'].head(2)['Model'].tolist()
    print(f"\nTop 2 models for tuning: {top_models}")
    
    # Hyperparameter tuning
    print(f"\n[6/6] Hyperparameter tuning...")
    best_model = None
    best_score = 0
    best_name = None
    
    for model_name in top_models:
        print(f"\n      Tuning {model_name}...")
        try:
            pipeline, params, score = tune_model(model_name, X_train, y_train)
            print(f"      Best params: {params}")
            print(f"      Best CV ROC-AUC: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_name = model_name
        except Exception as e:
            print(f"      Error tuning {model_name}: {e}")
    
    if best_model is None:
        print("\nERROR: No model could be trained successfully.")
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_name}")
    print(f"Best CV ROC-AUC: {best_score:.4f}")
    print(f"{'=' * 60}")
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    
    print("\nTraining complete!")
    print("Next step: Run evaluation with:")
    print(f"  python -m src.evaluate --data {args.data} --model {MODEL_PATH}")


if __name__ == '__main__':
    main()
