"""
Evaluation script for Heart Disease Prediction model.

Usage:
    python -m src.evaluate --data data/raw/heart_disease.csv --model models/best_model.joblib
"""
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_DATA_PATH, MODEL_PATH, FIG_DIR, RANDOM_SEED, TEST_SIZE, THRESHOLD,
    NUMERIC_COLS, CATEGORICAL_COLS
)
from .data_utils import load_data, standardize_columns, validate_schema, clean_data
from .modeling import (
    evaluate_model, threshold_analysis, compute_permutation_importance,
    get_roc_curve_data
)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal (0)', 'Heart Disease (1)'],
        yticklabels=['Normal (0)', 'Heart Disease (1)']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_roc_curve(fpr, tpr, auc_score: float, save_path: Path):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_permutation_importance(
    importances: np.ndarray,
    stds: np.ndarray,
    feature_names: list,
    save_path: Path,
    top_n: int = 15
):
    """Plot and save permutation importance."""
    # Sort and take top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    top_stds = stds[indices]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_importances, xerr=top_stds, align='center', color='steelblue')
    plt.yticks(y_pos, top_names)
    plt.xlabel('Mean Decrease in ROC-AUC')
    plt.title(f'Top {top_n} Feature Importances (Permutation)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_target_distribution(y, save_path: Path):
    """Plot and save target distribution."""
    target_counts = y.value_counts().sort_index()
    
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c']
    bars = plt.bar(['Normal (0)', 'Heart Disease (1)'], target_counts.values, color=colors)
    plt.xlabel('Target Class')
    plt.ylabel('Count')
    plt.title('Target Variable Distribution')
    
    for bar, count in zip(bars, target_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_correlation_heatmap(df, numeric_cols, save_path: Path):
    """Plot and save correlation heatmap."""
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='RdBu_r', 
        center=0,
        vmin=-1, vmax=1
    )
    plt.title('Correlation Heatmap (Numeric Features)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_numeric_distributions(df, numeric_cols, save_path: Path):
    """Plot histograms for numeric features."""
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], color='steelblue', edgecolor='white')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide extra subplots
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_numeric_boxplots(df, y, numeric_cols, save_path: Path):
    """Plot boxplots of numeric features by target class."""
    plot_df = df.copy()
    plot_df['target'] = y.values
    
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            plot_df.boxplot(column=col, by='target', ax=axes[i])
            axes[i].set_xlabel('Target')
            axes[i].set_ylabel(col)
            axes[i].set_title(f'{col} by Target')
    
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def plot_categorical_distributions(df, categorical_cols, save_path: Path):
    """Plot count plots for categorical features."""
    n_cols = len(categorical_cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts().sort_index()
            axes[i].bar(value_counts.index.astype(str), value_counts.values, color='steelblue')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{col} Distribution')
    
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate heart disease prediction model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help='Path to the heart disease CSV dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(MODEL_PATH),
        help='Path to the trained model (.joblib)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Classification threshold (if not specified, will be determined via analysis)'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Heart Disease Prediction - Model Evaluation")
    print("=" * 60)
    
    # Load model
    model_path = Path(args.model)
    print(f"\n[1/6] Loading model from: {model_path}")
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run training first: python -m src.train")
        sys.exit(1)
    
    pipeline = joblib.load(model_path)
    print("      Model loaded successfully.")
    
    # Load and process data
    print(f"\n[2/6] Loading data from: {args.data}")
    try:
        df = load_data(args.data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    df = standardize_columns(df)
    validate_schema(df)
    X, y, stats = clean_data(df)
    
    # Reproducible split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Test set: {len(X_test)} samples")
    
    # Create figures directory
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Threshold analysis
    print(f"\n[3/6] Threshold analysis...")
    threshold_df, recommended_threshold = threshold_analysis(pipeline, X_test, y_test)
    print("\nThreshold Analysis:")
    print(threshold_df.to_string(index=False))
    
    final_threshold = args.threshold if args.threshold else recommended_threshold
    print(f"\nRecommended threshold: {recommended_threshold:.2f}")
    print(f"Using threshold: {final_threshold:.2f}")
    print("(Prioritizing recall to minimize false negatives in medical screening)")
    
    # Evaluate model
    print(f"\n[4/6] Evaluating model on test set...")
    metrics = evaluate_model(pipeline, X_test, y_test, threshold=final_threshold)
    
    print(f"\n{'=' * 60}")
    print("TEST SET METRICS")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"{'=' * 60}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save plots
    print(f"\n[5/6] Saving evaluation plots...")
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        FIG_DIR / 'confusion_matrix.png'
    )
    
    # ROC curve
    fpr, tpr, _ = get_roc_curve_data(pipeline, X_test, y_test)
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], FIG_DIR / 'roc_curve.png')
    
    # Permutation importance
    print(f"\n[6/6] Computing permutation importance...")
    importances, stds, feature_names = compute_permutation_importance(
        pipeline, X_test, y_test
    )
    plot_permutation_importance(
        importances, stds, feature_names,
        FIG_DIR / 'permutation_importance.png'
    )
    
    # EDA Plots
    print(f"\n[7/7] Generating EDA visualizations...")
    
    # Target distribution (use full dataset)
    plot_target_distribution(y, FIG_DIR / 'target_distribution.png')
    
    # Correlation heatmap
    plot_correlation_heatmap(X, NUMERIC_COLS, FIG_DIR / 'corr_heatmap.png')
    
    # Numeric distributions
    plot_numeric_distributions(X, NUMERIC_COLS, FIG_DIR / 'numeric_distributions.png')
    
    # Numeric boxplots by target
    plot_numeric_boxplots(X, y, NUMERIC_COLS, FIG_DIR / 'numeric_boxplots.png')
    
    # Categorical distributions
    plot_categorical_distributions(X, CATEGORICAL_COLS, FIG_DIR / 'categorical_distributions.png')
    
    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
