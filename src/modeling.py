"""
Model building, training, and evaluation utilities.
Uses sklearn Pipeline and ColumnTransformer for preprocessing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.inspection import permutation_importance

from .config import (
    NUMERIC_COLS, CATEGORICAL_COLS, RANDOM_SEED, CV_FOLDS
)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.
    
    Numeric: Impute with median, then StandardScaler
    Categorical: Impute with most_frequent, then OneHotEncoder
    
    Returns:
        Fitted ColumnTransformer
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_COLS),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ])
    
    return preprocessor


def get_models() -> Dict[str, Any]:
    """
    Get a dictionary of models to compare.
    
    Returns:
        Dict mapping model names to model instances
    """
    models = {
        'DummyClassifier': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
        'LogisticRegression_Balanced': LogisticRegression(
            max_iter=500, class_weight='balanced', random_state=RANDOM_SEED
        ),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        'SVC': SVC(probability=True, random_state=RANDOM_SEED),
        'KNeighbors': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_SEED)
    }
    return models


def build_pipeline(model) -> Pipeline:
    """
    Build a complete pipeline with preprocessing and a classifier.
    
    Args:
        model: sklearn classifier instance
        
    Returns:
        Complete Pipeline
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        DataFrame with model comparison results (ROC-AUC, F1)
    """
    models = get_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    results = []
    
    for name, model in models.items():
        pipeline = build_pipeline(model)
        
        # Cross-validation for ROC-AUC
        try:
            roc_auc_scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring='roc_auc'
            )
        except Exception:
            roc_auc_scores = np.array([0.5] * CV_FOLDS)
        
        # Cross-validation for F1
        try:
            f1_scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring='f1'
            )
        except Exception:
            f1_scores = np.array([0.0] * CV_FOLDS)
        
        results.append({
            'Model': name,
            'ROC-AUC Mean': roc_auc_scores.mean(),
            'ROC-AUC Std': roc_auc_scores.std(),
            'F1 Mean': f1_scores.mean(),
            'F1 Std': f1_scores.std()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC Mean', ascending=False)
    
    return results_df


def get_tuning_grids() -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for top models.
    
    Returns:
        Dict mapping model names to parameter grids
    """
    grids = {
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5]
        },
        'LogisticRegression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        },
        'LogisticRegression_Balanced': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        }
    }
    return grids


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Perform GridSearchCV for a given model.
    
    Args:
        model_name: Name of the model (key in get_models())
        X_train: Training features
        y_train: Training target
        
    Returns:
        Tuple of (best pipeline, best params, best score)
    """
    models = get_models()
    grids = get_tuning_grids()
    
    model = models[model_name]
    pipeline = build_pipeline(model)
    
    param_grid = grids.get(model_name, {})
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a fitted pipeline on test data.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold
        
    Returns:
        Dict of evaluation metrics
    """
    # Get probabilities and predictions
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold': threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics


def threshold_analysis(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, float]:
    """
    Analyze precision/recall at different thresholds.
    Recommend a threshold that prioritizes recall.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_test: Test features
        y_test: Test target
        
    Returns:
        Tuple of (threshold analysis DataFrame, recommended threshold)
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        results.append({
            'Threshold': thresh,
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_test, y_pred)
        })
    
    results_df = pd.DataFrame(results)
    
    # Recommend threshold with recall >= 0.85, if possible
    # otherwise pick the one with best F1 where recall >= 0.80
    high_recall = results_df[results_df['Recall'] >= 0.85]
    if len(high_recall) > 0:
        # Among high recall, pick best F1
        best_idx = high_recall['F1'].idxmax()
        recommended = results_df.loc[best_idx, 'Threshold']
    else:
        # Fallback: best F1 with recall >= 0.75
        moderate_recall = results_df[results_df['Recall'] >= 0.75]
        if len(moderate_recall) > 0:
            best_idx = moderate_recall['F1'].idxmax()
            recommended = results_df.loc[best_idx, 'Threshold']
        else:
            # Just use 0.5
            recommended = 0.5
    
    return results_df, recommended


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute permutation importance for the fitted pipeline.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_test: Test features
        y_test: Test target
        n_repeats: Number of permutation repeats
        
    Returns:
        Tuple of (importance means, importance stds, feature names)
    """
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring='roc_auc'
    )
    
    # Get feature names from preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    
    feature_names = []
    # Numeric features
    feature_names.extend(NUMERIC_COLS)
    
    # Categorical features (one-hot encoded names)
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
        feature_names.extend(cat_features)
    except Exception:
        feature_names.extend(CATEGORICAL_COLS)
    
    return result.importances_mean, result.importances_std, feature_names


def get_roc_curve_data(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data for plotting.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_test: Test features
        y_test: Test target
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return fpr, tpr, thresholds
