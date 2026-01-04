"""
FastAPI application for Heart Disease Prediction.

Usage:
    uvicorn api.app:app --reload
"""
from pathlib import Path
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictionRequest, PredictionResponse, HealthResponse

# Path to the trained model (relative to project root)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"

# Default threshold (can be adjusted)
DEFAULT_THRESHOLD = 0.50

# Feature columns in the exact order expected by the model
FEATURE_COLS = [
    "age", "resting bp s", "cholesterol", "max heart rate", "oldpeak",
    "sex", "chest pain type", "fasting blood sugar", "resting ecg",
    "exercise angina", "ST slope"
]

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
_model = None


def get_model():
    """Load and cache the trained model."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at {MODEL_PATH}. Please run training first: python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict heart disease probability.
    
    Accepts JSON with feature values matching the dataset column names.
    Keys with spaces (e.g., "chest pain type") are supported.
    """
    try:
        model = get_model()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Build feature DataFrame in exact column order
    row_data = {
        "age": request.age,
        "resting bp s": request.resting_bp_s,
        "cholesterol": request.cholesterol,
        "max heart rate": request.max_heart_rate,
        "oldpeak": request.oldpeak,
        "sex": request.sex,
        "chest pain type": request.chest_pain_type,
        "fasting blood sugar": request.fasting_blood_sugar,
        "resting ecg": request.resting_ecg,
        "exercise angina": request.exercise_angina,
        "ST slope": request.st_slope
    }
    
    df = pd.DataFrame([row_data])
    
    # Reorder columns to match training order
    df = df[FEATURE_COLS]
    
    try:
        # Get probability of positive class (heart disease)
        proba = model.predict_proba(df)[0, 1]
        prediction = 1 if proba >= DEFAULT_THRESHOLD else 0
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(float(proba), 4),
            threshold=DEFAULT_THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "reports": "/reports"
    }


# Reports endpoints
from fastapi.responses import FileResponse
import base64

# Figures directory
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@app.get("/reports")
async def get_reports():
    """Get list of available report figures and model info."""
    figures = []
    
    # Check for each expected figure
    figure_files = [
        ("target_distribution.png", "Target Distribution", "Distribution of heart disease vs normal cases in the dataset"),
        ("corr_heatmap.png", "Correlation Heatmap", "Correlation matrix of numeric features"),
        ("confusion_matrix.png", "Confusion Matrix", "Model predictions vs actual values on test set"),
        ("roc_curve.png", "ROC Curve", "Receiver Operating Characteristic curve showing model performance"),
        ("permutation_importance.png", "Feature Importance", "Top features ranked by permutation importance"),
        ("numeric_distributions.png", "Numeric Distributions", "Histograms of numeric features"),
        ("numeric_boxplots.png", "Numeric Boxplots", "Boxplots of numeric features by target class"),
        ("categorical_distributions.png", "Categorical Distributions", "Count plots of categorical features"),
    ]
    
    for filename, title, description in figure_files:
        filepath = FIG_DIR / filename
        if filepath.exists():
            figures.append({
                "filename": filename,
                "title": title,
                "description": description,
                "available": True
            })
        else:
            figures.append({
                "filename": filename,
                "title": title,
                "description": description,
                "available": False
            })
    
    # Check if model exists
    model_exists = MODEL_PATH.exists()
    
    # Check if dataset exists
    dataset_path = DATA_DIR / "heart_disease.csv"
    dataset_info = None
    if dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            dataset_info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
        except Exception:
            pass
    
    return {
        "figures": figures,
        "model_trained": model_exists,
        "dataset_loaded": dataset_info is not None,
        "dataset_info": dataset_info
    }


@app.get("/reports/figures/{filename}")
async def get_figure(filename: str):
    """Serve a specific figure image."""
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = FIG_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Figure not found: {filename}. Run evaluation first: python -m src.evaluate"
        )
    
    return FileResponse(filepath, media_type="image/png")


@app.get("/reports/dataset-preview")
async def get_dataset_preview():
    """Get a preview of the dataset (first 10 rows)."""
    dataset_path = DATA_DIR / "heart_disease.csv"
    
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Place heart_disease.csv in data/raw/"
        )
    
    try:
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip()
        
        # Get first 10 rows as dict
        preview = df.head(10).to_dict(orient="records")
        
        # Get basic stats
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_distribution": df["target"].value_counts().to_dict() if "target" in df.columns else None,
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return {
            "columns": list(df.columns),
            "preview": preview,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")


@app.get("/reports/model-metrics")
async def get_model_metrics():
    """Get model performance metrics including classification report."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
    
    # Check if model exists
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model not found. Run training first: python -m src.train"
        )
    
    # Check if dataset exists
    dataset_path = DATA_DIR / "heart_disease.csv"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Place heart_disease.csv in data/raw/"
        )
    
    try:
        # Load model
        model = joblib.load(MODEL_PATH)
        
        # Load and preprocess data
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip()
        
        # Define columns
        numeric_cols = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
        categorical_cols = ["sex", "chest pain type", "fasting blood sugar", 
                           "resting ecg", "exercise angina", "ST slope"]
        feature_cols = numeric_cols + categorical_cols
        
        # Clean data
        df = df.drop_duplicates()
        for col in ["resting bp s", "cholesterol"]:
            df.loc[df[col] == 0, col] = df[col].median()
        
        X = df[feature_cols]
        y = (df["target"] > 0).astype(int)
        
        # Same split as training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        }
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Format classification report for frontend
        classification_data = {
            "normal": {
                "precision": round(report["0"]["precision"], 4),
                "recall": round(report["0"]["recall"], 4),
                "f1_score": round(report["0"]["f1-score"], 4),
                "support": int(report["0"]["support"])
            },
            "heart_disease": {
                "precision": round(report["1"]["precision"], 4),
                "recall": round(report["1"]["recall"], 4),
                "f1_score": round(report["1"]["f1-score"], 4),
                "support": int(report["1"]["support"])
            },
            "macro_avg": {
                "precision": round(report["macro avg"]["precision"], 4),
                "recall": round(report["macro avg"]["recall"], 4),
                "f1_score": round(report["macro avg"]["f1-score"], 4),
                "support": int(report["macro avg"]["support"])
            },
            "weighted_avg": {
                "precision": round(report["weighted avg"]["precision"], 4),
                "recall": round(report["weighted avg"]["recall"], 4),
                "f1_score": round(report["weighted avg"]["f1-score"], 4),
                "support": int(report["weighted avg"]["support"])
            }
        }
        
        return {
            "metrics": metrics,
            "classification_report": classification_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")

