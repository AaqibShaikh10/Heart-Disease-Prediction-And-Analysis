# ❤️ Heart Disease Prediction and Analysis

> **A Complete Data Mining Project using Machine Learning**

This project implements a full-stack heart disease prediction system using data mining techniques. It includes a Python machine learning pipeline, FastAPI backend, and React frontend for predicting heart disease risk based on clinical and physiological features.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61dafb?style=flat-square&logo=react)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Algorithms Used](#-algorithms-used)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Technical Details](#-technical-details)
- [Limitations](#-limitations)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

Heart disease is one of the leading causes of death worldwide. Early detection and prediction can significantly improve patient outcomes. This project applies **data mining and machine learning techniques** to predict the presence of heart disease based on 11 clinical features.

### Key Objectives:
1. **Exploratory Data Analysis (EDA)** - Understand patterns in the data
2. **Data Preprocessing** - Handle missing values and transformations
3. **Model Comparison** - Evaluate 5 ML algorithms using cross-validation
4. **Hyperparameter Tuning** - Optimize the best performing models
5. **Model Evaluation** - Assess performance using various metrics
6. **Deployment** - Create a web interface for predictions

---

## ✨ Features

### Machine Learning Pipeline
- ✅ Automated data cleaning and preprocessing
- ✅ Comparison of 5 classification algorithms
- ✅ Cross-validation with stratified K-fold
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Threshold analysis for optimal classification
- ✅ Permutation-based feature importance

### Web Application
- ✅ Interactive prediction form
- ✅ Real-time API predictions
- ✅ Model performance dashboard
- ✅ Dataset visualization
- ✅ Classification report display
- ✅ Generated figure gallery

### Visualizations
- 📊 Target distribution plot
- 📈 Correlation heatmap
- 📉 ROC curve with AUC score
- 🔢 Confusion matrix
- 📋 Feature importance chart
- 📊 Numeric feature distributions
- 📦 Boxplots by target class
- 📊 Categorical feature distributions

---

## 📁 Project Structure

```
Heart Disease Prediction and Analysis Project/
│
├── data/
│   ├── raw/                     # Raw dataset (heart_disease.csv)
│   └── processed/               # Processed data files
│
├── models/
│   └── best_model.joblib        # Trained model (generated)
│
├── notebooks/
│   ├── 01_eda.py                # Exploratory Data Analysis
│   └── 02_modeling.py           # Model training workflow
│
├── reports/
│   └── figures/                 # Generated visualization plots
│
├── src/                         # Python source code
│   ├── __init__.py
│   ├── config.py                # Configuration constants
│   ├── data_utils.py            # Data loading & preprocessing
│   ├── modeling.py              # ML modeling utilities
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── predict.py               # CLI prediction script
│
├── api/                         # FastAPI backend
│   ├── __init__.py
│   ├── app.py                   # API endpoints
│   └── schemas.py               # Pydantic models
│
├── web/                         # React frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── styles.css
│       └── pages/
│           ├── PredictPage.tsx
│           ├── ReportsPage.tsx
│           └── AboutPage.tsx
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📊 Dataset

### Source
The dataset is based on the **UCI Heart Disease dataset**, combining data from:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- University Hospital, Zurich
- V.A. Medical Center, Long Beach

### Features (11 Input Variables)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Numeric | Age in years |
| 2 | `sex` | Categorical | 0 = Female, 1 = Male |
| 3 | `chest pain type` | Categorical | 1-4 (Typical angina to Asymptomatic) |
| 4 | `resting bp s` | Numeric | Resting blood pressure (mm Hg) |
| 5 | `cholesterol` | Numeric | Serum cholesterol (mg/dl) |
| 6 | `fasting blood sugar` | Categorical | 0 = ≤120 mg/dl, 1 = >120 mg/dl |
| 7 | `resting ecg` | Categorical | 0-2 (Normal to LV hypertrophy) |
| 8 | `max heart rate` | Numeric | Maximum heart rate achieved |
| 9 | `exercise angina` | Categorical | 0 = No, 1 = Yes |
| 10 | `oldpeak` | Numeric | ST depression induced by exercise |
| 11 | `ST slope` | Categorical | 1-3 (Upsloping to Downsloping) |

### Target Variable
- `target`: 0 = No heart disease, 1 = Heart disease present

### Dataset Statistics
- **Total samples**: 1190 (before cleaning)
- **After cleaning**: 918 samples (duplicates removed)
- **Target distribution**: 508 positive (55%), 410 negative (45%)

---

## 🧠 Algorithms Used

This project compares **5 classification algorithms** commonly taught in Data Mining courses:

| # | Algorithm | Description | ROC-AUC |
|---|-----------|-------------|---------|
| 1 | **Naive Bayes** | Probabilistic classifier based on Bayes' theorem | 90.9% |
| 2 | **Logistic Regression** | Linear model for binary classification | 92.2% |
| 3 | **Decision Tree** | Tree-based rules for classification | 78.5% |
| 4 | **K-Nearest Neighbors (KNN)** | Classifies based on nearest neighbors | 89.8% |
| 5 | **Random Forest** | Ensemble of multiple decision trees | **92.5%** |

### Best Model: Random Forest Classifier

After hyperparameter tuning:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    random_state=42
)
```

### Data Preprocessing Pipeline

1. **Duplicate Removal**: 272 duplicate rows removed
2. **Missing Value Treatment**:
   - Zeros in `resting bp s` and `cholesterol` treated as missing
   - Numeric features: Median imputation + StandardScaler
   - Categorical features: Most frequent imputation + OneHotEncoder

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)

### Step 1: Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Add the Dataset

Place `heart_disease.csv` in `data/raw/` directory.

### Step 3: Train the Model

```bash
python -m src.train --data data/raw/heart_disease.csv
```

### Step 4: Generate Evaluation Reports

```bash
python -m src.evaluate --data data/raw/heart_disease.csv --model models/best_model.joblib
```

### Step 5: Start the Backend API

```bash
uvicorn api.app:app --reload
```
API available at: `http://localhost:8000`

### Step 6: Start the Frontend

```bash
cd web
npm install
npm run dev
```
Frontend available at: `http://localhost:5173`

---

## 💻 Usage

### Command Line

**Training:**
```bash
python -m src.train --data data/raw/heart_disease.csv
```

**Evaluation:**
```bash
python -m src.evaluate --data data/raw/heart_disease.csv --model models/best_model.joblib
```

**Single Prediction:**
```bash
python -m src.predict --model models/best_model.joblib --json '{"age": 55, "sex": 1, "chest pain type": 4, "resting bp s": 140, "cholesterol": 250, "fasting blood sugar": 0, "resting ecg": 0, "max heart rate": 150, "exercise angina": 1, "oldpeak": 1.5, "ST slope": 2}'
```

### Web Interface

1. **Predict Page**: Enter patient data and get predictions
2. **Reports Page**: View model metrics and visualizations
3. **About Page**: Learn about the project

---

## 📡 API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Make a prediction |
| GET | `/reports` | Get available figures |
| GET | `/reports/model-metrics` | Get classification report |

### Prediction Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": 1, "chest pain type": 4, ...}'
```

---

## 📈 Model Performance

### Cross-Validation Results (5-Fold)

| Model | ROC-AUC | F1-Score |
|-------|---------|----------|
| **Random Forest** | **92.5%** | **86.3%** |
| Logistic Regression | 92.2% | 86.3% |
| Naive Bayes | 90.9% | 76.1% |
| KNN | 89.8% | 85.9% |
| Decision Tree | 78.5% | 80.7% |

### Test Set Performance (Random Forest)

| Metric | Value |
|--------|-------|
| **Accuracy** | 89.13% |
| **Precision** | 87.96% |
| **Recall** | 93.14% |
| **F1-Score** | 90.48% |
| **ROC-AUC** | 92.65% |

### Classification Report

```
              precision    recall  f1-score   support

     Normal       0.91      0.84      0.87        82
Heart Disease     0.88      0.93      0.90       102

    accuracy                          0.89       184
```

### Confusion Matrix

|  | Predicted Normal | Predicted Disease |
|---|:---:|:---:|
| **Actual Normal** | 69 | 13 |
| **Actual Disease** | 7 | 95 |

---

## 🛠 Technical Details

### Dependencies

**Python:**
- numpy, pandas - Data manipulation
- scikit-learn - Machine learning
- matplotlib, seaborn - Visualization
- joblib - Model serialization
- fastapi, uvicorn - Web API
- pydantic - Data validation

**Frontend:**
- React 18, TypeScript, Vite

---

## ⚠️ Limitations

1. **Dataset Size**: ~918 samples may not capture all variations
2. **Geographic Bias**: Data from Western medical centers
3. **Feature Limitations**: Only 11 features included
4. **Binary Classification**: Does not predict disease severity

---

## ⚕️ Disclaimer

> **⚠️ This tool is for EDUCATIONAL PURPOSES ONLY**

This prediction model is **NOT** intended to replace professional medical diagnosis. Always consult a qualified healthcare professional for medical decisions.

This project was developed as part of a Data Warehousing and Data Mining course in Fall 2025 to demonstrate machine learning concepts and data mining techniques.

---

## 📄 License

This project is for educational purposes. The dataset is publicly available from the UCI Machine Learning Repository.
