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
- [Algorithms & Methodology](#-algorithms--methodology)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Technical Details](#-technical-details)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

Heart disease is one of the leading causes of death worldwide. Early detection and prediction can significantly improve patient outcomes. This project applies **data mining and machine learning techniques** to predict the presence of heart disease based on 11 clinical features.

### Key Objectives:
1. **Exploratory Data Analysis (EDA)** - Understand patterns and relationships in the data
2. **Data Preprocessing** - Handle missing values, duplicates, and data transformations
3. **Model Comparison** - Evaluate multiple ML algorithms using cross-validation
4. **Hyperparameter Tuning** - Optimize the best performing models
5. **Model Evaluation** - Assess performance using various metrics
6. **Deployment** - Create a web interface for predictions

---

## ✨ Features

### Machine Learning Pipeline
- ✅ Automated data cleaning and preprocessing
- ✅ Comparison of 7 different classifiers
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
├── 📂 data/
│   ├── raw/                     # Raw dataset (heart_disease.csv)
│   └── processed/               # Processed data files
│
├── 📂 models/
│   └── best_model.joblib        # Trained model (generated)
│
├── 📂 notebooks/
│   ├── 01_eda.py                # Exploratory Data Analysis
│   └── 02_modeling.py           # Model training workflow
│
├── 📂 reports/
│   └── figures/                 # Generated visualization plots
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── permutation_importance.png
│       ├── target_distribution.png
│       ├── corr_heatmap.png
│       ├── numeric_distributions.png
│       ├── numeric_boxplots.png
│       └── categorical_distributions.png
│
├── 📂 src/                      # Python source code
│   ├── __init__.py
│   ├── config.py                # Configuration constants
│   ├── data_utils.py            # Data loading & preprocessing
│   ├── modeling.py              # ML modeling utilities
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── predict.py               # CLI prediction script
│
├── 📂 api/                      # FastAPI backend
│   ├── __init__.py
│   ├── app.py                   # API endpoints
│   └── schemas.py               # Pydantic models
│
├── 📂 web/                      # React frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── styles.css
│       ├── api/client.ts
│       ├── components/
│       │   ├── FieldNumber.tsx
│       │   ├── FieldSelect.tsx
│       │   └── ResultCard.tsx
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
| 3 | `chest pain type` | Categorical | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic |
| 4 | `resting bp s` | Numeric | Resting blood pressure (mm Hg) |
| 5 | `cholesterol` | Numeric | Serum cholesterol (mg/dl) |
| 6 | `fasting blood sugar` | Categorical | 0 = ≤120 mg/dl, 1 = >120 mg/dl |
| 7 | `resting ecg` | Categorical | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| 8 | `max heart rate` | Numeric | Maximum heart rate achieved |
| 9 | `exercise angina` | Categorical | 0 = No, 1 = Yes |
| 10 | `oldpeak` | Numeric | ST depression induced by exercise |
| 11 | `ST slope` | Categorical | 1 = Upsloping, 2 = Flat, 3 = Downsloping |

### Target Variable
- `target`: 0 = No heart disease, 1 = Heart disease present

### Dataset Statistics
- **Total samples**: 1190 (before deduplication)
- **After cleaning**: ~918 samples
- **Target distribution**: Approximately 55% positive, 45% negative

---

## 🧠 Algorithms & Methodology

### Data Preprocessing Pipeline

```
Raw Data → Remove Duplicates → Handle Missing Values → Feature Engineering → Train/Test Split
```

1. **Duplicate Removal**: ~272 duplicate rows removed
2. **Missing Value Treatment**:
   - Zeros in `resting bp s` and `cholesterol` treated as missing (medically impossible)
   - Numeric: Median imputation
   - Categorical: Most frequent imputation
3. **Feature Scaling**: StandardScaler for numeric features
4. **Encoding**: OneHotEncoder for categorical features

### Models Compared

| Algorithm | Description |
|-----------|-------------|
| **Random Forest** | Ensemble of decision trees with bagging |
| **Logistic Regression** | Linear classifier with sigmoid activation |
| **Logistic Regression (Balanced)** | Class-weighted logistic regression |
| **Gradient Boosting** | Sequential ensemble with boosting |
| **Support Vector Classifier** | Kernel-based maximum margin classifier |
| **K-Nearest Neighbors** | Distance-based lazy learner |
| **Dummy Classifier** | Baseline (majority class prediction) |

### Model Selection Process

1. **Cross-Validation**: 5-fold stratified CV
2. **Primary Metric**: ROC-AUC (handles class imbalance)
3. **Secondary Metric**: F1-Score
4. **Top 2 Selection**: Best models proceed to hyperparameter tuning
5. **Grid Search**: Exhaustive search over parameter grid

### Best Model: Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    random_state=42
)
```

### Why Random Forest?
- ✅ Handles mixed feature types (numeric + categorical)
- ✅ Resistant to overfitting through ensemble averaging
- ✅ Provides feature importance for interpretability
- ✅ No strict assumptions about data distribution
- ✅ Works well with small-medium datasets

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher (for frontend)
- pip or uv (Python package manager)

### Step 1: Clone/Download the Project

```bash
cd "Heart Disease Prediction and Analysis Project"
```

### Step 2: Set Up Python Environment

**Option A: Using pip (standard)**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using uv (faster)**
```bash
# Install uv first
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

**Option C: Using conda**
```bash
conda create -n heart-disease python=3.10 numpy pandas scikit-learn matplotlib seaborn joblib fastapi uvicorn pydantic -y
conda activate heart-disease
```

### Step 3: Add the Dataset

Place your `heart_disease.csv` file in the `data/raw/` directory:
```
data/raw/heart_disease.csv
```

### Step 4: Train the Model

```bash
python -m src.train --data data/raw/heart_disease.csv
```

This will:
- Load and clean the data
- Compare 7 different models
- Tune the top 2 models
- Save the best model to `models/best_model.joblib`

### Step 5: Generate Evaluation Reports

```bash
python -m src.evaluate --data data/raw/heart_disease.csv --model models/best_model.joblib
```

This will:
- Evaluate the model on the test set
- Generate all visualization plots in `reports/figures/`
- Print classification metrics

### Step 6: Start the Backend API

```bash
uvicorn api.app:app --reload
```

The API will be available at: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Step 7: Start the Frontend

```bash
cd web
npm install
npm run dev
```

The frontend will be available at: `http://localhost:5173`

---

## 💻 Usage

### Command Line Interface

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

1. Navigate to `http://localhost:5173`
2. **Predict Page**: Enter patient data and get predictions
3. **Reports Page**: View model metrics, visualizations, and dataset overview
4. **About Page**: Learn about the project and data dictionary

---

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Make a prediction |
| GET | `/reports` | Get available figures |
| GET | `/reports/figures/{filename}` | Get a specific figure |
| GET | `/reports/dataset-preview` | Get dataset preview |
| GET | `/reports/model-metrics` | Get classification report |

### Example: Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "chest pain type": 4,
    "resting bp s": 140,
    "cholesterol": 250,
    "fasting blood sugar": 0,
    "resting ecg": 0,
    "max heart rate": 150,
    "exercise angina": 1,
    "oldpeak": 1.5,
    "ST slope": 2
  }'
```

### Example: Prediction Response

```json
{
  "prediction": 1,
  "probability": 0.8234,
  "threshold": 0.5
}
```

---

## 📈 Model Performance

### Cross-Validation Results (5-Fold Stratified)

| Model | ROC-AUC | F1-Score |
|-------|---------|----------|
| **Random Forest** | **0.9253** | **0.8633** |
| Logistic Regression | 0.9223 | 0.8633 |
| Gradient Boosting | 0.9187 | 0.8760 |
| SVC | 0.9178 | 0.8716 |
| KNeighbors | 0.8978 | 0.8594 |

### Test Set Performance (After Tuning)

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
   macro avg      0.89      0.89      0.89       184
weighted avg      0.89      0.89      0.89       184
```

### Confusion Matrix

```
                Predicted
              Normal  Disease
Actual Normal    69      13
       Disease    7      95
```

---

## 🛠 Technical Details

### Dependencies

**Python Backend:**
- numpy, pandas - Data manipulation
- scikit-learn - Machine learning
- matplotlib, seaborn - Visualization
- joblib - Model serialization
- fastapi, uvicorn - Web API
- pydantic - Data validation

**Frontend:**
- React 18 - UI framework
- TypeScript - Type safety
- Vite - Build tool
- React Router - Navigation

### Configuration

Key settings in `src/config.py`:
```python
RANDOM_SEED = 42          # Reproducibility
TEST_SIZE = 0.2           # 80% train, 20% test
THRESHOLD = 0.50          # Classification threshold
```

---

## ⚠️ Limitations

1. **Dataset Size**: ~918 samples may not capture all population variations
2. **Geographic Bias**: Data primarily from Western medical centers
3. **Feature Limitations**: Only 11 features; other risk factors not included
4. **Binary Classification**: Does not predict severity or type of heart disease
5. **Static Model**: Does not update with new data automatically

---

## 🔮 Future Improvements

- [ ] Add more features (smoking, BMI, family history)
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add SHAP values for explainability
- [ ] Create Docker containerization
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add user authentication
- [ ] Implement model retraining pipeline
- [ ] Add more visualization types (UMAP, t-SNE)

---

## ⚕️ Disclaimer

> **⚠️ IMPORTANT: This tool is for EDUCATIONAL PURPOSES ONLY**

This prediction model is **NOT** intended to replace professional medical diagnosis. The predictions made by this system should **NOT** be used as a basis for medical decisions. Always consult a qualified healthcare professional for any health concerns or medical decisions.

This project was developed as part of a Data Warehousing and Data Mining course in Fall 2025 to demonstrate machine learning concepts and data mining techniques.

---

## 📄 License

This project is for educational purposes. The dataset is publicly available from the UCI Machine Learning Repository.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- scikit-learn team for the excellent ML library
- FastAPI team for the modern Python web framework
- React team for the frontend framework
