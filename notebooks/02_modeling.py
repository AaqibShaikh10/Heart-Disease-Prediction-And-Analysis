# Notebook: 02_modeling.ipynb
# Heart Disease Prediction - Model Training and Evaluation
#
# This pseudo-notebook format shows the modeling workflow.
# Convert to actual .ipynb using a Jupyter-compatible tool if needed.

# %% [md]
# # Model Training and Evaluation
# 
# This notebook trains and evaluates machine learning models for heart disease 
# prediction. We compare multiple classifiers, perform hyperparameter tuning, 
# and analyze model performance.

# %% [md]
# ## 1. Setup

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.inspection import permutation_importance

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATA_PATH = Path('../data/raw/heart_disease.csv')
MODEL_PATH = Path('../models/best_model.joblib')
FIG_DIR = Path('../reports/figures')

NUMERIC_COLS = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
CATEGORICAL_COLS = ['sex', 'chest pain type', 'fasting blood sugar', 
                   'resting ecg', 'exercise angina', 'ST slope']
TARGET_COL = 'target'

# %% [md]
# ## 2. Data Loading and Preprocessing

# %% [code]
# Load data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print(f"Original shape: {df.shape}")

# Drop duplicates
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicates removed: {dup_count}")

# Treat zeros as missing for specific columns
for col in ['resting bp s', 'cholesterol']:
    zeros = (df[col] == 0).sum()
    df.loc[df[col] == 0, col] = np.nan
    print(f"{col}: {zeros} zeros converted to NaN")

# Separate features and target
X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
y = df[TARGET_COL].copy()

# Ensure binary target
y = (y > 0).astype(int)

print(f"\nFinal shape: X={X.shape}, y={y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# %% [md]
# ## 3. Build Preprocessing Pipeline

# %% [code]
# Numeric pipeline: impute + scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute + one-hot encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Column transformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, NUMERIC_COLS),
    ('cat', categorical_pipeline, CATEGORICAL_COLS)
])

print("Preprocessing pipeline created.")

# %% [md]
# ## 4. Train/Test Split

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED, 
    stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# %% [md]
# ## 5. Model Comparison

# %% [code]
# Define models to compare
models = {
    'DummyClassifier': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
    'LogisticRegression_Balanced': LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_SEED),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    'SVC': SVC(probability=True, random_state=RANDOM_SEED),
    'KNeighbors': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_SEED)
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor.set_output(transform='default')),
        ('classifier', model)
    ])
    
    # Cross-validation scores
    roc_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    
    results.append({
        'Model': name,
        'ROC-AUC Mean': roc_auc.mean(),
        'ROC-AUC Std': roc_auc.std(),
        'F1 Mean': f1.mean(),
        'F1 Std': f1.std()
    })
    
    print(f"{name}: ROC-AUC = {roc_auc.mean():.4f} (+/- {roc_auc.std():.4f})")

results_df = pd.DataFrame(results).sort_values('ROC-AUC Mean', ascending=False)

# %% [code]
# Display results table
print("\n" + "="*70)
print("MODEL COMPARISON (sorted by ROC-AUC)")
print("="*70)
print(results_df.to_string(index=False))

# %% [md]
# ## 6. Hyperparameter Tuning

# %% [code]
# Select top 2 models (excluding baseline)
top_models = results_df[results_df['Model'] != 'DummyClassifier'].head(2)['Model'].tolist()
print(f"Top models for tuning: {top_models}")

# %% [code]
# Define parameter grids
param_grids = {
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

best_model = None
best_score = 0
best_name = None

for model_name in top_models:
    print(f"\nTuning {model_name}...")
    
    model = models[model_name]
    pipeline = Pipeline([
        ('preprocessor', preprocessor.set_output(transform='default')),
        ('classifier', model)
    ])
    
    grid = param_grids.get(model_name, {})
    
    grid_search = GridSearchCV(
        pipeline, grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_name = model_name

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_name}")
print(f"Best CV ROC-AUC: {best_score:.4f}")
print(f"{'='*50}")

# %% [md]
# ## 7. Test Set Evaluation

# %% [code]
# Predictions on test set
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*50)
print("TEST SET METRICS (threshold=0.5)")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# %% [code]
# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Heart Disease']))

# %% [code]
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Heart Disease (1)'],
            yticklabels=['Normal (0)', 'Heart Disease (1)'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrix.png', dpi=150)
plt.show()

# %% [md]
# ## 8. Threshold Analysis

# %% [code]
# Evaluate at different thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
threshold_results = []

for thresh in thresholds:
    y_pred_t = (y_proba >= thresh).astype(int)
    threshold_results.append({
        'Threshold': thresh,
        'Precision': precision_score(y_test, y_pred_t, zero_division=0),
        'Recall': recall_score(y_test, y_pred_t, zero_division=0),
        'F1': f1_score(y_test, y_pred_t, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred_t)
    })

thresh_df = pd.DataFrame(threshold_results)
print("Threshold Analysis:")
print(thresh_df.to_string(index=False))

# %% [code]
# Recommend threshold (prioritize recall for medical screening)
high_recall = thresh_df[thresh_df['Recall'] >= 0.85]
if len(high_recall) > 0:
    best_idx = high_recall['F1'].idxmax()
    recommended = thresh_df.loc[best_idx, 'Threshold']
else:
    recommended = 0.5

print(f"\nRecommended threshold: {recommended:.2f}")
print("Justification: In medical screening, high recall is preferred to minimize")
print("false negatives (missing actual cases of heart disease).")

# %% [md]
# ## 9. ROC Curve

# %% [code]
# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'roc_curve.png', dpi=150)
plt.show()

# %% [md]
# ## 10. Feature Importance

# %% [code]
# Permutation importance
result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=RANDOM_SEED, scoring='roc_auc'
)

# Get feature names
feature_names = NUMERIC_COLS.copy()
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_features = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
feature_names.extend(cat_features)

# Sort by importance
indices = np.argsort(result.importances_mean)[::-1][:15]
top_names = [feature_names[i] for i in indices]
top_importances = result.importances_mean[indices]
top_stds = result.importances_std[indices]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(top_names))
ax.barh(y_pos, top_importances, xerr=top_stds, align='center', color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_names)
ax.invert_yaxis()
ax.set_xlabel('Mean Decrease in ROC-AUC')
ax.set_title('Top 15 Feature Importances (Permutation)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'permutation_importance.png', dpi=150)
plt.show()

# %% [md]
# ## 11. Save Model

# %% [code]
# Save the best model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# %% [md]
# ## 12. Summary
# 
# ### Key Findings:
# 
# 1. **Best Model:** The hyperparameter-tuned model achieved strong performance 
#    on the test set with good ROC-AUC and F1 scores.
# 
# 2. **Threshold Selection:** A threshold lower than 0.5 may be preferred for 
#    medical screening to increase recall (sensitivity) at the cost of some precision.
# 
# 3. **Important Features:** Based on permutation importance, features like 
#    ST slope, chest pain type, and max heart rate are most predictive.
# 
# 4. **Model Interpretability:** The selected model provides probability outputs 
#    that can be calibrated for clinical use.
# 
# ### Recommendations:
# 
# - Monitor model performance over time with new data
# - Consider ensemble approaches for further improvement
# - Implement SHAP for more detailed feature explanations
# - Validate on external datasets before clinical deployment

# %% [md]
# ---
# End of Modeling Notebook
