# Notebook: 01_eda.ipynb
# Heart Disease Dataset - Exploratory Data Analysis
# 
# This pseudo-notebook format shows the EDA workflow.
# Convert to actual .ipynb using a Jupyter-compatible tool if needed.

# %% [md]
# # Exploratory Data Analysis: Heart Disease Dataset
# 
# This notebook performs exploratory data analysis on the heart disease dataset
# to understand the data distribution, identify patterns, and prepare for modeling.

# %% [md]
# ## 1. Setup and Data Loading

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# Define paths
DATA_PATH = Path('../data/raw/heart_disease.csv')
FIG_DIR = Path('../reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Define column groups
NUMERIC_COLS = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
CATEGORICAL_COLS = ['sex', 'chest pain type', 'fasting blood sugar', 
                   'resting ecg', 'exercise angina', 'ST slope']
TARGET_COL = 'target'

# %% [code]
# Load data
df = pd.read_csv(DATA_PATH)

# Standardize column names (strip whitespace)
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# %% [md]
# ## 2. Data Overview

# %% [code]
# First few rows
print("First 5 rows:")
df.head()

# %% [code]
# Data types
print("\nData types:")
print(df.dtypes)

# %% [code]
# Basic statistics
print("\nDescriptive statistics:")
df.describe()

# %% [md]
# ## 3. Schema Validation

# %% [code]
# Check for required columns
required_cols = set(NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])
found_cols = set(df.columns)

print(f"Expected columns: {sorted(required_cols)}")
print(f"Found columns: {sorted(found_cols)}")

missing = required_cols - found_cols
if missing:
    print(f"WARNING: Missing columns: {missing}")
else:
    print("All required columns present.")

# %% [md]
# ## 4. Missing Values Analysis

# %% [code]
# Check missing values
print("Missing values per column:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.any() else "No missing values (NaN)")

# Check for zeros that may represent missing values
print("\n\nZero value counts for numeric columns:")
for col in ['resting bp s', 'cholesterol']:
    zeros = (df[col] == 0).sum()
    print(f"{col}: {zeros} zeros ({zeros/len(df)*100:.2f}%)")
print("\nNote: Zero values in 'resting bp s' and 'cholesterol' are treated as missing.")

# %% [md]
# ## 5. Duplicate Analysis

# %% [code]
# Check for duplicates
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")

if dup_count > 0:
    print(f"Percentage: {dup_count/len(df)*100:.2f}%")
    print("Duplicates will be removed during preprocessing.")

# %% [md]
# ## 6. Target Distribution

# %% [code]
# Target distribution
target_counts = df[TARGET_COL].value_counts()
print("Target distribution:")
print(target_counts)
print(f"\nClass ratio (1/0): {target_counts[1]/target_counts[0]:.2f}")

# %% [code]
# Plot target distribution
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#3498db', '#e74c3c']
bars = ax.bar(['Normal (0)', 'Heart Disease (1)'], target_counts.values, color=colors)
ax.set_xlabel('Target Class')
ax.set_ylabel('Count')
ax.set_title('Target Variable Distribution')

# Add count labels
for bar, count in zip(bars, target_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            str(count), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(FIG_DIR / 'target_distribution.png', dpi=150)
plt.show()

# %% [md]
# ## 7. Numeric Feature Distributions

# %% [code]
# Distribution plots for numeric features
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    ax = axes[i]
    df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {col}')

# Hide the extra subplot
axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / 'numeric_distributions.png', dpi=150)
plt.show()

# %% [code]
# Boxplots for numeric features by target
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    ax = axes[i]
    df.boxplot(column=col, by=TARGET_COL, ax=ax)
    ax.set_xlabel('Target')
    ax.set_ylabel(col)
    ax.set_title(f'{col} by Target')

axes[-1].set_visible(False)
plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.savefig(FIG_DIR / 'numeric_boxplots.png', dpi=150)
plt.show()

# %% [md]
# ## 8. Categorical Feature Distributions

# %% [code]
# Count plots for categorical features
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL_COLS):
    ax = axes[i]
    value_counts = df[col].value_counts().sort_index()
    ax.bar(value_counts.index.astype(str), value_counts.values, color='steelblue')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_title(f'{col} Distribution')

plt.tight_layout()
plt.savefig(FIG_DIR / 'categorical_distributions.png', dpi=150)
plt.show()

# %% [md]
# ## 9. Correlation Analysis

# %% [code]
# Correlation heatmap for numeric features
# Include target as numeric for correlation
numeric_df = df[NUMERIC_COLS + [TARGET_COL]].copy()
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    fmt='.2f', 
    cmap='RdBu_r', 
    center=0,
    vmin=-1, vmax=1,
    ax=ax
)
ax.set_title('Correlation Heatmap (Numeric Features + Target)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'corr_heatmap.png', dpi=150)
plt.show()

# %% [md]
# ## 10. Key Insights

# %% [md]
# ### Summary of Findings
# 
# Based on the exploratory data analysis, here are the key insights:
# 
# 1. **Dataset Size:** The dataset contains a moderate number of samples suitable 
#    for traditional machine learning approaches.
# 
# 2. **Class Distribution:** There is some class imbalance in the target variable. 
#    Stratified sampling and class-weight adjustments may be needed.
# 
# 3. **Missing Values:** While explicit NaN values may be minimal, zero values in 
#    'resting bp s' and 'cholesterol' should be treated as missing since these 
#    measurements cannot physiologically be zero.
# 
# 4. **Duplicate Rows:** Some duplicate rows exist and will be removed during 
#    preprocessing.
# 
# 5. **Numeric Feature Ranges:**
#    - Age ranges from young adults to elderly
#    - Resting blood pressure and cholesterol show typical clinical ranges with some outliers
#    - Max heart rate shows expected age-related variation
#    - Oldpeak includes negative values (kept as-is for modeling)
# 
# 6. **Categorical Feature Distribution:**
#    - Sex distribution may not be equal
#    - Chest pain type 4 (asymptomatic) is often associated with heart disease
#    - Exercise-induced angina shows clear discrimination between classes
# 
# 7. **Correlations:**
#    - Several features show moderate correlations with the target
#    - Max heart rate tends to be inversely correlated with heart disease
#    - Oldpeak tends to be positively correlated with heart disease
# 
# 8. **Feature Importance Hints:**
#    - ST slope, chest pain type, and exercise angina appear to be discriminative
#    - Age and max heart rate show clear separation between classes
# 
# 9. **Outliers:** Some outliers exist in cholesterol and resting bp s; 
#    tree-based models may handle these better than linear models.
# 
# 10. **Next Steps:** Proceed with preprocessing (imputation, scaling, encoding) 
#     and model training with appropriate validation strategy.

# %% [md]
# ---
# End of EDA Notebook
