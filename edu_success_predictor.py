"""
Project: EduSuccess Predictor
Author: Mohammad Warish
Description:
    Predicts whether a student will pass or fail based on study habits,
    attendance, past academic score, internet access, and sleep hours.

    Workflow:
    1. Load dataset
    2. Clean and preprocess
    3. Feature engineering
    4. Split data into train/test
    5. Train multiple ML models
    6. Evaluate and select best model
    7. Save final trained model pipeline
"""

# ==============================
# 1. IMPORT DEPENDENCIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 2. LOAD DATA
# ==============================
print("🔹 Loading dataset...")
df = pd.read_csv('data.csv')

print("\nFirst 5 rows of data:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# ==============================
# 3. DATA CLEANING
# ==============================
print("\n🔹 Cleaning and encoding data...")

# Convert numeric columns to numeric type
numeric_cols = ['StudyHours', 'Attendance', 'PastScore', 'SleepHours']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode categorical columns
df['Internet'] = df['Internet'].map({'Yes': 1, 'No': 0})
df['Passed'] = df['Passed'].map({'Yes': 1, 'No': 0})

# Handle missing values (if any)
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("\nData types after cleaning:")
print(df.dtypes)

# ==============================
# 4. FEATURE ENGINEERING
# ==============================
print("\n🔹 Adding new engineered features...")

# Engagement metric and interaction feature
df['Engagement'] = df['StudyHours'] * (df['Attendance'] / 100)
df['Study_Past_Interaction'] = df['StudyHours'] * df['PastScore']

# ==============================
# 5. DEFINE FEATURES AND TARGET
# ==============================
X = df[['StudyHours', 'Attendance', 'PastScore', 'Internet', 'SleepHours', 
        'Engagement', 'Study_Past_Interaction']]
y = df['Passed']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/Test split complete.")
print("Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])

# ==============================
# 6. MODEL TRAINING AND COMPARISON
# ==============================
print("\n🔹 Training multiple models...")

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=500))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train model and print key evaluation metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    try:
        proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, proba)
    except Exception:
        roc = None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc if roc else 'N/A'}")
    print(classification_report(y_test, preds))
    
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model.__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return acc, f1

scores = {}
for name, model in models.items():
    acc, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    scores[name] = (acc, f1)

# Display all model results
print("\n🔹 Model comparison (Accuracy / F1):")
for name, (acc, f1) in scores.items():
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")

# ==============================
# 7. MODEL TUNING (RANDOM FOREST)
# ==============================
print("\n🔹 Tuning best model (Random Forest)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

best_model = grid.best_estimator_
evaluate_model(best_model, X_train, y_train, X_test, y_test)

# ==============================
# 8. FEATURE IMPORTANCES
# ==============================
print("\n🔹 Feature Importance (Random Forest):")
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
print(importances)

plt.figure(figsize=(6,4))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance - EduSuccess Predictor")
plt.show()

# ==============================
# 9. SAVE FINAL MODEL
# ==============================
print("\n🔹 Saving final trained model pipeline...")

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', best_model)
])

final_pipeline.fit(X, y)
joblib.dump(final_pipeline, 'edu_success_predictor.pkl')

print("✅ Model saved successfully as 'edu_success_predictor.pkl'")

# ==============================
# 10. WRAP-UP
# ==============================
print("\n--- SUMMARY ---")
print("✔ Data cleaned and preprocessed")
print("✔ Models trained and evaluated")
print("✔ Best model tuned (Random Forest)")
print("✔ Model exported as pickle file")
print("Project completed: EduSuccess Predictor ✅")