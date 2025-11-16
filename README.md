# Customer Churn Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **supervised binary classification** project to predict customer churn in a telecom company using the **Telco Customer Churn Dataset**. The model helps identify at-risk customers for proactive retention strategies.

---

## Dataset

- **Source**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows**: 7,043  
- **Columns**: 21  
- **Target Variable**: `Churn` → `Yes` (1) or `No` (0)

### Key Features
| Category         | Features |
|------------------|---------|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Services**     | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Billing**      | `MonthlyCharges`, `TotalCharges` |

> **Class Distribution (Imbalanced)**  
> - No Churn: ~73.5%  
> - Churn: ~26.5%

---

## Project Structure

├── Customer_Churn_Prediction_using_ML.ipynb
├── customer_churn_model.pkl
├── encoders.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
text---

## Installation & Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
Note: Uses LabelEncoder for categorical encoding and SMOTE for oversampling.

Pipeline Overview

StepDescription1. Data LoadingLoad CSV into Pandas DataFrame2. Exploratory Analysis.head(), .shape, check for missing values3. PreprocessingDrop customerIDFix TotalCharges (coerce to float)Label encode all categorical columnsSave encoders → encoders.pkl4. Train-Test Split80% train, 20% test (train_test_split)5. Handle ImbalanceApply SMOTE to oversample minority class6. Model TrainingTrain: Decision TreeRandom ForestXGBoost7. Model SelectionRandom Forest selected (best balance of accuracy & generalization)8. EvaluationAccuracy, Confusion Matrix, Classification Report9. Save Modelpickle.dump(model + feature_names) → customer_churn_model.pkl10. Prediction SystemLoad model + encoders → Predict on new data

Model Performance
textAccuracy Score: 77.86%

Confusion Matrix:
[[878 158]
 [154 219]]

Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.85      0.85      1036
           1       0.58      0.59      0.58       373
    accuracy                           0.78      1409
   macro avg       0.72      0.72      0.72      1409
weighted avg       0.78      0.78      0.78      1409
Insight: Strong performance on majority class (No Churn), moderate on minority (Churn) due to imbalance.

How to Make Predictions
pythonimport pandas as pd
import pickle

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
features = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# New customer input
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

df = pd.DataFrame([input_data])

# Encode using saved LabelEncoders
for col, enc in encoders.items():
    df[col] = enc.transform(df[col])

# Predict
pred = model.predict(df)[0]
prob = model.predict_proba(df)[0]

print(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
print(f"Churn Probability: {prob[1]:.2%}")
Output Example:
textPrediction: No Churn
Churn Probability: 22.00%
