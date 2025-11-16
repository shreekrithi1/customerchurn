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
