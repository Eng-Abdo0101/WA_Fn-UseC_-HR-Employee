# Employee Attrition & Salary Prediction

A machine learning web app that predicts whether an employee is likely to leave a company (**classification**) and estimates a fair **Monthly Income** for an employee (**regression**), built on the IBM HR Analytics Employee Attrition dataset.

**Live app:** https://wafn-usec-hr-employee-a2t4hvzdqsxhwkwzyfqmxv.streamlit.app/

## Overview

This project tackles two related problems from the same HR dataset:

| Task | Target | Type |
|---|---|---|
| Attrition Risk | `Attrition` | Binary classification |
| Salary Estimate | `MonthlyIncome` | Regression |

Both models share the same feature engineering pipeline, built from the raw HR fields plus five engineered features:

- **AgeGroup** — employee age bucketed into Young / Adult / Middle_Aged / Senior
- **OverallSatisfaction** — sum of Job, Environment, Relationship satisfaction and Job Involvement
- **CompanyTenureRatio** — `YearsAtCompany / (TotalWorkingYears + 1)`
- **PromotionDelayRatio** — `YearsSinceLastPromotion / (YearsAtCompany + 1)`
- **OverTime_WorkLife** — interaction between OverTime and WorkLifeBalance

## Models

**Attrition (classification)**
- Algorithm: CatBoost Classifier, `class_weights=[1, 4]` to handle class imbalance
- Decision threshold tuned to **0.30** (rather than the default 0.5) for better F1 on the minority (leaver) class
- Trained on 51 features (raw + engineered + one-hot encoded categoricals)

**Salary (regression)**
- Algorithm: XGBoost Regressor, tuned via `GridSearchCV` (`learning_rate=0.05`, `max_depth=3`, `n_estimators=100`)
- Compared against Linear Regression, Decision Tree, CatBoost, LightGBM, and a Stacking ensemble — XGBoost was chosen as the best balance of accuracy (R² ≈ 0.944) and simplicity
- Inputs are standardized with `StandardScaler` before prediction

## App

Built with [Streamlit](https://streamlit.io/), the app has two tabs:

- 🚪 **Attrition Risk** — fill in an employee's profile and get a leave/stay prediction with probability
- 💰 **Salary Estimate** — fill in the same profile (minus income) and get a predicted fair Monthly Income

Try it here: **https://wafn-usec-hr-employee-a2t4hvzdqsxhwkwzyfqmxv.streamlit.app/**

## Running locally

1. Clone this repository and install dependencies:
   ```bash
   pip install streamlit pandas joblib catboost xgboost scikit-learn
   ```
2. Make sure the following model artifacts are in the same folder as `app.py`:
   - `employee_attrition_catboost_model.pkl`
   - `employee_features.pkl`
   - `employee_salary_xgb_model.pkl`
   - `employee_salary_scaler.pkl`
   - `employee_salary_features.pkl`
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project structure

```
.
├── app.py                                  # Streamlit app (attrition + salary tabs)
├── Employee.ipynb                          # Data exploration, feature engineering, model training/comparison
├── employee_attrition_catboost_model.pkl   # Trained attrition classifier
├── employee_features.pkl                   # Feature list for the attrition model
├── employee_salary_xgb_model.pkl           # Trained salary regressor
├── employee_salary_scaler.pkl              # Scaler used for the salary model's inputs
├── employee_salary_features.pkl            # Feature list for the salary model
└── README.md
```

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employee records with demographic, job, and satisfaction attributes.

## Tech stack

Python · pandas · scikit-learn · CatBoost · XGBoost · Streamlit
