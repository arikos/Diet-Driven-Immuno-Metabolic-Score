# DIMS-AI: Diet-Driven Immuno-Metabolic Score

This repository contains the Machine Learning Proof of Concept for the **Diet-Driven Immuno-Metabolic Score (DIMS-AI)**, a clinical decision-support tool designed for Type 2 Diabetes management for countries in Sub SAharan Africa.

## 📌 Project Overview
Current diabetes care heavily relies on HbA1c. This often misses unresolved chronic inflammation that drives complications. DIMS-AI combines:
1. **Glycaemic Control** (HbA1c)
2. **Inflammation Marker** (hs-CRP)
3. **Dietary Butyrate Potential** (via a 7-item localized Food Frequency Questionnaire)

The system outputs a transparent, 0–100 health score with explainable AI (SHAP) recommendations for clinicians and patients.

## 📁 Repository Structure
- `data_generator.py`: Script to create realistic synthetic clinical and dietary data for 1,000 virtual patients.
- `synthetic_dims_data.csv`: The generated dataset (safe for public sharing; contains no real patient data).
- `eda_visualizations.py`: Exploratory Data Analysis scripts generating distribution and correlation charts.
- `model_benchmarking.py`: Benchmarking of XGBoost vs. Random Forest for predicting the DIMS score.
- `shap_explainability.py`: Implementation of SHAP to provide transparent, clinician-friendly feature attribution.

## 🚀 Key Findings (Proof of Concept)
- **XGBoost** outperformed Random Forest in predicting the immuno-metabolic score (Lower MAE, Higher R²).
- **SHAP integration** successfully isolates the impact of specific local dietary factors (e.g., legume intake, fermented foods like *obushera*) on the final clinical score, proving the model is not a "black box."

## 🛠️ Requirements & Setup
To run this code locally, install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn shap
