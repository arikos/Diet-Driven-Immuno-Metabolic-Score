# DIMS-AI: Diet-Driven Immuno-Metabolic Score

Machine Learning Proof of Concept for a clinical decision-support tool designed to predict and manage systemic inflammation in Type 2 Diabetes patients within low-resource settings (e.g., Uganda).

## 1. Scientific & Biological Premise

**Context:** 
Current diabetes management heavily relies on HbA1c (glycaemic control), often missing unresolved chronic, low-grade inflammation that drives micro/macrovascular complications. DIMS-AI targets the [...]

**Reference Baselines:** 
The scoring logic adapts principles from the Dietary Inflammatory Index (DII) but is explicitly recalibrated for Sub-Saharan African nutritional transitions. It moves beyond Western-centric baseli[...]

## 2. Mathematical Formulation

The composite Immuno-Metabolic Score is derived from a biologically grounded, weighted linear combination of clinical and dietary features, normalized to a 0–100 scale.

**Raw Risk Calculation:**

![DIMS_raw equation](https://latex.codecogs.com/png.latex?DIMS_{raw}%20=%20%5Csum_{i=1}^{n}%20w_i%20%5Ccdot%20X_i%20%2B%20%5Cepsilon)

Where:
- $X_i$ represents the normalized nutritional and clinical feature inputs (e.g., HbA1c, hs-CRP, FFQ ordinal frequencies).
- $w_i$ represents the biologically informed inflammatory/metabolic effect weights (positive for risk factors like refined carbs; negative for protective factors like legumes/fermented foods).
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents Gaussian stochastic noise, simulating unmeasured biological confounders (e.g., genetics, stress) to prevent model overfitting.

**Normalized Composite Score (0–100):**

![DIMS_score equation](https://latex.codecogs.com/png.latex?DIMS_{score}%20=%20100%20%5Ctimes%20%5Cleft(1%20-%20%5Cfrac{DIMS_{raw}%20-%20%5Cmin(DIMS_{raw})}{%5Cmax(DIMS_{raw})%20-%20%5Cmin(DIMS_{raw})}%5Cright))

## 3. Pipeline & Software Architecture

**Data Sources:**
The model is currently validated on a biologically plausible synthetic bio-cohort ($N=1,000$ virtual Ugandan T2D patients). Distribution parameters are strictly informed by regional epidemiologica[...]

**Feature Pipeline:**
1. **Clinical Inputs:** Continuous biomarkers (HbA1c, hs-CRP).
2. **Dietary Inputs:** 7-item localized Food Frequency Questionnaire (FFQ) mapped to ordinal vectors (0=Rarely to 3=Daily).
3. **Preprocessing:** Ordinal encoding and min-max scaling to generate standardized inflammatory/metabolic vectors.

**Model & Algorithm:**
- **Core Engine:** Supervised regression via **XGBoost Regressor**, benchmarked against Random Forest. XGBoost was selected for its superior handling of non-linear feature interactions and lower M[...]
- **Explainability:** **SHAP (SHapley Additive exPlanations)** is integrated to provide transparent, clinician-friendly feature attribution, ensuring the model is not a "black box."

## ️ Requirements & Setup

To run the pipeline locally:
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn shap
```
