# DIMS-AI: Diet-Driven Immuno-Metabolic Score

Machine Learning Proof of Concept for a clinical decision-support tool designed to predict and manage systemic inflammation in Type 2 Diabetes patients within low-resource settings (e.g., Uganda).

## 1. Scientific & Biological Premise

**Context:** 
Current diabetes management heavily relies on HbA1c (glycaemic control), often missing unresolved chronic, low-grade inflammation that drives micro/macrovascular complications. DIMS-AI targets the gut-immune axis, estimating the systemic inflammatory burden (via hs-CRP) modulated by dietary short-chain fatty acid (SCFA) production, specifically butyrate.

**Reference Baselines:** 
The scoring logic adapts principles from the Dietary Inflammatory Index (DII) but is explicitly recalibrated for Sub-Saharan African nutritional transitions. It moves beyond Western-centric baselines by weighting local, culturally relevant staples (e.g., fermented *obushera*, millet, sorghum, and refined maize posho) based on their documented fiber content and fermentation-derived probiotic potential.

## 2. Mathematical Formulation

The composite Immuno-Metabolic Score is derived from a biologically grounded, weighted linear combination of clinical and dietary features, normalized to a 0–100 scale.

**Raw Risk Calculation:**
$$
\text{DIMS}_{\text{raw}} = \sum_{i=1}^{n} w_i \cdot X_i + \epsilon
$$

Where:
- $X_i$ represents the normalized nutritional and clinical feature inputs (e.g., HbA1c, hs-CRP, FFQ ordinal frequencies).
- $w_i$ represents the biologically informed inflammatory/metabolic effect weights (positive for risk factors like refined carbs; negative for protective factors like legumes/fermented foods).
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents Gaussian stochastic noise, simulating unmeasured biological confounders (e.g., genetics, stress) to prevent model overfitting.

**Normalized Composite Score (0–100):**
$$ \text{DIMS}_{\text{score}} = 100 \times \left( 1 - \frac{\text{DIMS}_{\text{raw}} - \min(\text{DIMS}_{\text{raw}})}{\max(\text{DIMS}_{\text{raw}}) - \min(\text{DIMS}_{\text{raw}})} \right) $$

## 3. Pipeline & Software Architecture

**Data Sources:**
The model is currently validated on a biologically plausible synthetic bio-cohort ($N=1,000$ virtual Ugandan T2D patients). Distribution parameters are strictly informed by regional epidemiological baselines (e.g., mean HbA1c $\approx 9.9\%$ in Ugandan cohorts; elevated baseline hs-CRP distributions in African populations).

**Feature Pipeline:**
1. **Clinical Inputs:** Continuous biomarkers (HbA1c, hs-CRP).
2. **Dietary Inputs:** 7-item localized Food Frequency Questionnaire (FFQ) mapped to ordinal vectors (0=Rarely to 3=Daily).
3. **Preprocessing:** Ordinal encoding and min-max scaling to generate standardized inflammatory/metabolic vectors.

**Model & Algorithm:**
- **Core Engine:** Supervised regression via **XGBoost Regressor**, benchmarked against Random Forest. XGBoost was selected for its superior handling of non-linear feature interactions and lower Mean Absolute Error (MAE).
- **Explainability:** **SHAP (SHapley Additive exPlanations)** is integrated to provide transparent, clinician-friendly feature attribution, ensuring the model is not a "black box."

## ️ Requirements & Setup

To run the pipeline locally:
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn shap
