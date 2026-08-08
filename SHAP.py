import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 1. Load data and retrain quickly (or use the model from previous step if still in memory)
df = pd.read_csv('synthetic_dims_data.csv')
X = df.drop('true_dims_score', axis=1)
y = df['true_dims_score']

model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X, y)

print("Initializing SHAP Explainer... (This might take a few seconds)")
# 2. Create the SHAP explainer for our tree-based model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ==========================================
# VISUALIZATION 1: Global Feature Importance
# ==========================================
print("\nGenerating Global Feature Importance Chart...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('What Drives the DIMS Score the Most? (Global View)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# VISUALIZATION 2: The "Doctor's Dashboard" for ONE Patient
# ==========================================
# Let's pick a specific patient to explain (e.g., Patient index 150)
patient_idx = 150
patient_data = X.iloc[patient_idx:patient_idx+1]
true_score = y.iloc[patient_idx]
predicted_score = model.predict(patient_data)[0]

print(f"\n--- DOCTOR'S DASHBOARD: Patient #{patient_idx} ---")
print(f"AI Predicted DIMS Score: {predicted_score:.1f} (True Score was: {true_score:.1f})")
print("\nCalculating individual feature impacts...")

# Create a waterfall plot for this single patient
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(values=shap_values[patient_idx], 
                                       base_values=explainer.expected_value, 
                                       data=X.iloc[patient_idx], 
                                       feature_names=X.columns), 
                     max_display=8, show=False)
plt.title(f'Why did Patient #{patient_idx} get a score of {predicted_score:.1f}?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print a simple text summary for the patient
print("\n📝 Plain Language Explanation for the Patient:")
print(f"Your baseline health score starts around {explainer.expected_value:.1f}.")

# Get the top 3 features that lowered the score
impacts = [(X.columns[i], X.iloc[patient_idx].values[i], shap_values[patient_idx][i]) for i in range(len(X.columns))]
impacts.sort(key=lambda x: x[2]) # Sort by impact (most negative first)

print("The main factors lowering your score today are:")
for i in range(3):
    feat, val, impact = impacts[i]
    print(f"  🔴 {feat.replace('_', ' ').title()} (Level: {val}) decreased your score by {abs(impact):.1f} points.")

print("\nThe main factors improving your score today are:")
# Get the top 2 features that raised the score
positive_impacts = [x for x in impacts if x[2] > 0]
positive_impacts.sort(key=lambda x: x[2], reverse=True)
for i in range(min(2, len(positive_impacts))):
    feat, val, impact = positive_impacts[i]
    print(f"  🟢 {feat.replace('_', ' ').title()} (Level: {val}) increased your score by {abs(impact):.1f} points.")