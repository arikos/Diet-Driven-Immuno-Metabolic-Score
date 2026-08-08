import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the data
df = pd.read_csv('synthetic_dims_data.csv')
X = df.drop('true_dims_score', axis=1)
y = df['true_dims_score']

# 2. Split data (Using random_state=42 to ensure it's the EXACT same split as XGBoost)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Random Forest model on 800 patients...")

# 3. Initialize and Train the Random Forest Model
# n_estimators=100 means we are building a forest of 100 decision trees
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

print("Testing the model on the 200 hidden patients...")

# 4. Make predictions
rf_pred = rf_model.predict(X_test)

# 5. Grade the AI
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# ==========================================
# THE SHOWDOWN: XGBoost vs Random Forest
# ==========================================
# (These are the numbers we got from your XGBoost model in the previous step)
xgb_mae = 6.26
xgb_r2 = 0.72

print("\n" + "="*50)
print("🏆 MODEL BENCHMARKING REPORT CARD 🏆")
print("="*50)
print(f"{'Metric':<20} | {'XGBoost':<10} | {'Random Forest':<15}")
print("-" * 50)
print(f"{'Mean Abs Error (MAE)':<20} | {xgb_mae:<10} | {rf_mae:<15}")
print(f"{'R-squared (Accuracy)':<20} | {xgb_r2:<10} | {rf_r2:<15}")
print("="*50)

if rf_mae < xgb_mae:
    print(" Winner: Random Forest has a lower error rate!")
else:
    print("👉 Winner: XGBoost has a lower error rate!")

# 6. Visualize: Actual vs. Predicted for Random Forest
plt.figure(figsize=(8, 8))
plt.scatter(y_test, rf_pred, alpha=0.6, color='orange', label='Random Forest Predictions')
plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction Line')

plt.title('Random Forest: AI Predictions vs. Actual DIMS Scores', fontsize=14, fontweight='bold')
plt.xlabel('Actual True Score', fontsize=12)
plt.ylabel('AI Predicted Score', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()