import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the data
df = pd.read_csv('synthetic_dims_data.csv')

# Separate the inputs (X) from the target we want to predict (y)
X = df.drop('true_dims_score', axis=1) # Everything except the score
y = df['true_dims_score']              # Only the score

# 2. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 1: Training the XGBoost model on 800 patients...")

# 3. Initialize and Train the XGBoost Model
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

print("Step 2: Testing the model on the 200 hidden patients...")

# 4. Make predictions on the hidden test set
y_pred = model.predict(X_test)

# 5. Grade the AI
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n🤖 AI Model Report Card:")
print(f"  Mean Absolute Error (MAE): {mae:.2f} points")
print(f"  R-squared Score (Accuracy): {r2:.2f} (Closer to 1.0 is perfect)")

# 6. Visualize: Actual vs. Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction Line') # The red line is perfection

plt.title('AI Predictions vs. Actual DIMS Scores', fontsize=14, fontweight='bold')
plt.xlabel('Actual True Score', fontsize=12)
plt.ylabel('AI Predicted Score', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()