import numpy as np
import pandas as pd

print("Starting Step 1: Generating Synthetic Data...")

# 1. SETUP
np.random.seed(42) # Ensures we get the exact same data every time we run this
n_patients = 1000

# 2. CLINICAL BIOMARKERS
# HbA1c: Bell curve, average 8.2%, clipped between realistic min/max
hba1c = np.clip(np.random.normal(8.2, 1.6, n_patients), 5.5, 13.0)

# hs-CRP: Log-normal curve (skewed), because most people have low inflammation, few have very high
hscrp = np.clip(np.random.lognormal(1.0, 0.8, n_patients), 0.1, 20.0)

# 3. DIETARY FFQ (0=Rarely, 1=1-2x/wk, 2=3-5x/wk, 3=Daily)
# The 'p' array represents the probability of each answer [0, 1, 2, 3]
ffq_legumes = np.random.choice([0, 1, 2, 3], n_patients, p=[0.2, 0.4, 0.3, 0.1])
ffq_whole_grains = np.random.choice([0, 1, 2, 3], n_patients, p=[0.3, 0.4, 0.2, 0.1])
ffq_fermented = np.random.choice([0, 1, 2, 3], n_patients, p=[0.5, 0.3, 0.15, 0.05])
ffq_refined_carbs = np.random.choice([0, 1, 2, 3], n_patients, p=[0.1, 0.2, 0.4, 0.3]) # High posho intake
ffq_red_meat = np.random.choice([0, 1, 2, 3], n_patients, p=[0.4, 0.4, 0.15, 0.05])
ffq_veggies = np.random.choice([0, 1, 2, 3], n_patients, p=[0.2, 0.3, 0.3, 0.2])
ffq_fruits = np.random.choice([0, 1, 2, 3], n_patients, p=[0.3, 0.4, 0.2, 0.1])

# Combine into a table (DataFrame)
df = pd.DataFrame({
    'hba1c': hba1c,
    'hscrp': hscrp,
    'legumes': ffq_legumes,
    'whole_grains': ffq_whole_grains,
    'fermented': ffq_fermented,
    'refined_carbs': ffq_refined_carbs,
    'red_meat': ffq_red_meat,
    'veggies': ffq_veggies,
    'fruits': ffq_fruits
})

# 4. CALCULATE GROUND TRUTH SCORE
# Bad things add to risk, good things subtract from risk
raw_risk = (
    (df['hba1c'] * 6) + 
    (df['hscrp'] * 3) + 
    (df['refined_carbs'] * 4) + 
    (df['red_meat'] * 3) - 
    (df['legumes'] * 5) - 
    (df['whole_grains'] * 4) - 
    (df['fermented'] * 5) - 
    (df['veggies'] * 3) - 
    (df['fruits'] * 2)
)

# Add realistic biological "noise" so it's not perfectly predictable
noise = np.random.normal(0, 8, n_patients)
raw_risk = raw_risk + noise

# Convert raw risk into a 0 to 100 Health Score (100 = Best, 0 = Worst)
min_r, max_r = raw_risk.min(), raw_risk.max()
dims_score = 100 - ((raw_risk - min_r) / (max_r - min_r) * 100)

df['true_dims_score'] = dims_score

# 5. SAVE TO CSV
csv_filename = 'synthetic_dims_data.csv'
df.to_csv(csv_filename, index=False)
print(f"✅ Success! Data saved to '{csv_filename}'")
print(f"Dataset shape: {df.shape[0]} patients, {df.shape[1]} columns.\n")

# Show the first 5 patients so you can see what it looks like
print("--- First 5 Virtual Patients ---")
print(df.head().to_string())