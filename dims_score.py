import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # We'll use seaborn for beautiful, easy boxplots

# Load the data
df = pd.read_csv('synthetic_dims_data.csv')

# Set up a figure with 2 plots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- PLOT 1: LEGUMES (The Good Food) ---
# We expect the score to go UP as legume intake goes from 0 to 3
sns.boxplot(x='legumes', y='true_dims_score', data=df, ax=axes[0], palette='Blues')
axes[0].set_title('Impact of Legumes/Beans on DIMS Score', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Frequency: 0=Rarely, 1=1-2x/wk, 2=3-5x/wk, 3=Daily', fontsize=11)
axes[0].set_ylabel('DIMS Score (0-100)', fontsize=11)
axes[0].set_xticklabels(['0', '1', '2', '3'])

# --- PLOT 2: REFINED CARBS (The Bad Food) ---
# We expect the score to go DOWN as refined carb intake goes from 0 to 3
sns.boxplot(x='refined_carbs', y='true_dims_score', data=df, ax=axes[1], palette='Reds')
axes[1].set_title('Impact of Refined Carbs (Posho/White Bread) on DIMS Score', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Frequency: 0=Rarely, 1=1-2x/wk, 2=3-5x/wk, 3=Daily', fontsize=11)
axes[1].set_ylabel('DIMS Score (0-100)', fontsize=11)
axes[1].set_xticklabels(['0', '1', '2', '3'])

plt.tight_layout()
plt.show()

# Print a quick summary of the averages to prove the trend
print("\n📊 Average DIMS Score by Food Frequency:")
print("\nLegumes (Should go UP):")
print(df.groupby('legumes')['true_dims_score'].mean().round(1))

print("\nRefined Carbs (Should go DOWN):")
print(df.groupby('refined_carbs')['true_dims_score'].mean().round(1))