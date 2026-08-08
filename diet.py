import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('synthetic_dims_data.csv')

# Count how many patients gave each answer (0, 1, 2, or 3)
refined_counts = df['refined_carbs'].value_counts().sort_index()
fermented_counts = df['fermented'].value_counts().sort_index()

# Set up the bar chart
x = np.array([0, 1, 2, 3])
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars side-by-side
bars1 = ax.bar(x - width/2, refined_counts, width, label='Refined Carbs (Posho/White Bread)', color='lightcoral', edgecolor='black')
bars2 = ax.bar(x + width/2, fermented_counts, width, label='Fermented Foods (Obushera/Millet)', color='lightgreen', edgecolor='black')

# Add labels and title
ax.set_title('Dietary Habits: Refined Carbs vs. Fermented Foods', fontsize=14, fontweight='bold')
ax.set_xlabel('Frequency (0=Rarely, 1=1-2x/wk, 2=3-5x/wk, 3=Daily)', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['0 (Rarely)', '1 (1-2x/week)', '2 (3-5x/week)', '3 (Daily)'])
ax.legend()

# Add a grid for easier reading
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print the exact percentages
print("\n📊 Dietary Percentages:")
print("Refined Carbs (Daily intake):", round((refined_counts.get(3, 0) / len(df)) * 100, 1), "%")
print("Fermented Foods (Daily intake):", round((fermented_counts.get(3, 0) / len(df)) * 100, 1), "%")