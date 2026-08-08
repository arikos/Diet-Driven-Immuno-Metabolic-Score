import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('synthetic_dims_data.csv')

# Create the hs-CRP histogram
plt.figure(figsize=(10, 6))
plt.hist(df['hscrp'], bins=30, color='salmon', edgecolor='black', alpha=0.7)

# Add a vertical line for the clinical threshold
# In clinical practice, hs-CRP > 3.0 mg/L is often considered "high risk" for inflammation
plt.axvline(x=3.0, color='red', linestyle='--', linewidth=2, label='High Inflammation Threshold (>3.0)')

# Labels and title
plt.title('Distribution of hs-CRP (Inflammation) in Synthetic Patients', fontsize=14, fontweight='bold')
plt.xlabel('hs-CRP (mg/L)', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()

# Print statistics
print(f"\n📊 hs-CRP Statistics:")
print(f"  Mean:   {df['hscrp'].mean():.2f} mg/L")
print(f"  Median: {df['hscrp'].median():.2f} mg/L")
print(f"  Max:    {df['hscrp'].max():.2f} mg/L")