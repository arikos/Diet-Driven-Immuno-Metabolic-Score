import pandas as pd
import matplotlib.pyplot as plt

# Load the data we created
df = pd.read_csv('synthetic_dims_data.csv')

# Create the HbA1c histogram
plt.figure(figsize=(10, 6))
plt.hist(df['hba1c'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Add vertical lines for context
plt.axvline(x=7.0, color='green', linestyle='--', linewidth=2, label='Target (<7.0%)')
plt.axvline(x=df['hba1c'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({df["hba1c"].mean():.2f}%)')

# Labels and title
plt.title('Distribution of HbA1c in Synthetic Ugandan T2D Patients', fontsize=14, fontweight='bold')
plt.xlabel('HbA1c (%)', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()

# Print statistics
print(f"\n📊 HbA1c Statistics:")
print(f"  Mean:   {df['hba1c'].mean():.2f}%")
print(f"  Median: {df['hba1c'].median():.2f}%")
print(f"  Min:    {df['hba1c'].min():.2f}%")
print(f"  Max:    {df['hba1c'].max():.2f}%")
print(f"  Std Dev:{df['hba1c'].std():.2f}%")