"""
Run this using:
python -m MainScripts.PerClassAnalysis
"""

# Load the uploaded CSV file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the uploaded CSV file
# file_path = "output/SENR_BASELINE/BASELINEeval_per_class_metrics.csv"
file_path = "output/D4_tomtest/MLP_Basic/eval_per_class_metrics.csv" 

df = pd.read_csv(file_path)

# Create grouped bar chart for accuracy, precision, and recall (in that order)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df))
bar_width = 0.25

# Plot results
ax.bar(x, df['precision'], width=bar_width, label='Precision', color='skyblue')
ax.bar(x + bar_width, df['recall'], width=bar_width, label='Recall', color='salmon')

ax.set_xticks(x)
ax.set_xticklabels(df['class_name'], rotation=45)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Evaluation after improving Dataset')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
print(df)
plt.show()
