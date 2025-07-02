import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv('output/C2_Results.csv', index_col=0).round(3)

# Remove columns whose name contains 'val'
df = df.loc[:, ~df.columns.str.contains('val')]

# Normalise for colour mapping
vmin, vmax = df.min().min(), df.max().max()
norm = plt.Normalize(vmin, vmax)
cmap = plt.cm.coolwarm

# Create compact figure
fig_height = 0.6 + 0.4 * len(df)  # Adjust height based on row count
fig, ax = plt.subplots(figsize=(6.5, fig_height))  # Smaller width
ax.axis('off')

# Create table
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 loc='center',
                 cellLoc='center')

# Format table cells
for (i, j), cell in table.get_celld().items():
    if i == 0 or j == -1:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')
    elif i > 0 and j >= 0:
        val = df.iloc[i-1, j]
        cell.set_facecolor(cmap(norm(val)))

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.2)  # Less horizontal scale for compactness

# Add compact colorbar below
cbar_ax = inset_axes(ax, width="50%", height="4%", loc='lower center', borderpad=1)
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                   cax=cbar_ax,
                   orientation='horizontal')
cb1.set_label('Value Range (Low → High)', fontsize=8)
cb1.ax.tick_params(labelsize=7)

# Save compact image
plt.savefig("output/C2_Results_Table.png", bbox_inches='tight', dpi=300)
plt.close()