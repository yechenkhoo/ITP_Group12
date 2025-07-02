import pandas as pd
from collections import defaultdict

folder = "C_gnn"
model = "gnn"
value = "test_predictions500"

if model == "gnn":
    df = pd.read_csv(f'output/{folder}/{value}.csv')
else:
    df = pd.read_csv(f'output/{folder}/{model}_{value}.csv')

# Map label numbers to actual names
label_names = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4', 4: 'P5', 5: 'P6', 6: 'P7', 7: 'P8', 8: 'P9', 9: 'P10'}
df['true_name'] = df['true_label'].map(label_names)
df['pred_name'] = df['pred_label'].map(label_names)

# Build confusion matrix counts: dict of dicts true_label -> pred_label -> count
confusion = defaultdict(lambda: defaultdict(int))
for true, pred in zip(df['true_name'], df['pred_name']):
    confusion[true][pred] += 1

# Collect summary data
summary_data = []
for true_label, preds in confusion.items():
    total = sum(preds.values())
    correct = preds.get(true_label, 0)
    correct_pct = correct / total if total else 0

    lines = []
    lines.append("===============================================================")
    if correct_pct > 0.9:
        lines.append(f"{true_label} is usually recognized correctly ({correct_pct:.0%} of the time).")
    else:
        lines.append(f"{true_label} is recognized correctly only {correct_pct:.0%} of the time.")

    # Find top misclassifications > 5%
    misclassifications = {k: v for k, v in preds.items() if k != true_label}
    filtered_mis = [(pred_label, count) for pred_label, count in misclassifications.items() if count / total > 0.05]
    if filtered_mis:
        sorted_mis = sorted(filtered_mis, key=lambda x: x[1], reverse=True)
        for pred_label, count in sorted_mis:
            pct = count / total
            lines.append(f"  - Often misrecognized as {pred_label} ({pct:.0%} of the time).")

    summary_data.append((correct_pct, "\n".join(lines)))

# Sort by correct_pct descending
summary_data.sort(reverse=True, key=lambda x: x[0])

# Print summary
for _, text in summary_data:
    print(text)



# DRAW GRAPH TO SHOW ASSOCIATIONS
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

G = nx.DiGraph()

# Add nodes
for label in label_names.values():
    G.add_node(label)

# Add edges weighted by confusion count (only misclassifications)
for true_label, preds in confusion.items():
    total = sum(preds.values())
    for pred_label, count in preds.items():
        if pred_label != true_label and count / total > 0.05:
            G.add_edge(true_label, pred_label, weight=count)

pos = nx.circular_layout(G)
weights = [G[u][v]['weight'] for u,v in G.edges()]

# Normalize weights for color mapping
norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
cmap = cm.Reds

# Map weights to colors
edge_colors = [cmap(norm(w)) for w in weights]

fig, ax = plt.subplots(figsize=(10,10))  # create fig and ax explicitly

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
nx.draw_networkx_edges(
    G, pos,
    width=[w/5 for w in weights],
    edge_color=edge_colors,
    arrowstyle='->',
    arrowsize=15,
    ax=ax
)

ax.set_title('Confusion Network Graph (Edges show common misrecognitions)')
ax.axis('off')

# Create ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Pass the axis to colorbar to place it correctly
cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
cbar.set_label('Confusion Count (Edge Weight)')

plt.savefig(f'output/{folder}/{model}_{value}_confusion_network_colored.png', dpi=300)
plt.show()