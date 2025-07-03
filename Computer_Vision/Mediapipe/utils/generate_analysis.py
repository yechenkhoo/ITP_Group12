import pandas as pd
from collections import defaultdict

model = "CNN_3_block"
folder = f"output/D3_tomtest/{model}"
value = "test_predictions"

if model == "gnn":
    df = pd.read_csv(f'{folder}/{value}.csv')
else:
    df = pd.read_csv(f'{folder}/{model}_{value}.csv')

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
    filtered_mis = [(pred_label, count) for pred_label, count in misclassifications.items() if count / total > 0]
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
    
# DRAW GRAPH TO SHOW ASSOCIATIONS WITH CLEARER DIRECTION
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
        if pred_label != true_label and count > 0:
            G.add_edge(true_label, pred_label, weight=count)

pos = nx.circular_layout(G)
weights = [G[u][v]['weight'] for u, v in G.edges()]

# Use a perceptually uniform colormap and clip the lower end to avoid very light colors
cmap = cm.get_cmap('Reds')
vmin = min(weights)
vmax = max(weights)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# To avoid very light colors, remap the normalized values to [0.3, 1.0] of the colormap
def shifted_color(val):
    normed = norm(val)
    normed = 0.3 + 0.7 * normed  # shift range up
    return cmap(normed)

edge_colors = [shifted_color(w) for w in weights]

fig, ax = plt.subplots(figsize=(10, 10))

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

# Draw edges with arrows and increased arrow size for clearer direction
nx.draw_networkx_edges(
    G, pos,
    width=[w / 5 for w in weights],
    edge_color=edge_colors,
    arrowstyle='-|>',
    arrowsize=22, 
    ax=ax,
    connectionstyle='arc3,rad=0.25',
    min_source_margin=25,
    min_target_margin=25
)

# for (u, v), color in zip(G.edges(), edge_colors):
#     nx.draw_networkx_edges(
#         G, pos,
#         edgelist=[(u, v)],
#         width=[G[u][v]['weight'] / 5],
#         edge_color=edge_colors,
#         arrowstyle='-|>',
#         arrowsize=22,
#         ax=ax,
#         alpha=0.3,
#         connectionstyle='arc3,rad=0.25',
#         min_source_margin=25,
#         min_target_margin=25
#     )

edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10, label_pos=0.6, ax=ax)

ax.set_title('Confusion Network Graph (Directed: True → Predicted)')
ax.axis('off')

# Create ScalarMappable for the colorbar (shifted as above)
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
cbar.set_label('Confusion Count (Edge Weight)')

plt.savefig(f'{folder}/{model}_{value}_confusion_network_colored.png', dpi=300)
plt.show()