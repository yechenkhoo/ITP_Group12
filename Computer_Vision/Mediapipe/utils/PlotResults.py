import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from mediapipe import solutions as mp
from matplotlib.patches import Rectangle

HAND_LANDMARKS = [15, 16, 17, 18, 19, 20, 21, 22]  # left_wrist to right_thumb_2

def plot_predictions_in_batches_from_csv(
    csv_path,
    dataset_dir="./Dataset/",
    output_dir="output/gnn/prediction_grid_batches",
    grid_cols=10,
    batch_size=100,
    hand_landmarks=HAND_LANDMARKS,
    connections=None,
    filter_correct=None,  # True for correct only, False for incorrect only, None for all
    class_filter=None     # Only show GT of this class (0-indexed), or None
):
    df = pd.read_csv(csv_path)
    df["landmarks"] = df["landmarks"].apply(lambda s: list(map(float, s.strip().split())))
    df["correct"] = df["true_label"] == df["pred_label"]

    if filter_correct is not None:
        df = df[df["correct"] == filter_correct]
    if class_filter is not None:
        df = df[df["true_label"] == class_filter]

    if df.empty:
        print("[WARN] No data to plot after filtering.")
        return

    os.makedirs(output_dir, exist_ok=True)
    total = len(df)
    for batch_start in range(0, total, batch_size):
        batch_df = df.iloc[batch_start:batch_start + batch_size]
        n = len(batch_df)
        grid_rows = (n + grid_cols - 1) // grid_cols
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
        axes = axes.flatten()

        for ax in axes[n:]:
            ax.axis('off')

        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = os.path.join(dataset_dir, row["img_path"])
            landmarks = row["landmarks"]
            gt = row["true_label"]
            pred = row["pred_label"]
            conf = row["confidence"]

            ax = axes[i]

            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARN] Could not load image: {img_path}")
                ax.axis('off')
                continue
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            denorm_landmarks = []
            for j in range(0, len(landmarks), 4):
                x = landmarks[j] * w
                y = landmarks[j + 1] * h
                v = landmarks[j + 3]
                denorm_landmarks.append((x, y, v))

            ax.imshow(image_rgb)

            # Plot keypoints
            for j, (x, y, v) in enumerate(denorm_landmarks):
                if v > 0.5:
                    color = 'blue' if j in hand_landmarks else 'red'
                    ax.scatter(x, y, c=color, s=10, alpha=0.8)

            # Plot skeleton connections
            if connections:
                for start, end in connections:
                    x1, y1, v1 = denorm_landmarks[start]
                    x2, y2, v2 = denorm_landmarks[end]
                    if v1 > 0.5 and v2 > 0.5:
                        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.6)

            # Add border around the image
            correct = gt == pred
            border_color = 'green' if correct else 'red'
            # border_width = 3
            
            # # Create a rectangle patch for the border
            # rect = Rectangle((0, 0), w-1, h-1, linewidth=border_width, 
            #                edgecolor=border_color, facecolor='none', alpha=0.8)
            # ax.add_patch(rect)

            # Set title with colored background box
            correct_flag = "✔" if correct else "✘"
            title_text = f"GT:{gt+1} → Pred:{pred+1}\n{correct_flag} | {conf:.2f} | {img_path}"
            title_bg_color = 'lightgreen' if correct else 'lightcoral'
            
            ax.set_title(title_text, fontsize=9, fontweight='bold', pad=8,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=title_bg_color, 
                                alpha=0.8, edgecolor=border_color, linewidth=2))

        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        title = f"Predictions Batch {batch_num}/{total_batches}"
        filename = os.path.join(output_dir, f"batch_{batch_num}_of_{total_batches}.png")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[SAVED] {filename}")


# csv_path="output/gnn/test_predictions.csv"
# output_dir="output/gnn/vis_batches"

csv_path="output/C5_NewDatasetCorrectPoses/CNN_Basic/CNN_Basic_test_predictions.csv"
output_dir="output/C5_NewDatasetCorrectPoses/CNN_Basic/test_vis"

plot_predictions_in_batches_from_csv(
    csv_path=csv_path,
    dataset_dir="./Dataset2/",
    output_dir=output_dir,
    grid_cols=10,
    batch_size=100,
    hand_landmarks=HAND_LANDMARKS,
    connections=list(mp.pose.POSE_CONNECTIONS),
    filter_correct=None,        # or True / False
    class_filter=None           # or 0 for P1, 1 for P2, etc.
)
