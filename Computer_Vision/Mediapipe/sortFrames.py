import pandas as pd
import os
import shutil

INPUT_CSV = "dataset1.csv"
IMAGE_BASE_DIR = "New Dataset"
NIL_OUTPUT_DIR = os.path.join(IMAGE_BASE_DIR, "NIL")
OUTPUT_CSV = "grouped_labeled_dataset.csv"

ANGLE_COLUMNS = ["left_elbow", "right_elbow", "shoulder_tilt", "hip_tilt"]
ANGLE_TOLERANCE = 8.0
MIN_MATCHES_REQUIRED = 2

# Load Dataset
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["Core"] = df["Core"].str.strip().str.lower()
    core_df = df[df["Core"] == "yes"]
    non_core_df = df[df["Core"] != "yes"]
    return core_df, non_core_df

# Frame Comparison
def is_similar_to_any_core(row, core_df):
    pose, group = row["Pose"], row["Group"]
    candidate_cores = core_df[(core_df["Pose"] == pose) & (core_df["Group"] == group)]

    if candidate_cores.empty:
        return "keep"

    for _, core_row in candidate_cores.iterrows():
        match_count = sum(
            abs(row[angle] - core_row[angle]) <= ANGLE_TOLERANCE 
            for angle in ANGLE_COLUMNS
        )
        if match_count >= MIN_MATCHES_REQUIRED:
            return "keep"

    return "NIL"

# Classify Frames
def classify_frames(core_df, non_core_df):
    non_core_df["type"] = non_core_df.apply(lambda row: is_similar_to_any_core(row, core_df), axis=1)
    core_df["type"] = "keep"
    return pd.concat([core_df, non_core_df], ignore_index=True)

# Move NIL frames 
def move_nil_frames(df, base_dir, nil_dir):
    os.makedirs(nil_dir, exist_ok=True)
    moved = 0

    for _, row in df[df["type"] == "NIL"].iterrows():
        filename = row["Frame"]
        if not filename.lower().endswith(".jpg"):
            filename += ".jpg"

        found = False
        for folder in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(subfolder_path) or folder == "NIL":
                continue

            frame_path = os.path.join(subfolder_path, filename)
            if os.path.exists(frame_path):
                shutil.move(frame_path, os.path.join(nil_dir, filename))
                moved += 1
                found = True
                break

        if not found:
            print(f"Frame not found: {filename}")

    return moved

# Save to CSV
def save_results(df, csv_path):
    df.to_csv(csv_path, index=False)
    print(f"Saved labeled dataset to: {csv_path}")
    print(df["type"].value_counts())

def main():
    core_df, non_core_df = load_dataset(INPUT_CSV)
    labeled_df = classify_frames(core_df, non_core_df)
    moved_count = move_nil_frames(labeled_df, IMAGE_BASE_DIR, NIL_OUTPUT_DIR)
    print(f"Moved {moved_count} NIL frames to: {NIL_OUTPUT_DIR}")
    save_results(labeled_df, OUTPUT_CSV)

if __name__ == "__main__":
    main()
